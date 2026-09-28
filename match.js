/**
 * match.js – Match TED notices in results.json to an exporter product description
 *
 * Usage:
 *   node match.js "leather football size 5 hand stitched"
 *
 * Environment:
 *   LLM_API_KEY   – required (Groq or xAI key)
 *   LLM_MODEL     – required (e.g. llama-3.3-70b-versatile or grok-4.1-fast)
 *   LLM_BASE_URL  – optional; auto-detected from model name if omitted
 *
 * Pipeline:
 *   1. LLM expands product → { exact[], family[], category[], cpvPrefixes[] }
 *   2. Keyword + CPV filter (skipped entirely if < 500 notices loaded)
 *   3. LLM scores notices 0-100 in batches of 8 (4s between batches)
 *   4. Failed batches → status "failed", not cached; one automatic retry pass at end
 *   5. scored-all.json = all (status scored|failed); matches.json = score >= 60
 *   6. Cache only successfully scored notices
 */

'use strict';

const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const RESULTS_FILE = path.join(__dirname, 'results.json');
const MATCHES_FILE = path.join(__dirname, 'matches.json');
const SCORED_ALL_FILE = path.join(__dirname, 'scored-all.json');
const CACHE_FILE = path.join(__dirname, 'cache.json');

const API_KEY = process.env.LLM_API_KEY;
const MODEL = process.env.LLM_MODEL;
const BASE_URL = resolveBaseUrl(process.env.LLM_BASE_URL, MODEL);

const SCORE_THRESHOLD = 60;
const BATCH_SIZE = 8;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1500;
const INTER_BATCH_DELAY_MS = 4000;
const SKIP_FILTER_BELOW = 500;

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
if (!API_KEY) {
  console.error('Error: LLM_API_KEY environment variable is required.');
  process.exit(1);
}
if (!MODEL) {
  console.error('Error: LLM_MODEL environment variable is required.');
  process.exit(1);
}

const product = process.argv.slice(2).join(' ').trim();
if (!product) {
  console.error('Usage: node match.js "<product description>"');
  process.exit(1);
}

if (!fs.existsSync(RESULTS_FILE)) {
  console.error(`Error: ${RESULTS_FILE} not found. Run ted-fetch.js first.`);
  process.exit(1);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function resolveBaseUrl(explicit, model) {
  if (explicit) return explicit.replace(/\/$/, '');
  const m = (model || '').toLowerCase();
  if (m.startsWith('grok') || m.includes('xai')) return 'https://api.x.ai/v1';
  return 'https://api.groq.com/openai/v1';
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

function normalize(str) {
  return String(str || '')
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase();
}

function loadJson(file, fallback) {
  try {
    if (fs.existsSync(file)) return JSON.parse(fs.readFileSync(file, 'utf8'));
  } catch (_) { /* ignore */ }
  return fallback;
}

function saveJson(file, data) {
  fs.writeFileSync(file, JSON.stringify(data, null, 2), 'utf8');
}

function cacheKey(productDesc, noticeUrl) {
  return `${normalize(productDesc)}||${noticeUrl}`;
}

function flattenTiers(tiers) {
  const all = [];
  for (const key of ['exact', 'family', 'category']) {
    if (Array.isArray(tiers[key])) all.push(...tiers[key]);
  }
  return [...new Set(all.map(k => String(k).trim()).filter(k => k.length >= 2))];
}

// ---------------------------------------------------------------------------
// LLM client – Retry-After on 429, exponential backoff otherwise
// ---------------------------------------------------------------------------

async function llmChat(messages, { temperature = 0.2, maxTokens = 2048 } = {}) {
  let lastErr;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const res = await fetch(`${BASE_URL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${API_KEY}`
        },
        body: JSON.stringify({
          model: MODEL,
          messages,
          temperature,
          max_tokens: maxTokens,
          response_format: { type: 'json_object' }
        })
      });

      if (res.status === 429 || res.status >= 500) {
        let delay = BASE_DELAY_MS * Math.pow(2, attempt);
        if (res.status === 429) {
          const ra = res.headers.get('retry-after');
          if (ra) {
            const secs = Number(ra);
            if (!Number.isNaN(secs) && secs > 0) {
              delay = Math.max(delay, secs * 1000);
            } else {
              // HTTP-date form – fall back to at least 4s
              delay = Math.max(delay, 4000);
            }
          } else {
            delay = Math.max(delay, 4000);
          }
        }
        console.warn(`  LLM ${res.status} – retry in ${delay}ms (attempt ${attempt + 1}/${MAX_RETRIES})`);
        await sleep(delay);
        continue;
      }

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`LLM HTTP ${res.status}: ${body.slice(0, 400)}`);
      }

      const data = await res.json();
      const content = data.choices?.[0]?.message?.content;
      if (!content) throw new Error('Empty LLM response');
      return content;
    } catch (err) {
      lastErr = err;
      // Don't retry non-HTTP parse errors beyond the loop; still backoff
      if (attempt < MAX_RETRIES - 1) {
        const delay = BASE_DELAY_MS * Math.pow(2, attempt);
        console.warn(`  LLM error: ${err.message} – retry in ${delay}ms`);
        await sleep(delay);
      }
    }
  }
  throw lastErr || new Error('LLM call failed after retries');
}

function parseJsonStrict(text) {
  let cleaned = text.trim();
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  }
  return JSON.parse(cleaned);
}

// ---------------------------------------------------------------------------
// Step 1 – Tiered keyword + CPV expansion
// ---------------------------------------------------------------------------

async function expandKeywords(productDesc) {
  console.log('\n[1/4] Expanding product into tiered keywords + CPV prefixes…');

  const system = `You are a procurement keyword expert helping a Pakistani exporter find EU public tenders on TED.

Given a product description, return ONLY valid JSON with this exact shape:
{
  "exact": ["..."],
  "family": ["..."],
  "category": ["..."],
  "cpvPrefixes": ["3741", "3742"]
}

Rules for keywords:
- exact: the product itself, close synonyms, size/material variants (English + DE/FR/ES/IT/PL/NL/CS/RO). Include single-word stems (e.g. "football", "ball", "leather").
- family: broader product family (e.g. ball games, team sports goods, sports balls). Same languages + stems.
- category: generic buyer/tender wording (e.g. sports equipment, Sportgeräte, matériels sportifs, gym equipment, school sports supplies, sporting goods). Same languages + short stems like "sport", "equip".
- Aim for 8-15 terms per tier. Mix phrases and single words. No duplicates across the whole response if possible.

Rules for cpvPrefixes:
- 3 to 6 likely CPV code PREFIXES (first 4 digits only, as strings).
- Examples: sports goods → "3741","3742"; surgical instruments → "3316","3314"; clothing → "1821","1831".

Return ONLY the JSON object. No markdown, no commentary.`;

  const user = `Product: ${productDesc}`;

  const raw = await llmChat([
    { role: 'system', content: system },
    { role: 'user', content: user }
  ], { temperature: 0.35, maxTokens: 1500 });

  const parsed = parseJsonStrict(raw);

  for (const key of ['exact', 'family', 'category']) {
    if (!Array.isArray(parsed[key])) parsed[key] = [];
    parsed[key] = parsed[key].map(k => String(k).trim()).filter(k => k.length >= 2);
  }
  if (!Array.isArray(parsed.cpvPrefixes)) parsed.cpvPrefixes = [];
  parsed.cpvPrefixes = parsed.cpvPrefixes
    .map(p => String(p).replace(/\D/g, '').slice(0, 4))
    .filter(p => p.length === 4);

  const totalKw = flattenTiers(parsed).length;
  if (totalKw < 5) {
    throw new Error(`Too few keywords from LLM (${totalKw}): ${raw.slice(0, 400)}`);
  }

  console.log(`  exact   (${parsed.exact.length}): ${parsed.exact.slice(0, 6).join(', ')}…`);
  console.log(`  family  (${parsed.family.length}): ${parsed.family.slice(0, 6).join(', ')}…`);
  console.log(`  category(${parsed.category.length}): ${parsed.category.slice(0, 6).join(', ')}…`);
  console.log(`  CPV prefixes: ${parsed.cpvPrefixes.join(', ') || '(none)'}`);

  return parsed;
}

// ---------------------------------------------------------------------------
// Step 2 – Filter (keywords any tier OR CPV prefix). Skip if < 500 notices.
// ---------------------------------------------------------------------------

function filterNotices(notices, tiers) {
  if (notices.length < SKIP_FILTER_BELOW) {
    console.log(`\n[2/4] Only ${notices.length} notices loaded (< ${SKIP_FILTER_BELOW}) → SKIP keyword filter, score ALL.`);
    return notices;
  }

  console.log('\n[2/4] Filtering by tiered keywords + CPV prefixes…');
  const keywords = flattenTiers(tiers).map(normalize);
  const prefixes = tiers.cpvPrefixes || [];

  const matched = notices.filter(n => {
    const haystack = normalize(`${n.title || ''} ${n.description || ''}`);
    const kwHit = keywords.some(kw => haystack.includes(kw));
    if (kwHit) return true;

    if (prefixes.length && n.cpv) {
      const codes = String(n.cpv).split(/[,\s]+/).map(c => c.replace(/\D/g, ''));
      if (codes.some(code => prefixes.some(p => code.startsWith(p)))) return true;
    }
    return false;
  });

  console.log(`  → ${matched.length} of ${notices.length} notices passed filter`);
  return matched;
}

// ---------------------------------------------------------------------------
// Scoring prompt (tightened)
// ---------------------------------------------------------------------------

const SCORING_SYSTEM = `You are an expert EU public-procurement advisor helping a Pakistani manufacturer/exporter.

For each tender notice, score 0-100: how well does this tender match the exact product the manufacturer makes?

Scoring rules (follow strictly):
- 75-100: ONLY if the tender text explicitly covers the exact product or a clearly matching product line (e.g. footballs, match balls, team sports balls named in the title/description).
- 55-70: generic category tenders that plausibly include the product (e.g. "sports equipment", "Sportgeräte", "matériels sportifs") BUT only when the category is the right one. Do NOT give 60+ just because something "could include" the product — the tender text itself must list sports balls, team sports equipment, or an equivalent clear product line for 60+.
- Below 40: playground equipment, construction works, teaching aids, advertising items, furniture, pure services, IT, medical devices unrelated to the product, or any stretch interpretation.
- 40-54: weak / tangential only.

Return ONLY JSON:
{ "scores": [ { "index": 0, "score": 65, "reason": "one short sentence" }, ... ] }`;

// ---------------------------------------------------------------------------
// Score a single batch → { ok: true, rows } | { ok: false, error }
// Cache only on success.
// ---------------------------------------------------------------------------

async function scoreBatch(productDesc, batch, cache) {
  const noticesPayload = batch.map((n, idx) => ({
    index: idx,
    title: n.title || '',
    description: (n.description || '').slice(0, 500),
    country: n.country || '',
    cpv: n.cpv || '',
    deadline: n.deadline || ''
  }));

  const user = `Product manufactured in Pakistan: ${productDesc}\n\nNotices to score:\n${JSON.stringify(noticesPayload, null, 2)}`;

  const raw = await llmChat([
    { role: 'system', content: SCORING_SYSTEM },
    { role: 'user', content: user }
  ], { temperature: 0.15, maxTokens: 2048 });

  const parsed = parseJsonStrict(raw);
  const scores = Array.isArray(parsed.scores) ? parsed.scores : [];

  const byIndex = new Map();
  for (const s of scores) {
    const idx = Number(s.index);
    if (Number.isNaN(idx) || idx < 0 || idx >= batch.length) continue;
    byIndex.set(idx, {
      score: Math.max(0, Math.min(100, Number(s.score) || 0)),
      reason: String(s.reason || '').slice(0, 220)
    });
  }

  // Require at least half the batch scored, otherwise treat as failure
  if (byIndex.size < Math.ceil(batch.length / 2)) {
    throw new Error(`LLM returned only ${byIndex.size}/${batch.length} scores`);
  }

  const rows = [];
  for (let idx = 0; idx < batch.length; idx++) {
    const notice = batch[idx];
    if (byIndex.has(idx)) {
      const scored = byIndex.get(idx);
      const key = cacheKey(productDesc, notice.noticeUrl || notice.title);
      cache[key] = {
        score: scored.score,
        reason: scored.reason,
        scoredAt: new Date().toISOString()
      };
      rows.push({
        ...notice,
        score: scored.score,
        reason: scored.reason,
        status: 'scored'
      });
    } else {
      // Missing index → mark failed (do not cache)
      rows.push({
        ...notice,
        score: null,
        reason: 'LLM did not return a score for this notice',
        status: 'failed'
      });
    }
  }

  saveJson(CACHE_FILE, cache);
  return rows;
}

// ---------------------------------------------------------------------------
// Step 3 – Score all notices; retry failures once at the end
// ---------------------------------------------------------------------------

async function scoreNotices(productDesc, notices, cache) {
  console.log(`\n[3/4] Scoring ${notices.length} notices with LLM (batches of ${BATCH_SIZE}, ${INTER_BATCH_DELAY_MS / 1000}s between batches)…`);

  const results = [];
  const toScore = [];
  const failed = [];

  for (const n of notices) {
    const key = cacheKey(productDesc, n.noticeUrl || n.title);
    if (cache[key] && typeof cache[key].score === 'number') {
      results.push({
        ...n,
        score: cache[key].score,
        reason: cache[key].reason,
        status: 'scored'
      });
    } else {
      toScore.push(n);
    }
  }

  console.log(`  Cached (successful only): ${results.length} | Need scoring: ${toScore.length}`);

  async function runBatches(list, label) {
    for (let i = 0; i < list.length; i += BATCH_SIZE) {
      const batch = list.slice(i, i + BATCH_SIZE);
      const batchNum = Math.floor(i / BATCH_SIZE) + 1;
      const totalBatches = Math.ceil(list.length / BATCH_SIZE);
      console.log(`  ${label} batch ${batchNum}/${totalBatches} (${batch.length} notices)…`);

      try {
        const rows = await scoreBatch(productDesc, batch, cache);
        for (const row of rows) {
          if (row.status === 'scored') {
            results.push(row);
          } else {
            failed.push(row);
          }
        }
      } catch (err) {
        console.error(`  Batch ${batchNum} failed after retries: ${err.message}`);
        for (const notice of batch) {
          // status failed, score null — do NOT cache
          failed.push({
            ...notice,
            score: null,
            reason: `Scoring failed: ${err.message.slice(0, 120)}`,
            status: 'failed'
          });
        }
      }

      if (i + BATCH_SIZE < list.length) {
        console.log(`  Waiting ${INTER_BATCH_DELAY_MS / 1000}s before next batch…`);
        await sleep(INTER_BATCH_DELAY_MS);
      }
    }
  }

  await runBatches(toScore, 'Scoring');

  // One automatic retry pass for all failed notices
  if (failed.length > 0) {
    console.log(`\n  Retrying ${failed.length} failed notices once…`);
    await sleep(INTER_BATCH_DELAY_MS);
    const retryList = failed.map(n => {
      // strip previous failure fields before re-scoring
      const { score, reason, status, ...rest } = n;
      return rest;
    });
    failed.length = 0;
    await runBatches(retryList, 'Retry');
  }

  // Anything still in failed stays failed
  results.push(...failed);
  return results;
}

// ---------------------------------------------------------------------------
// Step 4 – Output
// ---------------------------------------------------------------------------

function outputResults(scored) {
  console.log('\n[4/4] Writing scored-all.json and matches.json…');

  const scoredOk = scored.filter(n => n.status === 'scored');
  const failedCount = scored.filter(n => n.status === 'failed').length;

  // Sort scored first by score desc, failed at the end
  const sorted = [...scored].sort((a, b) => {
    if (a.status !== 'scored' && b.status === 'scored') return 1;
    if (a.status === 'scored' && b.status !== 'scored') return -1;
    return (b.score || 0) - (a.score || 0);
  });

  const top15 = scoredOk.sort((a, b) => b.score - a.score).slice(0, 15);
  console.log(`\n=== TOP 15 SCORES (of ${scoredOk.length} successfully scored) ===`);
  if (top15.length === 0) {
    console.log('(none scored successfully)');
  } else {
    console.table(top15.map(n => ({
      Score: n.score,
      Title: (n.title || '').slice(0, 48) + (n.title?.length > 48 ? '…' : ''),
      Country: n.country,
      Deadline: n.deadline || '—',
      Reason: (n.reason || '').slice(0, 55),
      URL: n.noticeUrl
    })));
  }

  const matches = scoredOk.filter(n => n.score >= SCORE_THRESHOLD);
  console.log(`\n=== MATCHES (score >= ${SCORE_THRESHOLD}) — ${matches.length} ===`);
  if (matches.length === 0) {
    console.log('(none above threshold)');
  } else {
    console.table(matches.map(n => ({
      Score: n.score,
      Title: (n.title || '').slice(0, 48) + (n.title?.length > 48 ? '…' : ''),
      Country: n.country,
      Deadline: n.deadline || '—',
      Reason: (n.reason || '').slice(0, 55),
      URL: n.noticeUrl
    })));
  }

  const toExport = (rows) => rows.map(n => ({
    title: n.title,
    country: n.country,
    deadline: n.deadline,
    score: n.score,
    reason: n.reason,
    status: n.status || 'scored',
    url: n.noticeUrl,
    cpv: n.cpv,
    buyerName: n.buyerName,
    description: n.description || ''
  }));

  saveJson(SCORED_ALL_FILE, toExport(sorted));
  saveJson(MATCHES_FILE, toExport(matches));

  console.log(`\n---------- SUMMARY ----------`);
  console.log(`Notices sent to scoring     : ${scored.length}`);
  console.log(`Successfully scored         : ${scoredOk.length}`);
  console.log(`Failed (status=failed)      : ${failedCount}`);
  console.log(`Score >= ${SCORE_THRESHOLD} (matches)      : ${matches.length}  → matches.json`);
  console.log(`All results                 : ${sorted.length}  → scored-all.json`);

  return matches;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

async function main() {
  console.log(`Product : "${product}"`);
  console.log(`Model   : ${MODEL}`);
  console.log(`Base URL: ${BASE_URL}`);

  const notices = loadJson(RESULTS_FILE, []);
  if (!Array.isArray(notices) || notices.length === 0) {
    console.error('results.json is empty. Run ted-fetch.js first.');
    process.exit(1);
  }
  console.log(`Loaded ${notices.length} notices from results.json`);

  const cache = loadJson(CACHE_FILE, {});

  const tiers = await expandKeywords(product);
  const filtered = filterNotices(notices, tiers);

  if (filtered.length === 0) {
    console.log('\nNo notices to score. Saving empty outputs.');
    saveJson(MATCHES_FILE, []);
    saveJson(SCORED_ALL_FILE, []);
    return;
  }

  console.log(`\n→ ${filtered.length} notices will be sent to LLM scoring`);

  const scored = await scoreNotices(product, filtered, cache);
  outputResults(scored);
}

main().catch(err => {
  console.error('\nFatal:', err.message);
  process.exit(1);
});
