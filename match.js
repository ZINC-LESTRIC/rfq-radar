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
 *   3. LLM scores notices from title/description only (no CPV interpretation)
 *   4. Programmatic strong-tier guard: exact/family keyword must appear in text
 *   5. Failed batches → status "failed", not cached; one automatic retry at end
 *   6. scored-all.json + matches.json (tier strong + check)
 *   7. Cache keyed by PROMPT_VERSION
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

// Bump when scoring prompt/rules change so old cache entries are ignored
const PROMPT_VERSION = 'v4-no-cpv-hallucinate-2026-09';

const API_KEY = process.env.LLM_API_KEY;
const MODEL = process.env.LLM_MODEL;
const BASE_URL = resolveBaseUrl(process.env.LLM_BASE_URL, MODEL);

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
  return `${PROMPT_VERSION}||${normalize(productDesc)}||${noticeUrl}`;
}

function assignTier(score) {
  if (typeof score !== 'number') return undefined;
  if (score >= 70) return 'strong';
  if (score >= 40) return 'check';
  return undefined;
}

function flattenTiers(tiers, keys) {
  const all = [];
  for (const key of keys) {
    if (Array.isArray(tiers[key])) all.push(...tiers[key]);
  }
  return [...new Set(all.map(k => String(k).trim()).filter(k => k.length >= 2))];
}

/** True if any exact/family keyword literally appears in title or description */
function hasExactOrFamilyHit(notice, tiers) {
  const haystack = normalize(`${notice.title || ''} ${notice.description || ''}`);
  const kws = flattenTiers(tiers, ['exact', 'family']).map(normalize);
  return kws.some(kw => kw.length >= 2 && haystack.includes(kw));
}

/** Cap score at 69 / tier check unless exact|family keyword is present */
function applyStrongGuard(notice, score, reason, tiers) {
  let finalScore = score;
  let tier = assignTier(finalScore);
  let finalReason = reason;

  if (tier === 'strong' && !hasExactOrFamilyHit(notice, tiers)) {
    finalScore = Math.min(finalScore, 69);
    tier = 'check';
    finalReason = `${reason} [capped: no exact/family keyword in title/description]`;
  }

  return { score: finalScore, reason: finalReason, tier };
}

// ---------------------------------------------------------------------------
// LLM client – Retry-After on 429, exponential backoff
// ---------------------------------------------------------------------------

async function llmChat(messages, { temperature = 0.2, maxTokens = 2048, forceJson = true } = {}) {
  let lastErr;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const body = {
        model: MODEL,
        messages,
        temperature,
        max_tokens: maxTokens
      };
      if (forceJson) {
        body.response_format = { type: 'json_object' };
      }

      const res = await fetch(`${BASE_URL}/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${API_KEY}`
        },
        body: JSON.stringify(body)
      });

      if (res.status === 429 || res.status >= 500) {
        let delay = BASE_DELAY_MS * Math.pow(2, attempt);
        if (res.status === 429) {
          const ra = res.headers.get('retry-after');
          if (ra) {
            const secs = Number(ra);
            delay = !Number.isNaN(secs) && secs > 0 ? Math.max(delay, secs * 1000) : Math.max(delay, 4000);
          } else {
            delay = Math.max(delay, 4000);
          }
        }
        console.warn(`  LLM ${res.status} – retry in ${delay}ms (attempt ${attempt + 1}/${MAX_RETRIES})`);
        await sleep(delay);
        continue;
      }

      // 400 often = json_validate_failed; retry without response_format or with shorter prompt upstream
      if (res.status === 400) {
        const bodyText = await res.text();
        lastErr = new Error(`LLM HTTP 400: ${bodyText.slice(0, 300)}`);
        if (attempt < MAX_RETRIES - 1) {
          const delay = BASE_DELAY_MS * Math.pow(2, attempt);
          console.warn(`  LLM 400 – retry in ${delay}ms (attempt ${attempt + 1}/${MAX_RETRIES})`);
          await sleep(delay);
          // On next attempts, drop strict json_object mode if it caused validate failures
          forceJson = false;
          continue;
        }
        throw lastErr;
      }

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`LLM HTTP ${res.status}: ${text.slice(0, 400)}`);
      }

      const data = await res.json();
      const content = data.choices?.[0]?.message?.content;
      if (!content) throw new Error('Empty LLM response');
      return content;
    } catch (err) {
      lastErr = err;
      if (attempt < MAX_RETRIES - 1 && !String(err.message).includes('HTTP 400')) {
        const delay = BASE_DELAY_MS * Math.pow(2, attempt);
        console.warn(`  LLM error: ${err.message} – retry in ${delay}ms`);
        await sleep(delay);
      } else if (attempt >= MAX_RETRIES - 1) {
        break;
      }
    }
  }
  throw lastErr || new Error('LLM call failed after retries');
}

function parseJsonStrict(text) {
  let cleaned = String(text).trim();
  // Strip markdown fences
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/s, '');
  }
  // Extract outermost JSON object if extra prose slipped in
  const start = cleaned.indexOf('{');
  const end = cleaned.lastIndexOf('}');
  if (start !== -1 && end !== -1 && end > start) {
    cleaned = cleaned.slice(start, end + 1);
  }
  return JSON.parse(cleaned);
}

// ---------------------------------------------------------------------------
// Step 1 – Keyword expansion (robust, flat schema, up to 5 attempts)
// ---------------------------------------------------------------------------

async function expandKeywords(productDesc) {
  console.log('\n[1/4] Expanding product into tiered keywords + CPV prefixes…');

  const shortSystem = `Return ONLY a JSON object with four arrays of short strings:
{"exact":[],"family":[],"category":[],"cpvPrefixes":[]}
exact = product synonyms + stems + translations DE/FR/ES/IT/PL/NL/CS/RO
family = product family terms + translations
category = generic tender wording (sports equipment, Sportgeräte, etc.) + translations
cpvPrefixes = 3-6 four-digit CPV prefixes as strings e.g. "3741"
8-12 terms per keyword array. No markdown.`;

  const longSystem = `You expand a product into search keywords for EU TED tenders.
Reply with ONLY this JSON (no markdown):
{"exact":["..."],"family":["..."],"category":["..."],"cpvPrefixes":["3741"]}
- exact: product + synonyms + single-word stems + DE FR ES IT PL NL CS RO
- family: broader family (ball games, team sports goods) + translations
- category: buyer wording (sports equipment, Sportgeräte, matériels sportifs) + stems
- cpvPrefixes: 3-6 four-digit prefixes only
Aim 8-12 terms per list.`;

  let lastErr;
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const useShort = attempt >= 2;
      const system = useShort ? shortSystem : longSystem;
      const raw = await llmChat([
        { role: 'system', content: system },
        { role: 'user', content: `Product: ${productDesc}` }
      ], {
        temperature: 0.3,
        maxTokens: useShort ? 900 : 1500,
        forceJson: attempt < 3 // drop json mode on later retries
      });

      const parsed = parseJsonStrict(raw);

      for (const key of ['exact', 'family', 'category']) {
        if (!Array.isArray(parsed[key])) parsed[key] = [];
        parsed[key] = parsed[key].map(k => String(k).trim()).filter(k => k.length >= 2);
      }
      if (!Array.isArray(parsed.cpvPrefixes)) parsed.cpvPrefixes = [];
      parsed.cpvPrefixes = parsed.cpvPrefixes
        .map(p => String(p).replace(/\D/g, '').slice(0, 4))
        .filter(p => p.length === 4);

      const totalKw = flattenTiers(parsed, ['exact', 'family', 'category']).length;
      if (totalKw < 5) {
        throw new Error(`Too few keywords (${totalKw})`);
      }

      console.log(`  exact   (${parsed.exact.length}): ${parsed.exact.slice(0, 6).join(', ')}…`);
      console.log(`  family  (${parsed.family.length}): ${parsed.family.slice(0, 6).join(', ')}…`);
      console.log(`  category(${parsed.category.length}): ${parsed.category.slice(0, 6).join(', ')}…`);
      console.log(`  CPV prefixes: ${parsed.cpvPrefixes.join(', ') || '(none)'}`);
      return parsed;
    } catch (err) {
      lastErr = err;
      console.warn(`  Keyword expand attempt ${attempt + 1}/${MAX_RETRIES} failed: ${err.message}`);
      if (attempt < MAX_RETRIES - 1) await sleep(BASE_DELAY_MS * Math.pow(2, attempt));
    }
  }
  throw lastErr || new Error('Keyword expansion failed');
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
  const keywords = flattenTiers(tiers, ['exact', 'family', 'category']).map(normalize);
  const prefixes = tiers.cpvPrefixes || [];

  const matched = notices.filter(n => {
    const haystack = normalize(`${n.title || ''} ${n.description || ''}`);
    if (keywords.some(kw => haystack.includes(kw))) return true;

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
// Scoring prompt – title/description only, no CPV interpretation
// ---------------------------------------------------------------------------

const SCORING_SYSTEM = `You are an expert EU public-procurement advisor helping a Pakistani manufacturer/exporter.

For each tender notice, score 0-100 based ONLY on the title and description text provided.

CRITICAL:
- Do NOT interpret, expand, or reason about CPV codes. Ignore any CPV field if present. CPV numbers in the data are for reference only and must not influence your score or reason.
- Judge solely from the title and description wording.

Scoring rules:
- 75-100: title or description explicitly names the product or a directly matching product line (e.g. footballs, balls, team sports balls, surgical scissors).
- 50-70: generic category wording that plausibly covers the product (e.g. "Sports goods and equipment", "Field and court sports equipment", school sports supplies, "surgical instruments") and does not rule the product out. The product need not be listed by name.
- 25-45: same broad sector but a stretch (playground equipment, fitness-only, therapy, teaching aids, advertising items, clothing/kits only).
- 0-20: construction, services, unrelated goods.

For the "reason" field: quote the exact short phrase from the title or description that supports the score, then add a brief note. If nothing relevant is written, set reason to exactly: no explicit mention

Return ONLY JSON:
{ "scores": [ { "index": 0, "score": 62, "reason": "\"sports equipment\" – generic category match" }, ... ] }`;

// ---------------------------------------------------------------------------
// Score a single batch
// ---------------------------------------------------------------------------

async function scoreBatch(productDesc, batch, cache, keywordTiers) {
  // Intentionally omit cpv from payload so the model cannot hallucinate CPV meanings
  const noticesPayload = batch.map((n, idx) => ({
    index: idx,
    title: n.title || '',
    description: (n.description || '').slice(0, 500),
    country: n.country || '',
    deadline: n.deadline || ''
  }));

  const user = `Product manufactured in Pakistan: ${productDesc}\n\nNotices to score (title + description only):\n${JSON.stringify(noticesPayload, null, 2)}`;

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
      reason: String(s.reason || 'no explicit mention').slice(0, 280)
    });
  }

  if (byIndex.size < Math.ceil(batch.length / 2)) {
    throw new Error(`LLM returned only ${byIndex.size}/${batch.length} scores`);
  }

  const rows = [];
  for (let idx = 0; idx < batch.length; idx++) {
    const notice = batch[idx];
    if (byIndex.has(idx)) {
      const scored = byIndex.get(idx);
      const guarded = applyStrongGuard(notice, scored.score, scored.reason, keywordTiers);

      const key = cacheKey(productDesc, notice.noticeUrl || notice.title);
      cache[key] = {
        score: guarded.score,
        reason: guarded.reason,
        tier: guarded.tier || null,
        promptVersion: PROMPT_VERSION,
        scoredAt: new Date().toISOString()
      };

      const row = {
        ...notice,
        score: guarded.score,
        reason: guarded.reason,
        status: 'scored'
      };
      if (guarded.tier) row.tier = guarded.tier;
      rows.push(row);
    } else {
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

async function scoreNotices(productDesc, notices, cache, keywordTiers) {
  console.log(`\n[3/4] Scoring ${notices.length} notices with LLM (batches of ${BATCH_SIZE}, ${INTER_BATCH_DELAY_MS / 1000}s between batches)…`);
  console.log(`  Prompt version: ${PROMPT_VERSION}`);

  const results = [];
  const toScore = [];
  const failed = [];

  for (const n of notices) {
    const key = cacheKey(productDesc, n.noticeUrl || n.title);
    if (cache[key] && typeof cache[key].score === 'number') {
      // Re-apply strong guard in case tiers changed relative to cached score
      const guarded = applyStrongGuard(n, cache[key].score, cache[key].reason, keywordTiers);
      const row = {
        ...n,
        score: guarded.score,
        reason: guarded.reason,
        status: 'scored'
      };
      if (guarded.tier) row.tier = guarded.tier;
      results.push(row);
    } else {
      toScore.push(n);
    }
  }

  console.log(`  Cached (this prompt version): ${results.length} | Need scoring: ${toScore.length}`);

  async function runBatches(list, label) {
    for (let i = 0; i < list.length; i += BATCH_SIZE) {
      const batch = list.slice(i, i + BATCH_SIZE);
      const batchNum = Math.floor(i / BATCH_SIZE) + 1;
      const totalBatches = Math.ceil(list.length / BATCH_SIZE);
      console.log(`  ${label} batch ${batchNum}/${totalBatches} (${batch.length} notices)…`);

      try {
        const rows = await scoreBatch(productDesc, batch, cache, keywordTiers);
        for (const row of rows) {
          if (row.status === 'scored') results.push(row);
          else failed.push(row);
        }
      } catch (err) {
        console.error(`  Batch ${batchNum} failed after retries: ${err.message}`);
        for (const notice of batch) {
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

  if (failed.length > 0) {
    console.log(`\n  Retrying ${failed.length} failed notices once…`);
    await sleep(INTER_BATCH_DELAY_MS);
    const retryList = failed.map(n => {
      const { score, reason, status, tier, ...rest } = n;
      return rest;
    });
    failed.length = 0;
    await runBatches(retryList, 'Retry');
  }

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

  const sorted = [...scored].sort((a, b) => {
    if (a.status !== 'scored' && b.status === 'scored') return 1;
    if (a.status === 'scored' && b.status !== 'scored') return -1;
    return (b.score || 0) - (a.score || 0);
  });

  const top15 = [...scoredOk].sort((a, b) => b.score - a.score).slice(0, 15);
  console.log(`\n=== TOP 15 SCORES (of ${scoredOk.length} successfully scored) ===`);
  if (top15.length === 0) {
    console.log('(none scored successfully)');
  } else {
    console.table(top15.map(n => ({
      Score: n.score,
      Tier: n.tier || '—',
      Title: (n.title || '').slice(0, 45) + (n.title?.length > 45 ? '…' : ''),
      Country: n.country,
      Deadline: n.deadline || '—',
      Reason: (n.reason || '').slice(0, 50),
      URL: n.noticeUrl
    })));
  }

  const strong = scoredOk.filter(n => n.tier === 'strong').sort((a, b) => b.score - a.score);
  const check = scoredOk.filter(n => n.tier === 'check').sort((a, b) => b.score - a.score);
  const matches = [...strong, ...check];

  console.log(`\n=== STRONG (score >= 70 + exact/family keyword hit) — ${strong.length} ===`);
  if (strong.length === 0) {
    console.log('(none)');
  } else {
    console.table(strong.map(n => ({
      Score: n.score,
      Title: (n.title || '').slice(0, 48) + (n.title?.length > 48 ? '…' : ''),
      Country: n.country,
      Deadline: n.deadline || '—',
      Reason: (n.reason || '').slice(0, 55),
      URL: n.noticeUrl
    })));
  }

  console.log(`\n=== CHECK (score 40-69) — ${check.length} ===`);
  if (check.length === 0) {
    console.log('(none)');
  } else {
    console.table(check.map(n => ({
      Score: n.score,
      Title: (n.title || '').slice(0, 48) + (n.title?.length > 48 ? '…' : ''),
      Country: n.country,
      Deadline: n.deadline || '—',
      Reason: (n.reason || '').slice(0, 55),
      URL: n.noticeUrl
    })));
  }

  const toExport = (rows) => rows.map(n => {
    const out = {
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
    };
    if (n.tier) out.tier = n.tier;
    return out;
  });

  saveJson(SCORED_ALL_FILE, toExport(sorted));
  saveJson(MATCHES_FILE, toExport(matches));

  console.log(`\n---------- SUMMARY ----------`);
  console.log(`Prompt version              : ${PROMPT_VERSION}`);
  console.log(`Notices sent to scoring     : ${scored.length}`);
  console.log(`Successfully scored         : ${scoredOk.length}`);
  console.log(`Failed (status=failed)      : ${failedCount}`);
  console.log(`Strong (tier)               : ${strong.length}`);
  console.log(`Check  (tier)               : ${check.length}`);
  console.log(`matches.json total          : ${matches.length}`);
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
  console.log(`Prompt  : ${PROMPT_VERSION}`);

  const notices = loadJson(RESULTS_FILE, []);
  if (!Array.isArray(notices) || notices.length === 0) {
    console.error('results.json is empty. Run ted-fetch.js first.');
    process.exit(1);
  }
  console.log(`Loaded ${notices.length} notices from results.json`);

  const cache = loadJson(CACHE_FILE, {});

  const keywordTiers = await expandKeywords(product);
  const filtered = filterNotices(notices, keywordTiers);

  if (filtered.length === 0) {
    console.log('\nNo notices to score. Saving empty outputs.');
    saveJson(MATCHES_FILE, []);
    saveJson(SCORED_ALL_FILE, []);
    return;
  }

  console.log(`\n→ ${filtered.length} notices will be sent to LLM scoring`);

  const scored = await scoreNotices(product, filtered, cache, keywordTiers);
  outputResults(scored);
}

main().catch(err => {
  console.error('\nFatal:', err.message);
  process.exit(1);
});
