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
 * Features:
 *   1. LLM expands product into 15-25 keywords (synonyms + DE/FR/ES/IT/PL/NL)
 *   2. Keyword-filter notices (title + description, accent-insensitive)
 *   3. LLM scores relevance 0-100 in batches of 10 (Pakistani exporter lens)
 *   4. Output score >= 60, sorted, console table + matches.json
 *   5. Cache scores in cache.json (never re-score same notice+product)
 *   6. Retry + exponential backoff on rate limits / transient errors
 */

'use strict';

const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const RESULTS_FILE = path.join(__dirname, 'results.json');
const MATCHES_FILE = path.join(__dirname, 'matches.json');
const CACHE_FILE = path.join(__dirname, 'cache.json');

const API_KEY = process.env.LLM_API_KEY;
const MODEL = process.env.LLM_MODEL;
const BASE_URL = resolveBaseUrl(process.env.LLM_BASE_URL, MODEL);

const SCORE_THRESHOLD = 60;
const BATCH_SIZE = 10;
const MAX_RETRIES = 5;
const BASE_DELAY_MS = 1500;

// ---------------------------------------------------------------------------
// Bootstrap checks
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
  // Default to Groq (OpenAI-compatible)
  return 'https://api.groq.com/openai/v1';
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms));
}

/** Strip diacritics / accents for case-insensitive matching */
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

// ---------------------------------------------------------------------------
// LLM client with retry + backoff
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
        const delay = BASE_DELAY_MS * Math.pow(2, attempt);
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
  // Strip possible markdown fences
  let cleaned = text.trim();
  if (cleaned.startsWith('```')) {
    cleaned = cleaned.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '');
  }
  const parsed = JSON.parse(cleaned);
  return parsed;
}

// ---------------------------------------------------------------------------
// Step 1 – Expand product into keywords
// ---------------------------------------------------------------------------

async function expandKeywords(productDesc) {
  console.log('\n[1/4] Expanding product into multilingual keywords…');

  const system = `You are a procurement keyword expert helping a Pakistani exporter find EU public tenders.
Given a product description, return a JSON object with a single key "keywords" whose value is an array of 15-25 short search terms.
Include:
- English synonyms and related product terms
- Common commercial names
- Translations / equivalents in German, French, Spanish, Italian, Polish, and Dutch
Return ONLY valid JSON. No commentary.`;

  const user = `Product: ${productDesc}`;

  const raw = await llmChat([
    { role: 'system', content: system },
    { role: 'user', content: user }
  ], { temperature: 0.3, maxTokens: 1024 });

  const parsed = parseJsonStrict(raw);
  if (!parsed || !Array.isArray(parsed.keywords) || parsed.keywords.length < 5) {
    throw new Error(`Invalid keywords JSON from LLM: ${raw.slice(0, 300)}`);
  }

  const keywords = [...new Set(parsed.keywords.map(k => String(k).trim()).filter(Boolean))];
  console.log(`  → ${keywords.length} keywords: ${keywords.slice(0, 8).join(', ')}…`);
  return keywords;
}

// ---------------------------------------------------------------------------
// Step 2 – Keyword filter (title + description)
// ---------------------------------------------------------------------------

function keywordFilter(notices, keywords) {
  console.log('\n[2/4] Keyword-filtering notices (title + description)…');
  const norms = keywords.map(normalize);

  const matched = notices.filter(n => {
    const haystack = normalize(`${n.title || ''} ${n.description || ''}`);
    return norms.some(kw => kw.length >= 2 && haystack.includes(kw));
  });

  console.log(`  → ${matched.length} of ${notices.length} notices matched keywords`);
  return matched;
}

// ---------------------------------------------------------------------------
// Step 3 – LLM relevance scoring (batches of 10) with cache
// ---------------------------------------------------------------------------

async function scoreNotices(productDesc, notices, cache) {
  console.log('\n[3/4] Scoring relevance with LLM (batches of 10)…');

  const results = [];
  const toScore = [];

  for (const n of notices) {
    const key = cacheKey(productDesc, n.noticeUrl || n.title);
    if (cache[key] && typeof cache[key].score === 'number') {
      results.push({ ...n, score: cache[key].score, reason: cache[key].reason });
    } else {
      toScore.push(n);
    }
  }

  console.log(`  Cached: ${results.length} | Need scoring: ${toScore.length}`);

  for (let i = 0; i < toScore.length; i += BATCH_SIZE) {
    const batch = toScore.slice(i, i + BATCH_SIZE);
    console.log(`  Scoring batch ${Math.floor(i / BATCH_SIZE) + 1}/${Math.ceil(toScore.length / BATCH_SIZE)} (${batch.length} notices)…`);

    const system = `You are an expert EU public-procurement advisor helping a Pakistani exporter.
For each tender notice, judge whether the exporter of the given product could realistically bid.
Consider: product fit, language of the notice, typical buyer needs, and that the exporter ships from Pakistan.
Return ONLY a JSON object: { "scores": [ { "index": 0, "score": 0-100, "reason": "one short sentence" }, ... ] }
Score 0 = completely irrelevant, 100 = perfect match. Be strict but fair.`;

    const noticesPayload = batch.map((n, idx) => ({
      index: idx,
      title: n.title || '',
      description: (n.description || '').slice(0, 400),
      country: n.country || '',
      cpv: n.cpv || '',
      deadline: n.deadline || ''
    }));

    const user = `Product: ${productDesc}\n\nNotices:\n${JSON.stringify(noticesPayload, null, 2)}`;

    try {
      const raw = await llmChat([
        { role: 'system', content: system },
        { role: 'user', content: user }
      ], { temperature: 0.1, maxTokens: 2048 });

      const parsed = parseJsonStrict(raw);
      const scores = Array.isArray(parsed.scores) ? parsed.scores : [];

      for (const s of scores) {
        const idx = Number(s.index);
        if (idx < 0 || idx >= batch.length) continue;
        const notice = batch[idx];
        const score = Math.max(0, Math.min(100, Number(s.score) || 0));
        const reason = String(s.reason || '').slice(0, 200);

        const key = cacheKey(productDesc, notice.noticeUrl || notice.title);
        cache[key] = { score, reason, scoredAt: new Date().toISOString() };
        results.push({ ...notice, score, reason });
      }

      // Persist cache after each batch
      saveJson(CACHE_FILE, cache);
    } catch (err) {
      console.error(`  Batch failed: ${err.message}`);
      // Continue with remaining batches
    }

    // Small pause between batches to respect rate limits
    if (i + BATCH_SIZE < toScore.length) await sleep(800);
  }

  return results;
}

// ---------------------------------------------------------------------------
// Step 4 – Output
// ---------------------------------------------------------------------------

function outputMatches(scored) {
  console.log('\n[4/4] Filtering score >= 60 and writing matches.json…');

  const top = scored
    .filter(n => n.score >= SCORE_THRESHOLD)
    .sort((a, b) => b.score - a.score);

  const display = top.map(n => ({
    Score: n.score,
    Title: (n.title || '').slice(0, 50) + (n.title?.length > 50 ? '…' : ''),
    Country: n.country,
    Deadline: n.deadline || '—',
    Reason: (n.reason || '').slice(0, 60),
    URL: n.noticeUrl
  }));

  if (display.length === 0) {
    console.log('\nNo notices scored 60 or above.');
  } else {
    console.log(`\n=== MATCHES (score >= ${SCORE_THRESHOLD}) — ${top.length} ===`);
    console.table(display);
  }

  const exportRows = top.map(n => ({
    title: n.title,
    country: n.country,
    deadline: n.deadline,
    score: n.score,
    reason: n.reason,
    url: n.noticeUrl,
    cpv: n.cpv,
    buyerName: n.buyerName,
    description: n.description || ''
  }));

  saveJson(MATCHES_FILE, exportRows);
  console.log(`\nSaved ${exportRows.length} matches → ${MATCHES_FILE}`);
  return exportRows;
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

  const keywords = await expandKeywords(product);
  const filtered = keywordFilter(notices, keywords);

  if (filtered.length === 0) {
    console.log('\nNo keyword matches. Saving empty matches.json.');
    saveJson(MATCHES_FILE, []);
    return;
  }

  const scored = await scoreNotices(product, filtered, cache);
  outputMatches(scored);
}

main().catch(err => {
  console.error('\nFatal:', err.message);
  process.exit(1);
});
