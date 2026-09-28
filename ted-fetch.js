/**
 * TED (Tenders Electronic Daily) EU public tender notices fetcher
 * Official API v3 – Search endpoint
 *
 * - Scope: ACTIVE (still-open / recent active notices)
 * - CPV 37400000 + 33169000 + keywords (football, sports equipment, surgical instruments)
 * - Last 30 days
 * - Deduplicated CPV codes
 * - hasDeadline flag; results.json = with deadline (future only), results-no-deadline.json = without
 * - When multiple lot deadlines exist, store the LATEST; drop notices whose latest deadline is before today
 * - description-proc / description-lot when available (official field names)
 * - Node.js 18+ native fetch, no extra dependencies
 */

'use strict';

const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_URL = 'https://api.ted.europa.eu/v3/notices/search';
const LIMIT = 250;
const SCOPE = 'ACTIVE';
const OUTPUT_WITH = path.join(__dirname, 'results.json');
const OUTPUT_WITHOUT = path.join(__dirname, 'results-no-deadline.json');

const FIELDS = [
  'publication-number',
  'notice-title',
  'buyer-name',
  'buyer-country',
  'classification-cpv',
  'publication-date',
  'deadline-receipt-tender-date-lot',
  'description-proc',
  'description-lot'
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function toTedDate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}${m}${day}`;
}

function todayISO() {
  return toTedDate(new Date()).replace(/(\d{4})(\d{2})(\d{2})/, '$1-$2-$3');
}

function extractLangText(obj) {
  if (!obj || typeof obj !== 'object') return '';
  if (typeof obj.eng === 'string') return obj.eng;
  if (Array.isArray(obj.eng) && obj.eng.length) return obj.eng.join(' | ');
  for (const v of Object.values(obj)) {
    if (typeof v === 'string' && v.trim()) return v;
    if (Array.isArray(v) && v.length) return v.filter(Boolean).join(' | ');
  }
  return '';
}

function extractTitle(titleObj) {
  return extractLangText(titleObj);
}

function extractBuyerName(buyerObj) {
  if (!buyerObj || typeof buyerObj !== 'object') return '';
  const names = [];
  for (const val of Object.values(buyerObj)) {
    if (Array.isArray(val)) names.push(...val);
    else if (typeof val === 'string') names.push(val);
  }
  return [...new Set(names.filter(Boolean))].join(' / ');
}

function dedupeCpv(arr) {
  if (!Array.isArray(arr)) return arr ? String(arr) : '';
  return [...new Set(arr.filter(Boolean))].join(', ');
}

function joinArr(arr) {
  return Array.isArray(arr) ? arr.filter(Boolean).join(', ') : (arr || '');
}

/**
 * Extract the LATEST deadline from one or more date strings.
 * Returns YYYY-MM-DD or empty string.
 */
function latestDeadline(raw) {
  if (!raw) return '';
  const parts = Array.isArray(raw) ? raw : [raw];
  const dates = parts
    .map(s => String(s).split('+')[0].split('T')[0].trim())
    .filter(d => /^\d{4}-\d{2}-\d{2}$/.test(d));
  if (dates.length === 0) return '';
  dates.sort(); // ISO dates sort chronologically
  return dates[dates.length - 1]; // latest
}

function cleanDate(raw) {
  if (!raw) return '';
  const s = Array.isArray(raw) ? raw.filter(Boolean).join(', ') : String(raw);
  return s.split('+')[0].split('T')[0].trim();
}

function buildQuery() {
  const now = new Date();
  const thirtyDaysAgo = new Date(now);
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  const from = toTedDate(thirtyDaysAgo);

  return [
    `publication-date >= ${from}`,
    `AND (`,
    `  classification-cpv IN (37400000 33169000)`,
    `  OR FT=(football OR "sports equipment" OR "surgical instruments")`,
    `)`,
    `SORT BY publication-date DESC`
  ].join(' ');
}

// ---------------------------------------------------------------------------
// API call with pagination
// ---------------------------------------------------------------------------
async function fetchAllNotices() {
  const query = buildQuery();
  console.log('Scope: ACTIVE');
  console.log('Query:', query);
  console.log('Fetching notices from TED API v3…\n');

  const allNotices = [];
  let page = 1;
  let totalCount = null;

  while (true) {
    const body = {
      query,
      fields: FIELDS,
      page,
      limit: LIMIT,
      scope: SCOPE,
      paginationMode: 'PAGE_NUMBER',
      onlyLatestVersions: false,
      checkQuerySyntax: false
    };

    let response;
    try {
      response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(body)
      });
    } catch (err) {
      throw new Error(`Network error on page ${page}: ${err.message}`);
    }

    if (!response.ok) {
      const text = await response.text();
      throw new Error(`HTTP ${response.status} on page ${page}: ${text.slice(0, 500)}`);
    }

    const data = await response.json();

    if (data.timedOut) {
      console.warn('Warning: API reported timedOut=true – results may be incomplete.');
    }

    if (totalCount === null) {
      totalCount = data.totalNoticeCount ?? 0;
      console.log(`Total matching notices reported by API: ${totalCount}`);
    }

    const notices = data.notices || [];
    if (notices.length === 0) break;

    allNotices.push(...notices);
    console.log(`  Page ${page}: retrieved ${notices.length} notices (running total: ${allNotices.length})`);

    if (allNotices.length >= totalCount || notices.length < LIMIT || allNotices.length >= 15000) {
      break;
    }
    page += 1;
  }

  return { notices: allNotices, totalCount };
}

// ---------------------------------------------------------------------------
// Transform
// ---------------------------------------------------------------------------
function transform(notices) {
  const today = todayISO();

  return notices.map(n => {
    const pubNum = n['publication-number'] || '';
    const deadline = latestDeadline(n['deadline-receipt-tender-date-lot']);
    const hasDeadline = Boolean(deadline);

    const descProc = extractLangText(n['description-proc']);
    const descLot = extractLangText(n['description-lot']);
    const description = descProc || descLot || '';

    const row = {
      title: extractTitle(n['notice-title']),
      buyerName: extractBuyerName(n['buyer-name']),
      country: joinArr(n['buyer-country']),
      cpv: dedupeCpv(n['classification-cpv']),
      publicationDate: cleanDate(n['publication-date']),
      deadline,
      hasDeadline,
      noticeUrl: pubNum ? `https://ted.europa.eu/en/notice/-/detail/${pubNum}` : ''
    };

    if (description) {
      row.description = description;
    }

    return row;
  }).filter(row => {
    // Keep notices without deadline (go to no-deadline file)
    if (!row.hasDeadline) return true;
    // Drop notices whose latest deadline is already past
    return row.deadline >= today;
  });
}

// ---------------------------------------------------------------------------
// Console table
// ---------------------------------------------------------------------------
function printTable(rows, label) {
  console.log(`\n=== ${label} (${rows.length}) ===`);
  if (rows.length === 0) {
    console.log('(none)');
    return;
  }

  const display = rows.map(r => ({
    Title: (r.title || '').slice(0, 55) + (r.title?.length > 55 ? '…' : ''),
    Buyer: (r.buyerName || '').slice(0, 30) + (r.buyerName?.length > 30 ? '…' : ''),
    Country: r.country,
    CPV: (r.cpv || '').slice(0, 22),
    Published: r.publicationDate,
    Deadline: r.deadline || '—',
    URL: r.noticeUrl
  }));

  console.table(display);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
async function main() {
  try {
    const { notices, totalCount } = await fetchAllNotices();
    const results = transform(notices);

    const withDeadline = results.filter(r => r.hasDeadline);
    const withoutDeadline = results.filter(r => !r.hasDeadline);

    printTable(withDeadline, 'Notices WITH future deadline (saved to results.json)');
    printTable(withoutDeadline, 'Notices WITHOUT deadline (saved to results-no-deadline.json)');

    console.log('\n---------- SUMMARY ----------');
    console.log(`Total after filtering past deadlines : ${results.length}`);
    if (totalCount !== null) {
      console.log(`(API reported ${totalCount} total matches before local filters)`);
    }
    console.log(`With future deadline                 : ${withDeadline.length}  → results.json`);
    console.log(`Without deadline                     : ${withoutDeadline.length}  → results-no-deadline.json`);

    fs.writeFileSync(OUTPUT_WITH, JSON.stringify(withDeadline, null, 2), 'utf8');
    fs.writeFileSync(OUTPUT_WITHOUT, JSON.stringify(withoutDeadline, null, 2), 'utf8');

    console.log(`\nFiles written:\n  ${OUTPUT_WITH}\n  ${OUTPUT_WITHOUT}`);
  } catch (err) {
    console.error('\nError:', err.message);
    process.exit(1);
  }
}

main();
