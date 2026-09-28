/**
 * TED (Tenders Electronic Daily) EU public tender notices fetcher
 * Official API v3 – Search endpoint
 *
 * Requirements covered:
 * - Last 30 days
 * - CPV 37400000 (sports goods/equipment) + 33169000 (surgical instruments)
 * - Keywords: football, sports equipment, surgical instruments
 * - Extracts: title, buyer name, country, CPV, publication date, deadline, notice URL
 * - Console table + results.json
 * - Node.js 18+ (native fetch), error handling, comments
 */

'use strict';

const fs = require('fs');
const path = require('path');

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
const API_URL = 'https://api.ted.europa.eu/v3/notices/search';
const LIMIT = 250;               // max allowed per page
const SCOPE = 'ALL';             // LATEST | ACTIVE | ALL
const OUTPUT_FILE = path.join(__dirname, 'results.json');

// Fields requested from the API (must be non-empty; official names only)
const FIELDS = [
  'publication-number',
  'notice-title',
  'buyer-name',
  'buyer-country',
  'classification-cpv',
  'publication-date',
  'deadline-receipt-tender-date-lot'
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Returns YYYYMMDD for a Date object */
function toTedDate(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, '0');
  const day = String(d.getDate()).padStart(2, '0');
  return `${y}${m}${day}`;
}

/** Prefer English title, fall back to first available language */
function extractTitle(titleObj) {
  if (!titleObj || typeof titleObj !== 'object') return '';
  if (titleObj.eng) return titleObj.eng;
  const first = Object.values(titleObj)[0];
  return typeof first === 'string' ? first : '';
}

/** Flatten multilingual buyer-name object into a single string */
function extractBuyerName(buyerObj) {
  if (!buyerObj || typeof buyerObj !== 'object') return '';
  const names = [];
  for (const lang of Object.keys(buyerObj)) {
    const val = buyerObj[lang];
    if (Array.isArray(val)) names.push(...val);
    else if (typeof val === 'string') names.push(val);
  }
  return [...new Set(names)].join(' / ');
}

/** Safe array → string join */
function joinArr(arr) {
  return Array.isArray(arr) ? arr.filter(Boolean).join(', ') : (arr || '');
}

/** Build the expert query for the last 30 days + CPV + keywords */
function buildQuery() {
  const now = new Date();
  const thirtyDaysAgo = new Date(now);
  thirtyDaysAgo.setDate(thirtyDaysAgo.getDate() - 30);
  const from = toTedDate(thirtyDaysAgo);

  // Expert-query syntax (official):
  // classification-cpv IN (...)   – exact CPV codes
  // FT=(...)                      – full-text search
  // publication-date >= YYYYMMDD
  // SORT BY publication-date DESC
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

    // Stop if we have everything or hit the 15 k pagination limit
    if (allNotices.length >= totalCount || notices.length < LIMIT || allNotices.length >= 15000) {
      break;
    }
    page += 1;
  }

  return { notices: allNotices, totalCount };
}

// ---------------------------------------------------------------------------
// Transform raw API notices into the required flat structure
// ---------------------------------------------------------------------------
function transform(notices) {
  return notices.map(n => {
    const pubNum = n['publication-number'] || '';
    return {
      title: extractTitle(n['notice-title']),
      buyerName: extractBuyerName(n['buyer-name']),
      country: joinArr(n['buyer-country']),
      cpv: joinArr(n['classification-cpv']),
      publicationDate: (n['publication-date'] || '').split('+')[0].split('T')[0], // YYYY-MM-DD
      deadline: joinArr(n['deadline-receipt-tender-date-lot']).split('+')[0].split('T')[0] || '',
      noticeUrl: pubNum
        ? `https://ted.europa.eu/en/notice/-/detail/${pubNum}`
        : ''
    };
  });
}

// ---------------------------------------------------------------------------
// Pretty console table
// ---------------------------------------------------------------------------
function printTable(rows) {
  if (rows.length === 0) {
    console.log('No notices found.');
    return;
  }

  // Truncate long fields for readability
  const display = rows.map(r => ({
    Title: (r.title || '').slice(0, 60) + (r.title?.length > 60 ? '…' : ''),
    Buyer: (r.buyerName || '').slice(0, 35) + (r.buyerName?.length > 35 ? '…' : ''),
    Country: r.country,
    CPV: (r.cpv || '').slice(0, 25),
    Published: r.publicationDate,
    Deadline: r.deadline,
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

    // Console output
    printTable(results);
    console.log(`\nTotal notices retrieved and saved: ${results.length}`);
    if (totalCount !== null && results.length < totalCount) {
      console.log(`(API reported ${totalCount} total matches – some may have been beyond the 15 000 pagination limit)`);
    }

    // Persist
    fs.writeFileSync(OUTPUT_FILE, JSON.stringify(results, null, 2), 'utf8');
    console.log(`\nResults written to ${OUTPUT_FILE}`);
  } catch (err) {
    console.error('\nError:', err.message);
    process.exit(1);
  }
}

main();
