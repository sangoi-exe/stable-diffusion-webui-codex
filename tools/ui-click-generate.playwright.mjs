#!/usr/bin/env node
// Headless UI click test using Playwright.
// Env:
//  - URL (default: http://127.0.0.1:7860)
//  - SELECTOR (default: #txt2img_generate)
//  - TIMEOUT_MS (default: 120000)
//  - CHROME_BIN (optional: path to system Chrome)
//  - BROWSER (optional: chromium|firefox|webkit; default chromium)
//  - SCREENSHOT (optional: path to save screenshot on success)

import fs from 'node:fs';
import path from 'node:path';

const URL = process.env.URL || 'http://127.0.0.1:7860';
const SELECTOR = process.env.SELECTOR || '#txt2img_generate';
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 120000);
const BROWSER = (process.env.BROWSER || 'chromium').toLowerCase();
const CHROME_BIN = process.env.CHROME_BIN || process.env.PUPPETEER_EXECUTABLE_PATH || '';
const SCREENSHOT = process.env.SCREENSHOT || path.join('artifacts', 'ui_headless', 'after-click.png');

let chromium, firefox, webkit;
try {
  ({ chromium, firefox, webkit } = await import('playwright'));
} catch (e) {
  console.error('[ui-test] Playwright is not installed: npm i -D playwright');
  process.exit(2);
}

function pickBrowser() {
  switch (BROWSER) {
    case 'firefox': return firefox;
    case 'webkit': return webkit;
    default: return chromium;
  }
}

async function main() {
  const engine = pickBrowser();
  const args = [
    '--no-sandbox',
    '--disable-setuid-sandbox',
    '--disable-gpu',
    '--disable-dev-shm-usage',
    '--mute-audio',
    '--no-first-run',
    '--no-default-browser-check',
    '--disable-crash-reporter',
  ];
  const launchOpts = { headless: true, args };
  if (engine === chromium) {
    if (CHROME_BIN) launchOpts.executablePath = CHROME_BIN; // use system Chrome if provided
    // else: use Playwright-managed Chromium from PLAYWRIGHT_BROWSERS_PATH
  }

  const browser = await engine.launch(launchOpts);
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  page.setDefaultTimeout(TIMEOUT_MS);
  console.log(`[ui-test] Navigating to ${URL} ...`);
  await page.goto(URL, { waitUntil: 'domcontentloaded', timeout: TIMEOUT_MS });

  // Ensure txt2img prompt is present, then prime a value
  // Prompt textarea is nested inside the container
  const promptBox = page.locator('#txt2img_prompt');
  await promptBox.waitFor({ state: 'visible', timeout: TIMEOUT_MS });
  const prompt = promptBox.locator('textarea, [contenteditable="true"], textarea >> nth=0');
  await prompt.first().fill('smoke test');

  // Try direct selector, then fallback to keyboard submit (Ctrl+Enter)
  let clicked = false;
  const btnLocator = page.locator(SELECTOR);
  if (await btnLocator.count().catch(()=>0)) {
    await btnLocator.first().click();
    clicked = true;
  } else {
    try {
      const textBtn = page.getByRole('button', { name: /generate/i });
      await textBtn.first().click();
      clicked = true;
    } catch {}
  }

  if (!clicked) {
    // Fallback: send Ctrl+Enter on prompt to trigger submit_named binding
    await prompt.focus();
    await prompt.press('Control+Enter');
    clicked = true;
  }

  if (!clicked) throw new Error('Generate button not found');
  console.log('[ui-test] Clicked Generate. Waiting for feedback...');

  // Feedback: button disabled or progress element present
  const ok = await Promise.race([
    page.waitForFunction(() => {
      const bts = Array.from(document.querySelectorAll('button'));
      const g = bts.find(b => /generate/i.test(b.textContent||''));
      return g && (g.disabled || g.getAttribute('aria-disabled') === 'true');
    }, { timeout: 30000 }).then(()=>true).catch(()=>false),
    page.waitForSelector('[role=progressbar], [aria-busy=true], .progress, .progress-bar', { timeout: 30000 }).then(()=>true).catch(()=>false)
  ]);

  fs.mkdirSync(path.dirname(SCREENSHOT), { recursive: true });
  await page.screenshot({ path: SCREENSHOT, fullPage: false }).catch(()=>{});

  await browser.close();
  if (!ok) throw new Error('No progress feedback after click');
  console.log('[ui-test] Success: Generate acknowledged.');
}

main().catch(e => { console.error('[ui-test] FAIL:', e.message); process.exit(1); });
