#!/usr/bin/env node
/*
 Headless UI click test: open WebUI page and click the Generate button.
 Usage envs:
   URL=http://127.0.0.1:7860  SELECTOR="#txt2img_generate"  TIMEOUT_MS=120000 node tools/ui-click-generate.js
 Exits 0 on success (button clicked and page remained alive), non-zero on failure.
*/

const URL = process.env.URL || 'http://127.0.0.1:7860';
const SELECTOR = process.env.SELECTOR || '#txt2img_generate';
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 120000);

async function main() {
  const puppeteer = await import('puppeteer');
  const launchOptions = {
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--disable-gpu',
      '--no-zygote',
      '--disable-dev-shm-usage',
      '--mute-audio',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-crash-reporter',
      '--no-crashpad',
      `--user-data-dir=${process.env.CHROME_USER_DATA_DIR || '.chromium-profile'}`,
      '--disable-features=IsolateOrigins,site-per-process,AudioServiceOutOfProcess',
    ],
    defaultViewport: { width: 1280, height: 900 },
  };
  const chromeBin = process.env.CHROME_BIN || process.env.PUPPETEER_EXECUTABLE_PATH;
  if (chromeBin) launchOptions.executablePath = chromeBin;
  const browser = await puppeteer.launch(launchOptions);
  const page = await browser.newPage();
  page.setDefaultTimeout(TIMEOUT_MS);
  console.log(`[ui-test] Navigating to ${URL} ...`);
  await page.goto(URL, { waitUntil: 'networkidle0', timeout: TIMEOUT_MS });

  // Wait for the Generate button (selector or text fallback)
  let handle = await page.$(SELECTOR);
  if (!handle) {
    // Fallback by text content contains 'Generate'
    handle = await page.evaluateHandle(() => {
      const buttons = Array.from(document.querySelectorAll('button'));
      return buttons.find(b => /generate/i.test(b.textContent || '')) || null;
    });
  }
  if (!handle) {
    throw new Error(`Generate button not found (selector=${SELECTOR}).`);
  }

  console.log('[ui-test] Clicking Generate...');
  await handle.click();

  // Wait briefly for visual feedback: button disabled or progress region appears.
  const ok = await page.waitForFunction(() => {
    const btns = Array.from(document.querySelectorAll('button'));
    const gen = btns.find(b => /generate/i.test(b.textContent || ''));
    if (gen && (gen.disabled || gen.getAttribute('aria-disabled') === 'true')) return true;
    const busy = document.querySelector('[aria-busy="true"], [role="progressbar"], .progress, .progress-bar');
    return !!busy;
  }, { timeout: 10000 }).then(() => true).catch(() => false);

  await browser.close();
  if (!ok) {
    throw new Error('No feedback detected after clicking Generate (button not disabled, no progress).');
  }
  console.log('[ui-test] Generate click acknowledged.');
}

main().catch(err => { console.error(`[ui-test] FAIL: ${err.message}`); process.exit(1); });
