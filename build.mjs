#!/usr/bin/env node
/**
 * build.mjs - regenerate every distributable artifact from the single source of
 * truth, console_inject.js.
 *
 * The project ships the same agent in five places (console paste, userscript,
 * Chrome extension, and two copies inlined in quick_start.html). Editing them by
 * hand caused silent drift, so they are now generated. Run `npm run build`
 * after changing console_inject.js, and `npm test` to check the result.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const SOURCE = 'console_inject.js';
const source = readFileSync(SOURCE, 'utf8');
const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
const version = pkg.version;

if (source.includes('</script>')) {
  throw new Error('console_inject.js contains a literal </script>, which cannot be inlined into HTML.');
}

const USCRIPT_HEADER = `// ==UserScript==
// @name         Slither.io Autonomous Machine Learning Agent (SlitherML)
// @namespace    https://slither.io/
// @version      ${version}
// @description  32-input Hybrid Neuroevolution + Online Imitation Learning AI for Slither.io. 16 proximity rays, 8 food sectors, self-collision detection, trajectory prediction, and teacher-mode backpropagation whose lessons must win an autonomous 3-life evaluation before being saved.
// @author       DeepMind Antigravity AI
// @match        *://slither.io/*
// @match        *://slither.com/io*
// @match        *://*.slither.io/*
// @grant        GM_getValue
// @grant        GM_setValue
// @grant        unsafeWindow
// @run-at       document-idle
// ==/UserScript==

`;

const artifacts = new Map([
  ['extension/agent.js', source],
  ['slither_ml_agent.user.js', USCRIPT_HEADER + source],
]);

for (const [path, content] of artifacts) {
  writeFileSync(path, content, 'utf8');
}

// quick_start.html inlines both variants so the copy buttons work from file://,
// where fetch() of a local file is blocked by CORS.
const html = readFileSync('quick_start.html', 'utf8');

const replaceBlock = (htmlText, id, content) => {
  const open = new RegExp(`(<script type="text/plain" id="${id}">)([\\s\\S]*?)(</script>)`);
  if (!open.test(htmlText)) throw new Error(`could not find inline block #${id} in quick_start.html`);
  return htmlText.replace(open, (_, head, _old, tail) => head + content + tail);
};

let nextHtml = replaceBlock(html, 'codeConsole', source);
nextHtml = replaceBlock(nextHtml, 'codeUserscript', USCRIPT_HEADER + source);
writeFileSync('quick_start.html', nextHtml, 'utf8');

// ---------------------------------------------------------------------------
// Verify: every artifact must now carry byte-identical agent code.
// ---------------------------------------------------------------------------
const digest = (s) => createHash('sha256').update(s).digest('hex').slice(0, 16);
const verifyHtml = readFileSync('quick_start.html', 'utf8');
const grab = (htmlText, id) => {
  const m = htmlText.match(new RegExp(`<script type="text/plain" id="${id}">([\\s\\S]*?)</script>`));
  return m ? m[1] : null;
};

const checks = [
  ['extension/agent.js', readFileSync('extension/agent.js', 'utf8'), source],
  ['slither_ml_agent.user.js', readFileSync('slither_ml_agent.user.js', 'utf8'), USCRIPT_HEADER + source],
  ['quick_start.html#codeConsole', grab(verifyHtml, 'codeConsole'), source],
  ['quick_start.html#codeUserscript', grab(verifyHtml, 'codeUserscript'), USCRIPT_HEADER + source],
];

let ok = true;
console.log(`source : ${SOURCE}  (${digest(source)})  v${version}\n`);
for (const [name, actual, expected] of checks) {
  const match = actual === expected;
  ok &&= match;
  console.log(`  ${match ? 'OK  ' : 'FAIL'}  ${name.padEnd(32)} ${digest(actual ?? '')}`);
}
if (!ok) {
  console.error('\nbuild verification FAILED: an artifact does not match the source.');
  process.exit(1);
}
console.log('\nAll 4 artifacts regenerated from console_inject.js and verified identical.');
