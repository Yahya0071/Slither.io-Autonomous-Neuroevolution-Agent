#!/usr/bin/env node
/**
 * build.mjs - regenerate every distributable artifact from ES modules under src/.
 *
 * Source code lives in modular ES files under src/:
 *   - src/network.js: Feedforward neural network, SGD with momentum, mutations, crossover
 *   - src/flight-recorder.js: Rolling telemetry buffer, death autopsy deductor, breadcrumbs
 *   - src/perception.js: 16 proximity rays, 8 food radar sectors, combat geometry, corridors
 *   - src/stances.js: 7 tactical stances, kinematic steering assist, boost governor
 *   - src/evolution.js: 3-episode rollout scoring, consistency gate, silver candidate, breeding
 *   - src/agent.js: SlitherMLAgent state machine, learning loop, and persistence
 *   - src/controller.js: SlitherMLController HUD overlay, inputs, canvas rendering
 *
 * This build step bundles them into console_inject.js (and the other 3 distribution
 * targets) so users get a single zero-dependency file that can be pasted or injected.
 * Run `npm run build` after editing files in src/, and `npm test` to verify.
 */
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';

const pkg = JSON.parse(readFileSync('package.json', 'utf8'));
const version = pkg.version;

// Ensure extension manifest version is generated from package.json
const MANIFEST = 'extension/manifest.json';
const manifest = JSON.parse(readFileSync(MANIFEST, 'utf8'));
if (manifest.version !== version) {
  manifest.version = version;
  writeFileSync(MANIFEST, JSON.stringify(manifest, null, 2) + '\n', 'utf8');
}

function cleanModule(src) {
  let cleaned = src.replace(/import\s*\{[\s\S]*?\}\s*from\s*['"][^'"]+['"];?/g, '');
  cleaned = cleaned.replace(/import\s+[a-zA-Z0-9_$*,\s{}]+from\s*['"][^'"]+['"];?/g, '');
  cleaned = cleaned.replace(/^\s*export\s*\{[\s\S]*?\};?\s*$/gm, '');
  cleaned = cleaned.replace(/^\s*export\s+(class|function|const|let|var)\s+/gm, '$1 ');
  return cleaned;
}

function cleanControllerModule(src) {
  let cleaned = cleanModule(src);
  cleaned = cleaned.replace(/const pageWindow = [^\n]+;\n*/g, '');
  cleaned = cleaned.replace(/const getPlayerSnake = [^\n]+;\n*/g, '');
  cleaned = cleaned.replace(/const getAllSnakes = [^\n]+;\n*/g, '');
  return cleaned;
}

const perceptionSrc = cleanModule(readFileSync('src/perception.js', 'utf8'));
const stancesSrc = cleanModule(readFileSync('src/stances.js', 'utf8'));
const evolutionSrc = cleanModule(readFileSync('src/evolution.js', 'utf8'));
const networkSrc = cleanModule(readFileSync('src/network.js', 'utf8'));
const flightRecorderSrc = cleanModule(readFileSync('src/flight-recorder.js', 'utf8'));
const agentSrc = cleanModule(readFileSync('src/agent.js', 'utf8'));
const controllerSrc = cleanControllerModule(readFileSync('src/controller.js', 'utf8'));

const source = `/**
 * Slither.io Autonomous Machine Learning Agent (Console Paste Version)
 * Hybrid Neuroevolution + Imitation Learning Engine v${version} (Forensic Deductor & Pathfinder Flight Recorder Edition)
 * 
 * SYNTHESIZED FROM THE BEST OPEN-SOURCE REPOSITORIES:
 *  - dzaczek/slbot: Apex Predator Lead Interception Strike & Active Exploratory Scanning
 *  - ErmiyaEskandary/Slither.io-bot: The Circle Method (COIL Stance) & Dual-Zone Arena Boundary Wall Guard
 *  - MEMWashburn/Slither.io-NEAT: Confirmed Kill Bounty Recognition & Anti-Stagnation Adaptive Mutation
 *  - JuiHsiu/Slither-DRL: Mass-ROI Boost Governor & Anti-Jitter Hydrodynamic Steering
 * 
 * MODES:
 *  - AUTOPILOT [T]: AI plays autonomously with 3-episode rollout neuroevolution.
 *  - TEACHER MODE [T]: You play manually. The AI shadows your moves and learns your reflexes via real-time backpropagation!
 * 
 * HOTKEYS:
 *  - [T] or [B] : Toggle Autopilot / Teacher Mode
 *  - [V]        : Toggle Vision Rays Overlay ON / OFF
 *  - [R]        : Toggle Auto-Respawn ON / OFF
 *  - [O]        : View Last Death Autopsy Report
 *  - [A]        : Load Apex Champion Preset (Gen 100)
 *  - [X]        : Reset Brain to Generation 1
 *  - [E]        : Export Brain Checkpoint
 *  - [I]        : Import Brain Checkpoint
 */

(function () {
  'use strict';

  console.log('%c[SlitherML] 🚀 Initializing Hybrid Neuroevolution + Imitation Learning Agent v${version} (Forensic Deductor & Pathfinder Flight Recorder Edition)...', 'color: #00f5d4; font-weight: bold; font-size: 14px;');

  const pageWindow = (typeof unsafeWindow !== 'undefined' && unsafeWindow) ? unsafeWindow : (typeof window !== 'undefined' ? window : {});

  if (typeof window !== 'undefined') {
    window.__slitherMLBridgeSave = (serialized) => {
      try {
        window.dispatchEvent(new CustomEvent('SlitherML_BridgeSave', { detail: serialized }));
      } catch (_) {}
    };
    window.__slitherMLBridgeClear = () => {
      try {
        window.dispatchEvent(new CustomEvent('SlitherML_BridgeClear'));
      } catch (_) {}
    };
  }

  const getPlayerSnake = () => pageWindow.snake || pageWindow.slither || null;
  const getAllSnakes = () => pageWindow.snakes || pageWindow.slithers || [];

  // ==========================================
  // PERCEPTION & SENSOR PIPELINE
  // ==========================================
${perceptionSrc}

  // ==========================================
  // TACTICAL STANCES & ACTUATION
  // ==========================================
${stancesSrc}

  // ==========================================
  // NEUROEVOLUTION & BREEDING
  // ==========================================
${evolutionSrc}

  // ==========================================
  // 1. NEURAL ARCHITECTURE
  // ==========================================
${networkSrc}

  // ==========================================
  // 2. FLIGHT RECORDER & AUTOPSY FORENSICS
  // ==========================================
${flightRecorderSrc}

  // ==========================================
  // 3. CORE AGENT
  // ==========================================
${agentSrc}

  // ==========================================
  // 4. CONTROLLER & BROWSER INTERFACE
  // ==========================================
${controllerSrc}

  if (pageWindow._slitherMLController && typeof pageWindow._slitherMLController.destroy === 'function') {
    try {
      pageWindow._slitherMLController.destroy();
    } catch (_) { }
  }
  pageWindow._slitherMLController = new SlitherMLController();
  pageWindow.__slitherML = {
    controller: pageWindow._slitherMLController,
    agent: pageWindow._slitherMLController.agent,
    getAutopsy: () => pageWindow._slitherMLController.agent.getAutopsyReport(),
    getTelemetry: () => pageWindow._slitherMLController.agent.flightRecorder.getTelemetrySummary(),
    exportCheckpoint: () => pageWindow._slitherMLController.agent.exportCheckpoint()
  };
  console.log('%c[SlitherML] 🚀 SlitherML v${version} (Forensic Deductor & Pathfinder Flight Recorder) is RUNNING!', 'color: #00f5d4; font-size: 16px; font-weight: bold;');
})();
`;

if (source.includes('</script>')) {
  throw new Error('bundled source contains a literal </script>, which cannot be inlined into HTML.');
}

const SOURCE = 'console_inject.js';
writeFileSync(SOURCE, source, 'utf8');

const USCRIPT_HEADER = `// ==UserScript==
// @name         Slither.io Autonomous Machine Learning Agent (SlitherML)
// @namespace    https://slither.io/
// @version      ${version}
// @description  32-input Hybrid Neuroevolution + Online Imitation Learning AI for Slither.io. 16 proximity rays, 8 food sectors, self-collision detection, trajectory prediction, and teacher-mode backpropagation whose lessons must win an autonomous 3-life evaluation before being saved.
// @author       ${pkg.author || 'Yahya0071'}
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
console.log('\nAll 4 artifacts regenerated from src/ via console_inject.js and verified identical.');
