/**
 * Shared helper: load the real NeuralNetwork / SlitherMLAgent classes out of the
 * shipped bundle, without pulling in the DOM-dependent controller.
 *
 * console_inject.js is a single-file IIFE, so the tests slice out the two pure
 * classes by brace matching and evaluate them with browser shims injected as
 * parameters. If the class layout ever changes in a way this cannot follow, the
 * compile step below fails loudly rather than silently testing nothing.
 */
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

export const projectRoot = dirname(fileURLToPath(import.meta.url));
// SLITHER_SOURCE lets the suite run against an alternate build, which is how the
// regression tests are proven to fail on the pre-fix code.
export const sourcePath = join(projectRoot, process.env.SLITHER_SOURCE || 'console_inject.js');
export const sourceCode = readFileSync(sourcePath, 'utf8');

/** Pull a top-level `class X { ... }` out of the bundle by matching braces. */
export function extractClass(src, name) {
  const header = `class ${name} {`;
  const start = src.indexOf(header);
  if (start < 0) throw new Error(`class ${name} not found in console_inject.js`);

  let depth = 0;
  for (let i = src.indexOf('{', start); i < src.length; i++) {
    const ch = src[i];
    if (ch === '{') depth++;
    else if (ch === '}') {
      depth--;
      if (depth === 0) return src.slice(start, i + 1);
    }
  }
  throw new Error(`unterminated class ${name} in console_inject.js`);
}

/** The storage key is read from source so a rename cannot silently pass tests. */
function readStorageKey(src) {
  const m = src.match(/const STORAGE_KEY = '([^']+)'/);
  if (!m) throw new Error('could not find STORAGE_KEY in console_inject.js');
  return m[1];
}

export class MemoryStorage {
  constructor() {
    this.map = new Map();
    this.failWrites = false;
  }
  getItem(k) {
    return this.map.has(k) ? this.map.get(k) : null;
  }
  setItem(k, v) {
    if (this.failWrites) throw new Error('QuotaExceededError (simulated)');
    this.map.set(k, String(v));
  }
  removeItem(k) {
    this.map.delete(k);
  }
  clear() {
    this.map.clear();
  }
}

/**
 * Build a sandbox holding the agent classes.
 * @returns {{NeuralNetwork:Function, SlitherMLAgent:Function, storage:MemoryStorage,
 *            window:object, logs:string[], STORAGE_KEY:string, newAgent:Function}}
 */
export function loadAgent({ storage = new MemoryStorage(), window: win = {}, collectLogs = false, clock } = {}) {
  const nnSrc = extractClass(sourceCode, 'NeuralNetwork');
  const recorderSrc = extractClass(sourceCode, 'FlightRecorder');
  const agentSrc = extractClass(sourceCode, 'SlitherMLAgent');
  const storageKey = readStorageKey(sourceCode);

  const logs = [];
  const shimConsole = collectLogs
    ? {
        log: (...a) => logs.push(a.join(' ')),
        warn: (...a) => logs.push('WARN ' + a.join(' ')),
        error: (...a) => logs.push('ERROR ' + a.join(' ')),
      }
    : { log() {}, warn() {}, error() {} };

  const factory = new Function(
    'localStorage',
    'window',
    'performance',
    'console',
    'pageWindow',
    `"use strict";
     if (typeof pageWindow === 'undefined') { var pageWindow = window; }
     const STORAGE_KEY = ${JSON.stringify(storageKey)};
     ${nnSrc}
     ${recorderSrc}
     ${agentSrc}
     return { NeuralNetwork, FlightRecorder, SlitherMLAgent };`,
  );

  const { NeuralNetwork, FlightRecorder, SlitherMLAgent } = factory(storage, win, clock ?? performance, shimConsole, win);

  return {
    NeuralNetwork,
    FlightRecorder,
    SlitherMLAgent,
    storage,
    window: win,
    logs,
    STORAGE_KEY: storageKey,
    newAgent: () => new SlitherMLAgent(),
  };
}

/** Sum of weights, used as a cheap genome fingerprint. */
export function fingerprint(net) {
  return net.weights.map((w) => Array.from(w, (v) => Number(v.toFixed(6))));
}

export function sameGenome(a, b) {
  if (!a || !b || a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i].length !== b[i].length) return false;
    for (let j = 0; j < a[i].length; j++) if (Math.abs(a[i][j] - b[i][j]) > 1e-9) return false;
  }
  return true;
}
