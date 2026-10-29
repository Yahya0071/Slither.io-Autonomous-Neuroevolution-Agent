#!/usr/bin/env node
/**
 * SlitherML Checkpoint & Flight Recorder Diagnostic Analyzer
 *
 * Usage:
 *   node sim/analyze_checkpoint.mjs [path_to_checkpoint.json]
 *   npm run analyze -- checkpoints/slither_ml_brain_gen128.json
 */
import { readFileSync, readdirSync, statSync } from 'node:fs';
import { resolve, join } from 'node:path';

let targetPath = process.argv[2];

if (!targetPath) {
  // Find newest checkpoint in checkpoints/ directory
  try {
    const cpDir = resolve('checkpoints');
    const files = readdirSync(cpDir)
      .filter((f) => f.startsWith('slither_ml_brain_') && f.endsWith('.json'))
      .map((f) => ({ name: f, path: join(cpDir, f), mtime: statSync(join(cpDir, f)).mtimeMs }))
      .sort((a, b) => b.mtime - a.mtime);
    if (files.length > 0) {
      targetPath = files[0].path;
      console.log(`[SlitherML] No file specified, inspecting newest checkpoint: ${files[0].name}\n`);
    }
  } catch (_) {}
}

if (!targetPath) {
  console.error('Usage: node sim/analyze_checkpoint.mjs <checkpoint_file.json>');
  process.exit(1);
}

const raw = readFileSync(targetPath, 'utf8');
const cp = JSON.parse(raw);

console.log('======================================================================');
console.log(`📊 SLITHER.IO ML CHECKPOINT ANALYSIS: ${targetPath}`);
console.log('======================================================================');

// 1. Milestone Metadata
console.log('\n--- 1. MILESTONE METRICS ---');
console.log(`Generation:            ${cp.generation ?? 'N/A'}`);
console.log(`Current Episode:       ${cp.currentEpisode ?? 'N/A'}`);
console.log(`Confirmed Kills:       ${cp.kills ?? 0}`);
console.log(`Teacher Lessons:       ${(cp.teacherSamples || 0).toLocaleString()}`);
console.log(`Best Fitness:          ${cp.bestFitness ? cp.bestFitness.toFixed(1) : 0}`);
console.log(`All-Time Peak Length:  ${cp.allTimePeakLength ?? 'N/A'}`);
console.log(`Stagnant Generations:  ${cp.stagnantGens ?? 0}`);
if (cp.timestamp) {
  console.log(`Saved Timestamp:       ${new Date(cp.timestamp).toISOString()}`);
}

// 2. Neural Architecture & Weights
const brain = cp.brain || cp;
const active = cp.activeBrain || null;
console.log('\n--- 2. NEURAL NETWORK ARCHITECTURE ---');
console.log(`Input Dimensions:      ${brain.inputNodes ?? 32}`);
console.log(`Hidden Layer Shapes:   [${(brain.hiddenLayers || [48, 24]).join(', ')}]`);
console.log(`Output Dimensions:     ${brain.outputNodes ?? 2}`);

if (brain.biases && brain.biases[2]) {
  const b2 = brain.biases[2];
  console.log(`Champion Turn Bias b2[0]:   ${b2[0].toFixed(6)}`);
  console.log(`Champion Boost Bias b2[1]:  ${b2[1].toFixed(6)}`);
}
if (active && active.biases && active.biases[2]) {
  const ab2 = active.biases[2];
  console.log(`Active Mutant Turn b2[0]:   ${ab2[0].toFixed(6)}`);
  console.log(`Active Mutant Boost b2[1]:  ${ab2[1].toFixed(6)}`);
}

// 3. Saliency Calculation (L1 norm from 32 inputs into Hidden 1)
const targetNet = active || brain;
if (targetNet.weights && targetNet.weights[0]) {
  const w0 = targetNet.weights[0];
  const numInputs = targetNet.inputNodes || 32;
  const numH1 = (targetNet.hiddenLayers && targetNet.hiddenLayers[0]) || 48;
  const saliencies = [];

  const inputNames = [
    'Ray  0 (-180° Tail)', 'Ray  1 (-158°)', 'Ray  2 (-135° Rear-Right)', 'Ray  3 (-113°)',
    'Ray  4 ( -90° Right Flank)', 'Ray  5 ( -68°)', 'Ray  6 ( -45° Front-Right)', 'Ray  7 ( -23°)',
    'Ray  8 (   0° Head-On Nose)', 'Ray  9 (  23° Front-Left)', 'Ray 10 (  45° Left Flank)', 'Ray 11 (  68°)',
    'Ray 12 (  90° Left Flank)', 'Ray 13 ( 113°)', 'Ray 14 ( 135° Rear-Left)', 'Ray 15 ( 158°)',
    'Octant 0 (Food Ahead)', 'Octant 1 (Food Front-Left)', 'Octant 2 (Food Left Flank)', 'Octant 3 (Food Rear-Left)',
    'Octant 4 (Food Behind)', 'Octant 5 (Food Rear-Right)', 'Octant 6 (Food Right Flank)', 'Octant 7 (Food Front-Right)',
    'Center Dist Normalized [24]', 'Center Angle Normalized [25]', 'Current Length [26]', 'Current Speed [27]',
    'Memory 28 (prevTurnEma)', 'Memory 29 (leftThreatEma)', 'Memory 30 (rightThreatEma)', 'Memory 31 (congestionRatio)'
  ];

  for (let i = 0; i < numInputs; i++) {
    let sum = 0;
    for (let h = 0; h < numH1; h++) {
      sum += Math.abs(w0[h * numInputs + i] || 0);
    }
    saliencies.push({ idx: i, name: inputNames[i] || `Input [${i}]`, salience: sum });
  }

  saliencies.sort((a, b) => b.salience - a.salience);
  console.log('\n--- 3. TOP 8 SENSORY INPUT SALIENCIES ---');
  for (let i = 0; i < Math.min(8, saliencies.length); i++) {
    const s = saliencies[i];
    console.log(`  #${i + 1}: ${s.name.padEnd(35)} (Saliency: ${s.salience.toFixed(4)})`);
  }
}

// 4. Telemetry & Autopsy
if (cp.telemetry) {
  const tel = cp.telemetry;
  console.log('\n======================================================================');
  console.log('💀 FLIGHT RECORDER: TELEMETRY & AUTOPSY REPORT');
  console.log('======================================================================');

  if (tel.lastDeathAutopsy) {
    const a = tel.lastDeathAutopsy;
    console.log(`\nLast Death Cause:      ${a.cause} (Confidence: ${Math.round((a.confidence || 0) * 100)}%)`);
    console.log(`Description:           ${a.description}`);
    console.log(`Fatal Stance:          ${a.fatalStance}`);
    console.log(`Survival Time:         ${a.survivalTime}s`);
    console.log(`Final Score / Length:  ${a.finalScore} / ${a.peakLength}`);
    if (a.killerProfile) {
      console.log(`Killer Threat Profile: Type: ${a.killerProfile.type} | Dist: ${a.killerProfile.dist} | MassRatio: ${a.killerProfile.massRatio}x | Speed: ${a.killerProfile.speed}`);
    }
    if (a.preCrashTelemetry) {
      const pc = a.preCrashTelemetry;
      console.log(`Pre-Crash State:       Pos: (${pc.x}, ${pc.y}) | Speed: ${pc.speed} | Boost: ${pc.isBoosting} | TurnDelta: ${pc.turnDelta} | Ray: #${pc.dominantRayIdx} (${pc.dominantRayDist})`);
    }
  }

  if (tel.stanceDutyCycles && Object.keys(tel.stanceDutyCycles).length > 0) {
    console.log('\n--- COMBAT STANCE DUTY CYCLES ---');
    for (const [st, pct] of Object.entries(tel.stanceDutyCycles)) {
      console.log(`  ${st.padEnd(14)}: ${pct}`);
    }
  }

  if (tel.zoneDistribution) {
    console.log('\n--- SPATIAL ARENA OCCUPANCY ---');
    console.log(`  Center Hub:     ${tel.zoneDistribution.center}`);
    console.log(`  Midfield:       ${tel.zoneDistribution.midfield}`);
    console.log(`  Perimeter Wall: ${tel.zoneDistribution.perimeter}`);
  }

  if (tel.growthMilestones && Object.keys(tel.growthMilestones).length > 0) {
    console.log('\n--- GROWTH MILESTONES ---');
    for (const [target, t] of Object.entries(tel.growthMilestones)) {
      console.log(`  Reached ${target.padStart(5)} mass: at ${t}s`);
    }
  }

  if (Array.isArray(tel.recentEpisodeHistory) && tel.recentEpisodeHistory.length > 0) {
    console.log('\n--- RECENT EPISODES HISTORY ---');
    console.log('Ep   | Gen  | Duration | Length | Score | Fatal Cause        | Stance');
    console.log('-----|------|----------|--------|-------|--------------------|--------');
    for (const ep of tel.recentEpisodeHistory) {
      const epStr = String(ep.ep ?? '-').padEnd(4);
      const genStr = String(ep.gen ?? '-').padEnd(4);
      const durStr = `${ep.durationS ?? 0}s`.padEnd(8);
      const lenStr = String(ep.peakLength ?? 0).padEnd(6);
      const scStr = String(ep.finalScore ?? 0).padEnd(5);
      const causeStr = (ep.deathCause ?? 'UNKNOWN').padEnd(18);
      const stStr = ep.fatalStance ?? '-';
      console.log(`${epStr} | ${genStr} | ${durStr} | ${lenStr} | ${scStr} | ${causeStr} | ${stStr}`);
    }
  }

  if (Array.isArray(tel.pathfinderBreadcrumbs) && tel.pathfinderBreadcrumbs.length > 0) {
    console.log(`\nPathfinder Breadcrumbs: ${tel.pathfinderBreadcrumbs.length} waypoints recorded in final life.`);
  }
} else {
  console.log('\n[SlitherML] Note: This checkpoint does not contain a "telemetry" block (saved before v5.10.0).');
}

console.log('\n======================================================================');

