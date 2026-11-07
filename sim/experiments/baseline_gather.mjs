// Experiment: Gathers baseline heading revolution and path efficiency numbers across multiple seeds.
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadAgent, MemoryStorage } from '../../agent-loader.mjs';
import { mulberry32 } from '../snake-sim.mjs';

const __dir = dirname(fileURLToPath(import.meta.url));
const root  = join(__dir, '../..');

const wrap = (a) => {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
};

function runSim(agent, clock, seed, frames, foodCount) {
  const rand = mulberry32(seed);
  const foods = [];
  for (let i = 0; i < foodCount; i++) {
    const r = Math.sqrt(rand()) * 1200;
    const a = rand() * Math.PI * 2;
    foods.push({ xx: Math.cos(a) * r, yy: Math.sin(a) * r, sz: 4 });
  }

  const snake = { id:'me', xx:0, yy:0, eang:0, sc:1+100/15000, sp:5.9, pts:[], dead:false };
  const fps=60, dt=1/fps, speed=354, maxTurnRate=4.5;
  let travel=0, prevH=0, headingSum=0;
  const turns=[];

  for (let f=0; f<frames; f++) {
    clock.t += dt*1000;
    const dec = agent.evaluate(snake, [], foods, []);
    turns.push(dec.rawSteer);
    const diff=wrap(dec.targetAngle-snake.eang), maxStep=maxTurnRate*dt;
    snake.eang=wrap(snake.eang+Math.max(-maxStep,Math.min(maxStep,diff)));
    const dh=wrap(snake.eang-prevH); headingSum+=dh; prevH=snake.eang;
    const v=speed;
    snake.xx+=Math.cos(snake.eang)*v*dt;
    snake.yy+=Math.sin(snake.eang)*v*dt;
    travel+=v*dt;
    const catchR = 14*snake.sc+12;
    for (let i=foods.length-1;i>=0;i--) {
      if (Math.hypot(foods[i].xx-snake.xx, foods[i].yy-snake.yy) < catchR) foods.splice(i,1);
    }
    snake.sc=1+100/15000;
  }

  const net=Math.hypot(snake.xx,snake.yy);
  return {
    pathEff: parseFloat((net/travel).toFixed(3)),
    meanTurn: parseFloat((turns.reduce((a,b)=>a+b,0)/turns.length).toFixed(4)),
    netRevs: parseFloat((headingSum/(Math.PI*2)).toFixed(3)),
  };
}

function makeHarness() {
  const storage = new MemoryStorage();
  const clock   = { t: 0, now() { return this.t; } };
  const harness = loadAgent({ storage, clock, window: { grd: 16384 } });
  return { storage, clock, harness };
}

const ckCandidates = [
  join(root, 'checkpoints/slither_ml_brain_gen337.json'),
  join(root, 'checkpoints_archive/slither_ml_brain_gen337.json'),
  join(root, 'checkpoints/slither_ml_brain_gen100.json'),
];
const ckFile = ckCandidates.find(existsSync);
const ck = ckFile ? JSON.parse(readFileSync(ckFile, 'utf8')) : null;
const seeds = [1, 7, 13, 42, 99];

console.log('\n=== Fresh agent, NO food, 600 frames ===');
for (const seed of seeds) {
  const { harness, clock } = makeHarness();
  const agent = harness.newAgent(); agent.resetBrain();
  const r = runSim(agent, clock, seed, 600, 0);
  console.log(`  seed ${seed}: meanTurn=${r.meanTurn}, netRevs=${r.netRevs}, pathEff=${r.pathEff}`);
}

console.log('\n=== Fresh agent, 15 food, 600 frames ===');
for (const seed of seeds) {
  const { harness, clock } = makeHarness();
  const agent = harness.newAgent(); agent.resetBrain();
  const r = runSim(agent, clock, seed, 600, 15);
  console.log(`  seed ${seed}: meanTurn=${r.meanTurn}, netRevs=${r.netRevs}, pathEff=${r.pathEff}`);
}

if (ck) {
  console.log(`\n=== Champion (${ckFile}), 15 food, 600 frames ===`);
  for (const seed of seeds) {
    const { harness, clock } = makeHarness();
    const agent = harness.newAgent(); agent.resetBrain();
    agent.applySavedState(ck);
    const r = runSim(agent, clock, seed, 600, 15);
    console.log(`  seed ${seed}: meanTurn=${r.meanTurn}, netRevs=${r.netRevs}, pathEff=${r.pathEff}`);
  }

  console.log(`\n=== Champion (${ckFile}), NO food, 600 frames ===`);
  for (const seed of seeds) {
    const { harness, clock } = makeHarness();
    const agent = harness.newAgent(); agent.resetBrain();
    agent.applySavedState(ck);
    const r = runSim(agent, clock, seed, 600, 0);
    console.log(`  seed ${seed}: meanTurn=${r.meanTurn}, netRevs=${r.netRevs}, pathEff=${r.pathEff}`);
  }
}
