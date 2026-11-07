// Experiment: Measures Gen 337 champion circling metrics with and without food.
import { makeAgent, mulberry32 } from '../snake-sim.mjs';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { loadAgent, MemoryStorage } from '../../agent-loader.mjs';

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
    pathEff: (net/travel).toFixed(3),
    meanTurn: (turns.reduce((a,b)=>a+b,0)/turns.length).toFixed(4),
    netRevs: (headingSum/(Math.PI*2)).toFixed(3),
  };
}

const storage = new MemoryStorage();
const clock   = { t: 0, now() { return this.t; } };
const harness = loadAgent({ storage, clock, window: { grd: 16384 } });
const agent   = harness.newAgent();
agent.resetBrain();

const ckCandidates = [
  join(root, 'checkpoints/slither_ml_brain_gen337.json'),
  join(root, 'checkpoints_archive/slither_ml_brain_gen337.json'),
  join(root, 'checkpoints/slither_ml_brain_gen100.json'),
];
const ckFile = ckCandidates.find(existsSync);
if (!ckFile) {
  console.log('No champion checkpoint found; exiting experiment.');
  process.exit(0);
}

const ck = JSON.parse(readFileSync(ckFile, 'utf8'));
agent.applySavedState(ck);

console.log(`Using checkpoint: ${ckFile}`);
console.log('Seed 42, 600 frames, 15 food:', JSON.stringify(runSim(agent, clock, 42, 600, 15)));

const ck2 = { t: 0, now() { return this.t; } };
const h2 = loadAgent({ storage: new MemoryStorage(), clock: ck2, window: { grd: 16384 } });
const ag2 = h2.newAgent(); ag2.resetBrain();
ag2.applySavedState(ck);
console.log('Seed 42, 600 frames,  0 food:', JSON.stringify(runSim(ag2, ck2, 42, 600, 0)));
