// Experiment: Measures path efficiency, heading revolutions, and steering bias drift across 600 frames.
import { makeAgent } from '../snake-sim.mjs';

const { agent, clock } = makeAgent(1);

const wrap = (a) => {
  while (a > Math.PI) a -= Math.PI * 2;
  while (a < -Math.PI) a += Math.PI * 2;
  return a;
};

const snake = { id:'me', xx:0, yy:0, eang:0, sc:1+100/15000, sp:5.9, pts:[], dead:false };
const fps=60, dt=1/fps, speed=354, maxTurnRate=4.5;
let travel=0, prevH=0, headingSum=0;
const turns=[];

for (let f=0; f<600; f++) {
  clock.t += dt*1000;
  const dec = agent.evaluate(snake, [], [], []);
  turns.push(dec.rawSteer);
  const diff=wrap(dec.targetAngle-snake.eang), maxStep=maxTurnRate*dt;
  snake.eang=wrap(snake.eang+Math.max(-maxStep,Math.min(maxStep,diff)));
  const dh=wrap(snake.eang-prevH); headingSum+=dh; prevH=snake.eang;
  const v=speed;
  snake.xx+=Math.cos(snake.eang)*v*dt;
  snake.yy+=Math.sin(snake.eang)*v*dt;
  travel+=v*dt;
}

const net=Math.hypot(snake.xx,snake.yy);
console.log('pathEff:', (net/travel).toFixed(3));
console.log('meanTurn:', (turns.reduce((a,b)=>a+b,0)/turns.length).toFixed(4));
console.log('netRevs:', (headingSum/(Math.PI*2)).toFixed(3));
console.log('first 10 rawSteer:', turns.slice(0,10).map(v=>v.toFixed(4)).join(', '));
console.log('targetAngle frame0:', (() => {
  const { agent: ag2, clock: ck2 } = makeAgent(1);
  const sn2 = { id:'me', xx:0, yy:0, eang:0, sc:1+100/15000, sp:5.9, pts:[], dead:false };
  ck2.t += dt*1000;
  const d = ag2.evaluate(sn2, [], [], []);
  return `targetAngle=${d.targetAngle.toFixed(4)}, rawSteer=${d.rawSteer.toFixed(4)}`;
})());
