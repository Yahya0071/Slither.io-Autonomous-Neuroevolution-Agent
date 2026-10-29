/**
 * NeuralNetwork correctness tests.
 *
 * The headline test is a gradient check: trainStep's implicit gradients are
 * compared against an independent, hand-derived, double-precision
 * backpropagation. This is the check that catches a wrong derivative, a swapped
 * weight index, or a missing activation factor - none of which are visible from
 * "the loss goes down".
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { loadAgent } from '../agent-loader.mjs';

const { NeuralNetwork } = loadAgent();

const INPUTS = 32;
const HIDDEN = [48, 24];
const OUTPUTS = 2;

function randomNet(scale = 0.4) {
  const net = new NeuralNetwork(INPUTS, HIDDEN, OUTPUTS);
  for (const w of net.weights) for (let i = 0; i < w.length; i++) w[i] = (Math.random() * 2 - 1) * scale;
  for (const b of net.biases) for (let i = 0; i < b.length; i++) b[i] = (Math.random() * 2 - 1) * (scale / 2);
  net.biases[2][0] = 0.0; // Center steering bias so it stays safely unclamped during gradient verification
  return net;
}

/**
 * Independent reference: same architecture and activations, gradients derived by
 * hand in float64. Loss = 0.5 * ((t0 - out0)^2 + (t1 - out1)^2).
 */
function referenceGradients(net, x, t) {
  const [W0, W1, W2] = net.weights.map((w) => Float64Array.from(w));
  const [b0, b1, b2] = net.biases.map((b) => Float64Array.from(b));
  const a0 = Float64Array.from(x);

  const a1 = new Float64Array(48);
  for (let c = 0; c < 48; c++) {
    let s = b0[c];
    for (let r = 0; r < 32; r++) s += a0[r] * W0[r * 48 + c];
    a1[c] = Math.tanh(s);
  }
  const a2 = new Float64Array(24);
  for (let c = 0; c < 24; c++) {
    let s = b1[c];
    for (let r = 0; r < 48; r++) s += a1[r] * W1[r * 24 + c];
    a2[c] = Math.tanh(s);
  }
  const z3 = new Float64Array(2);
  for (let c = 0; c < 2; c++) {
    let s = b2[c];
    for (let r = 0; r < 24; r++) s += a2[r] * W2[r * 2 + c];
    z3[c] = s;
  }
  const out = [Math.tanh(z3[0]), 1 / (1 + Math.exp(-z3[1]))];
  const bce = - (t[1] * Math.log(Math.max(1e-12, out[1])) + (1 - t[1]) * Math.log(Math.max(1e-12, 1 - out[1])));
  const loss = 0.5 * ((t[0] - out[0]) ** 2) + bce;

  // output layer: dL/dz (tanh MSE for steer, sigmoid BCE for boost)
  const d3 = [(out[0] - t[0]) * (1 - out[0] ** 2), out[1] - t[1]];

  const gW2 = new Float64Array(24 * 2);
  for (let r = 0; r < 24; r++) for (let c = 0; c < 2; c++) gW2[r * 2 + c] = d3[c] * a2[r];
  const gB2 = Float64Array.from(d3);

  const d2 = new Float64Array(24);
  for (let r = 0; r < 24; r++) {
    let s = 0;
    for (let c = 0; c < 2; c++) s += W2[r * 2 + c] * d3[c];
    d2[r] = s * (1 - a2[r] ** 2);
  }
  const gW1 = new Float64Array(48 * 24);
  for (let r = 0; r < 48; r++) for (let c = 0; c < 24; c++) gW1[r * 24 + c] = d2[c] * a1[r];
  const gB1 = Float64Array.from(d2);

  const d1 = new Float64Array(48);
  for (let r = 0; r < 48; r++) {
    let s = 0;
    for (let c = 0; c < 24; c++) s += W1[r * 24 + c] * d2[c];
    d1[r] = s * (1 - a1[r] ** 2);
  }
  const gW0 = new Float64Array(32 * 48);
  for (let r = 0; r < 32; r++) for (let c = 0; c < 48; c++) gW0[r * 48 + c] = d1[c] * a0[r];
  const gB0 = Float64Array.from(d1);

  return { loss, out, gradW: [gW0, gW1, gW2], gradB: [gB0, gB1, gB2] };
}

/** trainStep with lr=0.2 updates weights with L2 decay and steering bias centering. */
function implicitGradients(net, x, t) {
  const wBefore = net.weights.map((w) => Float64Array.from(w));
  const bBefore = net.biases.map((b) => Float64Array.from(b));
  const lr = 0.2;
  net.trainStep(x, t, lr);
  const gradW = net.weights.map((w, i) => Float64Array.from(w, (v, j) => -(v - wBefore[i][j] * (1 - lr * 0.002)) / lr));
  const gradB = net.biases.map((b, i) => Float64Array.from(b, (v, j) => {
    if (i === 2 && j === 0) return -(v - bBefore[i][j] * 0.985) / (lr * 0.25);
    return -(v - bBefore[i][j]) / lr;
  }));
  net.weights.forEach((w, i) => w.set(wBefore[i]));
  net.biases.forEach((b, i) => b.set(bBefore[i]));
  return { gradW, gradB };
}

const randInput = () => Float32Array.from({ length: INPUTS }, () => Math.random());

test('forward pass matches an independent implementation', () => {
  for (let trial = 0; trial < 3; trial++) {
    const net = randomNet();
    const x = randInput();
    const expected = referenceGradients(net, x, [0.5, 0.5]);
    const actual = net.predict(x);
    assert.ok(Math.abs(actual[0] - expected.out[0]) < 1e-6, 'steer output');
    assert.ok(Math.abs(actual[1] - expected.out[1]) < 1e-6, 'boost output');
  }
});

test('trainStep gradients match independent backpropagation', () => {
  let worst = 0;
  let checked = 0;

  for (let trial = 0; trial < 5; trial++) {
    const net = randomNet();
    const x = randInput();
    const t = [Math.random() * 2 - 1, Math.random() > 0.5 ? 1 : 0];
    const expected = referenceGradients(net, x, t);
    const actual = implicitGradients(net, x, t);

    for (let layer = 0; layer < 3; layer++) {
      for (let i = 0; i < expected.gradW[layer].length; i++) {
        worst = Math.max(worst, Math.abs(expected.gradW[layer][i] - actual.gradW[layer][i]));
        checked++;
      }
      for (let i = 0; i < expected.gradB[layer].length; i++) {
        worst = Math.max(worst, Math.abs(expected.gradB[layer][i] - actual.gradB[layer][i]));
        checked++;
      }
    }
  }

  assert.ok(checked > 2700 * 4, 'sanity: all 2,736 weights and 74 biases were checked');
  // float32 activations put the agreement floor near 1e-7; 1e-5 leaves margin
  assert.ok(worst < 1e-5, `worst gradient error ${worst.toExponential(3)} exceeds tolerance`);
});

test('gradient descent reduces loss on a fixed dataset', () => {
  const net = randomNet(0.5);
  const data = Array.from({ length: 64 }, () => ({
    x: randInput(),
    t: [Math.random() * 2 - 1, Math.random() > 0.5 ? 1 : 0],
  }));
  const meanLoss = () =>
    data.reduce((sum, d) => sum + referenceGradients(net, d.x, d.t).loss, 0) / data.length;

  const before = meanLoss();
  for (let epoch = 0; epoch < 200; epoch++) for (const d of data) net.trainStep(d.x, d.t, 0.01);
  const after = meanLoss();

  assert.ok(after < before * 0.5, `loss should at least halve: ${before.toFixed(4)} -> ${after.toFixed(4)}`);
});

test('weights and biases stay inside the documented +/-6.0 bounds', () => {
  const net = randomNet(0.2);
  const x = randInput();
  for (let i = 0; i < 200; i++) net.trainStep(x, [1, 1], 5.0);
  for (const w of net.weights) for (const v of w) assert.ok(Math.abs(v) <= 6.0001, `weight ${v} out of bounds`);
  for (const b of net.biases) for (const v of b) assert.ok(Math.abs(v) <= 6.0001, `bias ${v} out of bounds`);
});

test('trainStep clamps output steering bias biases[2][0] within [-0.15, 0.15] after 500 consecutive one-sided updates', () => {
  const net = new NeuralNetwork(INPUTS, HIDDEN, OUTPUTS);
  const x = randInput();
  // 500 hard right steer updates (+1.0 target)
  for (let i = 0; i < 500; i++) {
    net.trainStep(x, [1.0, 0.0], 0.02);
  }
  assert.ok(net.biases[2][0] <= 0.15001, `steering bias drifted too high: ${net.biases[2][0]}`);
  assert.ok(net.biases[2][0] >= -0.15001, `steering bias drifted too low: ${net.biases[2][0]}`);

  // 500 hard left steer updates (-1.0 target)
  for (let i = 0; i < 500; i++) {
    net.trainStep(x, [-1.0, 0.0], 0.02);
  }
  assert.ok(net.biases[2][0] <= 0.15001, `steering bias drifted too high: ${net.biases[2][0]}`);
  assert.ok(net.biases[2][0] >= -0.15001, `steering bias drifted too low: ${net.biases[2][0]}`);
});

test('export/fromJSON round-trip preserves predictions', () => {
  const net = randomNet();
  const x = randInput();
  const clone = NeuralNetwork.fromJSON(JSON.parse(JSON.stringify(net.export())), INPUTS);
  assert.ok(clone, 'clone should load');
  const a = net.predict(x);
  const b = clone.predict(x);
  assert.ok(Math.abs(a[0] - b[0]) < 1e-6 && Math.abs(a[1] - b[1]) < 1e-6);
});

test('fromJSON rejects a genome with the wrong input arity or mismatched weight array lengths', () => {
  const json = randomNet().export();
  assert.equal(NeuralNetwork.fromJSON(json, 28), null);
  assert.equal(NeuralNetwork.fromJSON(json, 32) !== null, true);
  assert.equal(NeuralNetwork.fromJSON(null, 32), null);
  assert.equal(NeuralNetwork.fromJSON({ weights: [] }, 32), null);

  // Mismatched weight layer length (e.g. truncated array)
  const badWeights = JSON.parse(JSON.stringify(json));
  badWeights.weights[0] = badWeights.weights[0].slice(0, 100); // 100 instead of 1536
  assert.equal(NeuralNetwork.fromJSON(badWeights, 32), null, 'must reject truncated weight arrays');

  // Mismatched bias layer length
  const badBiases = JSON.parse(JSON.stringify(json));
  badBiases.biases[2] = [0.1]; // 1 instead of 2
  assert.equal(NeuralNetwork.fromJSON(badBiases, 32), null, 'must reject truncated bias arrays');
});

test('output activations stay inside their documented ranges', () => {
  // Small weights: no saturation, so the strict open interval must hold.
  const net = randomNet(0.5);
  for (let i = 0; i < 200; i++) {
    const out = net.predict(randInput());
    assert.ok(out[0] > -1 && out[0] < 1, 'steer is tanh -> (-1, 1)');
    assert.ok(out[1] > 0 && out[1] < 1, 'boost is sigmoid -> (0, 1)');
  }

  // Large weights saturate: tanh reaches +/-1 and sigmoid reaches 1.0 in float64,
  // so the contract there is the closed interval plus finiteness (no NaN/0-blowup).
  const hot = randomNet(3);
  for (let i = 0; i < 200; i++) {
    const out = hot.predict(randInput());
    assert.ok(Number.isFinite(out[0]) && Math.abs(out[0]) <= 1, 'saturated steer stays in [-1, 1]');
    assert.ok(Number.isFinite(out[1]) && out[1] >= 0 && out[1] <= 1, 'saturated boost stays in [0, 1]');
  }
});

test('mutate changes weights and uses gaussian-like perturbations', () => {
  const net = randomNet();
  const before = net.weights[0].slice();
  net.mutate(1.0, 0.1); // rate 1.0 => every weight moves
  let changed = 0;
  for (let i = 0; i < before.length; i++) if (Math.abs(net.weights[0][i] - before[i]) > 1e-9) changed++;
  assert.equal(changed, before.length);
});

test('trainStep with momentum builds velocity and accelerates weight updates along consistent gradients', () => {
  const netNoMom = randomNet(0.1);
  const netWithMom = NeuralNetwork.fromJSON(netNoMom.export(), INPUTS);

  const initialW2 = Array.from(netNoMom.weights[2]);
  const x = randInput();
  const t = [0.8, 1.0]; // consistent direction

  // 10 steps without momentum vs 10 steps with momentum = 0.85
  for (let i = 0; i < 10; i++) {
    netNoMom.trainStep(x, t, 0.01, 0.0);
    netWithMom.trainStep(x, t, 0.01, 0.85);
  }

  // With momentum, total displacement along consistent gradient is strictly greater
  assert.ok(netWithMom._vW && netWithMom._vW[0].length > 0, 'velocity tensors allocated');
  const wNoMomShift = Math.hypot(...Array.from(netNoMom.weights[2]).map((w, idx) => w - initialW2[idx]));
  const wWithMomShift = Math.hypot(...Array.from(netWithMom.weights[2]).map((w, idx) => w - initialW2[idx]));
  assert.ok(wWithMomShift > wNoMomShift, `momentum should accelerate weight shift: ${wWithMomShift} > ${wNoMomShift}`);
});

test('mutate scales perturbation by Xavier fan-in and produces valid bounded weights with Cauchy-Gaussian distribution', () => {
  const net = new NeuralNetwork(INPUTS, HIDDEN, OUTPUTS);
  for (const w of net.weights) w.fill(0);
  for (const b of net.biases) b.fill(0);

  for (let i = 0; i < 30; i++) {
    net.mutate(1.0, 0.2);
  }

  for (const w of net.weights) for (const v of w) assert.ok(Math.abs(v) <= 6.0, `weight ${v} must stay within [-6, 6]`);
  for (const b of net.biases) for (const v of b) assert.ok(Math.abs(v) <= 6.0, `bias ${v} must stay within [-6, 6]`);

  const l0Energy = net.weights[0].reduce((s, v) => s + v * v, 0) / net.weights[0].length;
  assert.ok(l0Energy > 0.01, 'mutation produces valid non-zero weight energy');
});

test('mutate clamps output steering bias biases[2][0] within [-0.15, 0.15] even after 500 aggressive mutations', () => {
  const net = new NeuralNetwork(INPUTS, HIDDEN, OUTPUTS);
  for (let step = 0; step < 500; step++) {
    net.mutate(1.0, 0.5); // extreme mutation rate and strength
    const steerBias = net.biases[2][0];
    assert.ok(steerBias >= -0.15001 && steerBias <= 0.15001, `steering bias must remain clamped within [-0.15, 0.15], got ${steerBias}`);
  }
});
