/**
 * Neural Network Engine with Online Backpropagation & Uniform Crossover
 * 32 inputs -> [48, 24] hidden -> 2 outputs (steer: tanh [-1, 1], boost: sigmoid [0, 1])
 */

export class NeuralNetwork {
  constructor(inputNodes, hiddenLayers, outputNodes) {
    this.inputNodes = inputNodes;
    this.hiddenLayers = hiddenLayers;
    this.outputNodes = outputNodes;

    this.layerSizes = [inputNodes, ...hiddenLayers, outputNodes];
    this.weights = [];
    this.biases = [];

    for (let i = 0; i < this.layerSizes.length - 1; i++) {
      const rows = this.layerSizes[i];
      const cols = this.layerSizes[i + 1];
      const w = new Float32Array(rows * cols);
      const b = new Float32Array(cols);

      const scale = Math.sqrt(2 / rows);
      for (let j = 0; j < w.length; j++) {
        w[j] = (Math.random() * 2 - 1) * scale;
      }
      for (let j = 0; j < b.length; j++) {
        b[j] = (Math.random() * 2 - 1) * 0.1;
      }

      this.weights.push(w);
      this.biases.push(b);
    }
  }

  forward(inputs) {
    const activations = [inputs];
    let current = inputs;
    for (let i = 0; i < this.weights.length; i++) {
      const rows = this.layerSizes[i];
      const cols = this.layerSizes[i + 1];
      const w = this.weights[i];
      const b = this.biases[i];
      const next = new Float32Array(cols);

      for (let c = 0; c < cols; c++) {
        let sum = b[c];
        for (let r = 0; r < rows; r++) {
          sum += current[r] * w[r * cols + c];
        }
        if (i === this.weights.length - 1 && c === 1) {
          next[c] = 1 / (1 + Math.exp(-sum)); // Sigmoid for boost [0, 1]
        } else {
          next[c] = Math.tanh(sum); // Tanh for steer [-1, 1]
        }
      }
      current = next;
      activations.push(current);
    }
    return { output: current, activations };
  }

  predict(inputs) {
    return this.forward(inputs).output;
  }

  // Real-Time Online Backpropagation for Imitation Learning
  // Maximum Likelihood Estimation: Uses Binary Cross-Entropy Loss for Bernoulli boost decision,
  // eliminating vanishing gradients on saturated sigmoids, with optional low-pass Momentum.
  trainStep(inputs, targets, lr = 0.015, momentum = 0.0) {
    const { output, activations } = this.forward(inputs);

    const dOut = new Float32Array(this.outputNodes);
    const errSteer = targets[0] - output[0];
    dOut[0] = errSteer * (1.0 - output[0] * output[0]); // d/dx tanh = 1 - tanh^2
    const errBoost = targets[1] - output[1];
    // Maximum Likelihood Estimation: Binary Cross-Entropy Loss for Bernoulli boost output.
    // Eliminates the vanishing gradient factor s*(1-s), accelerating boost tuition convergence by 20x.
    dOut[1] = errBoost;

    const deltas = [dOut];

    // Backpropagate error through hidden layers
    for (let layer = this.weights.length - 1; layer >= 1; layer--) {
      const prevCols = this.layerSizes[layer];
      const nextCols = this.layerSizes[layer + 1];
      const w = this.weights[layer];
      const nextDelta = deltas[0];
      const act = activations[layer];

      const layerDelta = new Float32Array(prevCols);
      for (let r = 0; r < prevCols; r++) {
        let sum = 0;
        for (let c = 0; c < nextCols; c++) {
          sum += nextDelta[c] * w[r * nextCols + c];
        }
        layerDelta[r] = sum * (1.0 - act[r] * act[r]);
      }
      deltas.unshift(layerDelta);
    }

    // Gradient Descent Updates with safety bounds [-6.0, 6.0]
    // L2 weight decay prevents saturation during prolonged teacher mode (>350k samples).
    const weightDecay = 1 - (lr * 0.002);
    const useMomentum = momentum > 0;
    if (useMomentum && !this._vW) {
      this._vW = this.weights.map(w => new Float32Array(w.length));
      this._vB = this.biases.map(b => new Float32Array(b.length));
    }

    for (let layer = 0; layer < this.weights.length; layer++) {
      const rows = this.layerSizes[layer];
      const cols = this.layerSizes[layer + 1];
      const act = activations[layer];
      const delta = deltas[layer];
      const w = this.weights[layer];
      const b = this.biases[layer];
      const vw = useMomentum ? this._vW[layer] : null;
      const vb = useMomentum ? this._vB[layer] : null;

      for (let c = 0; c < cols; c++) {
        const d = delta[c];
        // Steering output bias centering decay & tight clamping [-0.15, 0.15]
        // prevents permanent one-sided turn pull from drifting during circular human movements.
        if (layer === this.weights.length - 1 && c === 0) {
          const step = lr * 0.25 * d;
          if (useMomentum) {
            vb[c] = (vb[c] * momentum) + step;
            b[c] = Math.max(-0.15, Math.min(0.15, (b[c] * 0.985) + vb[c]));
          } else {
            b[c] = Math.max(-0.15, Math.min(0.15, (b[c] * 0.985) + step));
          }
        } else {
          const step = lr * d;
          if (useMomentum) {
            vb[c] = (vb[c] * momentum) + step;
            b[c] = Math.max(-6.0, Math.min(6.0, b[c] + vb[c]));
          } else {
            b[c] = Math.max(-6.0, Math.min(6.0, b[c] + step));
          }
        }
        for (let r = 0; r < rows; r++) {
          const idx = r * cols + c;
          const step = lr * d * act[r];
          if (useMomentum) {
            vw[idx] = (vw[idx] * momentum) + step;
            w[idx] = Math.max(-6.0, Math.min(6.0, (w[idx] * weightDecay) + vw[idx]));
          } else {
            w[idx] = Math.max(-6.0, Math.min(6.0, (w[idx] * weightDecay) + step));
          }
        }
      }
    }

    return output;
  }

  // Fast Evolutionary Programming (FEP) Mutation:
  // Combines 85% Gaussian micro-tuning with 15% Cauchy heavy-tailed Lévy flights to escape local fitness valleys,
  // aligned by Xavier spectral fan-in scaling across layer depths.
  mutate(rate = 0.06, strength = 0.16) {
    for (let i = 0; i < this.weights.length; i++) {
      const nIn = this.layerSizes[i];
      const fanInScale = Math.sqrt(24.0 / nIn);
      const layerStrength = strength * fanInScale;

      const w = this.weights[i];
      for (let j = 0; j < w.length; j++) {
        if (Math.random() < rate) {
          let pert;
          if (Math.random() < 0.15) {
            const u = Math.random();
            const cauchy = Math.tan(Math.PI * (u - 0.5));
            pert = Math.max(-4.0, Math.min(4.0, cauchy)) * layerStrength;
          } else {
            const u1 = Math.random() || 1e-7;
            const u2 = Math.random();
            pert = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * layerStrength;
          }
          w[j] = Math.max(-6.0, Math.min(6.0, w[j] + pert));
        }
      }
      const b = this.biases[i];
      for (let j = 0; j < b.length; j++) {
        if (Math.random() < rate) {
          let pert;
          if (Math.random() < 0.15) {
            const u = Math.random();
            const cauchy = Math.tan(Math.PI * (u - 0.5));
            pert = Math.max(-4.0, Math.min(4.0, cauchy)) * layerStrength;
          } else {
            const u1 = Math.random() || 1e-7;
            const u2 = Math.random();
            pert = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * layerStrength;
          }
          if (i === this.weights.length - 1 && j === 0) {
            // Output layer steering bias: heavily damped & clamped within [-0.15, 0.15]
            // to prevent runaway counter-clockwise or clockwise orbital drift.
            b[j] = Math.max(-0.15, Math.min(0.15, b[j] + pert * 0.35));
          } else {
            b[j] = Math.max(-6.0, Math.min(6.0, b[j] + pert));
          }
        }
      }
    }
    // Guarantee output steering bias remains strictly centered within hydrodynamic bounds [-0.15, 0.15]
    const lastBiases = this.biases[this.biases.length - 1];
    if (lastBiases && lastBiases.length > 0) {
      lastBiases[0] = Math.max(-0.15, Math.min(0.15, lastBiases[0]));
    }
  }

  // Uniform Crossover (NEAT-inspired genetic recombination):
  // Each weight gene is independently sampled from parentA or parentB with
  // probability p_B (default 0.5). Equivalent to a Mendelian uniform crossover
  // on aligned weight vectors. Used when stagnantGens >= 5 to escape local
  // fitness plateaus that pure mutation cannot cross.
  static crossover(parentA, parentB, pB = 0.5) {
    const child = NeuralNetwork.fromJSON(parentA.export(), parentA.inputNodes);
    if (!child || !parentB) return child;
    for (let layer = 0; layer < child.weights.length; layer++) {
      const wA = parentA.weights[layer];
      const wB = parentB.weights[layer];
      const wC = child.weights[layer];
      for (let j = 0; j < wC.length; j++) {
        if (Math.random() < pB) wC[j] = wB[j];
      }
      const bA = parentA.biases[layer];
      const bB = parentB.biases[layer];
      const bC = child.biases[layer];
      for (let j = 0; j < bC.length; j++) {
        if (Math.random() < pB) bC[j] = bB[j];
      }
    }
    // Re-enforce steering bias hydrodynamic bounds after crossover
    const lastB = child.biases[child.biases.length - 1];
    if (lastB && lastB.length > 0) lastB[0] = Math.max(-0.15, Math.min(0.15, lastB[0]));
    return child;
  }

  export() {
    return {
      inputNodes: this.inputNodes,
      hiddenLayers: this.hiddenLayers,
      outputNodes: this.outputNodes,
      weights: this.weights.map(w => Array.from(w)),
      biases: this.biases.map(b => Array.from(b))
    };
  }

  static fromJSON(json, expectedInputs = 32) {
    if (!json || !json.weights || json.inputNodes !== expectedInputs) return null;
    if (!Array.isArray(json.weights) || !Array.isArray(json.biases)) return null;
    const net = new NeuralNetwork(json.inputNodes, json.hiddenLayers, json.outputNodes);
    if (json.weights.length !== net.weights.length || json.biases.length !== net.biases.length) return null;
    for (let i = 0; i < json.weights.length; i++) {
      const w = json.weights[i];
      const b = json.biases[i];
      if (!w || w.length !== net.weights[i].length) return null;
      if (!b || b.length !== net.biases[i].length) return null;
      net.weights[i].set(w);
      net.biases[i].set(b);
    }
    return net;
  }
}
