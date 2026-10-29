// ==UserScript==
// @name         Slither.io Autonomous Machine Learning Agent (SlitherML)
// @namespace    https://slither.io/
// @version      5.10.0
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

/**
 * Slither.io Autonomous Machine Learning Agent (Console Paste Version)
 * Hybrid Neuroevolution + Imitation Learning Engine v5.10.0 (Forensic Deductor & Pathfinder Flight Recorder Edition)
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

  console.log('%c[SlitherML] 🚀 Initializing Hybrid Neuroevolution + Imitation Learning Agent v5.10.0 (Forensic Deductor & Pathfinder Flight Recorder Edition)...', 'color: #00f5d4; font-weight: bold; font-size: 14px;');

  const STORAGE_KEY = 'slither_ml_champion_weights_v4';

  const pageWindow = (typeof unsafeWindow !== 'undefined' && unsafeWindow) ? unsafeWindow : window;

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
  // 1. NEURAL NETWORK ENGINE (With Backprop)
  // ==========================================
  class NeuralNetwork {
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

  // ==========================================
  // 1.5 FLIGHT RECORDER & FORENSIC DEDUCTOR
  // ==========================================
  class FlightRecorder {
    constructor(options = {}) {
      this.bufferCapacity = options.bufferCapacity || 60; // 60 frames (~1.0s at 60 FPS)
      this.frameBuffer = [];
      this.bufferIndex = 0;

      // Compact Pathfinder Breadcrumbs
      this.maxWaypoints = 60;
      this.waypointIntervalSec = 2.5;
      this.lastWaypointTime = -999;
      this.waypoints = []; // [[x, y, Math.round(t), Math.round(score), stance], ...]
      this.lastLifeWaypoints = [];

      // Growth Milestones
      this.milestoneTargets = [500, 1000, 2500, 5000, 10000, 15000, 20000];
      this.growthMilestones = {}; // { "500": 14.2, "1000": 32.5, ... }

      // Spatial Zone Distribution (frame counters)
      this.zoneCounts = { center: 0, midfield: 0, perimeter: 0 };
      this.totalFrames = 0;

      // Stance Frame Counters
      this.stanceCounts = {
        WANDER: 0,
        FORAGE: 0,
        ATTACK: 0,
        COIL: 0,
        CONSTRICT: 0,
        DEFEND: 0,
        VACUUM: 0
      };

      // Significant Incidents Log (max 25 per life)
      this.maxIncidents = 25;
      this.incidents = [];

      // Autopsy & Episode History
      this.lastDeathAutopsy = null;
      this.recentEpisodes = []; // max 10
      this.maxRecentEpisodes = 10;
    }

    resetLife() {
      if (this.waypoints.length > 0) {
        this.lastLifeWaypoints = this.waypoints.slice(-40);
      }
      this.frameBuffer = [];
      this.bufferIndex = 0;
      this.waypoints = [];
      this.lastWaypointTime = -999;
      this.growthMilestones = {};
      this.zoneCounts = { center: 0, midfield: 0, perimeter: 0 };
      this.totalFrames = 0;
      for (const k of Object.keys(this.stanceCounts)) {
        this.stanceCounts[k] = 0;
      }
      this.incidents = [];
    }

    recordFrame(snapshot) {
      if (!snapshot) return;
      this.totalFrames++;

      // 1. Rolling FIFO Frame Buffer
      if (this.frameBuffer.length < this.bufferCapacity) {
        this.frameBuffer.push(snapshot);
      } else {
        this.frameBuffer[this.bufferIndex] = snapshot;
        this.bufferIndex = (this.bufferIndex + 1) % this.bufferCapacity;
      }

      // 2. Stance Counter
      const st = snapshot.stance || 'FORAGE';
      if (typeof this.stanceCounts[st] === 'number') {
        this.stanceCounts[st]++;
      } else {
        this.stanceCounts[st] = 1;
      }

      // 3. Spatial Zone Tracking (Arena radius grd is ~21,600)
      const distFromOrigin = Math.hypot(snapshot.x || 0, snapshot.y || 0);
      const normDist = Number.isFinite(snapshot.normWallDist) ? snapshot.normWallDist : (distFromOrigin / 21600);
      if (normDist < 0.45) {
        this.zoneCounts.center++;
      } else if (normDist < 0.82) {
        this.zoneCounts.midfield++;
      } else {
        this.zoneCounts.perimeter++;
      }

      // 4. Growth Milestones
      const currentScore = snapshot.score || 0;
      for (let i = 0; i < this.milestoneTargets.length; i++) {
        const target = this.milestoneTargets[i];
        if (currentScore >= target && typeof this.growthMilestones[target] !== 'number') {
          this.growthMilestones[target] = Math.round((snapshot.t || 0) * 10) / 10;
        }
      }

      // 5. Pathfinder Breadcrumbs (compact downsampled waypoints)
      const t = snapshot.t || 0;
      if (t - this.lastWaypointTime >= this.waypointIntervalSec) {
        this.lastWaypointTime = t;
        if (this.waypoints.length >= this.maxWaypoints) {
          this.waypoints.shift();
        }
        this.waypoints.push([
          Math.round(snapshot.x || 0),
          Math.round(snapshot.y || 0),
          Math.round(t),
          Math.round(currentScore),
          snapshot.stance || 'FORAGE'
        ]);
      }
    }

    logIncident(type, details = {}) {
      if (this.incidents.length >= this.maxIncidents) return;
      this.incidents.push({
        t: Math.round((details.t || 0) * 10) / 10,
        type,
        x: Math.round(details.x || 0),
        y: Math.round(details.y || 0),
        info: details.info || ''
      });
    }

    deduceDeathAutopsy(meta = {}) {
      const orderedFrames = this._getOrderedFrames();
      const last = orderedFrames[orderedFrames.length - 1] || null;
      const prev = orderedFrames.length > 5 ? orderedFrames[orderedFrames.length - 5] : last;

      let cause = 'SUDDEN_IMPACT';
      let description = 'Sudden collision or server despawn';
      let confidence = 0.5;

      if (last) {
        const headRadius = 14 * ((last.score ? Math.sqrt(last.score / 15) : 1) || 1);
        const distFromOrigin = Math.hypot(last.x || 0, last.y || 0);
        const nearWall = (last.normWallDist >= 0.94) || (distFromOrigin >= 21600 - 350);

        if (nearWall) {
          cause = 'ARENA_WALL';
          description = 'Collision with outer perimeter boundary wall';
          confidence = 0.95;
        } else if (last.stance === 'COIL' || (prev && prev.stance === 'COIL')) {
          cause = 'ENCIRCLED_TRAP';
          description = 'Suffocated inside closed predator coil without open exit slit';
          confidence = 0.88;
        } else if (last.threatType === 'head' && last.threatDist <= headRadius * 2.8) {
          const isFacingUs = Math.abs(last.threatAngle || 0) <= Math.PI * 0.45;
          if (isFacingUs && (last.threatMassRatio || 1.0) > 1.1) {
            cause = 'HEAD_ON_RAM';
            description = 'Frontal collision with larger predator head';
            confidence = 0.92;
          } else {
            cause = 'HEAD_ON_JOUST';
            description = 'Simultaneous head-to-head collision';
            confidence = 0.85;
          }
        } else if (last.threatType === 'body' && last.threatDist <= headRadius * 3.2) {
          if (last.stance === 'ATTACK' || (prev && prev.stance === 'ATTACK')) {
            cause = 'ATTACK_OVEREXTENDED';
            description = 'Counter-intercepted while executing offensive strike';
            confidence = 0.86;
          } else if ((last.threatSpeed || 5.9) > 8.0) {
            cause = 'FLANK_CUTOFF';
            description = 'Cut off by high-speed enemy boost trajectory';
            confidence = 0.90;
          } else {
            cause = 'BODY_COLLISION';
            description = 'Impacted enemy body segments';
            confidence = 0.80;
          }
        } else if (last.isBoosting && meta.finalScore < 15 && meta.survivalTime > 10) {
          cause = 'BOOST_STARVATION';
          description = 'Starved mass down to zero fuel threshold during sprint';
          confidence = 0.82;
        } else if (last.threatDist <= headRadius * 4.0) {
          cause = 'BODY_COLLISION';
          description = 'Unavoidable collision with dense body segments';
          confidence = 0.75;
        }
      }

      const autopsy = {
        cause,
        description,
        confidence,
        fatalStance: last ? last.stance : 'UNKNOWN',
        survivalTime: Math.round((meta.survivalTime || 0) * 10) / 10,
        finalScore: Math.round(meta.finalScore || 0),
        peakLength: Math.round(meta.peakLength || 0),
        isHumanDriven: Boolean(meta.isHumanDriven),
        generation: meta.generation || 1,
        episode: meta.episode || 1,
        timestamp: Date.now(),
        killerProfile: last ? {
          type: last.threatType || 'unknown',
          dist: Math.round((last.threatDist || 0) * 10) / 10,
          massRatio: Math.round((last.threatMassRatio || 1.0) * 100) / 100,
          speed: Math.round((last.threatSpeed || 5.9) * 10) / 10
        } : null,
        preCrashTelemetry: last ? {
          x: Math.round(last.x || 0),
          y: Math.round(last.y || 0),
          speed: Math.round((last.sp || 5.9) * 10) / 10,
          turnDelta: Math.round((last.turnDelta || 0) * 1000) / 1000,
          isBoosting: Boolean(last.isBoosting),
          dominantRayIdx: last.dominantRayIdx ?? -1,
          dominantRayDist: Math.round((last.dominantRayDist || 1.0) * 100) / 100
        } : null,
        incidentsCount: this.incidents.length
      };

      this.lastDeathAutopsy = autopsy;
      this.lastLifeWaypoints = this.waypoints.slice(-40);

      this.recentEpisodes.push({
        gen: autopsy.generation,
        ep: autopsy.episode,
        durationS: autopsy.survivalTime,
        peakLength: autopsy.peakLength,
        finalScore: autopsy.finalScore,
        deathCause: autopsy.cause,
        fatalStance: autopsy.fatalStance
      });
      if (this.recentEpisodes.length > this.maxRecentEpisodes) {
        this.recentEpisodes.shift();
      }

      return autopsy;
    }

    _getOrderedFrames() {
      if (this.frameBuffer.length < this.bufferCapacity) {
        return [...this.frameBuffer];
      }
      return [
        ...this.frameBuffer.slice(this.bufferIndex),
        ...this.frameBuffer.slice(0, this.bufferIndex)
      ];
    }

    _calculateStanceDutyCycles() {
      const total = Object.values(this.stanceCounts).reduce((a, b) => a + b, 0);
      if (total === 0) return {};
      const result = {};
      for (const [stance, count] of Object.entries(this.stanceCounts)) {
        if (count > 0) {
          result[stance] = `${((count / total) * 100).toFixed(1)}%`;
        }
      }
      return result;
    }

    _calculateZoneDistribution() {
      const total = this.zoneCounts.center + this.zoneCounts.midfield + this.zoneCounts.perimeter;
      if (total === 0) return { center: '0%', midfield: '0%', perimeter: '0%' };
      return {
        center: `${((this.zoneCounts.center / total) * 100).toFixed(1)}%`,
        midfield: `${((this.zoneCounts.midfield / total) * 100).toFixed(1)}%`,
        perimeter: `${((this.zoneCounts.perimeter / total) * 100).toFixed(1)}%`
      };
    }

    getTelemetrySummary() {
      return {
        version: '1.0.0',
        lastDeathAutopsy: this.lastDeathAutopsy,
        recentEpisodeHistory: [...this.recentEpisodes],
        growthMilestones: { ...this.growthMilestones },
        stanceDutyCycles: this._calculateStanceDutyCycles(),
        zoneDistribution: this._calculateZoneDistribution(),
        pathfinderBreadcrumbs: this.lastLifeWaypoints.length > 0
          ? [...this.lastLifeWaypoints]
          : this.waypoints.slice(-40)
      };
    }

    loadTelemetry(telemetry) {
      if (!telemetry || typeof telemetry !== 'object') return;
      if (telemetry.lastDeathAutopsy) {
        this.lastDeathAutopsy = telemetry.lastDeathAutopsy;
      }
      if (Array.isArray(telemetry.recentEpisodeHistory)) {
        this.recentEpisodes = telemetry.recentEpisodeHistory.slice(-this.maxRecentEpisodes);
      }
      if (telemetry.growthMilestones && typeof telemetry.growthMilestones === 'object') {
        this.growthMilestones = { ...telemetry.growthMilestones };
      }
      if (Array.isArray(telemetry.pathfinderBreadcrumbs)) {
        this.lastLifeWaypoints = telemetry.pathfinderBreadcrumbs.slice(-40);
      }
    }
  }

  // ==========================================
  // 2. HYBRID AGENT (EVOLUTION + IMITATION)
  // ==========================================
  class SlitherMLAgent {
    constructor() {
      this.flightRecorder = new FlightRecorder();
      this.rayCount = 16;
      this.baseSensorRange = 520;
      this.foodSectorCount = 8;
      this.inputCount = 32;
      this.brain = new NeuralNetwork(this.inputCount, [48, 24], 2);
      this.championBrain = null;

      // Multi-Episode Evaluation Architecture
      this.episodesPerGenome = 3;
      this.currentEpisode = 1;
      this.episodeFitnesses = [];

      // A genome produced by teacher mode is only a CANDIDATE: it must win an
      // autonomous 3-life rollout before it is ever written to the champion slot.
      this.candidatePending = false;
      this.candidateSource = null;
      // Set when a life was partly human-driven, so it cannot score the genome.
      this.abandonCurrentEpisode = false;

      this.generation = 1;
      this.fitness = 0;
      this.bestFitness = 0;
      this.allTimePeakFitness = 0;
      this.peakLength = 0;
      this.previousScore = 0;
      this.survivalTime = 0;
      this.totalFoodEaten = 0;

      // Teacher / Imitation Learning Metrics
      this.teacherSamples = 0;
      this.lifeTeacherSamples = 0;
      this.touchedByHuman = false;

      // Enemy Velocity & Trajectory Tracker: Map(id -> { xx, yy, vx, vy, timestamp })
      this.enemyTracker = new Map();
      this.isCenteredWorld = true;

      // Short-Term Tactical Memory Buffers (EMA decay = 0.85)
      this.prevTurnEma = 0;
      this.leftThreatEma = 0;
      this.rightThreatEma = 0;

      this.activeRays = [];
      this.foodSectors = new Float32Array(this.foodSectorCount);
      this.detectedFoods = [];
      this.targetAngle = 0;
      this.isBoosting = false;
      this.boostDuration = 0;
      this.boostCooldown = 0;
      this.foodTargetError = null;

      // Boost governor budgets. Boosting must pay for itself: a sprint is only
      // allowed when it buys a dense food cluster, and the fuel meter caps the
      // duty cycle so the snake cannot bleed its length away.
      this.boostFuel = 1;              // 0..1 stamina
      this.boostBurstSeconds = 1.2;    // hard cap on one sprint
      this.boostBurnSeconds = 1.6;     // seconds of sprinting to drain a full meter
      this.boostRefillSeconds = 4.0;   // seconds to refill fully while gliding
      this.boostRestSeconds = 2.0;     // mandatory glide after a sprint
      this.minBoostScore = 80;         // never boost below this score
      this.foodSteerEma = 0;           // smoothed foraging command (damps target flip-flop)
      this.lastDecisionTime = performance.now();

      // Combat & Tactical Stance State Machine ('FORAGE' | 'DEFEND' | 'ATTACK' | 'COIL' | 'CONSTRICT' | 'VACUUM')
      this.combatStance = 'FORAGE';
      this.attackTarget = null;
      this.cutOffThreat = null;
      this.constrictTarget = null;
      this.constrictDuration = 0;
      this.coilSign = 0;
      this.lastKillEpicenter = null;
      this.vacuumActiveUntil = 0;
      this.massRatio = 1.0;
      this.threatScale = 1.0;
      this.closestObstacleType = 'none';
      this.frontObstacleType = 'none';
      this.selfShieldSide = 0;
      this.encirclementEscape = null;
      this.slipstreamTarget = null;
      this.teacherQuarantineBuffer = [];
      this.teacherQuarantineCapacity = 45; // ~0.75s Failure-Trauma Filter
      this.kills = 0;
      this.stagnantGens = 0;
      this.allTimePeakLength = 0;
      this.lastLoadedTimestamp = 0;
      this._saveCheckpoint = () => this._saveChampionWeights();

      this._loadChampionWeights();
    }

    applySavedState(parsed) {
      if (!parsed || typeof parsed !== 'object') return false;
      if (!parsed.brain) return false;

      const champion = NeuralNetwork.fromJSON(parsed.brain, this.inputCount);
      if (!champion) return false;

      if (parsed.hasChampion === true || (typeof parsed.bestFitness === 'number' && parsed.bestFitness > 0)) {
        this.championBrain = champion;
      } else {
        this.championBrain = null;
      }
      const active = parsed.activeBrain ? NeuralNetwork.fromJSON(parsed.activeBrain, this.inputCount) : null;
      this.brain = active || champion;

      // Sanitize steering bias on loaded brains to keep within hydrodynamic limits [-0.15, 0.15]
      if (this.championBrain && this.championBrain.biases && this.championBrain.biases.length > 0) {
        const champOutB = this.championBrain.biases[this.championBrain.biases.length - 1];
        if (champOutB && champOutB.length > 0) {
          champOutB[0] = Math.max(-0.15, Math.min(0.15, champOutB[0]));
        }
      }
      if (this.brain && this.brain.biases && this.brain.biases.length > 0) {
        const brainOutB = this.brain.biases[this.brain.biases.length - 1];
        if (brainOutB && brainOutB.length > 0) {
          brainOutB[0] = Math.max(-0.15, Math.min(0.15, brainOutB[0]));
        }
      }

      this.bestFitness = typeof parsed.bestFitness === 'number' ? parsed.bestFitness : 0;
      this.allTimePeakFitness = typeof parsed.allTimePeakFitness === 'number'
        ? Math.max(parsed.allTimePeakFitness, this.bestFitness)
        : Math.max(this.allTimePeakFitness || 0, this.bestFitness);
      this.generation = typeof parsed.generation === 'number' ? parsed.generation : 1;
      this.currentEpisode = typeof parsed.currentEpisode === 'number' ? parsed.currentEpisode : 1;
      this.episodeFitnesses = Array.isArray(parsed.episodeFitnesses) ? [...parsed.episodeFitnesses] : [];
      this.candidatePending = Boolean(parsed.candidatePending);
      this.candidateSource = parsed.candidateSource || null;
      this.kills = typeof parsed.kills === 'number' ? parsed.kills : 0;
      this.teacherSamples = typeof parsed.teacherSamples === 'number' ? parsed.teacherSamples : 0;
      this.lifeTeacherSamples = 0;
      this.allTimePeakLength = typeof parsed.allTimePeakLength === 'number' ? parsed.allTimePeakLength : (this.allTimePeakLength || 0);
      this.stagnantGens = typeof parsed.stagnantGens === 'number' ? parsed.stagnantGens : 0;
      this.lastLoadedTimestamp = parsed.timestamp || Date.now();
      if (parsed.telemetry) {
        this.flightRecorder.loadTelemetry(parsed.telemetry);
      }
      return true;
    }

    _loadChampionWeights() {
      // 1. Try loading v4 champion weights directly from localStorage
      let saved = null;
      try {
        saved = localStorage.getItem(STORAGE_KEY);
      } catch (err) {
        console.warn('[SlitherML] Could not read saved champion weights:', err && err.message ? err.message : err);
      }

      // Check userscript cross-origin storage if available
      if (typeof GM_getValue === 'function') {
        try {
          const gmSaved = GM_getValue(STORAGE_KEY, null);
          if (gmSaved) {
            if (!saved) {
              saved = gmSaved;
            } else {
              const gmParsed = JSON.parse(gmSaved);
              const localParsed = JSON.parse(saved);
              if (
                ((gmParsed.bestFitness || 0) > (localParsed.bestFitness || 0)) ||
                ((gmParsed.generation || 1) > (localParsed.generation || 1)) ||
                ((gmParsed.timestamp || 0) > (localParsed.timestamp || 0))
              ) {
                saved = gmSaved;
              }
            }
          }
        } catch (_) {}
      }

      if (saved) {
        try {
          const parsed = JSON.parse(saved);
          if (this.applySavedState(parsed)) {
            console.log(`%c[SlitherML] 🌟 Resumed Session! Loaded Champion Genome (Gen ${this.generation}, Best: ${Math.floor(this.bestFitness)})`, 'color: #00f5d4; font-weight: bold;');
            return;
          }
        } catch (err) {
          console.warn('[SlitherML] Could not parse saved weights:', err && err.message ? err.message : err);
        }
      }

      // 2. Backward compatibility fallback: Migrate v3 weights if found
      try {
        const oldSaved = localStorage.getItem('slither_ml_champion_weights_v3');
        if (oldSaved) {
          const parsed = JSON.parse(oldSaved);
          if (parsed && parsed.brain && parsed.brain.weights) {
            this._seedBaselineWeights();
            const w0 = this.brain.weights[0];
            const oldW0 = parsed.brain.weights[0];
            const minRows = Math.min(28, parsed.brain.inputNodes || 28);
            for (let r = 0; r < minRows; r++) {
              for (let c = 0; c < 48; c++) {
                w0[r * 48 + c] = oldW0[r * 48 + c];
              }
            }
            if (parsed.brain.weights[1]) this.brain.weights[1].set(parsed.brain.weights[1]);
            if (parsed.brain.weights[2]) this.brain.weights[2].set(parsed.brain.weights[2]);
            if (parsed.brain.biases[0]) this.brain.biases[0].set(parsed.brain.biases[0]);
            if (parsed.brain.biases[1]) this.brain.biases[1].set(parsed.brain.biases[1]);
            if (parsed.brain.biases[2]) this.brain.biases[2].set(parsed.brain.biases[2]);

            this.championBrain = NeuralNetwork.fromJSON(this.brain.export(), this.inputCount);
            this.bestFitness = parsed.bestFitness || 0;
            this.generation = parsed.generation || 1;
            this._saveChampionWeights();
            console.log(`%c[SlitherML] 🔄 Migrated Gen ${this.generation} Champion weights seamlessly from previous session! (Best: ${Math.floor(this.bestFitness)})`, 'color: #00f5d4; font-weight: bold;');
            return;
          }
        }
      } catch (err) {
        console.warn('[SlitherML] Could not migrate v3 champion weights:', err && err.message ? err.message : err);
      }

      this._seedBaselineWeights();
    }

    _seedBaselineWeights() {
      const w0 = this.brain.weights[0];
      const b0 = this.brain.biases[0];
      const w1 = this.brain.weights[1];
      const b1 = this.brain.biases[1];
      const w2 = this.brain.weights[2];
      const b2 = this.brain.biases[2];

      w0.fill(0); b0.fill(0);
      w1.fill(0); b1.fill(0);
      w2.fill(0); b2.fill(0);

      // 1. Ray Proximity Inputs [0..15] -> Hidden1 [0..15]
      for (let r = 0; r < this.rayCount; r++) {
        const relAngle = ((r / this.rayCount) * Math.PI * 2) - Math.PI;
        const lateral = Math.sin(relAngle);
        if (r === 8) {
          w0[8 * 48 + 8] = 2.5; // Keep straight-ahead ray active to inhibit boost
        } else {
          w0[r * 48 + r] = -lateral * 3.4;
        }
      }

      // 2. Food Sector Inputs [16..23] -> Hidden1 [16..23]
      for (let f = 0; f < this.foodSectorCount; f++) {
        const sectorCenter = ((f + 0.5) / this.foodSectorCount) * Math.PI * 2;
        let relAngle = sectorCenter;
        if (relAngle > Math.PI) relAngle -= Math.PI * 2;
        const lateral = Math.sin(relAngle);
        w0[(16 + f) * 48 + (16 + f)] = lateral * 2.8;
      }

      // 3. Arena Center Angle [25] -> Hidden1 [25]
      w0[25 * 48 + 25] = 3.2;

      // 4. Tactical Memory & Congestion [28..31] -> Hidden1 [28..31]
      w0[28 * 48 + 28] = 0.65;  // Turn momentum / smoothing
      w0[29 * 48 + 29] = -1.8;  // Left threat memory -> steer right
      w0[30 * 48 + 30] = 1.8;   // Right threat memory -> steer left
      w0[31 * 48 + 31] = 1.5;   // Congestion detector -> inhibits boost

      // Layer 1 (48 -> 24): Dedicated channels into Hidden2 [0..23]
      for (let i = 0; i < 16; i++) {
        w1[i * 24 + i] = 1.2; // Hidden2 [0..15]: Obstacle rays
      }
      for (let f = 0; f < 8; f++) {
        w1[(16 + f) * 24 + (16 + (f % 4))] = 1.0; // Hidden2 [16..19]: Food sectors
      }
      w1[25 * 24 + 20] = 1.2; // Hidden2 [20]: Center angle pull
      w1[28 * 24 + 21] = 1.0; // Hidden2 [21]: Turn momentum (28)
      w1[29 * 24 + 22] = 1.0; // Hidden2 [22]: Left threat memory (29)
      w1[30 * 24 + 22] = 1.0; // Hidden2 [22]: Right threat memory (30)
      w1[31 * 24 + 23] = 1.2; // Hidden2 [23]: Congestion detector (31)

      // Layer 2 (24 -> 2: output[0] = steer, output[1] = boost)
      for (let i = 0; i < 16; i++) {
        if (i !== 8) w2[i * 2 + 0] = 1.0; // Obstacle avoidance steering
      }
      for (let i = 16; i < 20; i++) {
        w2[i * 2 + 0] = 0.8; // Food hunting steering
      }
      w2[20 * 2 + 0] = 0.9; // Arena center steering
      w2[21 * 2 + 0] = 0.5; // Momentum smoothing
      w2[22 * 2 + 0] = 0.9; // Threat memory steering

      // Boost output (index 1) - Restrained to prevent mass bleeding
      w2[8 * 2 + 1] = -3.5;  // Front obstacle (ray 8) strongly inhibits boost
      w2[16 * 2 + 1] = 1.3;  // Forward food moderately encourages boost
      w2[19 * 2 + 1] = 1.3;  // Forward food moderately encourages boost
      w2[23 * 2 + 1] = -2.2; // High congestion inhibits boost
      b2[1] = -0.95;         // Restraining bias to prevent boost spam
    }

    resetBrain() {
      try {
        localStorage.removeItem(STORAGE_KEY);
      } catch (_) { }
      if (typeof window !== 'undefined' && typeof window.__slitherMLBridgeClear === 'function') {
        window.__slitherMLBridgeClear();
      }
      if (typeof GM_setValue === 'function') {
        try {
          GM_setValue(STORAGE_KEY, '');
        } catch (_) { }
      }
      this.brain = new NeuralNetwork(this.inputCount, [48, 24], 2);
      this.championBrain = null;
      this.generation = 1;
      this.currentEpisode = 1;
      this.episodeFitnesses = [];
      this.bestFitness = 0;
      this.fitness = 0;
      this.survivalTime = 0;
      this.peakLength = 0;
      this.allTimePeakLength = 0;
      this.lastLoadedTimestamp = 0;
      this.previousScore = 0;
      this.teacherSamples = 0;
      this.lifeTeacherSamples = 0;
      this.touchedByHuman = false;
      this.candidatePending = false;
      this.candidateSource = null;
      this.abandonCurrentEpisode = false;
      this.boostDuration = 0;
      this.boostCooldown = 0;
      this.boostFuel = 1;
      this.foodTargetError = null;
      this.enemyTracker.clear();
      this.prevTurnEma = 0;
      this.leftThreatEma = 0;
      this.rightThreatEma = 0;
      this.kills = 0;
      this.stagnantGens = 0;
      this.allTimePeakFitness = 0;
      this.coilSign = 0;
      this.constrictTarget = null;
      this.lastKillEpicenter = null;
      this.vacuumActiveUntil = 0;
      this.flightRecorder = new FlightRecorder();
      this._seedBaselineWeights();
      console.log('%c[SlitherML] 🔄 Brain reset to Gen 1 Baseline Weights.', 'color: #ffb703; font-weight: bold;');
    }

    static getApexPreset() {
      if (SlitherMLAgent._cachedApexPreset) return SlitherMLAgent._cachedApexPreset;
      const b64 = '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';
      const raw = typeof atob === 'function' ? atob(b64) : Buffer.from(b64, 'base64').toString('binary');
      const bytes = new Uint8Array(raw.length);
      for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
      const f32 = new Float32Array(bytes.buffer);
      if (f32.length !== 2810) {
        throw new Error(`[SlitherML] Corrupt Apex preset: expected 2810 Float32 elements, got ${f32.length}`);
      }

      const w0 = Array.from(f32.subarray(0, 1536));
      const w1 = Array.from(f32.subarray(1536, 2688));
      const w2 = Array.from(f32.subarray(2688, 2736));

      const b0 = Array.from(f32.subarray(2736, 2784));
      const b1 = Array.from(f32.subarray(2784, 2808));
      const b2 = Array.from(f32.subarray(2808, 2810));

      SlitherMLAgent._cachedApexPreset = {
        brain: {
          inputNodes: 32,
          hiddenLayers: [48, 24],
          outputNodes: 2,
          weights: [w0, w1, w2],
          biases: [b0, b1, b2]
        },
        hasChampion: true,
        bestFitness: 37089,
        allTimePeakFitness: 43634,
        generation: 100,
        currentEpisode: 1,
        episodeFitnesses: [],
        candidatePending: false,
        candidateSource: null,
        kills: 168,
        teacherSamples: 1079717,
        allTimePeakLength: 22783,
        stagnantGens: 0,
        provenance: 'checkpoints/slither_ml_brain_gen100.json (Century landmark; 100 generations of live play, 168 kills, 1.08M lessons, 22.8k peak length, 37.1k floor)',
        timestamp: 1793231453303
      };
      return SlitherMLAgent._cachedApexPreset;
    }

    loadApexPreset() {
      let preset;
      try {
        preset = SlitherMLAgent.getApexPreset();
      } catch (err) {
        console.error('%c[SlitherML] ❌ Apex preset load failed:', 'color: #ef4444; font-weight: bold;', err.message);
        return false;
      }
      if (preset && this.applySavedState(preset)) {
        console.log(`%c[SlitherML] 💎 Apex Preset Loaded: Gen ${preset.generation} (${preset.kills} kills, ${preset.teacherSamples.toLocaleString()} lessons, bestFitness ${preset.bestFitness}, b2=[${preset.brain.biases[2][0].toFixed(4)}, ${preset.brain.biases[2][1].toFixed(4)}])`, 'color: #00f5d4; font-weight: bold;');
        this.lifeTeacherSamples = 0;
        if (this.championBrain) {
          this.brain = NeuralNetwork.fromJSON(this.championBrain.export(), this.inputCount);
          this.brain.mutate(0.035, 0.08);
        }
        this.enemyTracker.clear();
        this.prevTurnEma = 0;
        this.leftThreatEma = 0;
        this.rightThreatEma = 0;
        this.foodTargetError = null;
        this.boostDuration = 0;
        this.boostCooldown = 0;
        this.boostFuel = 1;
        this.abandonCurrentEpisode = false;
        return true;
      }
      return false;
    }

    // Failure-Trauma Filter (v5.9.0):
    // Flushes surviving buffered teacher frames when a player transitions out of Teacher Mode alive
    flushTeacherQuarantine() {
      if (this.teacherQuarantineBuffer && this.teacherQuarantineBuffer.length > 0) {
        while (this.teacherQuarantineBuffer.length > 0) {
          const frame = this.teacherQuarantineBuffer.shift();
          this.brain.trainStep(frame.inputs, frame.targets, frame.lr, frame.momentum);
        }
      }
    }

    // Shared sensor extraction pipeline for both Autonomous and Teacher modes
    _extractInputs(snake, snakes, foods, preys, dt, now) {
      const arenaRadius = (typeof pageWindow.grd === 'number' && pageWindow.grd > 0) ? pageWindow.grd : 16384;
      const headX = snake.xx;
      const headY = snake.yy;
      const currentAng = (typeof snake.eang === 'number') ? snake.eang : (snake.ang || 0);
      const scale = snake.sc || 1.0;
      const sensorRange = this.baseSensorRange * Math.min(2.0, Math.max(1.0, scale * 0.9));
      const headRadius = 14 * scale;

      const currentScore = (typeof snake.sct === 'number' && snake.sct > 0)
        ? Math.floor(snake.sct)
        : Math.floor(15 * (scale - 1) / 0.001 || 10);
      if (currentScore > this.previousScore) {
        const growth = currentScore - this.previousScore;
        this.fitness += growth * 25.0;
        this.totalFoodEaten += growth;
        this.previousScore = currentScore;
      } else if (currentScore < this.previousScore) {
        // Mass loss penalty (punishes excessive boosting/mass bleeding)
        const loss = this.previousScore - currentScore;
        this.fitness = Math.max(0, this.fitness - loss * 6.0);
        this.previousScore = currentScore;
      }
      if (currentScore > this.peakLength) {
        this.peakLength = currentScore;
        if (this.peakLength > this.allTimePeakLength) {
          this.allTimePeakLength = this.peakLength;
        }
      }

      // 1. Enemy Head Velocity & Trajectory Tracking + Combat Analysis
      let nearbyEnemyCount = 0;
      let cutOffThreat = null;
      let attackCandidate = null;
      let bestAttackScore = -1;
      let constrictCandidate = null;
      let bestConstrictScore = -1;
      let nearestThreatDist = Infinity;
      let nearestThreatScale = scale;
      let nearestThreatSpeed = 5.9;
      let outerPredatorNearby = false;
      let slipstreamTarget = null;

      const mySpeed = (typeof snake.sp === 'number' && snake.sp > 0) ? snake.sp : 5.9;
      const cosHead = Math.cos(currentAng);
      const sinHead = Math.sin(currentAng);

      const activeIds = new Set();
      if (Array.isArray(snakes)) {
        for (let i = 0; i < snakes.length; i++) {
          const other = snakes[i];
          if (!other || other.dead || other === snake || other.id === snake.id) continue;

          const dxHead = other.xx - headX;
          const dyHead = other.yy - headY;
          const distHead = Math.hypot(dxHead, dyHead);
          if (distHead < sensorRange) {
            nearbyEnemyCount++;
          }

          const otherScale = other.sc || 1.0;
          const otherSpeed = (typeof other.sp === 'number' && other.sp > 0) ? other.sp : 5.9;
          if (distHead < nearestThreatDist) {
            nearestThreatDist = distHead;
            nearestThreatScale = otherScale;
            nearestThreatSpeed = otherSpeed;
          }

          // Third-Party Outer Threat: a larger predator roaming in the perimeter
          if (distHead < 420 && otherScale >= scale * 1.15) {
            outerPredatorNearby = true;
          }

          // Slipstream Detection (v5.9.0):
          // When an enemy is boosting ahead of us within a forward cone, flag their exhaust wake for drafting
          if (otherSpeed > 7.0 && distHead > 70 && distHead < 520) {
            let relBearing = Math.atan2(dyHead, dxHead) - currentAng;
            while (relBearing > Math.PI) relBearing -= Math.PI * 2;
            while (relBearing < -Math.PI) relBearing += Math.PI * 2;
            if (Math.abs(relBearing) < Math.PI * 0.35) {
              slipstreamTarget = {
                enemyId: other.id !== undefined ? other.id : other,
                dist: distHead,
                speed: otherSpeed,
                side: relBearing >= 0 ? 1 : -1
              };
            }
          }

          const id = other.id !== undefined ? other.id : other;
          activeIds.add(id);
          const prev = this.enemyTracker.get(id);
          let vx = 0;
          let vy = 0;
          let omega = 0;
          const otherAng = (typeof other.eang === 'number') ? other.eang : (other.ang || 0);

          if (prev) {
            const dtTrack = (now - prev.timestamp) / 1000;
            if (dtTrack > 0.005) {
              const instVx = (other.xx - prev.xx) / dtTrack;
              const instVy = (other.yy - prev.yy) / dtTrack;
              vx = prev.vx * 0.6 + instVx * 0.4;
              vy = prev.vy * 0.6 + instVy * 0.4;

              // Angular velocity tracking (omega = dAngle / dt) for CTRA trajectory prediction
              if (typeof prev.ang === 'number') {
                let dAng = otherAng - prev.ang;
                while (dAng > Math.PI) dAng -= Math.PI * 2;
                while (dAng < -Math.PI) dAng += Math.PI * 2;
                const instOmega = dAng / dtTrack;
                omega = (prev.omega || 0) * 0.6 + instOmega * 0.4;
              }

              this.enemyTracker.set(id, { xx: other.xx, yy: other.yy, ang: otherAng, vx, vy, omega, timestamp: now, wasClose: distHead < 300, dist: distHead, enemyScale: other.sc || 1.0 });
            } else {
              vx = prev.vx;
              vy = prev.vy;
              omega = prev.omega || 0;
            }
          } else {
            this.enemyTracker.set(id, { xx: other.xx, yy: other.yy, ang: otherAng, vx: 0, vy: 0, omega: 0, timestamp: now, wasClose: distHead < 300, dist: distHead, enemyScale: other.sc || 1.0 });
          }

          // Combat Geometry Evaluation for enemies within 420 units
          if (distHead < 420) {
            const otherSpeed = (typeof other.sp === 'number' && other.sp > 0) ? other.sp : 5.9;
            const enemySpeedSec = Math.hypot(vx, vy) || (otherSpeed * 60);
            const evx = vx !== 0 ? vx : Math.cos(otherAng) * enemySpeedSec;
            const evy = vy !== 0 ? vy : Math.sin(otherAng) * enemySpeedSec;

            let relBearing = Math.atan2(dyHead, dxHead) - currentAng;
            while (relBearing > Math.PI) relBearing -= Math.PI * 2;
            while (relBearing < -Math.PI) relBearing += Math.PI * 2;

            // A. DEFENSIVE THREAT: Is this enemy attempting to cut across our heading?
            if (Math.abs(relBearing) < Math.PI * 0.7) {
              const D = cosHead * evy - sinHead * evx;
              if (Math.abs(D) > 1e-4) {
                const t_self = (dxHead * evy - dyHead * evx) / D;
                const t_enemy = (dxHead * sinHead - dyHead * cosHead) / D;
                const myReachTime = t_self / (mySpeed * 60);

                if (t_enemy > 0 && t_enemy < 1.0 && t_self > 8 && t_self < 420 && t_enemy <= myReachTime + 0.35) {
                  const cutSide = relBearing >= 0 ? 1 : -1; // 1 = cutting from Left, -1 = cutting from Right
                  const relScale = otherScale / scale;
                  const scaleWeight = Math.max(0.5, Math.min(2.5, relScale));
                  const threatScore = (1 - distHead / 420) * (otherSpeed > 7.0 ? 1.5 : 1.0) * scaleWeight;
                  if (!cutOffThreat || threatScore > cutOffThreat.threatScore) {
                    cutOffThreat = { enemyId: id, cutSide, dist: distHead, threatScore, massRatio: relScale, enemyScale: otherScale };
                  }
                }
              }
            }

            // B. OFFENSIVE ATTACK OPPORTUNITY: Curvilinear trajectory intercept (Constant Turn-Rate Kinematics)
            if (currentScore >= this.minBoostScore && this.boostFuel > 0.35 && distHead >= 80 && distHead <= 380) {
              const enemyScale = other.sc || 1.0;
              if (enemyScale <= scale * 1.6) {
                const relSpeed = Math.hypot(evx - (cosHead * mySpeed * 60), evy - (sinHead * mySpeed * 60));
                const leadSec = Math.max(0.25, Math.min(0.60, distHead / (relSpeed + 1e-3)));

                // Constant Turn-Rate & Acceleration (CTRA) Curvilinear Projection:
                // If enemy is curving (|omega| > 0.20 rad/s), project along circular flight arc rather than linear tangent
                let interceptX, interceptY;
                if (Math.abs(omega) > 0.20 && Math.abs(omega) < 6.0) {
                  const enemySpeed = Math.hypot(evx, evy);
                  const turnRad = enemySpeed / omega;
                  interceptX = other.xx + (turnRad * (Math.sin(otherAng + omega * leadSec) - Math.sin(otherAng)));
                  interceptY = other.yy - (turnRad * (Math.cos(otherAng + omega * leadSec) - Math.cos(otherAng)));
                } else {
                  interceptX = other.xx + evx * leadSec;
                  interceptY = other.yy + evy * leadSec;
                }

                // Prevent predicting intercept coordinates through outer arena death wall
                const distToCenter = Math.hypot(interceptX, interceptY);
                if (distToCenter > arenaRadius * 0.96) {
                  const clampScale = (arenaRadius * 0.96) / (distToCenter + 1e-6);
                  interceptX *= clampScale;
                  interceptY *= clampScale;
                }

                const distToIntercept = Math.hypot(interceptX - headX, interceptY - headY);
                const sprintSpeedSec = mySpeed * 1.85 * 60;
                const timeToIntercept = distToIntercept / sprintSpeedSec;

                if (timeToIntercept <= leadSec + 0.30 && distToIntercept < 440) {
                  const attackScore = ((scale / enemyScale) * 120) / (distToIntercept + 40);
                  if (attackScore > bestAttackScore) {
                    bestAttackScore = attackScore;
                    attackCandidate = {
                      enemyId: id,
                      interceptX,
                      interceptY,
                      distToIntercept,
                      side: relBearing >= 0 ? 1 : -1,
                      score: attackScore
                    };
                  }
                }
              }
            }

            // C. OFFENSIVE BOA CONSTRICTOR (Circle Trap): Target smaller snakes on our flank to encircle
            // Pro Rule: Veto if an outer predator is lurking nearby (Third-Party Perimeter Guard)
            if (!outerPredatorNearby && (currentScore >= 500 || scale >= 1.25) && distHead >= 70 && distHead <= 380 && Math.abs(relBearing) >= 0.65) {
              const enemyScale = other.sc || 1.0;
              const enemyScore = (typeof other.sct === 'number' && other.sct > 0) ? other.sct : Math.floor(15 * (enemyScale - 1) / 0.001 || 10);
              // Target must be substantially smaller so our body length can safely close the loop
              if (enemyScale <= scale * 0.65 && enemyScore <= currentScore * 0.45) {
                const constrictScore = ((scale / enemyScale) * 100) / (distHead + 40);
                if (constrictScore > bestConstrictScore) {
                  bestConstrictScore = constrictScore;
                  constrictCandidate = {
                    enemyId: id,
                    enemyX: other.xx,
                    enemyY: other.yy,
                    enemyScale,
                    dist: distHead,
                    side: relBearing >= 0 ? 1 : -1
                  };
                }
              }
            }
          }
        }
      }

      // Check for confirmed enemy kills (from MEMWashburn/Slither.io-NEAT)
      for (const [id, record] of this.enemyTracker.entries()) {
        if (!activeIds.has(id)) {
          if (record.wasClose && (now - record.timestamp < 1000)) {
            this.kills++;
            // Dynamic Mass-Scaled Combat Kill Bounty:
            // Slaying equal or larger predators rewards up to +600 fitness,
            // while eliminating small hatchlings still awards a solid base +250.
            const victimScale = record.enemyScale || 1.0;
            const myScale = scale || 1.0;
            const killRatio = victimScale / Math.max(0.1, myScale);
            const combatBounty = Math.round(250 + Math.min(350, Math.max(0, (killRatio - 0.75) * 450)));
            this.fitness += combatBounty;
            this.lastKillEpicenter = { x: record.xx, y: record.yy, timestamp: now };
            this.vacuumActiveUntil = now + 4000;
            this.flightRecorder.logIncident('KILL_CONFIRMED', {
              t: this.survivalTime,
              x: record.xx,
              y: record.yy,
              info: `Kill #${this.kills} (+${combatBounty} Bounty, victim scale ${victimScale.toFixed(2)})`
            });
            console.log(`%c[SlitherML] ⚔️ ENEMY ELIMINATED! Confirmed Kill #${this.kills} (+${combatBounty} Fitness Bounty, victim scale ${victimScale.toFixed(2)} vs self ${myScale.toFixed(2)}). Activating 4s Post-Kill Corpse Vacuum Mode!`, 'color: #ff0055; font-weight: bold;');
          }
          this.enemyTracker.delete(id);
        } else if (now - record.timestamp > 2000) {
          this.enemyTracker.delete(id);
        }
      }

      // Hoisted Arena Geometry: detect centered [-grd, grd] vs offset [0, 2*grd] coordinates.
      // Default is centered. If coordinates strictly exceed arenaRadius * 1.05, the world is using offset geometry.
      if (headX > arenaRadius * 1.05 || headY > arenaRadius * 1.05) {
        this.isCenteredWorld = false;
      }
      const isOffsetCoords = !this.isCenteredWorld && typeof pageWindow.grd === 'number' && pageWindow.grd > 0;
      const cx = isOffsetCoords ? arenaRadius : 0;
      const cy = isOffsetCoords ? arenaRadius : 0;
      const relX = headX - cx;
      const relY = headY - cy;

      // 2. 16 Proximity Raycasters
      const inputs = new Float32Array(this.inputCount);
      this.activeRays = [];

      let minObstacleDist = sensorRange;
      // Nearest obstacle in a forward cone only. The 360-degree minimum is
      // dominated by the snake's own tail curling behind it, which used to pin
      // dangerUrgency at 1.0 and permanently suppress foraging.
      let frontObstacleDist = sensorRange;
      let frontThreatScale = scale;
      let frontThreatSpeed = 5.9;
      let closestObstacleType = 'none';
      let frontObstacleType = 'none';
      let forwardObstacleDanger = false;
      let leftClearance = 0;
      let rightClearance = 0;
      let leftSelfProximity = 0;
      let rightSelfProximity = 0;
      let encirclingThreatRays = 0;

      for (let r = 0; r < this.rayCount; r++) {
        const relAngle = ((r / this.rayCount) * Math.PI * 2) - Math.PI;
        const rayAngle = currentAng + relAngle;
        const cosR = Math.cos(rayAngle);
        const sinR = Math.sin(rayAngle);

        let closestDist = sensorRange;
        let hitType = 'none';
        let hitScale = scale;
        let hitSpeed = 5.9;

        // Outer arena boundary
        const arenaR = arenaRadius;
        const b = relX * cosR + relY * sinR;
        const c = relX * relX + relY * relY - arenaR * arenaR;
        const disc = b * b - c;
        if (disc >= 0) {
          const tWall = -b + Math.sqrt(disc);
          if (tWall > 0 && tWall < closestDist) {
            closestDist = tWall;
            hitType = 'wall';
            hitScale = scale;
          }
        }

        // Enemy snakes
        if (Array.isArray(snakes)) {
          for (let sIdx = 0; sIdx < snakes.length; sIdx++) {
            const other = snakes[sIdx];
            if (!other || other.dead || other === snake || other.id === snake.id) continue;

            const otherScale = other.sc || 1.0;
            const otherSpeed = (typeof other.sp === 'number' && other.sp > 0) ? other.sp : 5.9;
            const otherRadius = 14 * otherScale;
            const hitThreshold = headRadius + otherRadius + 4;
            const hitThresholdSq = hitThreshold * hitThreshold;

            // Current head
            const dxHead = other.xx - headX;
            const dyHead = other.yy - headY;
            const projHead = dxHead * cosR + dyHead * sinR;
            if (projHead > 0 && projHead < closestDist + hitThreshold) {
              const perpHeadSq = (dxHead * dxHead + dyHead * dyHead) - projHead * projHead;
              if (perpHeadSq < hitThresholdSq) {
                const hitDist = Math.max(0, projHead - Math.sqrt(Math.max(0, hitThresholdSq - perpHeadSq)));
                if (hitDist < closestDist) {
                  closestDist = hitDist;
                  hitType = 'head';
                  hitScale = otherScale;
                  hitSpeed = otherSpeed;
                }
              }
            }

            // Predicted head (speed-scaled dynamic future horizon)
            const trackId = other.id !== undefined ? other.id : other;
            const track = this.enemyTracker.get(trackId);
            if (track && (track.vx !== 0 || track.vy !== 0)) {
              const leadHorizon = otherSpeed > 7.0 ? 0.35 : 0.25;
              const predX = other.xx + track.vx * leadHorizon;
              const predY = other.yy + track.vy * leadHorizon;
              const predHitThreshold = headRadius + (otherRadius * 1.25) + 4;
              const predHitThresholdSq = predHitThreshold * predHitThreshold;

              const dxPred = predX - headX;
              const dyPred = predY - headY;
              const projPred = dxPred * cosR + dyPred * sinR;
              if (projPred > 0 && projPred < closestDist + predHitThreshold) {
                const perpPredSq = (dxPred * dxPred + dyPred * dyPred) - projPred * projPred;
                if (perpPredSq < predHitThresholdSq) {
                  const hitDist = Math.max(0, projPred - Math.sqrt(Math.max(0, predHitThresholdSq - perpPredSq)));
                  if (hitDist < closestDist) {
                    closestDist = hitDist;
                    hitType = 'head';
                    hitScale = otherScale;
                    hitSpeed = otherSpeed;
                  }
                }
              }
            }

            // Body Segments: dense sampling so no ray tunnels through an enemy body
            if (Array.isArray(other.pts)) {
              const segStep = other.pts.length > 250 ? 3 : (other.pts.length > 80 ? 2 : 1);
              for (let pIdx = 0; pIdx < other.pts.length; pIdx += segStep) {
                const pt = other.pts[pIdx];
                if (!pt || pt.dying) continue;
                const dxSeg = pt.xx - headX;
                const dySeg = pt.yy - headY;
                const projSeg = dxSeg * cosR + dySeg * sinR;
                if (projSeg > 0 && projSeg < closestDist + hitThreshold) {
                  const perpSegSq = (dxSeg * dxSeg + dySeg * dySeg) - projSeg * projSeg;
                  if (perpSegSq < hitThresholdSq) {
                    const hitDist = Math.max(0, projSeg - Math.sqrt(Math.max(0, hitThresholdSq - perpSegSq)));
                    if (hitDist < closestDist) {
                      closestDist = hitDist;
                      hitType = 'body';
                      hitScale = otherScale;
                      hitSpeed = otherSpeed;
                    }
                  }
                }
              }
            }
          }
        }

        // Self-body segments (skip first 18 neck segments to prevent false self-hits)
        if (Array.isArray(snake.pts)) {
          const selfRadius = headRadius;
          const selfHitThreshold = headRadius + selfRadius + 2;
          const selfHitThresholdSq = selfHitThreshold * selfHitThreshold;
          for (let pIdx = 18; pIdx < snake.pts.length; pIdx += 2) {
            const pt = snake.pts[pIdx];
            if (!pt || pt.dying) continue;
            const dxSelf = pt.xx - headX;
            const dySelf = pt.yy - headY;
            const projSelf = dxSelf * cosR + dySelf * sinR;
            if (projSelf > 0 && projSelf < closestDist + selfHitThreshold) {
              const perpSelfSq = (dxSelf * dxSelf + dySelf * dySelf) - projSelf * projSelf;
              if (perpSelfSq < selfHitThresholdSq) {
                const hitDist = Math.max(0, projSelf - Math.sqrt(Math.max(0, selfHitThresholdSq - perpSelfSq)));
                if (hitDist < closestDist) {
                  closestDist = hitDist;
                  hitType = 'self';
                  hitScale = scale;
                }
              }
            }
          }
        }

        const normDist = Math.max(0, Math.min(1.0, closestDist / sensorRange));
        inputs[r] = 1.0 - normDist;

        if (closestDist < minObstacleDist) {
          minObstacleDist = closestDist;
          closestObstacleType = hitType;
        }

        // +/-45 degrees around the heading
        if (Math.abs(relAngle) <= Math.PI / 4 && closestDist < frontObstacleDist) {
          frontObstacleDist = closestDist;
          frontThreatScale = hitScale;
          frontObstacleType = hitType;
          frontThreatSpeed = hitSpeed;
        }

        // Anti-Encirclement Threat Tracking (v5.9.0):
        if ((hitType === 'body' || hitType === 'head') && hitScale >= scale * 1.15 && closestDist < sensorRange * 0.85) {
          encirclingThreatRays++;
        }

        const degFromHead = Math.abs(relAngle * 180 / Math.PI);
        const obstacleThreshold = (headRadius + (hitScale * 14)) * 2.1;
        if (degFromHead <= 35 && closestDist < obstacleThreshold && hitType !== 'self') {
          forwardObstacleDanger = true;
        }

        if (relAngle > 0 && relAngle < Math.PI * 0.85) {
          leftClearance += closestDist;
        } else if (relAngle < 0 && relAngle > -Math.PI * 0.85) {
          rightClearance += closestDist;
        }

        // Own-Body Shield Flank Detection:
        // Detect if our own body segments are alongside on left (+1) vs right (-1) flank.
        if (hitType === 'self') {
          const proximity = 1.0 - normDist;
          if (relAngle > 0.15 && relAngle < Math.PI * 0.85) {
            leftSelfProximity += proximity;
          } else if (relAngle < -0.15 && relAngle > -Math.PI * 0.85) {
            rightSelfProximity += proximity;
          }
        }

        this.activeRays.push({
          angle: rayAngle,
          dist: closestDist,
          norm: normDist,
          hitType
        });
      }

      // Compute dominant own-body shield flank:
      let selfShieldSide = 0;
      if (leftSelfProximity > 0.25 && leftSelfProximity > rightSelfProximity * 1.35) {
        selfShieldSide = 1; // Left flank shielded by own body
      } else if (rightSelfProximity > 0.25 && rightSelfProximity > leftSelfProximity * 1.35) {
        selfShieldSide = -1; // Right flank shielded by own body
      }

      // Anti-Encirclement & Escape Slit Breaching Arbiter (v5.9.0):
      // When a larger enemy wraps around >= 7 of our 16 rays, find the widest open exit gap
      let encirclementEscape = null;
      if (encirclingThreatRays >= 7) {
        let maxSlitClearance = 0;
        let bestSlitAngle = currentAng;
        for (let r = 0; r < this.rayCount; r++) {
          const ray = this.activeRays[r];
          if (ray.hitType !== 'wall' && ray.dist > maxSlitClearance) {
            maxSlitClearance = ray.dist;
            bestSlitAngle = ray.angle;
          }
        }
        encirclementEscape = {
          threatRays: encirclingThreatRays,
          slitDist: maxSlitClearance,
          slitAngle: bestSlitAngle,
          isOpen: maxSlitClearance > headRadius * 4.5
        };
      }

      // 3. 8-Sector Food Radar & Debris/Prey Perception
      const foodSectorSpan = (Math.PI * 2) / this.foodSectorCount;
      this.foodSectors.fill(0);
      this.detectedFoods = [];

      // Dual-tier sensing radius:
      // Obstacle detection stays compact (~520) for collision safety.
      // Ambient pellets extend to ~700, while high-value dead snake remains
      // and evasive prey are perceived across the full screen (~1200 units).
      const maxNormalFoodDist = Math.max(700, sensorRange * 1.35);
      const maxDebrisDist = 1200;

      const processPellet = (fx, fy, size, isPrey = false) => {
        const dx = fx - headX;
        const dy = fy - headY;
        const dist = Math.hypot(dx, dy);
        const isHighValue = isPrey || size > 8;
        const maxDist = isHighValue ? maxDebrisDist : maxNormalFoodDist;
        if (dist > maxDist || dist < 2) return;

        let pelletAngle = Math.atan2(dy, dx) - currentAng;
        while (pelletAngle < 0) pelletAngle += Math.PI * 2;
        while (pelletAngle >= Math.PI * 2) pelletAngle -= Math.PI * 2;

        const sectorIdx = Math.floor(pelletAngle / foodSectorSpan) % this.foodSectorCount;
        const multiplier = isPrey ? 15.0 : (size > 8 ? 8.0 : 1.0);
        const weight = (size * multiplier) / (dist * 0.04 + 1);
        this.foodSectors[sectorIdx] += weight;

        if (this.detectedFoods.length < 250) {
          this.detectedFoods.push({ dx, dy, dist, sz: size, isPrey });
        } else if (isHighValue) {
          // If candidate pool is saturated with ordinary pellets, make room for dead snake drops
          const replaceIdx = this.detectedFoods.findIndex(f => !f.isPrey && f.sz <= 8);
          if (replaceIdx !== -1) {
            this.detectedFoods[replaceIdx] = { dx, dy, dist, sz: size, isPrey };
          }
        }
      };
      // Scan all food pellets (natural pellets, dead snake drops, boost drops)
      const foodList = Array.isArray(foods) ? foods : (Array.isArray(pageWindow.foods) ? pageWindow.foods : []);
      const foodCount = (typeof pageWindow.foods_c === 'number' && pageWindow.foods_c > 0)
        ? Math.max(pageWindow.foods_c, foodList.length)
        : foodList.length;

      for (let i = 0; i < foodCount; i++) {
        const f = foodList[i];
        if (!f || f.eaten || f.dying) continue;
        // MUST use true map coordinates (xx, yy), NEVER render/screen coordinates (rx, ry)
        const fx = typeof f.xx === 'number' ? f.xx : (typeof f.x === 'number' ? f.x : NaN);
        const fy = typeof f.yy === 'number' ? f.yy : (typeof f.y === 'number' ? f.y : NaN);
        if (!isNaN(fx) && !isNaN(fy)) {
          const sz = f.sz || f.rad || 4;
          processPellet(fx, fy, sz, false);
        }
      }

      // Scan all moving preys (glowing evasive orbs)
      const preyList = Array.isArray(preys) ? preys : (Array.isArray(pageWindow.preys) ? pageWindow.preys : []);
      for (let i = 0; i < preyList.length; i++) {
        const p = preyList[i];
        if (!p || p.eaten || p.dying) continue;
        const px = typeof p.xx === 'number' ? p.xx : (typeof p.x === 'number' ? p.x : NaN);
        const py = typeof p.yy === 'number' ? p.yy : (typeof p.y === 'number' ? p.y : NaN);
        if (!isNaN(px) && !isNaN(py)) {
          const sz = p.sz || p.rad || 25;

          // Kinematic Lead Interception for Fleeing Preys:
          // Chase pellets actively flee snakes at ~7.0-9.0 speed.
          // Project future intercept coordinates rather than trailing behind.
          let targetPx = px;
          let targetPy = py;
          const pSpeed = (typeof p.sp === 'number' && p.sp > 0) ? p.sp : 0;
          const pAng = (typeof p.ang === 'number') ? p.ang : ((typeof p.dir === 'number') ? p.dir : null);

          if (pSpeed > 0 && pAng !== null) {
            const dxPrey = px - headX;
            const dyPrey = py - headY;
            const distPrey = Math.hypot(dxPrey, dyPrey);
            const sprintSpeedSec = mySpeed * 1.85 * 60;
            const leadSec = Math.max(0.10, Math.min(0.60, distPrey / (sprintSpeedSec + 1e-3)));
            const preySpeedSec = pSpeed * 60;
            targetPx = px + Math.cos(pAng) * (preySpeedSec * leadSec);
            targetPy = py + Math.sin(pAng) * (preySpeedSec * leadSec);
          }

          processPellet(targetPx, targetPy, sz, true);
        }
      }

      // Keep only the highest value & nearest pellets for the foraging assist / HUD markers,
      // rather than an arbitrary first-80 slice of world-scan order.
      if (this.detectedFoods.length > 80) {
        this.detectedFoods.sort((a, b) => {
          const valA = a.isPrey ? 25 : (a.sz > 8 ? 8 : 1);
          const valB = b.isPrey ? 25 : (b.sz > 8 ? 8 : 1);
          const scoreA = a.dist / Math.sqrt(valA);
          const scoreB = b.dist / Math.sqrt(valB);
          return scoreA - scoreB;
        });
        this.detectedFoods.length = 80;
      }

      // Sector sums are typically 0.3 (sparse) to ~10 (dense cluster). Dividing
      // by 40 squashed every reading to ~0.05, so food was effectively invisible
      // next to rays that use the full 0..1 range. Scale so a normal cluster is
      // genuinely salient while a lone pellet stays quiet.
      for (let f = 0; f < this.foodSectorCount; f++) {
        inputs[16 + f] = Math.min(1.0, this.foodSectors[f] / 8.0);
      }

      // 4. Arena Physics & Geometry
      const distFromCenter = Math.hypot(relX, relY);
      const normWallDist = distFromCenter / arenaRadius;
      // The direction to the middle is only meaningful when the boundary is
      // actually near. At the arena centre atan2(-0,-0) is degenerate (+/-pi),
      // which used to command a permanent U-turn and made the snake circle
      // instead of foraging. Gate the channel by wall pressure and neutralise it
      // at the origin.
      const wallPressure = Math.max(0, Math.min(1, (normWallDist - 0.55) / 0.35));
      const angleToCenter = distFromCenter > 1 ? Math.atan2(-relY, -relX) : currentAng;
      let diffCenterAngle = angleToCenter - currentAng;
      while (diffCenterAngle < -Math.PI) diffCenterAngle += Math.PI * 2;
      while (diffCenterAngle > Math.PI) diffCenterAngle -= Math.PI * 2;

      inputs[24] = normWallDist;
      inputs[25] = (diffCenterAngle / Math.PI) * wallPressure;
      inputs[26] = Math.min(1.0, scale / 8.0);
      inputs[27] = snake.sp > 7.0 ? 1.0 : 0.0;

      // 5. Short-Term Tactical Memory Buffers
      let maxLeftThreat = 0;
      for (let r = 9; r <= 15; r++) {
        if (inputs[r] > maxLeftThreat) maxLeftThreat = inputs[r];
      }
      let maxRightThreat = 0;
      for (let r = 1; r <= 7; r++) {
        if (inputs[r] > maxRightThreat) maxRightThreat = inputs[r];
      }

      this.leftThreatEma = this.leftThreatEma * 0.85 + maxLeftThreat * 0.15;
      this.rightThreatEma = this.rightThreatEma * 0.85 + maxRightThreat * 0.15;

      inputs[28] = Math.max(-1.0, Math.min(1.0, this.prevTurnEma));
      inputs[29] = Math.max(0.0, Math.min(1.0, this.leftThreatEma));
      inputs[30] = Math.max(0.0, Math.min(1.0, this.rightThreatEma));
      inputs[31] = Math.min(1.0, nearbyEnemyCount / 6);

      return {
        inputs,
        minObstacleDist,
        frontObstacleDist,
        frontThreatScale,
        nearestThreatScale,
        outerPredatorNearby,
        leftClearance,
        rightClearance,
        forwardObstacleDanger,
        normWallDist,
        diffCenterAngle,
        headRadius,
        currentAng,
        currentScore,
        cutOffThreat,
        attackCandidate,
        constrictCandidate,
        closestObstacleType,
        frontObstacleType,
        selfShieldSide,
        frontThreatSpeed,
        nearestThreatSpeed,
        encirclementEscape,
        slipstreamTarget
      };
    }

    // Pick the pellet worth steering at: close, valuable, and cheap to reach.
    // Sharp turns are penalised so the snake will not spin around for a pellet
    // that is already behind it.
    _pickFoodTarget(currentAng) {
      const list = this.detectedFoods;
      if (!list || list.length === 0) return null;

      const now = performance.now();
      const isVacuumActive = Boolean(this.lastKillEpicenter && now < this.vacuumActiveUntil);

      let best = null;
      let bestCost = Infinity;
      for (let i = 0; i < list.length; i++) {
        const f = list[i];
        const isHighValue = f.isPrey || f.sz > 8;
        const maxTargetDist = (isHighValue || isVacuumActive) ? 1200 : 650;
        if (f.dist < 4 || f.dist > maxTargetDist) continue;

        let error = Math.atan2(f.dy, f.dx) - currentAng;
        while (error > Math.PI) error -= Math.PI * 2;
        while (error < -Math.PI) error += Math.PI * 2;

        let value = f.isPrey ? 25 : (f.sz > 8 ? 8 : 1);
        if (isVacuumActive && (f.sz > 8 || f.dist < 500)) {
          value *= 3.0; // 3x loot priority during post-kill corpse vacuum!
        }
        // Slipstream Wake Bonus (v5.9.0):
        if (this.slipstreamTarget && f.dist < 450 && Math.abs(error) < 0.6) {
          value *= 2.0; // 2x priority to draft in boosting enemy's mass exhaust trail
        }
        const turnPenalty = 1 + 2.2 * (1 - Math.cos(error));
        const cost = ((f.dist + 60) * turnPenalty) / value;
        if (cost < bestCost) {
          bestCost = cost;
          best = { error, dist: f.dist, value, dx: f.dx, dy: f.dy, sz: f.sz, isPrey: f.isPrey };
        }
      }
      return best;
    }

    // AUTOPILOT MODE: Autonomous Evaluation & Decision Making
    evaluate(snake, snakes, foods, preys) {
      const now = performance.now();
      const dt = Math.min(0.1, Math.max(0.001, (now - this.lastDecisionTime) / 1000));
      this.lastDecisionTime = now;

      if (!snake || snake.dead) {
        return { targetAngle: 0, isBoosting: false, rawSteer: 0, rawBoost: 0 };
      }

      const survivalBonusRate = 4.0 + Math.min(6.0, (this.stagnantGens || 0) * 0.5);
      this.survivalTime += dt;
      this.fitness += dt * survivalBonusRate;

      const scale = snake.sc || 1.0;
      const state = this._extractInputs(snake, snakes, foods, preys, dt, now);
      const {
        inputs,
        minObstacleDist,
        frontObstacleDist,
        frontThreatScale = scale,
        nearestThreatScale = scale,
        outerPredatorNearby = false,
        leftClearance,
        rightClearance,
        forwardObstacleDanger,
        normWallDist,
        diffCenterAngle,
        headRadius,
        currentAng,
        currentScore,
        cutOffThreat,
        attackCandidate,
        constrictCandidate,
        closestObstacleType = 'none',
        frontObstacleType = 'none',
        selfShieldSide = 0,
        frontThreatSpeed = 5.9,
        nearestThreatSpeed = 5.9,
        encirclementEscape = null,
        slipstreamTarget = null
      } = state;

      this.closestObstacleType = closestObstacleType;
      this.frontObstacleType = frontObstacleType;
      this.selfShieldSide = selfShieldSide;
      this.encirclementEscape = encirclementEscape;
      this.slipstreamTarget = slipstreamTarget;

      const outputs = this.brain.predict(inputs);
      let turnDelta = outputs[0];
      const boostProb = outputs[1];

      const threatScale = Number.isFinite(frontThreatScale) ? frontThreatScale : (nearestThreatScale || scale);
      const threatRadius = 14 * threatScale;
      const massRatio = threatScale / Math.max(0.01, scale);
      this.threatScale = threatScale;
      this.massRatio = massRatio;

      const threatSpeed = (typeof frontThreatSpeed === 'number' && frontThreatSpeed > 0)
        ? frontThreatSpeed
        : ((typeof nearestThreatSpeed === 'number' && nearestThreatSpeed > 0) ? nearestThreatSpeed : 5.9);

      // Kinetic Minkowski Danger Clearance (v5.9.0):
      // Expands avoidance buffer using combined radii of self and threat, plus both our speed AND enemy closing speed
      const currentSpeed = (typeof snake.sp === 'number' && snake.sp > 0) ? snake.sp : 5.9;
      const mySpeedBuffer = Math.max(0, (currentSpeed - 5.5) * 15);
      const enemySpeedBuffer = Math.max(0, (threatSpeed - 5.5) * 18);
      const dangerDistance = (headRadius + threatRadius) * 2.25 + mySpeedBuffer + enemySpeedBuffer;
      const dangerSource = Number.isFinite(frontObstacleDist) ? frontObstacleDist : minObstacleDist;
      let dangerUrgency = Math.max(0, Math.min(1, (dangerDistance - dangerSource) / ((headRadius + threatRadius) * 1.5)));

      // Behemoth Evasion: escalate urgency when facing an opponent with massRatio > 1.5
      if (massRatio > 1.5 && dangerUrgency > 0.05) {
        dangerUrgency = Math.min(1.0, dangerUrgency * 1.35);
      }

      // Semantic Object-Aware Danger Scaling (v5.8.0):
      // 1. Own Body Shield ('self'): In Slither.io, own-body collisions cause ZERO damage.
      //    Our own body is an impenetrable shield and pocket sanctuary, not a lethal threat.
      //    Dampen danger urgency by 75% when facing own body so the agent does not panic.
      if (frontObstacleType === 'self') {
        dangerUrgency *= 0.25;
      }
      // 2. Enemy Head ('head'): Active kinetic predator capable of turning, striking, and boosting.
      //    Amplify evasion urgency by +35% to trigger immediate proactive banking.
      else if (frontObstacleType === 'head' && dangerUrgency > 0.05) {
        dangerUrgency = Math.min(1.0, dangerUrgency * 1.35);
      }
      const reflexSign = leftClearance >= rightClearance ? 1 : -1;

      const forwardClear = inputs[7] < 0.35 && inputs[8] < 0.35 && inputs[9] < 0.35;
      const healthyLength = currentScore >= this.minBoostScore &&
        !(this.peakLength > 120 && currentScore < this.peakLength * 0.9);
      const calmTraffic = inputs[31] < 0.5;

      // Arena Boundary Wall Proximity Check (from ErmiyaEskandary Death Wall Guard)
      const wallDanger = normWallDist > 0.91;

      // ---- TACTICAL COMBAT STANCE ARBITER -----------------------------------
      let stance = 'FORAGE';
      // 1. ANTI-ENCIRCLEMENT SLIT BREACHING vs COIL STANCE (v5.9.0):
      // When large enough and surrounded by multiple close threats or pinched on both flanks
      const canCoil = currentScore >= 300 || scale >= 1.15;
      const pinched = inputs[29] > 0.65 && inputs[30] > 0.65;
      const surrounded = dangerUrgency >= 0.55 && inputs[31] >= 0.33; // >=2 nearby enemies
      if (encirclementEscape && encirclementEscape.isOpen) {
        // Encirclement detected, but an open exit gap exists: VETO COIL and execute Slit Breach!
        stance = 'DEFEND';
      } else if (canCoil && (pinched || surrounded)) {
        stance = 'COIL';
      }
      // 2. DEFEND STANCE: Kinematic cut-off threat, acute obstacle danger, or death wall
      else if (cutOffThreat || dangerUrgency >= 0.5 || wallDanger) {
        stance = 'DEFEND';
      }
      // 3. CONSTRICT STANCE: Offensive Boa Constrictor (Circle Trap around smaller enemies)
      // Pro Rule: Veto if an outer predator is nearby (Third-Party Perimeter Guard)
      else if (constrictCandidate && !outerPredatorNearby && massRatio < 1.15 && dangerUrgency < 0.25 && normWallDist < 0.88) {
        stance = 'CONSTRICT';
      }
      // 4. ATTACK STANCE: Lead intercept predator strike (from dzaczek/slbot)
      // Never attempt head-on strike against behemoth threats (massRatio <= 1.5)
      else if (attackCandidate && massRatio <= 1.5 && inputs[8] < 0.65 && dangerUrgency < 0.3 && currentScore >= this.minBoostScore && this.boostFuel > 0.35 && this.boostCooldown <= 0) {
        stance = 'ATTACK';
      }
      // 5. VACUUM STANCE: Post-kill corpse loot gathering
      else if (this.lastKillEpicenter && now < this.vacuumActiveUntil && dangerUrgency < 0.3) {
        stance = 'VACUUM';
      }
      this.combatStance = stance;
      this.attackTarget = (stance === 'ATTACK') ? attackCandidate : null;
      this.cutOffThreat = (stance === 'DEFEND') ? cutOffThreat : null;
      this.constrictTarget = (stance === 'CONSTRICT') ? constrictCandidate : null;
      this.constrictDuration = (stance === 'CONSTRICT') ? ((this.constrictDuration || 0) + dt) : 0;

      // A. COIL STANCE: The Apex Defensive Fortress (Circle Method)
      if (stance === 'COIL') {
        this.foodSteerEma *= 0.2; // Suppress food greed while coiling
        if (!this.coilSign) {
          this.coilSign = Math.abs(this.prevTurnEma) > 0.25
            ? Math.sign(this.prevTurnEma)
            : reflexSign;
        }
        turnDelta = this.coilSign * 0.85;
      } else {
        this.coilSign = 0;
        // B. DEFEND STANCE: 100% Evasion Priority (Anti-Cut-Off, Death Wall & Obstacle Avoidance)
        if (stance === 'DEFEND') {
          this.foodSteerEma *= 0.5; // Suppress food greed
          if (encirclementEscape && encirclementEscape.isOpen) {
            // Anti-Encirclement Slit Breaching (v5.9.0):
            // Steer hard toward the open exit slit before the encircling predator's loop closes
            this.flightRecorder.logIncident('SLIT_BREACH_ESCAPE', {
              t: this.survivalTime,
              x: snake.xx,
              y: snake.yy,
              info: `Slit angle: ${encirclementEscape.slitAngle.toFixed(2)}`
            });
            let slitError = encirclementEscape.slitAngle - currentAng;
            while (slitError > Math.PI) slitError -= Math.PI * 2;
            while (slitError < -Math.PI) slitError += Math.PI * 2;
            turnDelta = Math.max(-1, Math.min(1, slitError / 1.25));
          } else if (wallDanger) {
            // Emergency bank away from death wall toward arena center
            turnDelta = diffCenterAngle > 0 ? 1.0 : -1.0;
          } else if (cutOffThreat) {
            // Pocket Defense (v5.8.0):
            // If enemy is cutting from one flank and our impenetrable body shield is on the opposite flank,
            // tuck head safely against our own body shield!
            if (selfShieldSide !== 0 && selfShieldSide === -cutOffThreat.cutSide) {
              this.flightRecorder.logIncident('POCKET_SHIELD_DEFENSE', {
                t: this.survivalTime,
                x: snake.xx,
                y: snake.yy,
                info: `Shield side: ${selfShieldSide}`
              });
              turnDelta = selfShieldSide * 1.0;
            } else {
              // Standard bank sharply AWAY from incoming cut trajectory:
              // Enemy cutting from Left (cutSide=1) -> steer hard Right (-1)
              // Enemy cutting from Right (cutSide=-1) -> steer hard Left (+1)
              turnDelta = -cutOffThreat.cutSide * 1.0;
            }
          } else {
            // Obstacle Avoidance with Pocket Shield Awareness:
            // If facing an enemy head and we have a verified body shield flank, bank toward our shield
            if (frontObstacleType === 'head' && selfShieldSide !== 0) {
              turnDelta = selfShieldSide * 1.0;
            } else {
              const evasionTurn = (massRatio > 1.5) ? 1.0 : Math.max(0.9, Math.abs(turnDelta));
              turnDelta = reflexSign * evasionTurn;
            }
          }
        }
        // C. ATTACK STANCE: Predator Lead-Intercept Cut-Off Strike
        else if (stance === 'ATTACK' && attackCandidate) {
          this.foodSteerEma *= 0.5;
          let attackAngle = Math.atan2(attackCandidate.interceptY - snake.yy, attackCandidate.interceptX - snake.xx);
          let attackError = attackAngle - currentAng;
          while (attackError > Math.PI) attackError -= Math.PI * 2;
          while (attackError < -Math.PI) attackError += Math.PI * 2;

          const closeInFactor = Math.max(0, Math.min(1, (headRadius * 8 - attackCandidate.distToIntercept) / (headRadius * 5)));
          const crossingCurl = (attackCandidate.side > 0 ? 0.22 : -0.22) * closeInFactor;
          turnDelta = Math.max(-1, Math.min(1, (attackError / 1.35) + crossingCurl));
        }
        // D. CONSTRICT STANCE: Offensive Boa Constrictor (Circle Trap around smaller prey)
        else if (stance === 'CONSTRICT' && constrictCandidate) {
          this.foodSteerEma *= 0.2;
          const angleToEnemy = Math.atan2(constrictCandidate.enemyY - snake.yy, constrictCandidate.enemyX - snake.xx);
          const wrapSign = constrictCandidate.side; // 1 = Left, -1 = Right

          // Archimedean Spiral Tightening:
          // Gradually contracts the orbit inward (from 0.45pi to 0.62pi) as the loop closes,
          // progressively suffocating the trapped enemy's turning radius.
          const tightenFactor = Math.min(1.0, (this.constrictDuration || 0) / 2.5);
          const inwardOffset = Math.PI * (0.45 + tightenFactor * 0.17);
          const orbitAngle = angleToEnemy + (wrapSign > 0 ? -inwardOffset : inwardOffset);

          let orbitError = orbitAngle - currentAng;
          while (orbitError > Math.PI) orbitError -= Math.PI * 2;
          while (orbitError < -Math.PI) orbitError += Math.PI * 2;
          turnDelta = Math.max(-1, Math.min(1, orbitError / 1.25));
        }
        // E. VACUUM & FORAGE STANCE: Dual-Tier Food & Corpse Scavenging + Exploratory Sweep
        else {
          const foodTarget = this._pickFoodTarget(currentAng);
          this.foodTargetError = foodTarget ? foodTarget.error : null;
          if (foodTarget) {
            let foodSteer = Math.max(-1, Math.min(1, foodTarget.error / 1.85));

            // "The Cross" Perpendicular Loot Sweep (Pro Rule):
            // When vacuuming a large corpse drop (value >= 8), approach with a slight tangential
            // crossing offset rather than a dead-on head-first collision course.
            // This sweeps a broader pellet swath with our body flank while shielding against rival scavengers.
            if (stance === 'VACUUM' && foodTarget.value >= 8 && foodTarget.dist > 120 && foodTarget.dist < 600) {
              const crossOffset = foodTarget.error >= 0 ? 0.20 : -0.20;
              foodSteer = Math.max(-1, Math.min(1, (foodTarget.error / 1.85) + crossOffset));
            }

            const steerWeight = stance === 'VACUUM' ? 0.65 : 0.45;
            this.foodSteerEma = this.foodSteerEma * (1 - steerWeight) + foodSteer * steerWeight;
            const assist = (stance === 'VACUUM' ? 0.95 : 0.85) * (1 - dangerUrgency);
            turnDelta = turnDelta * (1 - assist) + this.foodSteerEma * assist;
          } else {
            this.foodSteerEma *= 0.8;
            // Active Exploratory Patrol (Anti-Stall from dzaczek/slbot):
            if (dangerUrgency < 0.1 && normWallDist < 0.82) {
              turnDelta += Math.sin(now * 0.003) * 0.08;
            }
          }

          if (dangerUrgency > 0.05) {
            turnDelta = turnDelta * (1 - dangerUrgency) + reflexSign * dangerUrgency;
          } else if (normWallDist > 0.86) {
            turnDelta = Math.sign(diffCenterAngle) * Math.min(1.0, Math.abs(diffCenterAngle) + 0.35);
          } else if (normWallDist < 0.18 && inputs[31] >= 0.15 && dangerUrgency < 0.10) {
            // Goldilocks Ring Guidance (Pro Rule):
            // When near the congested dead-center mosh-pit (< 0.18 radius) with traffic present,
            // steer gently outward to patrol the fertile outer-center hunting grounds.
            const awayAngle = diffCenterAngle > 0 ? diffCenterAngle - Math.PI : diffCenterAngle + Math.PI;
            turnDelta += Math.sign(awayAngle) * 0.10;
          }
        }
      }

      if (dangerUrgency > 0.05) {
        this.fitness = Math.max(0, this.fitness - dt * 15.0 * dangerUrgency);
      }

      // Smooth steering to prevent high-frequency jitter/head shaking (JuiHsiu/Slither-DRL filter)
      const smoothedTurn = turnDelta * 0.7 + this.prevTurnEma * 0.3;
      this.prevTurnEma = this.prevTurnEma * 0.82 + turnDelta * 0.18;
      this.targetAngle = currentAng + smoothedTurn * 1.85;

      // ---- Boost Governor (Mass-ROI Governed from JuiHsiu/Slither-DRL) --------
      if (this.boostCooldown > 0) {
        this.boostCooldown = Math.max(0, this.boostCooldown - dt);
      }
      if (this.boostFuel < 1) {
        this.boostFuel = Math.min(1, this.boostFuel + dt / this.boostRefillSeconds);
      }

      let shouldBoost = false;

      if (stance === 'COIL') {
        // In defensive coil, never boost: conserve mass inside body shield
        shouldBoost = false;
      } else if (stance === 'DEFEND') {
        // Defensive Boost Protocol:
        if (encirclementEscape && encirclementEscape.isOpen) {
          // Emergency Slit Breach Sprint (v5.9.0):
          let slitError = encirclementEscape.slitAngle - currentAng;
          while (slitError > Math.PI) slitError -= Math.PI * 2;
          while (slitError < -Math.PI) slitError += Math.PI * 2;
          shouldBoost = Math.abs(slitError) < 0.48 && healthyLength && this.boostFuel > 0.15 && this.boostCooldown <= 0;
        }
        // 1. Direct cut-offs, wall danger, or massive behemoth proximity -> BRAKE so enemy cuts past harmlessly ahead
        else if (wallDanger || (cutOffThreat && cutOffThreat.threatScore > 0.75) || (massRatio > 1.5 && dangerUrgency >= 0.4)) {
          shouldBoost = false;
        }
        // 2. Flank pursuit or rear pinch with clear forward exit -> EMERGENCY ESCAPE SPRINT
        else if (forwardClear && forwardObstacleDanger < 0.20 && minObstacleDist > headRadius * 5.0 &&
                 healthyLength && this.boostFuel > 0.25 && this.boostCooldown <= 0) {
          shouldBoost = true;
        } else {
          shouldBoost = false;
        }
      } else if (stance === 'ATTACK') {
        // Strike sprint: accelerate across the enemy nose if stamina available
        shouldBoost = inputs[8] < 0.65 && this.boostCooldown <= 0 && this.boostFuel > 0.25;
      } else if (stance === 'CONSTRICT') {
        // Gapless Constriction (Pro Tournament Rule):
        // Never boost during coiling. Cruising speed keeps body segments tightly spaced
        // with zero gaps, preventing trapped prey from finding an exit slit while wrapping.
        shouldBoost = false;
      } else if (stance === 'VACUUM') {
        // Corpse vacuum sprint: rapid rush to snatch loot before rivals arrive
        const foodTarget = this._pickFoodTarget(currentAng);
        shouldBoost = forwardClear && healthyLength && Boolean(foodTarget) && foodTarget.dist < 400 &&
                      this.boostFuel > 0.20 && this.boostCooldown <= 0;
      } else {
        // Foraging sprint: Mass-ROI governed sprint (from JuiHsiu/Slither-DRL)
        const foodTarget = this._pickFoodTarget(currentAng);
        const estBurnTimeSec = foodTarget ? (foodTarget.dist / (currentSpeed * 1.85 * 60)) : 1.0;
        const estMassBurned = estBurnTimeSec * 16.0;
        const positiveROI = Boolean(foodTarget) && (foodTarget.value * 2.5 >= estMassBurned);

        const worthSprinting = Boolean(foodTarget) &&
          foodTarget.value >= 8 &&
          foodTarget.dist < 450 &&
          Math.abs(foodTarget.error) < 0.6 &&
          positiveROI;

        // Dedicated Chase Pellet (Prey) Interception Sprint:
        // Chase pellets actively flee; boosting is required to capture them before they escape.
        const isPreyChase = Boolean(foodTarget && foodTarget.isPrey &&
          foodTarget.dist < 700 && Math.abs(foodTarget.error) < 0.70);

        // Slipstream Wake Drafting Sprint (v5.9.0):
        const isSlipstreamSprint = Boolean(slipstreamTarget && foodTarget && foodTarget.dist < 420 && Math.abs(foodTarget.error) < 0.55);

        const maySprint = forwardClear && healthyLength && calmTraffic && (worthSprinting || isPreyChase || isSlipstreamSprint) &&
          this.boostCooldown <= 0 && this.boostFuel > 0.15;

        const policyVeto = !isPreyChase && !isSlipstreamSprint && boostProb < 0.15;
        shouldBoost = maySprint && !policyVeto;
      }

      // Angular Boost Gating (Slither.io Turning Physics: R_turn = v / omega):
      // Pro Rule: Boosting expands the turn radius by ~100%, causing the head to swing outward into obstacles.
      // Strictly cut boost during sharp turns (|turnDelta| > 0.48 rad ~28 deg),
      // except during Emergency Escape Sprint where forward exit is 100% clear.
      if (shouldBoost && stance !== 'DEFEND' && Math.abs(turnDelta) > 0.48) {
        shouldBoost = false;
      }

      if (shouldBoost) {
        this.isBoosting = true;
        this.boostDuration += dt;
        this.boostFuel = Math.max(0, this.boostFuel - dt / this.boostBurnSeconds);
        if (this.boostDuration >= this.boostBurstSeconds || this.boostFuel <= 0) {
          this.isBoosting = false;
          this.boostDuration = 0;
          this.boostCooldown = this.boostRestSeconds;
        }
      } else {
        this.isBoosting = false;
        this.boostDuration = 0;
      }

      let dominantRayIdx = 8;
      let dominantRayDist = inputs[8];
      for (let r = 0; r < 16; r++) {
        if (inputs[r] > dominantRayDist) {
          dominantRayDist = inputs[r];
          dominantRayIdx = r;
        }
      }

      this.flightRecorder.recordFrame({
        t: this.survivalTime,
        x: snake.xx ?? 0,
        y: snake.yy ?? 0,
        ang: currentAng,
        sp: currentSpeed,
        score: currentScore,
        len: (typeof snake.sct === 'number' && snake.sct > 0) ? snake.sct : Math.floor(currentScore / 10),
        stance: this.combatStance || 'FORAGE',
        normWallDist,
        threatDist: dangerSource,
        threatType: frontObstacleType !== 'none' ? frontObstacleType : closestObstacleType,
        threatAngle: 0,
        threatMassRatio: this.massRatio || 1.0,
        threatSpeed,
        turnDelta,
        isBoosting: this.isBoosting,
        selfShieldSide,
        dominantRayIdx,
        dominantRayDist
      });

      return {
        targetAngle: this.targetAngle,
        isBoosting: this.isBoosting,
        rawSteer: outputs[0],
        rawBoost: boostProb,
        fitness: Math.floor(this.fitness),
        bestFitness: Math.floor(this.bestFitness),
        generation: this.generation,
        currentEpisode: this.currentEpisode,
        score: currentScore,
        stance: this.combatStance,
        kills: this.kills,
        massRatio: Number(this.massRatio.toFixed(2)),
        threatScale: Number(this.threatScale.toFixed(2)),
        closestObstacleType,
        frontObstacleType,
        selfShieldSide,
        frontThreatSpeed,
        nearestThreatSpeed,
        encirclementEscape,
        slipstreamTarget
      };
    }

    // TEACHER MODE: Shadows the human player and trains via Backpropagation
    learnFromHuman(snake, snakes, foods, preys, targetX, targetY, isHumanBoosting) {
      const now = performance.now();
      const dt = Math.min(0.1, Math.max(0.001, (now - this.lastDecisionTime) / 1000));
      this.lastDecisionTime = now;

      if (!snake || snake.dead) return;

      const survivalBonusRate = 4.0 + Math.min(6.0, (this.stagnantGens || 0) * 0.5);
      this.survivalTime += dt;
      this.fitness += dt * survivalBonusRate;
      this.touchedByHuman = true;

      const state = this._extractInputs(snake, snakes, foods, preys, dt, now);
      const { inputs, currentAng } = state;

      // Extract Human Intention as Ground Truth Targets
      const humanAimAngle = Math.atan2(targetY, targetX);
      let diffAngle = humanAimAngle - currentAng;
      while (diffAngle < -Math.PI) diffAngle += Math.PI * 2;
      while (diffAngle > Math.PI) diffAngle -= Math.PI * 2;

      const humanSteerTarget = Math.max(-1.0, Math.min(1.0, diffAngle / 1.85));
      const humanBoostTarget = isHumanBoosting ? 1.0 : 0.0;

      // Salience-Gated Adaptive Learning Rate with low-pass Momentum (mu = 0.80) to smooth human mouse jitter
      // v5.8.0: Combat Reflex Priority - when dodging or banking against an active enemy head,
      // escalate learning rate to 0.008 so tactical evasion reflexes are permanently baked into neural weights.
      const isTurning = Math.abs(humanSteerTarget) > 0.12;
      const isNearObstacle = state.minObstacleDist < state.headRadius * 8.0;
      const isCombatReflex = state.frontObstacleType === 'head' || state.closestObstacleType === 'head';
      const isTacticalLesson = isTurning || isNearObstacle || isHumanBoosting || isCombatReflex;
      const adaptiveLr = isCombatReflex ? 0.008 : (isTacticalLesson ? 0.006 : 0.0004);

      this.teacherSamples++;
      this.lifeTeacherSamples = (this.lifeTeacherSamples || 0) + 1;

      // Failure-Trauma Filter (v5.9.0):
      // Buffer training frames into a quarantined FIFO queue (~0.75s).
      // Frames are only backpropagated into weights once they survive past the quarantine window.
      // If the snake dies, the quarantined death frames are incinerated, preventing the AI
      // from learning the human's fatal crash trajectory.
      if (!this.teacherQuarantineBuffer) this.teacherQuarantineBuffer = [];
      this.teacherQuarantineBuffer.push({
        inputs: new Float32Array(inputs),
        targets: [humanSteerTarget, humanBoostTarget],
        lr: adaptiveLr,
        momentum: 0.80
      });

      if (this.teacherQuarantineBuffer.length > (this.teacherQuarantineCapacity || 45)) {
        const safeFrame = this.teacherQuarantineBuffer.shift();
        this.brain.trainStep(safeFrame.inputs, safeFrame.targets, safeFrame.lr, safeFrame.momentum);
      }

      this.prevTurnEma = this.prevTurnEma * 0.85 + humanSteerTarget * 0.15;
      this.targetAngle = humanAimAngle;
      this.isBoosting = isHumanBoosting;
    }

    onDeath(isHumanDriven = false) {
      const runFitness = Math.max(0, this.fitness);
      const wasHumanAssisted = isHumanDriven || this.touchedByHuman;

      const autopsy = this.flightRecorder.deduceDeathAutopsy({
        finalScore: this.previousScore || 0,
        peakLength: this.peakLength || 0,
        survivalTime: this.survivalTime || 0,
        isHumanDriven: wasHumanAssisted,
        generation: this.generation,
        episode: this.currentEpisode
      });
      console.log(`%c[SlitherML 💀 Autopsy] Life Ended: ${autopsy.cause} (${autopsy.description}) | Stance: ${autopsy.fatalStance} | Time: ${autopsy.survivalTime}s | Length: ${autopsy.peakLength}`, 'color: #f43f5e; font-weight: bold;');

      if (this.abandonCurrentEpisode) {
        console.log('%c[SlitherML] ⏭️ Life excluded (mixed-mode play detected: was partly human-driven). Neither rollout nor candidate recorded.', 'color: #94a3b8;');
      } else if (wasHumanAssisted) {
        this._queueTeacherCandidate(isHumanDriven, runFitness);
      } else {
        this._recordEvaluationEpisode(runFitness);
      }

      this.flightRecorder.resetLife();
      this.fitness = 0;
      this.survivalTime = 0;
      this.previousScore = 0;
      this.peakLength = 0;
      this.enemyTracker.clear();
      this.prevTurnEma = 0;
      this.leftThreatEma = 0;
      this.rightThreatEma = 0;
      this.boostDuration = 0;
      this.boostCooldown = 0;
      this.boostFuel = 1;
      this.foodTargetError = null;
      this.touchedByHuman = false;
      this.abandonCurrentEpisode = false;
      this.coilSign = 0;
      this.lifeTeacherSamples = 0;
      this.closestObstacleType = 'none';
      this.frontObstacleType = 'none';
      this.selfShieldSide = 0;
      this.encirclementEscape = null;
      this.slipstreamTarget = null;
      this.teacherQuarantineBuffer = [];
    }

    // A teacher-driven life is NOT an evaluation episode: it never scores the
    // genome. It only nominates the trained brain as a candidate, which the
    // autonomous 3-life rollout must then beat the champion to promote.
    _queueTeacherCandidate(humanDriven, runFitness) {
      const MIN_TEACHER_SECONDS = 15;
      const MIN_TEACHER_SAMPLES = 300;
      const longEnough = this.survivalTime >= MIN_TEACHER_SECONDS;
      const samplesThisLife = this.lifeTeacherSamples > 0
        ? this.lifeTeacherSamples
        : (this.lastLoadedTimestamp > 0 ? 0 : this.teacherSamples);
      const sawEnoughTuition = samplesThisLife >= MIN_TEACHER_SAMPLES;

      if (!humanDriven || !longEnough || !sawEnoughTuition) {
        if (this.championBrain) {
          this.brain = NeuralNetwork.fromJSON(this.championBrain.export(), this.inputCount);
        }
        this.candidatePending = false;
        this.candidateSource = null;
        console.log(`%c[SlitherML] ⚠️ Teacher run discarded (${this.survivalTime.toFixed(1)}s, ${samplesThisLife} lessons, fitness ${Math.floor(runFitness)}). Champion genome untouched.`, 'color: #ffb703; font-weight: bold;');
        return;
      }

      this.candidatePending = true;
      this.candidateSource = 'teacher';
      this.currentEpisode = 1;
      this.episodeFitnesses = [];
      console.log(`%c[SlitherML] 🧪 Teacher brain queued as CANDIDATE (${samplesThisLife} lessons, fitness ${Math.floor(runFitness)}). It must beat the champion (${Math.floor(this.bestFitness)}) across 3 autonomous lives before it is saved.`, 'color: #f59e0b; font-weight: bold;');
      this._saveCheckpoint();
    }

    // Autonomous rollout: 3 lives per genome, then strict elitist selection.
    _recordEvaluationEpisode(runFitness) {
      // A life that was partly human-driven is not a fair sample of this
      // genome's ability, so it is dropped instead of being scored.
      if (this.abandonCurrentEpisode) {
        console.log('%c[SlitherML] ⏭️ Life excluded from the rollout (was partly human-driven, so it would bias this genome\'s score).', 'color: #94a3b8;');
        return;
      }

      // Mulligan filter: Server spawn-kills (instant collision on re-entry before snake can maneuver)
      // are unavoidable noise that poison 3-life rollouts. Exclude them from the rollout.
      const isSpawnKill = this.survivalTime > 0 && (
        (this.survivalTime < 3.5 && runFitness < 60) ||
        (this.survivalTime < 5.0 && runFitness < 40 && (!this.peakLength || this.peakLength <= 10))
      );
      if (isSpawnKill) {
        console.log(`%c[SlitherML] 🔄 Spawn-kill detected (${this.survivalTime.toFixed(1)}s, fitness ${Math.floor(runFitness)}, length ${this.peakLength || 0}). Mulligan granted: life excluded from rollout.`, 'color: #94a3b8;');
        return;
      }

      this.episodeFitnesses.push(runFitness);

      if (this.currentEpisode < this.episodesPerGenome) {
        this.currentEpisode++;
        this._saveCheckpoint();
        return;
      }

      // If fewer than 2 valid runs were recorded due to spawn-kill mulligans:
      if (this.episodeFitnesses.length < 2) {
        console.log(`%c[SlitherML] 🔄 Insufficient rollout data (${this.episodeFitnesses.length}/3 valid episodes due to spawn-kill mulligans). Extending rollout to gather reliable signal...`, 'color: #94a3b8;');
        this.currentEpisode = Math.max(1, this.episodeFitnesses.length + 1);
        this._saveCheckpoint();
        return;
      }

      const sorted = [...this.episodeFitnesses].sort((a, b) => b - a);
      const aggregateFitness = (sorted[0] + sorted[1]) / 2;
      const meanFitness = this.episodeFitnesses.reduce((a, b) => a + b, 0) / this.episodeFitnesses.length;
      const isTeacherCandidate = this.candidatePending && this.candidateSource === 'teacher';
      const origin = this.candidatePending ? 'teacher candidate' : 'mutant';
      const previousBest = this.bestFitness;
      // Two-Tier Tuition Preservation:
      // Tier 2: Candidate achieves >= 35% of champion OR >= 5,000 points -> blend 20% into championBrain.
      // Tier 3: Candidate achieved rollout under Tier 2 -> championBrain is protected, but 15% is blended
      //         into baseBrain (mutant breeding seed) so valuable human imitation learning never evaporates.
      const teacherWorthyOfChampBlend = isTeacherCandidate && (aggregateFitness >= previousBest * 0.35 || aggregateFitness >= 5000);
      const teacherWorthyOfBreedingBlend = isTeacherCandidate && !teacherWorthyOfChampBlend && aggregateFitness <= previousBest;
      const isAutonomousMutant = !this.candidatePending;

      // Consistency gate: To prevent 1 lucky fluke run from crowning an erratic genome,
      // the median run (sorted[1]) must be at least 40% of previousBest.
      const passesConsistencyGate = sorted[1] >= previousBest * 0.40;

      // Mutant soft trait retention: autonomous mutants scoring >= 75% of previousBest
      // blend 15% weights into the baseBrain breeding seed (championBrain untouched).
      const mutantWorthyOfBlend = isAutonomousMutant && aggregateFitness <= previousBest && aggregateFitness >= previousBest * 0.75;

      // Prodigy Single-Run Rule: An autonomous mutant that achieved a heroic single life
      // (>= 22,000 points or >= 55% of all-time peak fitness) but missed the 75% aggregate bar
      // due to bad spawn-luck on remaining runs should NOT have its breakthrough weights discarded.
      // Blends 20% of its weights into the breeding template.
      const isProdigyRun = isAutonomousMutant && !mutantWorthyOfBlend && aggregateFitness <= previousBest &&
        sorted[0] >= Math.min(22000, (this.allTimePeakFitness || previousBest) * 0.55);

      // Strict elitism with Consistency Gate:
      if (aggregateFitness > this.bestFitness && passesConsistencyGate) {
        this.stagnantGens = 0;
        this.bestFitness = aggregateFitness;
        this.allTimePeakFitness = Math.max(this.allTimePeakFitness || 0, aggregateFitness);
        this.championBrain = NeuralNetwork.fromJSON(this.brain.export(), this.inputCount);
        console.log(`%c[SlitherML] 🏆 NEW CHAMPION (${origin})! Gen ${this.generation} — Best-2-of-3 aggregate ${Math.floor(aggregateFitness)} (mean ${Math.floor(meanFitness)}, median ${Math.floor(sorted[1])}), previous ${Math.floor(previousBest)}`, 'color: #00f5d4; font-weight: bold;');
      } else {
        if (aggregateFitness > this.bestFitness && !passesConsistencyGate) {
          console.log(`%c[SlitherML] ⚠️ Candidate aggregate ${Math.floor(aggregateFitness)} exceeded champion ${Math.floor(this.bestFitness)} but was rejected by Consistency Gate (median ${Math.floor(sorted[1])} < 40% of champ ${Math.floor(previousBest * 0.40)}).`, 'color: #ffb703;');
        }
        this.stagnantGens = (this.stagnantGens || 0) + 1;
        // Plateau relief: after 6 consecutive stagnant generations, gently ease bar by 0.5%
        // (accelerating to 1.0% when stagnantGens >= 15) to prevent 1-in-1000 lucky spawn
        // outliers from permanently freezing evolution.
        // Adaptive Plateau Floor: Standard floor is 85% of peak fitness. When deep stagnation persists
        // (>= 50 gens without a champion), ease to 75%, and at >= 60 gens ease to 70% to prevent
        // infinite stagnation traps (e.g., Gen 134 at 67 stagnant gens).
        this.allTimePeakFitness = Math.max(this.allTimePeakFitness || 0, this.bestFitness);
        const floorRatio = this.stagnantGens >= 60 ? 0.70 : (this.stagnantGens >= 50 ? 0.75 : 0.85);
        const minBar = this.allTimePeakFitness * floorRatio;
        if (this.stagnantGens >= 15) {
          const decayed = Math.max(minBar, this.bestFitness * 0.990);
          console.log(`%c[SlitherML] 📉 Deep Plateau Relief (Stagnant Gen ${this.stagnantGens}): Fitness bar eased from ${this.bestFitness.toFixed(1)} to ${decayed.toFixed(1)} (-1.0%${decayed === minBar ? `, capped at ${Math.round(floorRatio * 100)}% floor` : ''})`, 'color: #94a3b8;');
          this.bestFitness = decayed;
        } else if (this.stagnantGens >= 6) {
          const decayed = Math.max(minBar, this.bestFitness * 0.995);
          console.log(`%c[SlitherML] 📉 Plateau Relief (Stagnant Gen ${this.stagnantGens}): Fitness bar gently eased from ${this.bestFitness.toFixed(1)} to ${decayed.toFixed(1)} (-0.5%${decayed === minBar ? `, capped at ${Math.round(floorRatio * 100)}% floor` : ''})`, 'color: #94a3b8;');
          this.bestFitness = decayed;
        }

        if (this.candidatePending) {
          if (teacherWorthyOfChampBlend && this.championBrain) {
            console.log(`%c[SlitherML] 🧬 Teacher candidate achieved competitive rollout (${Math.floor(aggregateFitness)} vs ${Math.floor(this.bestFitness)}). Integrating 20% tuition into Champion template!`, 'color: #a855f7; font-weight: bold;');
            const candidateBrain = this.brain;
            for (let layer = 0; layer < this.championBrain.weights.length; layer++) {
              const chW = this.championBrain.weights[layer];
              const cW = candidateBrain.weights[layer];
              for (let j = 0; j < chW.length; j++) {
                chW[j] = (chW[j] * 0.8) + (cW[j] * 0.2);
              }
              const chB = this.championBrain.biases[layer];
              const cB = candidateBrain.biases[layer];
              for (let j = 0; j < chB.length; j++) {
                chB[j] = (chB[j] * 0.8) + (cB[j] * 0.2);
              }
            }
            if (this.championBrain.biases && this.championBrain.biases.length > 0) {
              const chOutB = this.championBrain.biases[this.championBrain.biases.length - 1];
              if (chOutB && chOutB.length > 0) {
                chOutB[0] = Math.max(-0.15, Math.min(0.15, chOutB[0]));
              }
            }
            this.stagnantGens = 0;
          } else {
            console.log(`%c[SlitherML] 🚫 Teacher candidate rejected from champion: aggregate ${Math.floor(aggregateFitness)} did not beat champion ${Math.floor(this.bestFitness)}. Champion genome preserved; infusing 15% tuition into mutant breeding template.`, 'color: #ffb703; font-weight: bold;');
          }
        }
      }

      this.generation++;
      this.candidatePending = false;
      this.candidateSource = null;

      // Breed the next genome with Anti-Stagnation Adaptive Mutation (from MEMWashburn/NEAT)
      // and High-Fitness Annealed Mutation (for elite champions > 30000 fitness).
      if (this.championBrain) {
        const baseBrain = NeuralNetwork.fromJSON(this.championBrain.export(), this.inputCount);
        if (baseBrain) {
          if ((mutantWorthyOfBlend || teacherWorthyOfBreedingBlend || isProdigyRun) && this.brain) {
            let blendRatio = 0.15;
            let blendReason = '';
            if (teacherWorthyOfBreedingBlend) {
              blendRatio = 0.15;
              blendReason = `Preserving human tuition (${Math.floor(aggregateFitness)} aggregate): Blending 15% teacher candidate weights into mutant breeding template!`;
            } else if (isProdigyRun) {
              blendRatio = 0.20;
              blendReason = `🌟 Prodigy single-run breakthrough (${Math.floor(sorted[0])} peak life, aggregate ${Math.floor(aggregateFitness)} vs champ ${Math.floor(this.bestFitness)}): Blending 20% weights into mutant breeding seed!`;
            } else {
              blendRatio = 0.15;
              blendReason = `Autonomous mutant showed breakthrough performance (${Math.floor(aggregateFitness)} aggregate vs champ ${Math.floor(this.bestFitness)}). Blending 15% weights into mutant breeding template!`;
            }
            console.log(`%c[SlitherML] 🧬 ${blendReason}`, 'color: #a855f7; font-weight: bold;');
            const candidateBrain = this.brain;
            const keepRatio = 1.0 - blendRatio;
            for (let layer = 0; layer < baseBrain.weights.length; layer++) {
              const bW = baseBrain.weights[layer];
              const cW = candidateBrain.weights[layer];
              for (let j = 0; j < bW.length; j++) {
                bW[j] = (bW[j] * keepRatio) + (cW[j] * blendRatio);
              }
              const bB = baseBrain.biases[layer];
              const cB = candidateBrain.biases[layer];
              for (let j = 0; j < bB.length; j++) {
                bB[j] = (bB[j] * keepRatio) + (cB[j] * blendRatio);
              }
            }
            if (baseBrain.biases && baseBrain.biases.length > 0) {
              const baseOutB = baseBrain.biases[baseBrain.biases.length - 1];
              if (baseOutB && baseOutB.length > 0) {
                baseOutB[0] = Math.max(-0.15, Math.min(0.15, baseOutB[0]));
              }
            }
          }

          const isStagnant = this.stagnantGens >= 3;
          const isDeepStagnant = this.stagnantGens >= 8;
          const isElite = this.bestFitness > 30000;

          let mutRate = isDeepStagnant ? 0.18 : (isStagnant ? 0.12 : 0.06);
          let mutStrength = isDeepStagnant ? 0.35 : (isStagnant ? 0.28 : 0.16);

          if (isElite) {
            mutRate = isDeepStagnant ? 0.09 : (isStagnant ? 0.06 : 0.035);
            mutStrength = isDeepStagnant ? 0.20 : (isStagnant ? 0.14 : 0.08);
          }

          if (isDeepStagnant) {
            console.log(`%c[SlitherML] 🔥 Deep Stagnation Shake (Gen ${this.generation}, Stagnant ${this.stagnantGens}): Escalating mutation to escape deep local minimum (rate ${mutRate}, strength ${mutStrength}${isElite ? ', annealed' : ''})!`, 'color: #f43f5e; font-weight: bold;');
          } else if (isStagnant) {
            console.log(`%c[SlitherML] ⚡ Anti-Stagnation Activated (Gen ${this.generation}): Escalating mutation (rate ${mutRate}, strength ${mutStrength}${isElite ? ', annealed' : ''}) to escape local minima!`, 'color: #a855f7; font-weight: bold;');
          }
          baseBrain.mutate(mutRate, mutStrength);
          this.brain = baseBrain;
        }
      } else {
        this.brain.mutate(0.08, 0.20);
      }

      this.currentEpisode = 1;
      this.episodeFitnesses = [];
      this._saveCheckpoint();
    }

    _saveChampionWeights() {
      try {
        const championExport = this.championBrain
          ? this.championBrain.export()
          : this.brain.export();
        const activeExport = this.brain.export();
        const data = {
          brain: championExport,
          activeBrain: activeExport,
          hasChampion: Boolean(this.championBrain),
          bestFitness: this.bestFitness,
          allTimePeakFitness: this.allTimePeakFitness || this.bestFitness,
          generation: this.generation,
          currentEpisode: this.currentEpisode,
          episodeFitnesses: this.episodeFitnesses,
          candidatePending: this.candidatePending,
          candidateSource: this.candidateSource,
          kills: this.kills,
          teacherSamples: this.teacherSamples,
          allTimePeakLength: this.allTimePeakLength,
          stagnantGens: this.stagnantGens,
          telemetry: this.flightRecorder.getTelemetrySummary(),
          timestamp: Date.now()
        };
        const serialized = JSON.stringify(data);
        localStorage.setItem(STORAGE_KEY, serialized);
        if (typeof window.__slitherMLBridgeSave === 'function') {
          window.__slitherMLBridgeSave(serialized);
        }
        if (typeof GM_setValue === 'function') {
          try {
            GM_setValue(STORAGE_KEY, serialized);
          } catch (_) {}
        }
      } catch (e) {
        console.warn('[SlitherML] Could not persist checkpoint:', e && e.message ? e.message : e);
      }
    }

    getAutopsyReport() {
      const a = this.flightRecorder.lastDeathAutopsy;
      if (!a) {
        console.log('%c[SlitherML 💀 Autopsy] No death autopsy on record (snake has not died yet in this session).', 'color: #94a3b8;');
        return null;
      }
      const lines = [
        '======================================================',
        '💀 SLITHER ML FLIGHT RECORDER: LAST DEATH AUTOPSY',
        '======================================================',
        `Cause:        ${a.cause} (Confidence: ${Math.round(a.confidence * 100)}%)`,
        `Description:  ${a.description}`,
        `Fatal Stance: ${a.fatalStance}`,
        `Episode Info: Gen ${a.generation} | Episode ${a.episode} | ${a.isHumanDriven ? 'Teacher Mode' : 'Autonomous'}`,
        `Survival:     ${a.survivalTime}s | Final Score: ${a.finalScore} | Peak Length: ${a.peakLength}`,
        a.killerProfile ? `Killer:       Type: ${a.killerProfile.type} | Dist: ${a.killerProfile.dist} | Mass Ratio: ${a.killerProfile.massRatio}x | Speed: ${a.killerProfile.speed}` : 'Killer:       Unknown / Off-screen',
        a.preCrashTelemetry ? `Pre-Crash:    Pos: (${a.preCrashTelemetry.x}, ${a.preCrashTelemetry.y}) | Speed: ${a.preCrashTelemetry.speed} | Boost: ${a.preCrashTelemetry.isBoosting} | TurnDelta: ${a.preCrashTelemetry.turnDelta}` : '',
        '======================================================'
      ].filter(Boolean);
      const text = lines.join('\n');
      console.log(`%c${text}`, 'color: #38bdf8; font-family: monospace; font-size: 11px;');
      return a;
    }

    exportCheckpoint() {
      const data = {
        brain: (this.championBrain || this.brain).export(),
        activeBrain: this.brain.export(),
        hasChampion: Boolean(this.championBrain),
        bestFitness: this.bestFitness,
        allTimePeakFitness: this.allTimePeakFitness || this.bestFitness,
        generation: this.generation,
        currentEpisode: this.currentEpisode,
        episodeFitnesses: this.episodeFitnesses,
        candidatePending: this.candidatePending,
        candidateSource: this.candidateSource,
        kills: this.kills,
        teacherSamples: this.teacherSamples,
        allTimePeakLength: this.allTimePeakLength,
        stagnantGens: this.stagnantGens,
        telemetry: this.flightRecorder.getTelemetrySummary(),
        timestamp: Date.now()
      };
      return JSON.stringify(data, null, 2);
    }
  }

  // ==========================================
  // 3. CONTROLLER & HYBRID INTERFACE
  // ==========================================
  class SlitherMLController {
    constructor() {
      this.agent = new SlitherMLAgent();
      this.enabled = true; // true = Autopilot (AI), false = Teacher Mode (You play, AI learns)
      this.visualsEnabled = true;
      this.autoRespawn = true;
      this.isAlive = false;
      this.hasPlayed = false;

      // Real Human Mouse Coordinates (Separated from synthetic bot targets)
      this.humanMouseX = 0;
      this.humanMouseY = 0;

      this.overlayCanvas = null;
      this.overlayCtx = null;
      this.hudElement = null;
      this._keydownHandler = null;
      this._bridgeLoadHandler = null;

      this._createHUD();
      this._createCanvasOverlay();
      this._bindInputs();
      this._startRespawnSupervisor();
      this._startLoop();
    }

    _createHUD() {
      const existing = document.getElementById('slither-ml-hud');
      if (existing) existing.remove();

      const hud = document.createElement('div');
      hud.id = 'slither-ml-hud';
      hud.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:space-between;padding:10px 14px;background:rgba(0,245,212,0.1);border-bottom:1px solid rgba(0,245,212,0.25);">
          <div style="display:flex;align-items:center;gap:6px;font-weight:700;font-size:12px;color:#00f5d4;">
            <span id="smlStatusDot" style="width:8px;height:8px;background:#00f5d4;border-radius:50%;box-shadow:0 0 8px #00f5d4;"></span>
            <span>SLITHER.IO HYBRID ML</span>
          </div>
          <span id="smlGenBadge" style="background:#7b2cbf;color:#fff;padding:2px 8px;border-radius:20px;font-size:10px;font-weight:700;">GEN ${this.agent.generation} | EP ${this.agent.currentEpisode}/3</span>
        </div>
        <div style="padding:10px 14px;display:flex;flex-direction:column;gap:6px;font-size:12px;color:#cbd5e1;">
          <div style="display:flex;justify-content:space-between;"><span>Mode:</span><span id="smlStatus" style="color:#10b981;font-weight:700;">AUTOPILOT (AI)</span></div>
          <div style="display:flex;justify-content:space-between;"><span>Stance:</span><span id="smlStance" style="font-weight:700;color:#00f5d4;">🌾 FORAGE</span></div>
          <div style="display:flex;justify-content:space-between;"><span>Radar:</span><span id="smlRadar" style="font-weight:700;color:#00f5d4;">🌾 CLEAR</span></div>
          <div style="display:flex;justify-content:space-between;"><span>Kills:</span><span id="smlKills" style="font-weight:700;color:#f43f5e;">0</span></div>
          <div style="display:flex;justify-content:space-between;"><span id="smlLabelFit">Run Fitness:</span><span id="smlFitness" style="font-weight:700;color:#fff;">0</span></div>
          <div style="display:flex;justify-content:space-between;"><span id="smlLabelSub">3-Run Avg:</span><span id="smlAvgFitness" style="font-weight:700;color:#38bdf8;">0</span></div>
          <div style="display:flex;justify-content:space-between;"><span>Champion:</span><span id="smlBestFitness" style="font-weight:700;color:#00f5d4;">${Math.floor(this.agent.bestFitness)}</span></div>
          <div style="display:flex;justify-content:space-between;"><span>Peak Score:</span><span id="smlPeakLength" style="font-weight:700;color:#fff;">${Math.floor(this.agent.allTimePeakLength || this.agent.peakLength || 0)}</span></div>
          <div style="display:flex;justify-content:space-between;"><span>Survival:</span><span id="smlTime" style="font-weight:700;color:#fff;">0s</span></div>
          <div style="display:flex;justify-content:space-between;"><span>Last Autopsy:</span><span id="smlAutopsy" style="font-weight:700;color:#94a3b8;font-size:10px;">NONE</span></div>
        </div>
        <div style="padding:10px 14px;display:flex;flex-direction:column;gap:6px;background:rgba(0,0,0,0.3);border-top:1px solid rgba(255,255,255,0.06);">
          <button id="smlBtnToggle" style="padding:6px;border-radius:6px;background:rgba(0,245,212,0.2);border:1px solid #00f5d4;color:#00f5d4;font-weight:700;font-size:11px;cursor:pointer;">Autopilot: ON [T]</button>
          <button id="smlBtnVisuals" style="padding:6px;border-radius:6px;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#fff;font-size:11px;cursor:pointer;">Vision Rays: ON [V]</button>
          <button id="smlBtnRespawn" style="padding:6px;border-radius:6px;background:rgba(255,255,255,0.1);border:1px solid rgba(255,255,255,0.2);color:#fff;font-size:11px;cursor:pointer;">Auto-Respawn: ON [R]</button>
          <div style="display:flex;gap:6px;">
            <button id="smlBtnExport" style="flex:1;padding:6px;border-radius:6px;background:rgba(0,245,212,0.15);border:1px solid rgba(0,245,212,0.3);color:#00f5d4;font-size:11px;cursor:pointer;font-weight:600;">Export [E]</button>
            <button id="smlBtnImport" style="flex:1;padding:6px;border-radius:6px;background:rgba(168,85,247,0.15);border:1px solid rgba(168,85,247,0.3);color:#c084fc;font-size:11px;cursor:pointer;font-weight:600;">Import [I]</button>
            <button id="smlBtnApex" style="flex:1;padding:6px;border-radius:6px;background:rgba(255,183,3,0.15);border:1px solid rgba(255,183,3,0.3);color:#ffb703;font-size:11px;cursor:pointer;font-weight:600;" title="Load Gen 100 Apex Champion (168 Kills, 1.08M Lessons)">Apex [A]</button>
          </div>
          <button id="smlBtnAutopsy" style="padding:6px;border-radius:6px;background:rgba(56,189,248,0.15);border:1px solid rgba(56,189,248,0.3);color:#38bdf8;font-size:11px;cursor:pointer;font-weight:600;">View Autopsy [O]</button>
          <button id="smlBtnReset" style="padding:6px;border-radius:6px;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.4);color:#f87171;font-size:11px;cursor:pointer;font-weight:600;">Reset Brain [X]</button>
          <input type="file" id="smlFileInput" accept=".json" style="display:none;" />
        </div>
      `;

      Object.assign(hud.style, {
        position: 'fixed',
        top: '16px',
        right: '16px',
        zIndex: '999999',
        width: '240px',
        background: 'rgba(11, 15, 25, 0.94)',
        backdropFilter: 'blur(10px)',
        border: '1px solid rgba(0, 245, 212, 0.35)',
        boxShadow: '0 10px 30px rgba(0,0,0,0.7), 0 0 16px rgba(0,245,212,0.25)',
        borderRadius: '12px',
        color: '#f8fafc',
        fontFamily: 'system-ui, -apple-system, sans-serif'
      });

      document.body.appendChild(hud);
      this.hudElement = hud;

      this._domRefs = {
        elFit: document.getElementById('smlFitness'),
        elAvg: document.getElementById('smlAvgFitness'),
        elBest: document.getElementById('smlBestFitness'),
        elPeak: document.getElementById('smlPeakLength'),
        elTime: document.getElementById('smlTime'),
        elKills: document.getElementById('smlKills'),
        elStance: document.getElementById('smlStance'),
        elRadar: document.getElementById('smlRadar'),
        elAutopsy: document.getElementById('smlAutopsy'),
        smlGenBadge: document.getElementById('smlGenBadge'),
        smlStatus: document.getElementById('smlStatus'),
        smlStatusDot: document.getElementById('smlStatusDot'),
        smlLabelSub: document.getElementById('smlLabelSub')
      };

      const btnToggle = document.getElementById('smlBtnToggle');
      if (btnToggle) btnToggle.onclick = () => this.toggleEnabled();
      const btnVisuals = document.getElementById('smlBtnVisuals');
      if (btnVisuals) btnVisuals.onclick = () => this.toggleVisuals();
      const btnRespawn = document.getElementById('smlBtnRespawn');
      if (btnRespawn) btnRespawn.onclick = () => this.toggleAutoRespawn();
      const btnAutopsy = document.getElementById('smlBtnAutopsy');
      if (btnAutopsy) btnAutopsy.onclick = () => this.agent.getAutopsyReport();
      const btnReset = document.getElementById('smlBtnReset');
      if (btnReset) btnReset.onclick = () => this.resetBrain();
      const btnExport = document.getElementById('smlBtnExport');
      if (btnExport) btnExport.onclick = () => this.exportBrain();
      const btnImport = document.getElementById('smlBtnImport');
      if (btnImport) btnImport.onclick = () => this.importBrain();
      const btnApex = document.getElementById('smlBtnApex');
      if (btnApex) btnApex.onclick = () => this.loadApexPreset();

      const fileInput = document.getElementById('smlFileInput');
      if (fileInput) {
        fileInput.onchange = (e) => {
          const file = e.target.files && e.target.files[0];
          if (!file) return;
          const reader = new FileReader();
          reader.onload = (evt) => {
            try {
              const parsed = JSON.parse(evt.target.result);
              if (this.agent.applySavedState(parsed)) {
                this.agent._saveChampionWeights();
                this._updateHUDMetrics();
                console.log(`%c[SlitherML] 📥 Successfully imported brain backup! Resumed Gen ${this.agent.generation} (Best: ${Math.floor(this.agent.bestFitness)})`, 'color: #00f5d4; font-weight: bold;');
              } else {
                console.warn('[SlitherML] Invalid brain backup file format.');
              }
            } catch (err) {
              console.warn('[SlitherML] Failed to parse brain file:', err);
            }
          };
          reader.readAsText(file);
        };
      }
      this._updateHUDMetrics();
    }

    _createCanvasOverlay() {
      const existing = document.getElementById('slither-ml-overlay-canvas');
      if (existing) existing.remove();

      const canvas = document.createElement('canvas');
      canvas.id = 'slither-ml-overlay-canvas';
      Object.assign(canvas.style, {
        position: 'fixed',
        top: '0',
        left: '0',
        width: '100vw',
        height: '100vh',
        pointerEvents: 'none',
        zIndex: '100'
      });
      document.body.appendChild(canvas);
      this.overlayCanvas = canvas;
      this.overlayCtx = canvas.getContext('2d');

      const resize = () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
      };
      window.addEventListener('resize', resize);
      resize();
    }

    _bindInputs() {
      if (!pageWindow._originalSlitherMouseMove && typeof pageWindow.onmousemove === 'function') {
        pageWindow._originalSlitherMouseMove = pageWindow.onmousemove;
      }
      this.originalMouseMove = pageWindow._originalSlitherMouseMove || pageWindow.onmousemove;

      pageWindow.onmousemove = (e) => {
        if (!e._isSyntheticBotEvent && typeof e.clientX === 'number') {
          this.humanMouseX = e.clientX - (window.innerWidth / 2);
          this.humanMouseY = e.clientY - (window.innerHeight / 2);
        }
        const snake = getPlayerSnake();
        const isAliveNow = Boolean(snake && !snake.dead && typeof snake.xx === 'number');
        if (this.enabled && isAliveNow && !e._isSyntheticBotEvent) {
          return; // Ignore physical human mouse when Autopilot is active
        }
        if (typeof this.originalMouseMove === 'function') {
          this.originalMouseMove(e);
        }
      };

      this._mouseMoveHandler = (e) => {
        if (!e._isSyntheticBotEvent && typeof e.clientX === 'number') {
          this.humanMouseX = e.clientX - (window.innerWidth / 2);
          this.humanMouseY = e.clientY - (window.innerHeight / 2);
        }
      };
      window.addEventListener('mousemove', this._mouseMoveHandler, { capture: true, passive: true });

      this._keydownHandler = (e) => {
        if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
        if (e.code === 'KeyT' || e.code === 'KeyB') {
          this.toggleEnabled();
        } else if (e.code === 'KeyV') {
          this.toggleVisuals();
        } else if (e.code === 'KeyR') {
          this.toggleAutoRespawn();
        } else if (e.code === 'KeyX') {
          this.resetBrain();
        } else if (e.code === 'KeyE') {
          this.exportBrain();
        } else if (e.code === 'KeyI') {
          this.importBrain();
        } else if (e.code === 'KeyO') {
          this.agent.getAutopsyReport();
        } else if (e.code === 'KeyA') {
          this.loadApexPreset();
        }
      };
      window.addEventListener('keydown', this._keydownHandler);

      const handleLifecycleFlush = () => {
        const snake = getPlayerSnake();
        const isAliveNow = Boolean(snake && !snake.dead && typeof snake.xx === 'number');
        if (isAliveNow && this.agent.survivalTime >= 10) {
          if (!this.enabled && this.agent.lifeTeacherSamples >= 100) {
            this.agent.candidatePending = true;
            this.agent.candidateSource = 'teacher';
          }
          if (this.enabled && this.agent.fitness > this.agent.bestFitness && this.agent.bestFitness === 0) {
            this.agent.bestFitness = this.agent.fitness;
            this.agent.championBrain = NeuralNetwork.fromJSON(this.agent.brain.export(), this.agent.inputCount);
          }
        }
        this.agent._saveChampionWeights();
      };

      this._lifecycleFlushHandler = handleLifecycleFlush;
      this._visibilityChangeHandler = () => {
        if (document.visibilityState === 'hidden') {
          handleLifecycleFlush();
        }
      };

      window.addEventListener('pagehide', this._lifecycleFlushHandler);
      window.addEventListener('beforeunload', this._lifecycleFlushHandler);
      document.addEventListener('visibilitychange', this._visibilityChangeHandler);

      this._bridgeLoadHandler = (e) => {
        try {
          const raw = e && e.detail;
          if (!raw) return;
          const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw;
          if (!parsed || !parsed.brain) return;
          const isBetterOrNewer =
            ((parsed.bestFitness || 0) > (this.agent.bestFitness || 0)) ||
            ((parsed.generation || 1) > (this.agent.generation || 1)) ||
            ((parsed.timestamp || 0) > (this.agent.lastLoadedTimestamp || 0) &&
             (parsed.bestFitness || 0) >= (this.agent.bestFitness || 0));
          if (isBetterOrNewer && this.agent.applySavedState(parsed)) {
            localStorage.setItem(STORAGE_KEY, typeof raw === 'string' ? raw : JSON.stringify(parsed));
            this._updateHUDMetrics();
            console.log(`%c[SlitherML] 🔗 Synced checkpoint from Extension Storage (Gen ${this.agent.generation}, Best: ${Math.floor(this.agent.bestFitness)})`, 'color: #00f5d4; font-weight: bold;');
          }
        } catch (_) {}
      };
      window.addEventListener('SlitherML_BridgeLoad', this._bridgeLoadHandler);
      window.dispatchEvent(new CustomEvent('SlitherML_BridgeRequest'));
    }

    exportBrain() {
      try {
        const championExport = this.agent.championBrain
          ? this.agent.championBrain.export()
          : this.agent.brain.export();
        const activeExport = this.agent.brain.export();
        const data = {
          brain: championExport,
          activeBrain: activeExport,
          hasChampion: Boolean(this.agent.championBrain),
          bestFitness: this.agent.bestFitness,
          allTimePeakFitness: this.agent.allTimePeakFitness || this.agent.bestFitness,
          generation: this.agent.generation,
          currentEpisode: this.agent.currentEpisode,
          episodeFitnesses: this.agent.episodeFitnesses,
          candidatePending: this.agent.candidatePending,
          candidateSource: this.agent.candidateSource,
          kills: this.agent.kills,
          teacherSamples: this.agent.teacherSamples,
          allTimePeakLength: this.agent.allTimePeakLength,
          stagnantGens: this.agent.stagnantGens,
          timestamp: Date.now()
        };
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `slither_ml_brain_gen${this.agent.generation}.json`;
        document.body.appendChild(a);
        a.click();
        setTimeout(() => {
          a.remove();
          URL.revokeObjectURL(url);
        }, 100);
        console.log(`%c[SlitherML] 💾 Exported Gen ${this.agent.generation} brain to slither_ml_brain_gen${this.agent.generation}.json`, 'color: #00f5d4; font-weight: bold;');
      } catch (err) {
        console.warn('[SlitherML] Could not export brain:', err && err.message ? err.message : err);
      }
    }

    importBrain() {
      const fileInput = document.getElementById('smlFileInput');
      if (fileInput) {
        fileInput.value = '';
        fileInput.click();
      }
    }

    loadApexPreset() {
      try {
        const preset = SlitherMLAgent.getApexPreset();
        if (preset && typeof pageWindow.confirm === 'function' && this.agent.bestFitness > preset.bestFitness) {
          const ok = pageWindow.confirm(
            `[SlitherML] ⚠️ Active Champion Warning:\n\nYour active champion has higher fitness (${Math.floor(this.agent.bestFitness)}) than the Apex preset (${Math.floor(preset.bestFitness)}).\n\nOverwrite your trained champion with the Gen ${preset.generation} Apex preset?`
          );
          if (!ok) {
            console.log('%c[SlitherML] ℹ️ Apex preset load cancelled by user.', 'color: #94a3b8;');
            return false;
          }
        }
        if (this.agent.loadApexPreset()) {
          this.agent._saveChampionWeights();
          this._updateHUDMetrics();
          console.log(`%c[SlitherML] 👑 Successfully loaded Apex Champion Preset! (Gen ${this.agent.generation}, Best: ${Math.floor(this.agent.bestFitness)}, Peak: ${Math.floor(this.agent.allTimePeakLength)})`, 'color: #ffb703; font-weight: bold;');
          return true;
        }
      } catch (err) {
        console.error('%c[SlitherML] ❌ Controller failed to load Apex preset:', 'color: #ef4444; font-weight: bold;', err);
      }
      return false;
    }

    _updateHUDMetrics() {
      const badge = document.getElementById('smlGenBadge');
      if (badge) {
        badge.textContent = !this.enabled
          ? `TEACHER MODE | ${this.agent.teacherSamples} LESSONS`
          : (this.agent.candidatePending
            ? `CANDIDATE EVAL | EP ${this.agent.currentEpisode}/${this.agent.episodesPerGenome}`
            : `GEN ${this.agent.generation} | EP ${this.agent.currentEpisode}/${this.agent.episodesPerGenome}`);
      }
      const elBest = document.getElementById('smlBestFitness');
      if (elBest) elBest.textContent = Math.floor(this.agent.bestFitness);
      const elPeak = document.getElementById('smlPeakLength');
      if (elPeak) elPeak.textContent = Math.floor(this.agent.allTimePeakLength || this.agent.peakLength);
      const elKills = document.getElementById('smlKills');
      if (elKills) elKills.textContent = this.agent.kills || 0;
      const elAvg = document.getElementById('smlAvgFitness');
      if (elAvg) {
        if (this.enabled) {
          const snake = getPlayerSnake();
          const isAliveNow = Boolean(snake && !snake.dead && typeof snake.xx === 'number');
          const episodeRuns = (isAliveNow || this.agent.fitness > 0)
            ? [...this.agent.episodeFitnesses, this.agent.fitness]
            : [...this.agent.episodeFitnesses];
          const avg = episodeRuns.length > 0
            ? episodeRuns.reduce((a, b) => a + b, 0) / episodeRuns.length
            : 0;
          elAvg.textContent = Math.floor(avg);
        } else {
          elAvg.textContent = `${this.agent.teacherSamples}`;
        }
      }
    }

    toggleEnabled() {
      this.enabled = !this.enabled;

      // Handing control back to the AI: any life that received human input is
      // not a fair evaluation episode, so drop it rather than let a stray [T]
      // tap bias the genome's score or hijack the end-of-life bookkeeping.
      if (this.enabled && this.agent.touchedByHuman) {
        this.agent.flushTeacherQuarantine();
        this.agent.touchedByHuman = false;
        this.agent.abandonCurrentEpisode = true;
        this.agent.fitness = 0;
      } else if (!this.enabled) {
        // Switching into Teacher Mode: ensure human trains on top of the clean Champion
        // genome rather than a random exploration mutant.
        if (this.agent.championBrain && !this.agent.candidatePending) {
          this.agent.brain = NeuralNetwork.fromJSON(this.agent.championBrain.export(), this.agent.inputCount);
          console.log('%c[SlitherML] 🧑‍🏫 Teacher Mode started from pristine Champion genome.', 'color: #f59e0b;');
        }
      }

      const btn = document.getElementById('smlBtnToggle');
      const status = document.getElementById('smlStatus');
      const dot = document.getElementById('smlStatusDot');
      const labelSub = document.getElementById('smlLabelSub');

      if (btn && status) {
        if (this.enabled) {
          btn.textContent = 'Autopilot: ON [T]';
          btn.style.color = '#00f5d4';
          btn.style.borderColor = '#00f5d4';
          btn.style.background = 'rgba(0,245,212,0.2)';
          status.textContent = 'AUTOPILOT (AI)';
          status.style.color = '#10b981';
          if (dot) {
            dot.style.background = '#00f5d4';
            dot.style.boxShadow = '0 0 8px #00f5d4';
          }
          if (labelSub) labelSub.textContent = '3-Run Avg:';
          console.log('%c[SlitherML] 🤖 Autopilot ACTIVE. Teacher lessons are validated over 3 autonomous lives before they can become Champion.', 'color: #00f5d4; font-weight: bold;');
        } else {
          btn.textContent = 'Teacher: ON [T]';
          btn.style.color = '#f59e0b';
          btn.style.borderColor = '#f59e0b';
          btn.style.background = 'rgba(245,158,11,0.2)';
          status.textContent = 'TEACHER MODE (YOU PLAY)';
          status.style.color = '#f59e0b';
          if (dot) {
            dot.style.background = '#f59e0b';
            dot.style.boxShadow = '0 0 8px #f59e0b';
          }
          if (labelSub) labelSub.textContent = 'Teacher Lessons:';
          console.log('%c[SlitherML] 🎓 Teacher Mode ACTIVE: Play manually! The AI is shadowing your moves and learning your reflexes.', 'color: #f59e0b; font-weight: bold;');
        }
      }

      if (!this.enabled && typeof pageWindow.setAcceleration === 'function') {
        pageWindow.setAcceleration(0);
      }
    }

    toggleVisuals() {
      this.visualsEnabled = !this.visualsEnabled;
      const btn = document.getElementById('smlBtnVisuals');
      if (btn) btn.textContent = `Vision Rays: ${this.visualsEnabled ? 'ON' : 'OFF'} [V]`;
      if (!this.visualsEnabled && this.overlayCtx && this.overlayCanvas) {
        this.overlayCtx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
      }
    }

    toggleAutoRespawn() {
      this.autoRespawn = !this.autoRespawn;
      const btn = document.getElementById('smlBtnRespawn');
      if (btn) btn.textContent = `Auto-Respawn: ${this.autoRespawn ? 'ON' : 'OFF'} [R]`;
    }

    resetBrain() {
      this.agent.resetBrain();
      this._lastAutosaveTime = 0;
      const badge = document.getElementById('smlGenBadge');
      if (badge) badge.textContent = `GEN 1 | EP 1/3`;
      const elBest = document.getElementById('smlBestFitness');
      if (elBest) elBest.textContent = '0';
      const elAvg = document.getElementById('smlAvgFitness');
      if (elAvg) elAvg.textContent = '0';
      const elPeak = document.getElementById('smlPeakLength');
      if (elPeak) elPeak.textContent = '0';
      const elKills = document.getElementById('smlKills');
      if (elKills) elKills.textContent = '0';
    }

    destroy() {
      this.enabled = false;
      this.destroyed = true;
      if (this.hudElement) this.hudElement.remove();
      if (this.overlayCanvas) this.overlayCanvas.remove();
      if (this.supervisorInterval) clearInterval(this.supervisorInterval);
      if (this._keydownHandler) window.removeEventListener('keydown', this._keydownHandler);
      if (this._bridgeLoadHandler) window.removeEventListener('SlitherML_BridgeLoad', this._bridgeLoadHandler);
      if (this.originalMouseMove) pageWindow.onmousemove = this.originalMouseMove;
      if (this._mouseMoveHandler) {
        window.removeEventListener('mousemove', this._mouseMoveHandler, { capture: true });
      }
      if (this._lifecycleFlushHandler) {
        window.removeEventListener('pagehide', this._lifecycleFlushHandler);
        window.removeEventListener('beforeunload', this._lifecycleFlushHandler);
      }
      if (this._visibilityChangeHandler) {
        document.removeEventListener('visibilitychange', this._visibilityChangeHandler);
      }
    }

    _startRespawnSupervisor() {
      let lastAttempt = 0;
      this.supervisorInterval = setInterval(() => {
        if (this.destroyed || !this.autoRespawn || !this.hasPlayed) return;
        const snake = getPlayerSnake();
        const isPlaying = Boolean(snake && !snake.dead && typeof snake.xx === 'number');
        if (!isPlaying) {
          const now = Date.now();
          if (now - lastAttempt > 2200) {
            lastAttempt = now;
            this._attemptSpawn();
          }
        }
      }, 1000);
    }

    _triggerRespawn() {
      if (!this.autoRespawn) return;
      setTimeout(() => this._attemptSpawn(), 1500);
    }

    _attemptSpawn() {
      const snake = getPlayerSnake();
      const isAliveNow = Boolean(snake && !snake.dead && typeof snake.xx === 'number');
      if (isAliveNow) return;

      const nickInput = document.getElementById('nick') || document.querySelector('input[type="text"]') || document.querySelector('input');
      if (nickInput && (!nickInput.value || !nickInput.value.trim())) {
        nickInput.value = 'SlitherML-Bot';
      }

      if (pageWindow.play_btn) {
        if (pageWindow.play_btn.btnf && typeof pageWindow.play_btn.btnf.click === 'function') {
          pageWindow.play_btn.btnf.click();
          return;
        } else if (typeof pageWindow.play_btn.click === 'function') {
          pageWindow.play_btn.click();
          return;
        }
      }

      const selectors = [
        '#play_btn .btnf',
        '.btnf',
        '#play_btn',
        '#playh button',
        'button[type="submit"]',
        '.btnt.nsi'
      ];
      for (const sel of selectors) {
        const el = document.querySelector(sel);
        if (el) {
          try {
            el.click();
            el.dispatchEvent(new MouseEvent('mousedown', { bubbles: true, cancelable: true }));
            el.dispatchEvent(new MouseEvent('mouseup', { bubbles: true, cancelable: true }));
            return;
          } catch (_) { }
        }
      }
    }

    _startLoop() {
      const step = () => {
        if (this.destroyed) return;
        try {
          this._update();
        } catch (err) {
          console.error('[SlitherML] Update error:', err);
        }
        requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
    }

    _update() {
      const snake = getPlayerSnake();
      const isAliveNow = Boolean(snake && !snake.dead && typeof snake.xx === 'number');

      if (this.isAlive && !isAliveNow) {
        this.isAlive = false;
        const isHumanDriven = !this.enabled;
        this.agent.onDeath(isHumanDriven);

        const dom = this._domRefs || {};
        const autopsy = this.agent.flightRecorder.lastDeathAutopsy;
        const elAutopsy = dom.elAutopsy || document.getElementById('smlAutopsy');
        if (elAutopsy && autopsy) {
          elAutopsy.textContent = autopsy.cause;
          elAutopsy.style.color = '#f43f5e';
          elAutopsy.title = autopsy.description;
        }

        const badge = document.getElementById('smlGenBadge');
        if (badge) {
          badge.textContent = isHumanDriven && !this.enabled
            ? `TEACHER MODE | ${this.agent.teacherSamples} LESSONS`
            : (this.agent.candidatePending
              ? `CANDIDATE EVAL | EP ${this.agent.currentEpisode}/${this.agent.episodesPerGenome}`
              : `GEN ${this.agent.generation} | EP ${this.agent.currentEpisode}/${this.agent.episodesPerGenome}`);
        }

        const status = document.getElementById('smlStatus');
        if (status) {
          status.textContent = 'RESPAWNING...';
          status.style.color = '#ffb703';
        }
        this._triggerRespawn();
      } else if (!this.isAlive && isAliveNow) {
        this.isAlive = true;
        this.hasPlayed = true;
        this._lastAutosaveTime = performance.now();
        this.agent.previousScore = (typeof snake.sct === 'number' && snake.sct > 0)
          ? Math.floor(snake.sct)
          : Math.floor(15 * (((snake.sc || 1)) - 1) / 0.001 || 10);
        const status = document.getElementById('smlStatus');
        if (status) {
          status.textContent = this.enabled ? 'AUTOPILOT (AI)' : 'TEACHER MODE (YOU PLAY)';
          status.style.color = this.enabled ? '#10b981' : '#f59e0b';
        }
      }

      // Periodic live autosave (every 10s while alive)
      if (isAliveNow) {
        const now = performance.now();
        if (!this._lastAutosaveTime) this._lastAutosaveTime = now;
        if (now - this._lastAutosaveTime >= 10000) {
          this._lastAutosaveTime = now;
          if (this.enabled && this.agent.fitness > this.agent.bestFitness && !this.agent.championBrain) {
            this.agent.championBrain = NeuralNetwork.fromJSON(this.agent.brain.export(), this.agent.inputCount);
          }
          this.agent._saveChampionWeights();
        }
      }

      // HUD Telemetry (Throttled to ~10 Hz to prevent DOM churn at 60 FPS)
      const nowMs = performance.now();
      if (!this._lastHudUpdate || nowMs - this._lastHudUpdate >= 100) {
        this._lastHudUpdate = nowMs;
        const dom = this._domRefs || {};
        const elFit = dom.elFit || document.getElementById('smlFitness');
        const elAvg = dom.elAvg || document.getElementById('smlAvgFitness');
        const elBest = dom.elBest || document.getElementById('smlBestFitness');
        const elPeak = dom.elPeak || document.getElementById('smlPeakLength');
        const elTime = dom.elTime || document.getElementById('smlTime');

        if (elFit) elFit.textContent = Math.floor(this.agent.fitness);
        if (elBest) elBest.textContent = Math.floor(this.agent.bestFitness);
        if (elPeak) elPeak.textContent = Math.floor(this.agent.allTimePeakLength || this.agent.peakLength);
        if (elTime) elTime.textContent = `${Math.floor(this.agent.survivalTime)}s`;

        const elKills = dom.elKills || document.getElementById('smlKills');
        if (elKills) elKills.textContent = this.agent.kills || 0;

        const elStance = dom.elStance || document.getElementById('smlStance');
        if (elStance) {
          const stance = this.agent.combatStance || 'FORAGE';
          if (stance === 'COIL') {
            elStance.textContent = '🌀 COIL';
            elStance.style.color = '#a855f7';
          } else if (stance === 'CONSTRICT') {
            elStance.textContent = '🐍 CONSTRICT';
            elStance.style.color = '#c084fc';
          } else if (stance === 'ATTACK') {
            elStance.textContent = '⚔️ ATTACK';
            elStance.style.color = '#ff0055';
          } else if (stance === 'DEFEND') {
            const isBehemoth = (this.agent.massRatio || 1.0) > 1.5;
            elStance.textContent = isBehemoth ? '🛡️ DEFEND (BEHEMOTH)' : '🛡️ DEFEND';
            elStance.style.color = isBehemoth ? '#ef4444' : '#ffb703';
          } else if (stance === 'VACUUM') {
            elStance.textContent = '🍖 VACUUM';
            elStance.style.color = '#38bdf8';
          } else {
            elStance.textContent = '🌾 FORAGE';
            elStance.style.color = '#00f5d4';
          }
        }

        const elRadar = dom.elRadar || document.getElementById('smlRadar');
        if (elRadar) {
          const front = this.agent.frontObstacleType || 'none';
          const closest = this.agent.closestObstacleType || 'none';
          const shield = this.agent.selfShieldSide || 0;
          const slit = this.agent.encirclementEscape;
          const slipstream = this.agent.slipstreamTarget;

          if (slit && slit.isOpen) {
            elRadar.textContent = '🚪 SLIT BREACH';
            elRadar.style.color = '#38bdf8';
          } else if (front === 'head') {
            elRadar.textContent = '🚨 HEAD ALERT';
            elRadar.style.color = '#ef4444';
          } else if (shield !== 0) {
            elRadar.textContent = shield > 0 ? '🛡️ SHIELD (L)' : '🛡️ SHIELD (R)';
            elRadar.style.color = '#c084fc';
          } else if (front === 'body') {
            elRadar.textContent = '🧱 ENEMY FLANK';
            elRadar.style.color = '#f59e0b';
          } else if (front === 'wall') {
            elRadar.textContent = '⚡ ARENA WALL';
            elRadar.style.color = '#eab308';
          } else if (slipstream) {
            elRadar.textContent = '💨 SLIPSTREAM';
            elRadar.style.color = '#34d399';
          } else if (closest === 'head') {
            elRadar.textContent = '⚠️ PREDATOR NEAR';
            elRadar.style.color = '#fb923c';
          } else {
            elRadar.textContent = '🌾 CLEAR';
            elRadar.style.color = '#00f5d4';
          }
        }

        if (elAvg) {
          if (this.enabled) {
            const episodeRuns = (isAliveNow || this.agent.fitness > 0)
              ? [...this.agent.episodeFitnesses, this.agent.fitness]
              : [...this.agent.episodeFitnesses];
            const avg = episodeRuns.length > 0
              ? episodeRuns.reduce((a, b) => a + b, 0) / episodeRuns.length
              : 0;
            elAvg.textContent = Math.floor(avg);
          } else {
            elAvg.textContent = `${this.agent.teacherSamples}`;
          }
        }
      }

      // -------------------------------------------------------------
      // 1. AUTOPILOT MODE (AI DRIVING)
      // -------------------------------------------------------------
      if (this.enabled && isAliveNow) {
        const decision = this.agent.evaluate(snake, getAllSnakes(), pageWindow.foods, pageWindow.preys);

        const steerDistance = 300;
        const targetX = Math.cos(decision.targetAngle) * steerDistance;
        const targetY = Math.sin(decision.targetAngle) * steerDistance;
        const clientX = (window.innerWidth / 2) + targetX;
        const clientY = (window.innerHeight / 2) + targetY;

        pageWindow.xm = targetX;
        pageWindow.ym = targetY;
        pageWindow.fxm = targetX;
        pageWindow.fym = targetY;

        const slitherHandler = this.originalMouseMove || pageWindow._originalSlitherMouseMove;
        if (typeof slitherHandler === 'function') {
          try {
            slitherHandler({
              clientX,
              clientY,
              pageX: clientX,
              pageY: clientY,
              _isSyntheticBotEvent: true
            });
          } catch (_) { }
        }

        snake.wang = decision.targetAngle;
        // Do not force snake.ehang = decision.targetAngle; Slither's engine naturally interpolates 
        // the visual head/eyes angle smoothly. Overriding it here caused violent head shaking.

        // Primary Slither.io boost hook: setAcceleration(1|0) signals the client network engine to send boost packets.
        // We also mirror to snake.wmd for compatibility with older canvas-render forks that read wmd directly.
        if (typeof pageWindow.setAcceleration === 'function') {
          pageWindow.setAcceleration(decision.isBoosting ? 1 : 0);
        }
        if (snake && typeof snake.wmd !== 'undefined') {
          snake.wmd = decision.isBoosting;
        }
      }
      // -------------------------------------------------------------
      // 2. TEACHER MODE (YOU PLAY, AI LEARNS VIA BACKPROPAGATION)
      // -------------------------------------------------------------
      else if (!this.enabled && isAliveNow) {
        const targetX = this.humanMouseX || (typeof pageWindow.xm === 'number' ? pageWindow.xm : 0);
        const targetY = this.humanMouseY || (typeof pageWindow.ym === 'number' ? pageWindow.ym : 0);
        const isHumanBoosting = Boolean(snake.wmd);

        // Feed human mouse actions to neural network backprop
        this.agent.learnFromHuman(snake, getAllSnakes(), pageWindow.foods, pageWindow.preys, targetX, targetY, isHumanBoosting);
      }

      if (this.visualsEnabled && this.overlayCtx && this.overlayCanvas) {
        this._renderOverlay(snake, isAliveNow);
      }
    }

    _renderOverlay(snake, isAlive) {
      const ctx = this.overlayCtx;
      const canvas = this.overlayCanvas;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (!isAlive || !snake) return;

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const zoom = pageWindow.gsc || 0.9;

      const rays = this.agent.activeRays;
      if (rays && rays.length > 0) {
        ctx.save();
        ctx.lineWidth = 1.6;

        for (let i = 0; i < rays.length; i++) {
          const ray = rays[i];
          const distPx = ray.dist * zoom;
          const endX = centerX + Math.cos(ray.angle) * distPx;
          const endY = centerY + Math.sin(ray.angle) * distPx;

          const danger = 1.0 - ray.norm;
          let strokeStyle = 'rgba(0, 245, 212, 0.4)';
          if (ray.hitType === 'wall') {
            strokeStyle = 'rgba(255, 183, 3, 0.85)';
          } else if (ray.hitType === 'head') {
            strokeStyle = `rgba(255, 0, 85, ${0.55 + danger * 0.45})`;
          } else if (ray.hitType === 'body') {
            strokeStyle = `rgba(255, 140, 0, ${0.4 + danger * 0.55})`;
          } else if (ray.hitType === 'self') {
            strokeStyle = 'rgba(168, 85, 247, 0.75)';
          }

          ctx.strokeStyle = strokeStyle;
          ctx.beginPath();
          ctx.moveTo(centerX, centerY);
          ctx.lineTo(endX, endY);
          ctx.stroke();

          if (ray.norm < 0.98) {
            ctx.fillStyle = strokeStyle;
            ctx.beginPath();
            ctx.arc(endX, endY, 3.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.restore();
      }

      // Draw Nearby Detected Food Markers
      const detectedFoods = this.agent.detectedFoods;
      if (detectedFoods && detectedFoods.length > 0) {
        ctx.save();
        for (let i = 0; i < detectedFoods.length; i++) {
          const df = detectedFoods[i];
          const px = centerX + df.dx * zoom;
          const py = centerY + df.dy * zoom;

          if (df.isPrey) {
            // Evasive glowing prey orb
            ctx.fillStyle = '#ffb703';
            ctx.shadowColor = '#ffb703';
            ctx.shadowBlur = 8;
            ctx.beginPath();
            ctx.arc(px, py, 6 * zoom, 0, Math.PI * 2);
            ctx.fill();
          } else if (df.sz > 8) {
            // High-value dead snake drop / cluster
            ctx.fillStyle = 'rgba(0, 245, 212, 0.9)';
            ctx.shadowColor = '#00f5d4';
            ctx.shadowBlur = 6;
            ctx.beginPath();
            ctx.arc(px, py, 4.5 * zoom, 0, Math.PI * 2);
            ctx.fill();
          } else {
            // Standard food pellet
            ctx.fillStyle = 'rgba(0, 245, 212, 0.5)';
            ctx.beginPath();
            ctx.arc(px, py, 2.5 * zoom, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.restore();
      }

      // 8-Sector Food Radar Halo around Snake Head
      if (this.agent.foodSectors) {
        ctx.save();
        const radarRadius = 55 * zoom;
        const sectorSpan = (Math.PI * 2) / 8;
        const currentAng = (typeof snake.eang === 'number') ? snake.eang : (snake.ang || 0);

        for (let f = 0; f < 8; f++) {
          const foodIntensity = Math.min(1.0, this.agent.foodSectors[f] / 8.0);
          if (foodIntensity > 0.05) {
            const startAng = currentAng + (f * sectorSpan);
            const endAng = startAng + sectorSpan;
            ctx.strokeStyle = `rgba(0, 245, 212, ${0.25 + foodIntensity * 0.65})`;
            ctx.lineWidth = 3 + foodIntensity * 5;
            ctx.beginPath();
            ctx.arc(centerX, centerY, radarRadius + foodIntensity * 12, startAng, endAng);
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      if (this.agent.targetAngle !== undefined) {
        ctx.save();
        const targetLen = (this.agent.isBoosting ? 160 : 110) * zoom;
        const tx = centerX + Math.cos(this.agent.targetAngle) * targetLen;
        const ty = centerY + Math.sin(this.agent.targetAngle) * targetLen;

        const stance = this.agent.combatStance || 'FORAGE';
        let mainColor = '#00f5d4';
        if (!this.enabled) {
          mainColor = '#f59e0b';
        } else if (stance === 'COIL') {
          mainColor = '#a855f7';
        } else if (stance === 'CONSTRICT') {
          mainColor = '#c084fc';
        } else if (stance === 'ATTACK') {
          mainColor = '#ff0055';
        } else if (stance === 'DEFEND') {
          mainColor = '#ffb703';
        } else if (stance === 'VACUUM') {
          mainColor = '#38bdf8';
        } else if (this.agent.isBoosting) {
          mainColor = '#ff007f';
        }

        ctx.strokeStyle = mainColor;
        ctx.lineWidth = 3.5;
        ctx.shadowColor = mainColor;
        ctx.shadowBlur = 10;

        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(tx, ty);
        ctx.stroke();

        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.arc(tx, ty, 5.5, 0, Math.PI * 2);
        ctx.fill();

        // If in COIL stance, draw protective orbital ring around head (from ErmiyaEskandary Circle Method)
        if (stance === 'COIL') {
          ctx.save();
          ctx.strokeStyle = 'rgba(168, 85, 247, 0.8)';
          ctx.lineWidth = 3;
          ctx.shadowColor = '#a855f7';
          ctx.shadowBlur = 12;
          ctx.beginPath();
          ctx.arc(centerX, centerY, 48 * zoom, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }

        // If in ATTACK stance and attackTarget is set, draw intercept crosshair on target
        if (stance === 'ATTACK' && this.agent.attackTarget) {
          const at = this.agent.attackTarget;
          const ax = centerX + (at.interceptX - snake.xx) * zoom;
          const ay = centerY + (at.interceptY - snake.yy) * zoom;
          ctx.strokeStyle = '#ff0055';
          ctx.lineWidth = 2;
          ctx.shadowColor = '#ff0055';
          ctx.shadowBlur = 8;
          ctx.beginPath();
          ctx.arc(ax, ay, 10 * zoom, 0, Math.PI * 2);
          ctx.moveTo(ax - 14 * zoom, ay);
          ctx.lineTo(ax + 14 * zoom, ay);
          ctx.moveTo(ax, ay - 14 * zoom);
          ctx.lineTo(ax, ay + 14 * zoom);
          ctx.stroke();
        }
        ctx.restore();
      }
    }
  }

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
  console.log('%c[SlitherML] 🚀 SlitherML v5.10.0 (Forensic Deductor & Pathfinder Flight Recorder) is RUNNING!', 'color: #00f5d4; font-size: 16px; font-weight: bold;');
})();