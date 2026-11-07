# SlitherML: Autonomous Neuroevolution & Imitation Learning Agent

[![Node.js CI](https://github.com/Yahya0071/Slither.io-Autonomous-Neuroevolution-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Yahya0071/Slither.io-Autonomous-Neuroevolution-Agent/actions)
![Node Version](https://img.shields.io/badge/node-%3E%3D22.8-brightgreen)
![Dependencies](https://img.shields.io/badge/dependencies-0-blue)
![Tests](https://img.shields.io/badge/tests-146%20passing-success)

> [!CAUTION]
> **Educational & Research Purpose Only — Terms of Service Notice**
> This repository is developed strictly for academic study and algorithmic research into multi-agent continuous control, neuroevolution, online imitation learning, and kinematic trajectory prediction.
> Automated interaction with live multiplayer game servers violates the [Slither.io Terms of Service](https://slither.io/tos/). This software is intended for local study and offline simulation environments. Users assume full responsibility for compliance with third-party service terms.

SlitherML is a zero-dependency autonomous agent for the continuous 2D environment of Slither.io. It pairs a **32-input feedforward neural network** (32 → 48 → 24 → 2) with a **7-stance tactical combat state machine**, a **multi-episode neuroevolution engine**, an **online imitation learning subsystem (Teacher Mode)**, and a **flight recorder** for post-death trajectory autopsy and telemetry.

The project combines and refines techniques from prior open-source Slither.io agents:
- **Lead Interception & Scanning**: Kinematic lead intercept and exploratory sweeping (`dzaczek/slbot`).
- **Orbital Defense**: Circular coiling when surrounded, and dual-zone boundary wall repulsion (`ErmiyaEskandary/Slither.io-bot`).
- **Combat Rewards & Mutation**: Confirmed kill detection with mass-scaled bounties and anti-stagnation mutation schedules (`MEMWashburn/Slither.io-NEAT`).
- **Actuation Economics**: Mass-ROI sprint governor and inertia-filtered steering (`JuiHsiu/Slither-DRL`).

---

## Architecture Overview

```
                      ┌────────────────────────────────────────┐
                      │          ENVIRONMENT SENSORS           │
                      │ 16 Proximity Rays  •  8 Food Sectors   │
                      │ Kinematic Threat & Arena Center Fields │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │          NEURAL POLICY (32-D)          │
                      │       Dense MLP: 32 -> 48 -> 24 -> 2   │
                      │    Turn Delta (tanh) • Boost Prob (σ)  │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │       TACTICAL COMBAT ARBITER          │
                      │ FORAGE • DEFEND • ATTACK • COIL        │
                      │ CONSTRICT • VACUUM • HUNT              │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │         KINEMATIC ACTUATION            │
                      │ Steering Assist  •  Mass-ROI Boost Gov │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │      FLIGHT RECORDER & TELEMETRY       │
                      │ 60-Frame FIFO Buffer  •  Death Autopsy │
                      └────────────────────────────────────────┘
```

For detailed specifications of the sensor pipeline, network activations, and training loops, see [ARCHITECTURE.md](./ARCHITECTURE.md).

---

## Tactical Stance Matrix

The tactical arbiter evaluates relative mass `ρ = scale_threat / scale_self`, local obstacle geometry, and corridor clearances to select one of seven operational postures:

| Stance | Primary Objective | Trigger Conditions | Actuation Profile |
| :--- | :--- | :--- | :--- |
| **`FORAGE`** | Food collection & map exploration | No immediate lethal threats within buffer distance | Dual-tier food targeting (ambient pellets ≤ 650u, debris ≤ 1,200u); exploratory sweeps (±14° at 0.6 Hz) when empty. |
| **`DEFEND`** | Collision avoidance & boundary safety | Frontal collision risk, high enemy mass ratio, or wall proximity | Dynamic Minkowski buffer `(R_self + R_threat) × mult + speedBuffer`; counter-banks away from threat; emergency escape sprint when path is clear. |
| **`ATTACK`** | Intercept strike across enemy path | Vulnerable target identified, score ≥ 80, enemy not a behemoth (`ρ ≤ 1.5`) | Constant Turn-Rate and Acceleration (CTRA) trajectory lead intercept with crossing curl assist and sprint. |
| **`COIL`** | Self-protective orbit | Score ≥ 80, pinched by multiple threats with no open escape corridor | Enters continuous circular loop inside own body coils until surrounding threats disperse or an escape corridor opens. |
| **`CONSTRICT`** | Entrapment orbit | Parallel to smaller prey (`ρ ≤ 0.65`) with flank bearing | Cruising-speed orbit tightening around target while checking for outer third-party predators. |
| **`VACUUM`** | Rapid corpse harvesting | Confirmed kill recorded or dense corpse cluster detected | 6.5s priority lock on loot epicenter; high-value cluster attraction with rival scavenger avoidance. |
| **`HUNT`** | Stalking and pressure | Smaller snake nearby in open space with low danger urgency | Tracks target trajectory with leading offset to force defensive errors while earning proximity shaping reward. |

---

## Dual Learning Paradigms

### 1. Neuroevolution (Autopilot)
- **Multi-Episode Evaluation**: Evaluates each candidate genome over 3 separate episodes. Drops the single lowest score and averages the top 2: `Aggregate = (Run_top1 + Run_top2) / 2`.
- **Consistency Gate**: Rejects single-life lucky flukes; requires median life fitness `Median ≥ 0.40 × previousBest`.
- **Mulligan Filter**: Immediate respawn deaths (< 3.5s survival and score growth < 10) are discarded without penalizing candidate fitness.
- **Plateau Relief & Floor**: When stagnant for ≥ 6 generations without a new champion, fitness bar eases by 0.5% per generation (accelerating to 1.0% at ≥ 15 generations). The decay is bounded from below by an adaptive floor based on all-time peak fitness:
  - Standard floor: **85%** of peak (`stagnantGens < 50`)
  - Intermediate floor: **75%** of peak (`50 ≤ stagnantGens < 60`)
  - Deep relief floor: **70%** of peak (`stagnantGens ≥ 60`)
- **Fast Evolutionary Programming (FEP)**: Mutations sample from a hybrid distribution (85% Gaussian micro-tuning + 15% Cauchy heavy-tailed perturbations) with Xavier fan-in scaling (`σ_l = σ_0 × √(24 / n_in)`).
- **Uniform Crossover & Silver Brain**: When stagnation reaches ≥ 5 generations and a qualifying secondary genome (displaced champion or mutant scoring ≥ 40% of champion) exists, offspring are bred via 50/50 uniform crossover between the champion and silver genomes.

### 2. Online Imitation Learning (Teacher Mode)
- **Real-Time Backpropagation**: Toggle with `[T]` to take manual control. The network shadows human inputs and updates weights online using stochastic gradient descent with momentum (`μ = 0.80`).
- **Binary Cross-Entropy for Boost**: Boost activation is trained under BCE loss, providing a linear gradient error signal (`y - ŷ`) that prevents saturation.
- **Quarantined Tuition Buffer**: Input samples are stored in a rolling buffer. If a death occurs, the terminal crash frames (last 0.5–1.0s) are discarded to prevent learning panic errors.
- **Candidate Validation**: After completing a teacher session, the trained brain must prove itself across a full 3-episode autonomous evaluation before claiming the champion slot.

---

## Repository Structure

```
├── src/                          # Modular ES source files
│   ├── network.js                # NeuralNetwork class (forward, SGD, mutate, crossover)
│   ├── flight-recorder.js        # FlightRecorder class (telemetry buffer, autopsy deductor)
│   ├── perception.js             # 16-ray proximity sensor, food radar, corridor search
│   ├── stances.js                # Tactical state machine, steering assists, boost governor
│   ├── evolution.js              # Rollout scoring, consistency gate, breeding schedules
│   ├── agent.js                  # SlitherMLAgent state machine & persistence
│   ├── controller.js             # SlitherMLController HUD, event bindings, canvas overlay
│   └── index.js                  # Public API module export index
├── extension/                    # Manifest V3 Chrome/Edge extension
├── tests/                        # Node.js built-in test runner test suite (146 tests)
├── sim/                          # Offline headless simulation & analysis tools
├── docs/                         # Baseline, consistency audit, and evolution diagnosis
├── build.mjs                     # Zero-dependency bundler and artifact parity verifier
├── console_inject.js             # Generated console distribution bundle
├── slither_ml_agent.user.js      # Generated userscript distribution bundle
├── quick_start.html              # Generated local testing & dashboard interface
└── package.json                  # Project manifest (Node >= 22.8, zero dependencies)
```

---

## Installation & Usage

### Option A: Browser DevTools Console
1. Open [https://slither.io](https://slither.io) in any modern browser.
2. Open DevTools (`F12` or `Ctrl + Shift + J`).
3. Copy the contents of [`console_inject.js`](./console_inject.js) into the console and press Enter.
4. Start a game; the agent and HUD overlay activate automatically.

### Option B: Userscript (Tampermonkey / Violentmonkey)
1. Install Tampermonkey or Violentmonkey.
2. Create a new userscript and paste the contents of [`slither_ml_agent.user.js`](./slither_ml_agent.user.js).
3. Navigate to [https://slither.io](https://slither.io).

### Option C: Chrome / Edge Extension (Manifest V3)
1. Go to `chrome://extensions` and enable **Developer mode**.
2. Click **Load unpacked** and select the [`extension/`](./extension) directory.

---

## In-Game Keybindings

| Key | Function |
| :---: | :--- |
| **`[T]`** / **`[B]`** | Toggle between Autopilot (autonomous AI) and Teacher Mode (manual play + online learning). |
| **`[V]`** | Toggle sensor ray visualization and food radar HUD overlay. |
| **`[R]`** | Toggle auto-respawn supervisor on/off. |
| **`[O]`** | Output the root-cause autopsy of the last death to the browser console. |
| **`[A]`** | Load the Gen 100 Apex Champion preset checkpoint (prompts confirmation if active champion is higher). |
| **`[E]`** | Export the current neural weights and flight telemetry as a JSON checkpoint. |
| **`[I]`** | Open file selector to import a previously saved JSON checkpoint. |
| **`[X]`** | Reset weights to Generation 1 random baseline. |

---

## Development & Testing

The project has zero production and development dependencies, relying strictly on Node.js built-in modules (`node:fs`, `node:crypto`, `node:test`, `node:assert/strict`).

### Prerequisites
- Node.js ≥ 22.8

### Commands
```bash
# Verify build and regenerate all 4 distribution artifacts from src/
npm run build

# Run the complete test suite (146 tests)
npm test

# Launch the offline local dashboard
npm start

# Inspect synaptic weights, saliencies, and autopsy telemetry of a checkpoint
npm run analyze -- checkpoints/slither_ml_brain_gen100.json
```

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](./CONTRIBUTING.md) for details on code modularity, test requirements, and zero-dependency constraints.

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE) for details.
