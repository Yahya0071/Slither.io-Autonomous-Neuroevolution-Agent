# Slither.io Autonomous Neuroevolution & Imitation Learning Agent (v5.9.0 — Grandmaster Combat & Failure-Trauma Filter Edition)

A master hybrid agent that plays in the real, official [Slither.io](https://slither.io) game. Powered by a **32-input -> 48 -> 24 -> 2 deep neural network** combining **Multi-Episode Neuroevolution**, **Real-Time Online Imitation Learning with Failure-Trauma Filtering (Teacher Mode)**, and an **Apex Tactical Combat State Machine** equipped with **Semantic Object Radar Perception, Pocket Shield Defense, Anti-Encirclement Slit Breaching, and Kinetic Boost Clearance**.

Synthesized from the best breakthroughs across the 4 most prominent open-source Slither.io AI repositories:
- **`dzaczek/slbot`**: Apex Predator Lead Interception Strike ($\tau$-kinematics) & Active Exploratory Scanning
- **`ErmiyaEskandary/Slither.io-bot`**: The Legendary Circle Method (`🌀 COIL` Stance) & Dual-Zone Arena Boundary Wall Guard
- **`MEMWashburn/Slither.io-NEAT`**: Confirmed Kill Bounty Recognition (+250 Fitness) & Anti-Stagnation Adaptive Mutation
- **`JuiHsiu/Slither-DRL`**: Mass-ROI Boost Governor & Anti-Jitter Hydrodynamic Steering Filter

---

## 🏛️ Open-Source Repository Benchmark & Synthesis

| Repository | Original Approach | Original Vulnerability / Trap | What We Synthesized Into Our "Super One" Engine |
| :--- | :--- | :--- | :--- |
| **[`dzaczek/slbot`](https://github.com/dzaczek/slbot)** | Dueling DQN + PyTorch via Selenium | 70–120ms Python-to-browser IPC latency caused fatal cut-offs | **Velocity-scaled lead intercept strikes**: predicts enemy future head coordinates ($\tau = 0.25\text{--}0.60\text{s}$) with crossing curls, plus exploratory sweeps to eliminate passive drifting. |
| **[`ErmiyaEskandary/Slither.io-bot`](https://github.com/ErmiyaEskandary/Slither.io-bot)** (~2.5k stars) | Community Heuristic Rule Engine | Static & predictable; lacked machine learning adaptability | **The Circle Method (`🌀 COIL` Stance)**: forms an impenetrable body fortress when surrounded or pinched. **Dual-Zone Boundary Defense**: prevents perimeter wall deaths. |
| **[`MEMWashburn/Slither.io-NEAT`](https://github.com/MEMWashburn/Slither.io-NEAT)** | Genetic Algorithm (NEAT) | Random networks died on spawn; degenerate center circles | **Confirmed Kill Recognition**: tracks vanished rival snakes and awards $+150$ fitness bounties. **Anti-Stagnation Mutation**: escalates mutation rate ($0.06 \to 0.12$) to escape local minima. |
| **[`JuiHsiu/Slither-DRL`](https://github.com/JuiHsiu/Slither-DRL)** | DRL (Actor-Critic / A2C) via Gym | "Boost-Bleed": held spacebar continuously, shrinking from 300 to 10 | **Mass-ROI Boost Governor**: strictly calculates expected mass burned vs pellet value; hard-stops sprints below minimum score thresholds. Double-smoothed steering eliminates jitter. |

---

## 📁 Project Structure

| File / Folder | Description |
| :--- | :--- |
| **[`console_inject.js`](./console_inject.js)** | **Single source of truth.** Single-file IIFE script to paste directly into your browser's DevTools Console (`F12`). |
| **[`slither_ml_agent.user.js`](./slither_ml_agent.user.js)** | Tampermonkey / Violentmonkey userscript. **Generated** from `console_inject.js`. |
| **[`extension/`](./extension)** | Chrome / Edge Extension (Manifest V3) that injects the agent automatically. [`extension/agent.js`](./extension/agent.js) is **generated**. |
| **[`quick_start.html`](./quick_start.html)** | Local launcher dashboard with 1-click script copy buttons. Its two inlined copies are **generated**. |
| **[`build.mjs`](./build.mjs)** | Regenerates every distribution artifact from `console_inject.js` and verifies they match. |
| **[`tests/`](./tests)** | Node test suite: 90 unit and behavioural tests (kinematics, combat stances, anti-encirclement slit breach, failure-trauma filter, kinetic clearance, slipstream drafting, semantic radar, pocket defense). |

> ⚠️ Only edit `console_inject.js`. The other copies are generated — run `npm run build`, then `npm test` to confirm they match.

---

## 🎮 How to Install & Run

### Method 1: Browser DevTools Console (Fastest — 5 Seconds)
1. Navigate to [https://slither.io](https://slither.io) in Chrome, Edge, Brave, or Firefox.
2. Press `F12` (or `Ctrl + Shift + J` / Right-Click ➔ **Inspect**) and click the **Console** tab.
3. Open [`console_inject.js`](./console_inject.js), copy the entire content, paste it into the console, and press `Enter`.
4. Click **Play** on the Slither home screen — the agent takes steering control immediately with the neon HUD.

### Method 2: Tampermonkey / Violentmonkey Userscript (Auto-Loads)
1. Install the [Tampermonkey](https://www.tampermonkey.net/) or [Violentmonkey](https://violentmonkey.github.io/) browser extension.
2. Open the extension dashboard and create a **New Userscript**.
3. Copy the entire content of [`slither_ml_agent.user.js`](./slither_ml_agent.user.js), paste it in, and save (`Ctrl + S`).
4. Visit [https://slither.io](https://slither.io) — the agent will activate automatically every match.

### Method 3: Unpacked Chrome Extension
1. Open `chrome://extensions` in your browser.
2. Enable **Developer mode** (top-right toggle).
3. Click **Load unpacked** and select the [`extension/`](./extension) folder in this repository.
4. Open [https://slither.io](https://slither.io).

---

## 🛡️ The 6-Stance Tactical Matrix & Mass-Ratio Radar

The agent dynamically evaluates surrounding game state, calculates relative scale ($\rho = \frac{\text{scale}_{\text{threat}}}{\text{scale}_{\text{self}}}$) and Minkowski collision clearance, and arbitrates between six distinct combat postures:

```
                          ┌───────────────────────────┐
                          │   GAME SENSORY PIPELINE   │
                          │   16 Rays + 8 Food Octants │
                          └─────────────┬─────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
        [Severe Threat /       [Vulnerable Flank       [Surrounded /
         Kinematic Cut-Off]      Score >= 80]           Score >= 300]
                 │                      │                      │
                 ▼                      ▼                      ▼
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │   🛡️ DEFEND     │    │   ⚔️ ATTACK     │    │    🌀 COIL      │
        │ Counter-Bank &  │    │ Lead Intercept  │    │ Circle Fortress │
        │ Emergency Brake │    │ Sprint Across   │    │ Body Shielding  │
        └─────────────────┘    └─────────────────┘    └─────────────────┘
                 ▲                      ▲                      ▲
                 └──────────────────────┼──────────────────────┘
                                        │
                            [Normal Scavenging & Radar]
                                        ▼
                               ┌─────────────────┐
                               │   🌾 FORAGE     │
                               │ Dual-Tier Radar │
                               │ Mass-ROI Boost  │
                               └─────────────────┘
```

1. **`🌾 FORAGE` (Cyan `#00f5d4`)**: Dual-tier food scavenging (ambient pellets up to 650px, dead snake debris & glowing preys up to 1,200px). When no food is in sight, activates exploratory patrol sweeping ($\pm 14^\circ$ at $0.6\text{Hz}$) to scan dark map areas.
2. **`🛡️ DEFEND` (Amber `#ffb703` / Crimson `#ef4444` on Behemoth)**: Kinematic 2D ray-intersection collision avoidance with **Dynamic Minkowski Clearance** ($D_{\text{danger}} = (R_{\text{self}} + R_{\text{threat}}) \times 2.25 + \text{speedBuffer}$). If facing a **Behemoth** ($\rho > 1.5$), escalates urgency $+35\%$, widens counter-banking arcs, and vetoes sprints. When pinched on flanks with forward open space, engages **Emergency Escape Sprint**. Also acts as **Boundary Wall Guard** near arena perimeter (`normWallDist > 0.91`).
3. **`⚔️ ATTACK` (Crimson `#ff0055`)**: Constant Turn-Rate and Acceleration (CTRA) Curvilinear Trajectory Interception ($\omega = \frac{\Delta \theta}{\Delta t}$). Projects intercepts along curved flight paths. Strictly vetoed against behemoths ($\rho > 1.5$) to prevent suicidal head-on collisions.
4. **`🌀 COIL` (Deep Violet `#a855f7`)**: When snake score $\ge 300$ and surrounded by multiple close enemies or pinched on both flanks, loops in a tight continuous circular orbit. Head stays protected inside its own body coils until threats disperse.
5. **`🐍 CONSTRICT` (Purple `#c084fc`)**: Offensive Boa Constrictor trap. When alongside smaller flank prey ($\rho \le 0.65, |\text{relBearing}| \ge 0.65$), circles around prey with short boost pulses to entrap them inside body coils.
6. **`🍖 VACUUM` (Sky Blue `#38bdf8`)**: Post-Kill Corpse Vacuum. Following confirmed kills, latches the 1,200px loot epicenter for 4.0s with a 3x weight multiplier and rush sprints to harvest mass before rivals arrive.

### 📐 Relative Mass Ratio & Minkowski Danger Expansion Engine
- **Minkowski Danger Clearance**: Traditional bots calculate danger clearance using only their own head radius ($R_{\text{self}} \times 4.5$), underestimating collision distances with giant snakes. SlitherML expands the clearance space to $(R_{\text{self}} + R_{\text{threat}}) \times 2.25 + \text{speedBuffer}$, providing realistic spatial safety regardless of enemy size.
- **Scale-Weighted Threat Severity**: Cut-off threats are weighted by $\min(2.5, \max(0.5, \rho))$, preventing small newborn worms from causing panic 180° turns while prioritizing giant cutting threats.
- **Apex Dominance**: When the agent is substantially larger than surrounding snakes ($\rho < 0.65$), threat dampening allows the agent to maintain lucrative feeding runs without yielding food to tiny worms.

### 👁️ Semantic Object Radar & Pocket Shield Defense (v5.8.0)
- **Object-Classified Ray Perception**: Rather than treating all ray intersections as generic lethal obstacles, SlitherML identifies the true semantic type of every surrounding entity:
  - 🚨 **`'head'` (Enemy Head)**: Active kinetic predator capable of turning, cutting, and boosting. Evasion urgency escalated $+35\%$ to trigger immediate proactive counter-banking.
  - 🛡️ **`'self'` (Own Body Shield)**: In Slither.io, own-body collisions cause **ZERO** damage. Danger urgency dampened by $75\%$ when facing own body coils, eliminating false panic turns and allowing smooth coil navigation.
  - 🧱 **`'body'` (Enemy Flank)**: Passive moving wall; safe for parallel drafting and pinning.
  - ⚡ **`'wall'` (Arena Boundary)**: Lethal boundary void; commands hard inward center-seeking turn with boost shutdown.
- **Pocket Shield Defense**: When cut off or threatened by an enemy head on one flank while our own body shield is flanking the opposite side, the agent executes **Pocket Defense** — steering hard into its own impenetrable body shield (`turnDelta = selfShieldSide * 1.0`), safely hiding behind its own armor coils where the enemy cannot strike without dying.
- **Combat Reflex Tuition**: During Teacher Mode, encountering an active enemy head (`frontObstacleType === 'head'`) automatically elevates the adaptive learning rate to $\eta = 0.008$, baking critical anti-head evasion and counter-banking reflexes deeply into the neural network weights.
- **Live Semantic Radar HUD**: Displays real-time obstacle classifications (`🚨 HEAD ALERT`, `🛡️ SHIELD (L/R)`, `🧱 ENEMY FLANK`, `⚡ ARENA WALL`, `🌾 CLEAR`) with color-coded raycaster beams.

### 🏆 Grandmaster Tactics & Failure-Trauma Filter (v5.9.0)
- **🚪 Anti-Encirclement Slit Breaching**: When an enemy behemoth attempts to trap our snake inside a tightening body loop ($\ge 7$ rays blocked by a larger predator), the agent scans for the widest open exit gap (`slitDist > headRadius * 4.5`). Instead of passively waiting inside in `COIL` stance, it vetoes coiling, aligns directly with the slit, and executes an **Emergency Slit Breach Sprint** before the loop seals shut.
- **🛡️ Failure-Trauma Filter (Quarantined Tuition)**: Human player deaths in Teacher Mode are often preceded by $0.5\text{--}1.0\text{s}$ of fatal error inputs. SlitherML buffers incoming tuition in a quarantined FIFO queue. Updates are backpropagated into weights only after surviving past the quarantine window; upon death, all quarantined frames are permanently incinerated so the AI never learns human crash errors.
- **⚡ Kinetic Enemy Boost Velocity Clearance**: `dangerDistance` expands based on both our speed and the approaching enemy's speed:
  $$D_{\text{danger}} = (R_{\text{self}} + R_{\text{threat}}) \times 2.25 + \text{speedBuffer}_{\text{self}} + \max(0, (v_{\text{threat}} - 5.5) \times 18)$$
  Sprinting projectile enemies traveling at $12\text{--}14$ speed trigger proactive counter-banking and braking over $120\text{px}$ earlier.
- **💨 Slipstream Wake Drafting**: Detects sprinting enemies ahead and doubles the forage priority of their ejected boost exhaust trails, allowing the snake to draft behind rivals and harvest mass at zero fuel cost.

---

## 🕹️ Hybrid Learning Modes

### 1. Autopilot Mode (Neuroevolution) — Default
- **How it works**: The AI pilots the snake autonomously using its neural policy.
- **Best-2-of-3 Rollout Scoring with Consistency Gate**: Every genome is tested across 3 full lives. To prevent unlucky server lag spikes or instant cuts from discarding elite breakthrough genomes, the single lowest life is dropped and the top 2 runs are averaged:
  $$\text{Aggregate Fitness} = \frac{\text{Run}_{\text{top 1}} + \text{Run}_{\text{top 2}}}{2}$$
  To prevent erratic 1-run fluke mutants from hijacking the champion slot, a **Consistency Gate** requires the median run to be at least $40\%$ of the champion's score ($\text{Median} \ge 0.40 \times \text{previousBest}$). Unavoidable server spawn-kills ($< 3.5\text{s}$ survival with $< 60$ fitness, or $< 5.0\text{s}$ with $< 40$ fitness and zero growth) grant an immediate mulligan and are excluded from the rollout.
- **Strict Elitism, Persistent Peak Floor & Fast Evolutionary Programming (FEP)**:
  - **Calibrated Rewards**: Confirmed kills yield $+250$ fitness, ensuring stable length scaling isn't eclipsed by lucky combat RNG spikes. Survival bonus dynamically escalates during stagnation ($4.0 + \min(6.0, \text{stagnantGens} \times 0.5)$) to incentivize endurance.
  - **Fast Evolutionary Programming (Cauchy Lévy Flights & Xavier Spectral Scaling)**: Mutations draw from a hybrid distribution (85% Gaussian micro-tuning + 15% heavy-tailed Cauchy perturbations $\tan(\pi(U - 0.5))$). Because Cauchy distributions have heavy polynomial tails, genomes execute occasional Lévy flights over fitness barriers, provably escaping local minima plateaus. Mutation strengths scale inversely with layer fan-in ($\sigma_l = \sigma_0 \sqrt{24 / n_{l, \text{in}}}$) to preserve activation variances across deep layers.
  - **Persistent Plateau Floor**: If evolution stagnates for $\ge 6$ consecutive generations, gentle $0.5\%$ plateau relief eases the bar; at $\ge 15$ generations, relief accelerates to $1.0\%$. `allTimePeakFitness` is persisted across sessions and page reloads, enforcing an uncompromised **85% floor** ($\text{bestFitness} \ge 0.85 \times \text{allTimePeakFitness}$).
  - **Spawn Protection**: Rollouts with fewer than 2 valid episodes due to mulligans automatically extend rather than granting unearned generation advances.
- **Autonomous Mutant Soft Trait Retention (Polyak Blending)**: When an autonomous mutant demonstrates solid performance by achieving an aggregate score $\ge 75\%$ of the champion ($\text{Aggregate} \ge 0.75 \times \text{previousBest}$), $15\%$ of its weights are softly blended into the breeding template (`baseBrain`) for subsequent generations ($W_{\text{seed}} = 0.85 \cdot W_{\text{champ}} + 0.15 \cdot W_{\text{mutant}}$), accelerating evolutionary search while keeping `championBrain` strictly untouched.

### 2. Teacher Mode (Online Imitation Learning) — Press `[T]`
- **How it works**: Press `[T]` or `[B]` at any point to take manual control. Starting Teacher Mode initializes tuition from the pristine `championBrain` rather than an exploration mutant.
- **Maximum Likelihood BCE Loss & Low-Pass Momentum Filter**: Boost probability is optimized via Binary Cross-Entropy loss ($\mathcal{L}_{\text{BCE}} = -[y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$). Under BCE, pre-activation error gradient is strictly linear ($\delta_{\text{boost}} = y - \hat{y}$), eliminating the vanishing gradient of MSE + Sigmoid and accelerating boost tuition by $20\times$. Updates incorporate low-pass inertial momentum ($\mu = 0.80$) to filter out high-frequency 60 FPS human mouse jitter.
- **L2 Decay & Centering**: L2 weight decay ($1 - \text{lr} \times 0.002$) prevents network saturation, while output steering bias centering decay ($0.985$) and tight clamping ($[-0.15, 0.15]$) eliminate permanent one-sided turn drift.
- **Candidate Evaluation & Two-Tier Tuition Preservation**: Trained teacher networks prove themselves over a 3-autonomous-life evaluation rollout without ever losing tuition:
  - **Tier 1 (Champion Crown)**: Beat the current champion fitness ($\text{Aggregate} > \text{bestFitness}$) while passing the Consistency Gate ($\text{Median} \ge 0.40 \times \text{previousBest}$) to be crowned as the new champion.
  - **Tier 2 (Champion Soft-Blend - 20%)**: When a candidate achieves $\ge 35\%$ of champion fitness OR $\ge 5,000$ points, $20\%$ of its learned weights are permanently integrated into `championBrain` and `stagnantGens` resets to 0 without lowering `bestFitness`.
  - **Tier 3 (Breeding Seed Blend - 15%)**: If a chaotic public server cuts the candidate's rollout short ($< 35\%$ and $< 5,000$), `championBrain` remains 100% untouched and protected, but $15\%$ of the candidate's weights are blended into the mutant breeding template (`baseBrain`). Human tuition never evaporates!

---

## ⌨️ In-Game Hotkeys

| Hotkey | Action | Description |
| :---: | :--- | :--- |
| **`[T]`** or **`[B]`** | **Toggle Autopilot / Teacher** | Switch between autonomous AI driving and human teacher shadowing. |
| **`[V]`** | **Toggle Vision Rays** | Show/hide the real-time neon raycaster and food sector radar canvas. |
| **`[R]`** | **Toggle Auto-Respawn** | Automatically click play and re-enter games 24/7 upon death. |
| **`[A]`** | **Load Apex Preset** | Instantly load pre-trained Gen 100 Champion (168 kills, 1.08M lessons, 37.1k floor) with overwrite protection. |
| **`[X]`** | **Reset Brain** | Wipe stored champion weights and restart evolution at Generation 1. |
| **`[E]`** | **Export Brain** | Download complete JSON checkpoint backup of current brain and training state. |
| **`[I]`** | **Import Brain** | Load and resume from a saved JSON brain checkpoint file. |

---

## 🛠️ Development

Requires Node.js ≥ 22.8. Zero external dependencies.

```bash
npm run build   # regenerate the 4 distribution artifacts from console_inject.js
npm test        # 90 unit and behavioural tests (flight simulator, combat stances, neuroevolution)
```

### What the 90 tests cover

| Area | What is asserted |
| :--- | :--- |
| **Tactical Combat** | Minkowski dynamic clearance & Behemoth evasion (`🛡️ DEFEND`); Anti-Encirclement Slit Breaching (`BREACH`); kinetic enemy boost velocity clearance expansion; Slipstream wake drafting; semantic object radar dampening on own body shield (`'self'`); Pocket Shield Defense against enemy head attacks; predator dynamic lead sprint (`⚔️ ATTACK`); defensive coil fortress (`🌀 COIL`); gapless offensive boa constrictor with Third-Party Perimeter Guard (`🐍 CONSTRICT`); post-kill corpse loot vacuum with "The Cross" perpendicular sweep (`🍖 VACUUM`); Angular Boost Gating ($R_{\text{turn}} = v/\omega$); fleeing chase pellet lead interception; Emergency Escape Sprint; Apex dominance; death wall repulsion. |
| **Mass & Boost Economy** | No boosting for plain pellets; unsafe sprint rejection; Mass-ROI boost governor; length preservation. |
| **Perception** | 1,200px corpse radar; dense 300-segment body sampling with zero ray tunneling; forward-only obstacle urgency. |
| **Kill Recognition** | Vanished nearby enemies register confirmed kills, increment live HUD counter, and award $+500$ fitness. |
| **Neuroevolution** | Anti-stagnation mutation escalation; high-fitness annealed mutation; spawn-kill mulligan; soft teacher tuition blending; strict elitism; 3-life rollout aggregate; candidate validation. |
| **Gradient Check** | `trainStep` gradients verified against analytical float64 backpropagation across all 2,736 weights and 74 biases. |
| **Artifact Drift** | All 4 distribution copies byte-identical; package/manifest versions synchronized. |

---

## ⚠️ Disclaimer

This automates play on a live multiplayer game. Doing so likely violates
Slither.io's terms of service and may be detected or blocked. It depends on
reverse-engineered game internals (`window.snake`, `window.xm/ym`,
`setAcceleration`, `grd`, `gsc`, `foods`, `preys`, `wmd`, …) that are not a
public API, so a game update can break it without warning. Use at your own risk.

