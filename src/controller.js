/**
 * SlitherMLController: HUD, Overlay, Inputs, and Browser Runtime Controller
 */

import { NeuralNetwork } from './network.js';
import { SlitherMLAgent, STORAGE_KEY } from './agent.js';

const pageWindow = (typeof unsafeWindow !== 'undefined' && unsafeWindow) ? unsafeWindow : (typeof window !== 'undefined' ? window : {});
const getPlayerSnake = () => pageWindow.snake || pageWindow.slither || null;
const getAllSnakes = () => pageWindow.snakes || pageWindow.slithers || [];

export class SlitherMLController {
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
        const serialized = this.agent.exportCheckpoint();
        const blob = new Blob([serialized], { type: 'application/json' });
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
        console.log(`%c[SlitherML] 💾 Exported Gen ${this.agent.generation} brain + flight telemetry to slither_ml_brain_gen${this.agent.generation}.json`, 'color: #00f5d4; font-weight: bold;');
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
          } else if (stance === 'HUNT') {
            elStance.textContent = '🎯 HUNT';
            elStance.style.color = '#fb923c';
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
            ctx.beginPath();
            ctx.arc(px, py, 6 * zoom, 0, Math.PI * 2);
            ctx.fill();
          } else if (df.sz > 8) {
            // High-value dead snake drop / cluster
            ctx.fillStyle = 'rgba(0, 245, 212, 0.9)';
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
        } else if (stance === 'HUNT') {
          mainColor = '#fb923c';
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
