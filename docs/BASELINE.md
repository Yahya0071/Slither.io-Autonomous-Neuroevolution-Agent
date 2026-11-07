# Baseline System State (Phase 0)

Recorded: 2026-11-07  
Platform: Windows (PowerShell)  
Runtime: Node.js v24.14.0  
Package Version: 5.10.8  

## 1. Test Suite Execution

- **Command**: `npm test` (`node --test`)
- **Total Tests**: 145
- **Passing**: 145
- **Failing**: 0
- **Suites**: 0 (flat `node:test` execution)
- **Suite Duration**: ~1,200 ms – 1,350 ms

### Test Counts by File

| File | Tests | Focus Area |
| :--- | :---: | :--- |
| `tests/artifacts.test.mjs` | 7 | Distribution artifact parity, manifest synchronization |
| `tests/behaviour.test.mjs` | 55 | Kinematics, combat stances, spatial raycasting, evasion |
| `tests/champion-selection.test.mjs` | 16 | Rollouts, consistency gate, plateau decay, candidate blending |
| `tests/circling.test.mjs` | 6 | Directional bias regression, const→let crash proof, sim parity |
| `tests/crossover-silver.test.mjs` | 15 | Uniform crossover, silver brain lifecycle, fallback candidate |
| `tests/flight-recorder.test.mjs` | 21 | Telemetry buffer, autopsy deductor, zone distribution |
| `tests/neural-network.test.mjs` | 13 | Forward pass, BCE / MSE backpropagation, Xavier mutation |
| `tests/persistence-v5.test.mjs` | 12 | LocalStorage serialization, Apex preset restoration, episode recovery |
| **Total** | **145** | |

## 2. Artifact Drift & Verification Coverage

The build process (`build.mjs`) generates 4 output artifacts from `console_inject.js`:
1. `extension/agent.js` (byte-identical clone for Chrome/Edge MV3 extension)
2. `slither_ml_agent.user.js` (Tampermonkey/Violentmonkey script with userscript metadata header)
3. `quick_start.html` (`#codeConsole` raw inlined script block)
4. `quick_start.html` (`#codeUserscript` userscript inlined script block)

### What `tests/artifacts.test.mjs` Covers
- **Source Integrity**: Ensures `console_inject.js` exists, contains >1KB, parses without JavaScript syntax errors, and does not contain literal `</script>` tags that would break HTML inlining.
- **Byte Identity**: Asserts `extension/agent.js` strictly equals `console_inject.js` byte-for-byte.
- **Userscript Header Isolation**: Asserts `slither_ml_agent.user.js` is exactly `// ==UserScript== ... // ==/UserScript==\n\n` followed by `console_inject.js` with no agent code injected into the metadata header.
- **HTML Dashboard Synchronization**: Asserts `#codeConsole` in `quick_start.html` equals `console_inject.js`, and `#codeUserscript` equals `slither_ml_agent.user.js`.
- **Cross-Artifact Consistency**: Asserts all four artifacts agree with each other on content and length.
- **Version Manifest Synchronization**: Asserts version in `package.json` equals `@version` in userscript, `version` in `extension/manifest.json`, and banner in `console_inject.js`.
- **Extension Scope Security**: Asserts `extension/manifest.json` uses Manifest V3 and host permissions exclusively target `*://*.slither.io/*` and `*://*.slither.com/*`.
