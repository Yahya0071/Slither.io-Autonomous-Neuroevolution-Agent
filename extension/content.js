// SlitherML Extension Content Script: Injects the Neural Agent and bridges chrome.storage.local
(function() {
  const sendStoredCheckpoint = () => {
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
      chrome.storage.local.get(['slither_ml_champion_weights_v4'], (res) => {
        if (res && res.slither_ml_champion_weights_v4) {
          window.dispatchEvent(new CustomEvent('SlitherML_BridgeLoad', {
            detail: res.slither_ml_champion_weights_v4
          }));
        }
      });
    }
  };

  // Dispatch on initial load
  sendStoredCheckpoint();

  // Listen for agent bridge request
  window.addEventListener('SlitherML_BridgeRequest', sendStoredCheckpoint);

  // Listen for saves from page script and persist to chrome.storage.local
  window.addEventListener('SlitherML_BridgeSave', (e) => {
    if (e && e.detail && typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
      chrome.storage.local.set({ slither_ml_champion_weights_v4: e.detail });
    }
  });

  // Listen for reset/clear from page script and remove from chrome.storage.local
  window.addEventListener('SlitherML_BridgeClear', () => {
    if (typeof chrome !== 'undefined' && chrome.storage && chrome.storage.local) {
      chrome.storage.local.remove(['slither_ml_champion_weights_v4']);
    }
  });

  // Inject agent script into main execution context
  const script = document.createElement('script');
  script.src = chrome.runtime.getURL('agent.js');
  script.onload = function() {
    this.remove();
    sendStoredCheckpoint();
  };
  (document.head || document.documentElement).appendChild(script);
})();
