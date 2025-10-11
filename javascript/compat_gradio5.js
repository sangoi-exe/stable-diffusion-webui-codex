(function(){
  if (window.__forgeCompatInstalled) return; 
  window.__forgeCompatInstalled = true;

  const originalOnUiUpdate = window.onUiUpdate;
  const originalOnUiLoaded = window.onUiLoaded;

  function safeWrap(cb, tag){
    if (typeof cb !== 'function') return cb;
    return function(){
      try { return cb.apply(this, arguments); }
      catch(e){
        // Avoid console flood: log only first 10 of a given tag
        const key = '__forgeCompat_'+tag;
        window[key] = (window[key]||0)+1;
        if (window[key] <= 10) console.warn('for ge compat suppressed callback error in', tag, e);
        return undefined;
      }
    }
  }

  if (typeof originalOnUiUpdate === 'function') {
    window.onUiUpdate = function(cb){ return originalOnUiUpdate(safeWrap(cb,'onUiUpdate')); };
  }
  if (typeof originalOnUiLoaded === 'function') {
    window.onUiLoaded = function(cb){ return originalOnUiLoaded(safeWrap(cb,'onUiLoaded')); };
  }

  // Guard executeCallbacks to prevent a single bad callback from breaking the page
  if (typeof window.executeCallbacks === 'function') {
    const __origExec = window.executeCallbacks;
    window.executeCallbacks = function(callbacks){
      try {
        if (!Array.isArray(callbacks)) return __origExec.apply(this, arguments);
        for (const cb of callbacks) {
          try { typeof cb === 'function' && cb(); }
          catch(e){
            const key = '__forgeCompat_exec';
            window[key] = (window[key]||0)+1;
            if (window[key] <= 10) console.warn('forge compat suppressed executeCallbacks error', e);
          }
        }
      } catch(e) {
        // as last resort, fall back to original behavior
        return __origExec.apply(this, arguments);
      }
    }
  }
})();
