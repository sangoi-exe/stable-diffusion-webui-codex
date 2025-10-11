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
})();

