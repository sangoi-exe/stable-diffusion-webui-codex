(function () {
  try {
    window.Codex = window.Codex || {};
    window.Codex.Components = window.Codex.Components || {};
    const bus = {
      listeners: {},
      on(event, fn) {
        (this.listeners[event] = this.listeners[event] || []).push(fn);
      },
      emit(event, payload) {
        (this.listeners[event] || []).forEach((fn) => {
          try { fn(payload); } catch (e) { console.warn('Codex bus error', e); }
        });
      }
    };
    window.Codex.bus = bus;
  } catch (e) {
    console.warn('Codex core bootstrap failed', e);
  }
})();

