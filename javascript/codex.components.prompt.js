(function () {
  const NS = (window.Codex = window.Codex || {});
  const C = (NS.Components = NS.Components || {});

  function $id(id) {
    try { return gradioApp().getElementById(id) || document.getElementById(id); } catch (_) { return document.getElementById(id); }
  }

  function readTextbox(id) {
    const root = $id(id);
    if (!root) return '';
    const t = root.querySelector('textarea');
    return t && t.value ? String(t.value) : '';
  }

  function writeTextbox(id, text) {
    const root = $id(id);
    if (!root) return false;
    const t = root.querySelector('textarea');
    if (t) { t.value = String(text ?? ''); updateInput(t); return true; }
    return false;
  }

  C.Prompt = {
    get(tab /* 'txt2img'|'img2img'|'txt2vid'|'img2vid' */) {
      return {
        prompt: readTextbox(`${tab}_prompt`),
        negative: readTextbox(`${tab}_neg_prompt`),
      };
    },
    set(tab, values) {
      const ok1 = writeTextbox(`${tab}_prompt`, values?.prompt ?? '');
      const ok2 = writeTextbox(`${tab}_neg_prompt`, values?.negative ?? '');
      return ok1 || ok2;
    },
  };

  function _bindKey(el, tab) {
    if (!el || el.dataset.codexBound === '1') return;
    el.dataset.codexBound = '1';
    el.addEventListener('keydown', function (ev) {
      try {
        if (ev.key === 'Enter' && (ev.ctrlKey || ev.metaKey)) {
          const btn = gradioApp().querySelector(`#${tab}_generate button, #${tab}_generate`) || document.getElementById(`${tab}_generate`);
          if (btn && btn.click) { ev.preventDefault(); btn.click(); return; }
        }
        if (ev.key === 'Enter' && ev.altKey) {
          const btn = gradioApp().querySelector(`#${tab}_skip`) || document.getElementById(`${tab}_skip`);
          if (btn && btn.click) { ev.preventDefault(); btn.click(); return; }
        }
        if (ev.key === 'Escape') {
          const btn = gradioApp().querySelector(`#${tab}_interrupt`) || document.getElementById(`${tab}_interrupt`);
          if (btn && btn.click) { ev.preventDefault(); btn.click(); return; }
        }
      } catch (e) {
        console.warn('Codex.Prompt keybind error', e);
      }
    }, { capture: true });
  }

  function _scanAndBind() {
    ['txt2img','img2img','txt2vid','img2vid'].forEach((tab) => {
      const pos = $id(`${tab}_prompt`);
      const neg = $id(`${tab}_neg_prompt`);
      const posTa = pos && pos.querySelector('textarea');
      const negTa = neg && neg.querySelector('textarea');
      if (posTa) _bindKey(posTa, tab);
      if (negTa) _bindKey(negTa, tab);
    });
  }

  try { if (typeof window.onUiLoaded === 'function') window.onUiLoaded(_scanAndBind); } catch (_) {}
})();
