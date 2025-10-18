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
})();

