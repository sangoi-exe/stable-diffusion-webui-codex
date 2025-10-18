(function () {
  const NS = (window.Codex = window.Codex || {});
  const C = (NS.Components = NS.Components || {});

  function $root() {
    try { return gradioApp(); } catch (_) { return document; }
  }
  function $id(id) {
    try { return $root().getElementById(id) || document.getElementById(id); } catch (_) { return document.getElementById(id); }
  }

  function readText(id) {
    const el = $id(id);
    const ta = el && el.querySelector('textarea');
    return (ta && typeof ta.value === 'string') ? ta.value : '';
  }
  function readDropdownValue(id) {
    const el = $id(id);
    const sel = el && el.querySelector('select');
    return (sel && sel.value) ? sel.value : null;
  }
  function readDropdownOrRadioValue(id) {
    const el = $id(id);
    const sel = el && el.querySelector('select');
    if (sel && sel.value) return sel.value;
    const radios = el && el.querySelectorAll('input[type="radio"]');
    if (radios) {
      for (const r of Array.from(radios)) if (r.checked) return r.value || null;
    }
    return null;
  }
  function readInt(id) {
    const el = $id(id);
    const num = el && el.querySelector('input');
    if (!num) return 0;
    const v = Number.parseInt(String(num.value ?? '0'), 10);
    return Number.isNaN(v) ? 0 : v;
  }
  function readFloat(id) {
    const el = $id(id);
    const num = el && el.querySelector('input');
    if (!num) return 0.0;
    const v = Number.parseFloat(String(num.value ?? '0'));
    return Number.isNaN(v) ? 0.0 : v;
  }
  function readRadioValue(id) {
    const el = $id(id);
    if (!el) return null;
    const sel = el.querySelector('input[type="radio"]:checked');
    return sel ? sel.value : null;
  }
  function readRadioIndex(id) {
    const el = $id(id);
    if (!el) return 0;
    const radios = el.querySelectorAll('input[type="radio"]');
    let idx = 0;
    for (let i = 0; i < radios.length; i += 1) if (radios[i].checked) { idx = i; break; }
    return idx;
  }
  function readSeedValue(id) {
    const el = $id(id);
    const num = el && el.querySelector('input');
    if (!num) return -1;
    const v = Number.parseInt(String(num.value ?? '-1'), 10);
    return Number.isNaN(v) ? -1 : v;
  }

  C.Readers = { $id, readText, readDropdownValue, readDropdownOrRadioValue, readInt, readFloat, readRadioValue, readRadioIndex, readSeedValue };
})();

