(function () {
  const NS = (window.Codex = window.Codex || {});
  const C = (NS.Components = NS.Components || {});

  function $id(id) {
    try { return gradioApp().getElementById(id) || document.getElementById(id); } catch (_) { return document.getElementById(id); }
  }

  function readNumber(id) {
    const root = $id(id);
    if (!root) return null;
    const num = root.querySelector('input');
    if (num && num.value !== '') {
      const v = Number(num.value);
      return Number.isNaN(v) ? null : v;
    }
    return null;
  }

  function readDropdown(id) {
    const root = $id(id);
    const sel = root ? root.querySelector('select') : null;
    return sel ? sel.value : null;
  }

  function readCheckbox(id) {
    const root = $id(id);
    const cb = root ? root.querySelector('input[type="checkbox"]') : null;
    return !!(cb && cb.checked);
  }

  C.Hires = {
    get() {
      return {
        enable: readCheckbox('txt2img_hr_enable'),
        steps: readNumber('txt2img_hires_steps'),
        denoise: readNumber('txt2img_denoising_strength'),
        scale: readNumber('txt2img_hr_scale'),
        upscaler: readDropdown('txt2img_hr_upscaler'),
      };
    }
  };
})();

