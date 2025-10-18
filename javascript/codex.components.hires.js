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

  function _r() { return (window.Codex && window.Codex.Components && window.Codex.Components.Readers) || null; }

  C.Hires = {
    /**
     * Return a normalized hires object for given tab ('txt2img' only for now).
     */
    get(tab) {
      const R = _r();
      const isT2I = (tab === 'txt2img' || !tab);
      if (!isT2I) return { enable: false };
      const enable = R ? R.readCheckbox('txt2img_hr_enable') : readCheckbox('txt2img_hr_enable');
      if (!enable) return { enable: false };
      const steps = R ? R.readInt('txt2img_hires_steps') : readNumber('txt2img_hires_steps');
      const denoise = R ? R.readFloat('txt2img_denoising_strength') : readNumber('txt2img_denoising_strength');
      const scale = R ? R.readFloat('txt2img_hr_scale') : readNumber('txt2img_hr_scale');
      const upscaler = R ? R.readDropdownValue('txt2img_hr_upscaler') : readDropdown('txt2img_hr_upscaler');
      const resize_x = R ? R.readInt('txt2img_hr_resize_x') : readNumber('txt2img_hr_resize_x');
      const resize_y = R ? R.readInt('txt2img_hr_resize_y') : readNumber('txt2img_hr_resize_y');
      const hr_checkpoint = R ? R.readDropdownValue('hr_checkpoint') : readDropdown('hr_checkpoint');
      const hr_vae_te = R ? (R.readDropdownValue('hr_vae_te') || []) : (readDropdown('hr_vae_te') || []);
      const hr_sampler = R ? R.readDropdownValue('hr_sampler') : readDropdown('hr_sampler');
      const hr_scheduler = R ? R.readDropdownValue('hr_scheduler') : readDropdown('hr_scheduler');
      const hr_prompt = R ? R.readText('hires_prompt') : '';
      const hr_negative_prompt = R ? R.readText('hires_neg_prompt') : '';
      const hr_cfg = R ? R.readFloat('txt2img_hr_cfg') : readNumber('txt2img_hr_cfg');
      const hr_distilled_cfg = R ? R.readFloat('txt2img_hr_distilled_cfg') : readNumber('txt2img_hr_distilled_cfg');
      return {
        enable, steps, denoise, scale, upscaler, resize_x, resize_y,
        hr_checkpoint, hr_vae_te, hr_sampler, hr_scheduler,
        hr_prompt, hr_negative_prompt, hr_cfg, hr_distilled_cfg,
      };
    }
  };
})();
