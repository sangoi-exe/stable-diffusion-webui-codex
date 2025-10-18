(function () {
  const NS = (window.Codex = window.Codex || {});
  const C = (NS.Components = NS.Components || {});

  function $id(id) {
    try { return gradioApp().getElementById(id) || document.getElementById(id); } catch (_) { return document.getElementById(id); }
  }

  C.Canvas = {
    /**
     * @param {string} id e.g. 'img2img_image' or 'img2vid_image'
     * @returns {HTMLCanvasElement | HTMLImageElement | null}
     */
    getSurface(id) {
      const root = $id(id);
      if (!root) return null;
      const img = root.querySelector('img');
      if (img) return img;
      const canvas = root.querySelector('canvas');
      if (canvas) return canvas;
      return null;
    },
    /**
     * Return normalized inpainting settings for img2img when an inpaint tab is active.
     */
    getInpaint() {
      const R = (window.Codex && window.Codex.Components && window.Codex.Components.Readers) || null;
      const rInt = R ? R.readInt : (id) => { const el = $id(id); const n = el && el.querySelector('input'); const v = Number.parseInt(String(n?.value ?? '0'), 10); return Number.isNaN(v) ? 0 : v; };
      const rFloat = R ? R.readFloat : (id) => { const el = $id(id); const n = el && el.querySelector('input'); const v = Number.parseFloat(String(n?.value ?? '0')); return Number.isNaN(v) ? 0.0 : v; };
      const rRadioIdx = R ? R.readRadioIndex : (id) => 0;
      return {
        mask_blur: rInt('img2img_mask_blur'),
        mask_alpha: rFloat('img2img_mask_alpha'),
        inpainting_fill: rRadioIdx('img2img_inpainting_fill'),
        inpaint_full_res: (rRadioIdx('img2img_inpaint_full_res') === 1),
        inpaint_full_res_padding: rInt('img2img_inpaint_full_res_padding'),
        inpainting_mask_invert: rRadioIdx('img2img_mask_mode'),
      };
    }
  };
})();
