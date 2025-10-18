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
  };
})();

