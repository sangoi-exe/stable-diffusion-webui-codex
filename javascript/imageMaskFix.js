"use strict";
// @ts-check

/**
 * temporary fix for https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/668
 * @see https://github.com/gradio-app/gradio/issues/1721
 */
function imageMaskResize() {
    const canvases = Array.from(gradioApp().querySelectorAll('#img2maskimg .touch-none canvas'))
        .filter((canvas) => canvas instanceof HTMLCanvasElement);

    const firstCanvas = canvases[0];
    if (!firstCanvas) {
        window.removeEventListener('resize', imageMaskResize);
        return;
    }

    const wrapper = firstCanvas.closest('.touch-none');
    if (!(wrapper instanceof HTMLElement)) return;

    const prevSibling = wrapper.previousElementSibling;
    const previewImage = prevSibling instanceof HTMLImageElement ? prevSibling : null;
    if (!previewImage) return;

    if (!previewImage.complete) {
        previewImage.addEventListener('load', imageMaskResize, { once: true });
        return;
    }

    const { width: w, height: h, naturalWidth: nw, naturalHeight: nh } = previewImage;
    if (!nw || !nh) return;

    const portrait = nh > nw;
    const projectedWidth = portrait ? (h / nh) * nw : w;
    const projectedHeight = portrait ? h : (w / nw) * nh;

    const wW = Math.min(w, projectedWidth);
    const wH = Math.min(h, projectedHeight);

    wrapper.style.width = `${wW}px`;
    wrapper.style.height = `${wH}px`;
    wrapper.style.left = '0px';
    wrapper.style.top = '0px';

    canvases.forEach((canvas) => {
        canvas.style.width = '';
        canvas.style.height = '';
        canvas.style.maxWidth = '100%';
        canvas.style.maxHeight = '100%';
        canvas.style.objectFit = 'contain';
    });
}

onAfterUiUpdate(imageMaskResize);
window.addEventListener('resize', imageMaskResize);
