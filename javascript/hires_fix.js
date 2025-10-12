"use strict";
// @ts-check

/**
 * Toggle inactive state for hires fix controls.
 * @param {HTMLElement | null} elem
 * @param {boolean} inactive
 */
function setInactive(elem, inactive) {
    if (elem) {
        elem.classList.toggle('inactive', !!inactive);
    }
}

/**
 * Safe helper for grabbing elements by id from the Gradio app.
 * @param {string} id
 * @returns {HTMLElement | null}
 */
function getAppElementById(id) {
    const root = gradioApp();
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        const element = root.getElementById(id);
        if (element instanceof HTMLElement) return element;
    }
    const fallback = document.getElementById(id);
    return fallback instanceof HTMLElement ? fallback : null;
}

/**
 * Update hires fix resolution UI state.
 * @param {boolean} enable
 * @param {number} width
 * @param {number} height
 * @param {number} hr_scale
 * @param {number} hr_resize_x
 * @param {number} hr_resize_y
 * @returns {[boolean, number, number, number, number, number]}
 */
function onCalcResolutionHires(enable, width, height, hr_scale, hr_resize_x, hr_resize_y) {
    const hrUpscaleBy = getAppElementById('txt2img_hr_scale');
    const hrResizeX = getAppElementById('txt2img_hr_resize_x');
    const hrResizeY = getAppElementById('txt2img_hr_resize_y');
    const hiresRow = getAppElementById('txt2img_hires_fix_row2');

    if (hiresRow) {
        hiresRow.style.display = opts.use_old_hires_fix_width_height ? 'none' : '';
    }

    const disableScale = Boolean(opts.use_old_hires_fix_width_height || hr_resize_x > 0 || hr_resize_y > 0);
    const disableResizeX = Boolean(opts.use_old_hires_fix_width_height || hr_resize_x === 0);
    const disableResizeY = Boolean(opts.use_old_hires_fix_width_height || hr_resize_y === 0);

    setInactive(hrUpscaleBy, disableScale);
    setInactive(hrResizeX, disableResizeX);
    setInactive(hrResizeY, disableResizeY);

    return [enable, width, height, hr_scale, hr_resize_x, hr_resize_y];
}
