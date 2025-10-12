"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Kick off Textual Inversion training and wire the queue progress panel.
 - Safety: All ID lookups via getAppElementById with DOM fallback; progress handler guards textinfo.
 - Contract: start_training_textual_inversion(...args) → returns updated args with task id in [0].
*/

/**
 * Safe lookup for an element by id inside the Gradio root with a document fallback.
 * @param {string} id
 * @returns {HTMLElement | null}
 */
function getAppElementById(id) {
    const root = gradioApp();
    if (root && 'getElementById' in root && typeof root.getElementById === 'function') {
        const el = root.getElementById(id);
        if (el instanceof HTMLElement) return el;
    }
    const fallback = document.getElementById(id);
    return fallback instanceof HTMLElement ? fallback : null;
}

function start_training_textual_inversion() {
    const errorEl = gradioApp().querySelector('#ti_error');
    if (errorEl instanceof HTMLElement) errorEl.innerHTML = '';

    var id = randomId();
    const output = getAppElementById('ti_output');
    const gallery = getAppElementById('ti_gallery');

    /** @param {{ textinfo?: string }} progress */
    const onProgress = (progress) => {
        const tiProg = getAppElementById('ti_progress');
        if (tiProg) tiProg.innerHTML = progress.textinfo || '';
    };

    requestProgress(id, /** @type {HTMLElement} */ (output), /** @type {HTMLElement | null} */ (gallery), function() {}, onProgress);

    var res = Array.from(arguments);

    res[0] = id;

    return res;
}
