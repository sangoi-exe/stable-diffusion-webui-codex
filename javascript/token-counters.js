"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Atualiza contadores de tokens dos prompts (txt2img/img2img).
 - Safety: onEdit com debounce; resolve IDs sob Gradio root; evita rebind duplo.
*/

/** @type {Record<string, () => void>} */
let promptTokenCountUpdateFunctions = {};

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

/** @param {...unknown} args */
function update_txt2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("txt2img_token_button");
    update_token_counter("txt2img_negative_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

/** @param {...unknown} args */
function update_img2img_tokens(...args) {
    // Called from Gradio
    update_token_counter("img2img_token_button");
    update_token_counter("img2img_negative_token_button");
    if (args.length == 2) {
        return args[0];
    }
    return args;
}

/** @param {string} button_id */
function update_token_counter(button_id) {
    promptTokenCountUpdateFunctions[button_id]?.();
}


/** @param {string} name */
function recalculatePromptTokens(name) {
    promptTokenCountUpdateFunctions[name]?.();
}

function recalculate_prompts_txt2img() {
    // Called from Gradio
    recalculatePromptTokens('txt2img_prompt');
    recalculatePromptTokens('txt2img_neg_prompt');
    return Array.from(arguments);
}

function recalculate_prompts_img2img() {
    // Called from Gradio
    recalculatePromptTokens('img2img_prompt');
    recalculatePromptTokens('img2img_neg_prompt');
    return Array.from(arguments);
}

/** @param {string} id @param {string} id_counter @param {string} id_button */
function setupTokenCounting(id, id_counter, id_button) {
    const prompt = getAppElementById(id);
    const counter = getAppElementById(id_counter);
    const textarea = gradioApp().querySelector(`#${id} > label > textarea`);

    if (!prompt || !counter || !textarea) {
        return; // UI not ready yet in Gradio 5; skip safely
    }

    if (prompt.dataset.tokenCounterBound === 'true') {
        return;
    }
    prompt.dataset.tokenCounterBound = 'true';

    if (counter.parentElement && prompt.parentElement && counter.parentElement == prompt.parentElement) {
        return;
    }

    if (prompt.parentElement) {
        prompt.parentElement.insertBefore(counter, prompt);
        prompt.parentElement.style.position = "relative";
    }

    const func = onEdit(id, /** @type {HTMLTextAreaElement} */ (textarea), 800, function() {
        if (counter.classList.contains("token-counter-visible")) {
            getAppElementById(id_button)?.click();
        }
    });
    promptTokenCountUpdateFunctions[id] = func;
    promptTokenCountUpdateFunctions[id_button] = func;
}

/** @param {string} id @param {string} id_counter @param {string} id_button */
function toggleTokenCountingVisibility(id, id_counter, id_button) {
    void id; // not used but kept for signature stability
    const counter = getAppElementById(id_counter);
    if (!counter) {
        return;
    }

    counter.style.display = opts.disable_token_counters ? "none" : "block";
    counter.classList.toggle("token-counter-visible", !opts.disable_token_counters);
}

/**
 * @param {(id: string, id_counter: string, id_button: string) => void} fun
 */
function runCodeForTokenCounters(fun) {
    fun('txt2img_prompt', 'txt2img_token_counter', 'txt2img_token_button');
    fun('txt2img_neg_prompt', 'txt2img_negative_token_counter', 'txt2img_negative_token_button');
    fun('img2img_prompt', 'img2img_token_counter', 'img2img_token_button');
    fun('img2img_neg_prompt', 'img2img_negative_token_counter', 'img2img_negative_token_button');
}

onUiLoaded(function() {
    runCodeForTokenCounters(setupTokenCounting);
});

onUiUpdate(function() {
    runCodeForTokenCounters(setupTokenCounting);
});

onOptionsChanged(function() {
    runCodeForTokenCounters(toggleTokenCountingVisibility);
});
