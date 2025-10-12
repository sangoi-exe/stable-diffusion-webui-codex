"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Apply i18n titles/placeholders using backend-provided dictionary.
 - Behaviour: Traverses Gradio tree, translates eligible nodes, supports RTL via media queries.
 - Safety: Guards for node kinds, component resolution via elem_id or fallback.
*/

/** @typedef {{ [key: string]: string | boolean | undefined; rtl?: boolean }} LocalizationDict */
/** @typedef {{ id: number; props?: { elem_id?: string; webui_tooltip?: boolean; placeholder?: string } }} GradioComponent */
/** @typedef {Window & { localization?: LocalizationDict; gradio_config?: { components?: GradioComponent[] } }} LocalizationWindow */

/** @type {LocalizationWindow} */
const localizationWindow = window;

/** @type {Record<string, string>} */
const ignoreIdsForLocalization = {
    setting_sd_hypernetwork: 'OPTION',
    setting_sd_model_checkpoint: 'OPTION',
    modelmerger_primary_model_name: 'OPTION',
    modelmerger_secondary_model_name: 'OPTION',
    modelmerger_tertiary_model_name: 'OPTION',
    train_embedding: 'OPTION',
    train_hypernetwork: 'OPTION',
    txt2img_styles: 'OPTION',
    img2img_styles: 'OPTION',
    setting_random_artist_categories: 'OPTION',
    setting_face_restoration_model: 'OPTION',
    setting_realesrgan_enabled_models: 'OPTION',
    extras_upscaler_1: 'OPTION',
    extras_upscaler_2: 'OPTION',
};

const reNum = /^[.\d]+$/;
const reEmoji = /[\p{Extended_Pictographic}\u{1F3FB}-\u{1F3FF}\u{1F9B0}-\u{1F9B3}]/u;

/** @type {Record<string, number>} */
const originalLines = {};
/** @type {Record<string, number>} */
const translatedLines = {};

function hasLocalization() {
    const dict = localizationWindow.localization;
    return !!(dict && Object.keys(dict).length > 0);
}

/**
 * @param {Element} el
 * @returns {Text[]}
 */
function textNodesUnder(el) {
    const nodes = [];
    const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT, null);
    let current = walker.nextNode();
    while (current) {
        nodes.push(/** @type {Text} */ (current));
        current = walker.nextNode();
    }
    return nodes;
}

/**
 * @param {Node} node
 * @param {string} text
 */
function canBeTranslated(node, text) {
    if (!text) return false;
    if (!(node.parentElement instanceof HTMLElement)) return false;

    const parentType = node.parentElement.nodeName;
    if (parentType === 'SCRIPT' || parentType === 'STYLE' || parentType === 'TEXTAREA') return false;

    if (parentType === 'OPTION' || parentType === 'SPAN') {
        let level = 0;
        let pnode = node.parentElement instanceof HTMLElement ? node.parentElement : null;
        while (pnode instanceof HTMLElement && level < 4) {
            if (ignoreIdsForLocalization[pnode.id] === parentType) return false;
            pnode = pnode.parentElement;
            level += 1;
        }
    }

    if (reNum.test(text)) return false;
    if (reEmoji.test(text)) return false;
    return true;
}

/**
 * @param {string} text
 * @returns {string | undefined}
 */
function getTranslation(text) {
    if (!text) return undefined;

    if (translatedLines[text] === undefined) {
        originalLines[text] = 1;
    }

    const dict = localizationWindow.localization;
    const translated = dict ? dict[text] : undefined;
    if (typeof translated === 'string') {
        translatedLines[translated] = 1;
        return translated;
    }

    return undefined;
}

/**
 * @param {Text} node
 */
function processTextNode(node) {
    const text = node.textContent?.trim() ?? '';
    if (!canBeTranslated(node, text)) return;

    const translated = getTranslation(text);
    if (translated !== undefined) {
        node.textContent = translated;
    }
}

/**
 * @param {Element | Node} node
 */
function processNode(node) {
    if (node instanceof Text) {
        processTextNode(node);
        return;
    }

    if (node instanceof Document || node instanceof ShadowRoot) {
        node.childNodes.forEach((child) => processNode(child));
        return;
    }

    if (!(node instanceof HTMLElement)) return;

    if (node.title) {
        const tl = getTranslation(node.title);
        if (tl !== undefined) node.title = tl;
    }

    if (node instanceof HTMLInputElement || node instanceof HTMLTextAreaElement) {
        const tl = getTranslation(node.placeholder);
        if (tl !== undefined) node.placeholder = tl;
    }

    textNodesUnder(node).forEach(processTextNode);
}

function localizeWholePage() {
    const root = gradioApp();
    processNode(root);

    const components = /** @type {GradioComponent[]} */ (localizationWindow.gradio_config?.components ?? []);
    /**
     * @param {GradioComponent} comp
     * @returns {HTMLElement | null}
     */
    const resolveComponentElement = (comp) => {
        const elemId = comp.props?.elem_id ? String(comp.props.elem_id) : `component-${comp.id}`;
        const appRoot = gradioApp();
        if ('getElementById' in appRoot && typeof appRoot.getElementById === 'function') {
            const el = appRoot.getElementById(elemId);
            if (el instanceof HTMLElement) return el;
        }
        const fallback = document.getElementById(elemId);
        return fallback instanceof HTMLElement ? fallback : null;
    };

    for (const comp of components) {
        const element = resolveComponentElement(comp);
        if (!element) continue;

        if (comp.props?.webui_tooltip) {
            const tl = getTranslation(element.title);
            if (tl !== undefined) element.title = tl;
        }
        if (comp.props?.placeholder) {
            const textbox = element.querySelector('input[placeholder], textarea[placeholder]');
            if (textbox instanceof HTMLInputElement || textbox instanceof HTMLTextAreaElement) {
                const tl = getTranslation(textbox.placeholder);
                if (tl !== undefined) textbox.placeholder = tl;
            }
        }
    }
}

function dumpTranslations() {
    if (!hasLocalization()) {
        localizeWholePage();
    }

    /** @type {Record<string, string | boolean>} */
    const dumped = {};
    const dict = localizationWindow.localization ?? {};

    if (dict.rtl) {
        dumped.rtl = true;
    }

    for (const text of Object.keys(originalLines)) {
        if (dumped[text] !== undefined) continue;
        const value = dict[text];
        dumped[text] = typeof value === 'string' ? value : text;
    }

    return dumped;
}

function download_localization() {
    const text = JSON.stringify(dumpTranslations(), null, 4);

    const element = document.createElement('a');
    element.href = `data:text/plain;charset=utf-8,${encodeURIComponent(text)}`;
    element.download = 'localization.json';
    element.style.display = 'none';
    document.body.appendChild(element);

    element.click();

    document.body.removeChild(element);
}

/**
 * @param {MutationRecord[] | MutationRecord | null | undefined} mutations
 */
function handleLocalizationMutations(mutations) {
    const records = Array.isArray(mutations) ? mutations : mutations ? [mutations] : [];
    for (const record of records) {
        if (!record) continue;
        record.addedNodes.forEach((node) => {
            processNode(node);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    if (!hasLocalization()) return;

    onUiUpdate(handleLocalizationMutations);
    localizeWholePage();

    if (localizationWindow.localization?.rtl) {
        const observer = new MutationObserver((mutations, obs) => {
            for (const mutation of mutations) {
                mutation.addedNodes.forEach((node) => {
                    if (node instanceof HTMLStyleElement && node.sheet) {
                        obs.disconnect();
                        const rules = node.sheet.cssRules;
                        for (const rule of Array.from(rules)) {
                            if (rule instanceof CSSMediaRule && Array.from(rule.media).includes('rtl')) {
                                rule.media.appendMedium('all');
                            }
                        }
                    }
                });
            }
        });
        observer.observe(document.head, { childList: true });
    }
});
