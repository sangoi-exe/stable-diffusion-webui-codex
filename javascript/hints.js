"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Tooltips automáticos conforme elementos são inseridos/alterados.
 - Safety: Batching com debounce; busca por classes/dataset/label traduzidos.
*/

// mouseover tooltips for various UI elements

/** @typedef {Window & { localization?: Record<string, string>; __forgeHintsWarned?: boolean; gradio_config?: unknown }} HintsWindow */

/** @type {HintsWindow} */
const hintsWindow = window;

/** @type {Record<string, string>} */
const titles = {
    "Sampling steps": "How many times to improve the generated image iteratively; higher values take longer; very low values can produce bad results",
    "Sampling method": "Which algorithm to use to produce the image",
    GFPGAN: "Restore low quality faces using GFPGAN neural network",
    "Euler a": "Euler Ancestral - very creative, each can get a completely different picture depending on step count, setting steps higher than 30-40 does not help",
    DDIM: "Denoising Diffusion Implicit Models - best at inpainting",
    UniPC: "Unified Predictor-Corrector Framework for Fast Sampling of Diffusion Models",
    "DPM adaptive": "Ignores step count - uses a number of steps determined by the CFG and resolution",

    "\u{1F4D0}": "Auto detect size from img2img",
    "Batch count": "How many batches of images to create (has no impact on generation performance or VRAM usage)",
    "Batch size": "How many image to create in a single batch (increases generation performance at cost of higher VRAM usage)",
    "CFG Scale": "Classifier Free Guidance Scale - how strongly the image should conform to prompt - lower values produce more creative results",
    Seed: "A value that determines the output of random number generator - if you create an image with same parameters and seed as another image, you'll get the same result",
    "\u{1f3b2}\ufe0f": "Set seed to -1, which will cause a new random number to be used every time",
    "\u267b\ufe0f": "Reuse seed from last generation, mostly useful if it was randomized",
    "\u2199\ufe0f": "Read generation parameters from prompt or last generation if prompt is empty into user interface.",
    "\u{1f4c2}": "Open images output directory",
    "\u{1f4be}": "Save style",
    "\u{1f5d1}\ufe0f": "Clear prompt",
    "\u{1f4cb}": "Apply selected styles to current prompt",
    "\u{1f4d2}": "Paste available values into the field",
    "\u{1f3b4}": "Show/hide extra networks",
    "\u{1f300}": "Restore progress",

    "Inpaint a part of image": "Draw a mask over an image, and the script will regenerate the masked area with content according to prompt",
    "SD upscale": "Upscale image normally, split result into tiles, improve each tile using img2img, merge whole image back",

    "Just resize": "Resize image to target resolution. Unless height and width match, you will get incorrect aspect ratio.",
    "Crop and resize": "Resize the image so that entirety of target resolution is filled with the image. Crop parts that stick out.",
    "Resize and fill": "Resize the image so that entirety of image is inside target resolution. Fill empty space with image's colors.",

    "Mask blur": "How much to blur the mask before processing, in pixels.",
    "Masked content": "What to put inside the masked area before processing it with Stable Diffusion.",
    fill: "fill it with colors of the image",
    original: "keep whatever was there originally",
    "latent noise": "fill it with latent space noise",
    "latent nothing": "fill it with latent space zeroes",
    "Inpaint at full resolution": "Upscale masked region to target resolution, do inpainting, downscale back and paste into original image",

    "Denoising strength": "Determines how little respect the algorithm should have for image's content. At 0, nothing will change, and at 1 you'll get an unrelated image. With values below 1.0, processing will take less steps than the Sampling Steps slider specifies.",

    Skip: "Stop processing current image and continue processing.",
    Interrupt: "Stop processing images and return any results accumulated so far.",
    Save: "Write image to a directory (default - log/images) and generation parameters into csv file.",

    "X values": "Separate values for X axis using commas.",
    "Y values": "Separate values for Y axis using commas.",

    None: "Do not do anything special",
    "Prompt matrix": "Separate prompts into parts using vertical pipe character (|) and the script will create a picture for every combination of them (except for the first part, which will be present in all combinations)",
    "X/Y/Z plot": "Create grid(s) where images will have different parameters. Use inputs below to specify which parameters will be shared by columns and rows",
    "Custom code": "Run Python code. Advanced user only. Must run program with --allow-code for this to work",

    "Prompt S/R": "Separate a list of words with commas, and the first word will be used as a keyword: script will search for this word in the prompt, and replace it with others",
    "Prompt order": "Separate a list of words with commas, and the script will make a variation of prompt with those words for their every possible order",

    Tiling: "Produce an image that can be tiled.",
    "Tile overlap": "For SD upscale, how much overlap in pixels should there be between tiles. Tiles overlap so that when they are merged back into one picture, there is no clearly visible seam.",

    "Variation seed": "Seed of a different picture to be mixed into the generation.",
    "Variation strength": "How strong should the influence of the variation seed be - at 0, the seed will have no effect, and at 1, you'll just see the image associated with the seed.",
    "Variation seed strength": "Deprecated alias for Variation strength.",

    "Randomize": "Randomize the values",
    "Add style": "Add style to prompt",
    "Refresh": "Refresh the list",
};

/**
 * @param {HTMLElement} element
 */
function updateTooltip(element) {
    const localization = typeof hintsWindow.localization === 'object' && hintsWindow.localization
        ? hintsWindow.localization
        : /** @type {Record<string, string>} */ ({});

    let tooltip = element.title || '';
    if (!tooltip) {
        const label = (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement)
            ? element.placeholder || ''
            : element.innerText || '';
        if (label && titles[label]) {
            tooltip = localization[titles[label]] || titles[label];
        }
    }

    if (!tooltip && element.dataset) {
        const dataValue = element.dataset.value;
        if (dataValue && titles[dataValue]) {
            tooltip = localization[titles[dataValue]] || titles[dataValue];
        }
    }

    if (!tooltip) {
        for (const cls of Array.from(element.classList)) {
            if (titles[cls]) {
                tooltip = localization[titles[cls]] || titles[cls];
                break;
            }
        }
    }

    if (tooltip) {
        element.title = tooltip;
    }
}

/** @type {Set<HTMLElement>} */
const tooltipCheckNodes = new Set();
/** @type {ReturnType<typeof setTimeout> | null} */
let tooltipCheckTimer = null;

function processTooltipCheckNodes() {
    tooltipCheckNodes.forEach((node) => updateTooltip(node));
    tooltipCheckNodes.clear();
}

/**
 * @param {HTMLElement} element
 */
function enqueueTooltipCheck(element) {
    tooltipCheckNodes.add(element);
}

onUiUpdate((/** @type {MutationRecord[] | MutationRecord | null | undefined} */ mutationRecords) => {
    const records = Array.isArray(mutationRecords)
        ? mutationRecords
        : mutationRecords
            ? [mutationRecords]
            : [];

    for (const record of records) {
        if (!record || !(record.target instanceof Element)) continue;

        if (record.type === 'childList' && record.target.classList.contains('options')) {
            const wrap = record.target.parentElement;
            const input = wrap instanceof HTMLElement ? wrap.querySelector('input') : null;
            if (input instanceof HTMLElement) {
                input.title = '';
                enqueueTooltipCheck(input);
            }
        }

        if (record.addedNodes && record.addedNodes.length) {
            record.addedNodes.forEach((/** @type {Node} */ node) => {
                if (!(node instanceof HTMLElement) || node.classList.contains('hide')) return;
                if (!node.title) {
                    const tag = node.tagName;
                    if (tag === 'SPAN' || tag === 'BUTTON' || tag === 'P' || tag === 'INPUT' || (tag === 'LI' && node.classList.contains('item'))) {
                        enqueueTooltipCheck(node);
                    }
                }
                node.querySelectorAll('span, button, p').forEach((child) => {
                    if (child instanceof HTMLElement) enqueueTooltipCheck(child);
                });
            });
        }
    }

    if (tooltipCheckNodes.size) {
        if (tooltipCheckTimer !== null) clearTimeout(tooltipCheckTimer);
        tooltipCheckTimer = window.setTimeout(processTooltipCheckNodes, 1000);
    }
});

/**
 * Warn once if localization is missing.
 */
function ensureLocalization() {
    if (hintsWindow.__forgeHintsWarned) return;
    hintsWindow.__forgeHintsWarned = true;
    if (!hintsWindow.localization) {
        console.warn('Localization dictionary missing, tooltips will use defaults.');
    }
}

onUiLoaded(() => {
    ensureLocalization();
    processTooltipCheckNodes();
});

if (hintsWindow.gradio_config) {
    const root = gradioApp();
    let wrap = null;
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        wrap = root.getElementById('txt2img_settings');
    }
    if (!(wrap instanceof HTMLElement)) {
        wrap = document.getElementById('txt2img_settings');
    }
    if (wrap instanceof HTMLElement) enqueueTooltipCheck(wrap);
}
