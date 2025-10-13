"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: UI glue for submit flows, gallery helpers, and small UX toggles.
 - Safety: Use getAppElementById for all ID lookups; never mutate submit arg order.
 - Helpers: submitWithProgress, normalizeSubmitArgs, updateInput, restoreProgress.
 - Events: listeners annotated (Keyboard/Mouse), guarded against nulls.
*/

/**
 * Various client-side helpers used by the Gradio UI build in `ui.py`.
 */

/** @typedef {Window & {
 *   submit?: (...args: unknown[]) => unknown[];
 *   submit_txt2img_upscale?: (...args: unknown[]) => unknown[];
 *   restoreProgressTxt2img?: () => string | null;
 *   restoreProgressImg2img?: () => string | null;
 *   args_to_array?: typeof Array.from;
 * }} UIWindow */

/** @type {UIWindow} */
const uiWindow = window;

/**
 * Shortcut to the Gradio app root.
 * @returns {Document | ShadowRoot | HTMLElement}
 */
function gradioRoot() {
    return gradioApp();
}

/**
 * Retrieve an element by id, looking inside the Gradio root first with a DOM fallback.
 * @param {string} id
 * @returns {HTMLElement | null}
 */
function getAppElementById(id) {
    const root = gradioRoot();
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        const el = root.getElementById(id);
        if (el instanceof HTMLElement) return el;
    }
    const fallback = document.getElementById(id);
    return fallback instanceof HTMLElement ? fallback : null;
}

/**
 * @returns {HTMLElement[]}
 */
function all_gallery_buttons() {
    const buttons = gradioRoot().querySelectorAll('[style="display: block;"] .thumbnail-item.thumbnail-small');
    /** @type {HTMLElement[]} */
    const visible = [];
    buttons.forEach((elem) => {
        if (elem instanceof HTMLElement && elem.parentElement?.offsetParent) {
            visible.push(elem);
        }
    });
    return visible;
}

/**
 * @returns {HTMLElement | null}
 */
function selected_gallery_button() {
    return all_gallery_buttons().find((elem) => elem.classList.contains('selected')) ?? null;
}

/**
 * @returns {number}
 */
function selected_gallery_index() {
    return all_gallery_buttons().findIndex((elem) => elem.classList.contains('selected'));
}

/**
 * @param {string} galleryContainer
 * @returns {NodeListOf<HTMLElement>}
 */
function gallery_container_buttons(galleryContainer) {
    return gradioRoot().querySelectorAll(`#${galleryContainer} .thumbnail-item.thumbnail-small`);
}

/**
 * @param {string} galleryContainer
 * @returns {number}
 */
function selected_gallery_index_id(galleryContainer) {
    return Array.from(gallery_container_buttons(galleryContainer)).findIndex((elem) => elem.classList.contains('selected'));
}

/**
 * @template T
 * @param {T[]} gallery
 * @returns {[T | null][]}
 */
function extract_image_from_gallery(gallery) {
    if (gallery.length === 0) {
        return [[null]];
    }
    let index = selected_gallery_index();
    if (index < 0 || index >= gallery.length) {
        index = 0;
    }
    return [[gallery[index] ?? null]];
}

uiWindow.args_to_array = Array.from;

/** @param {string} theme */
function set_theme(theme) {
    const gradioURL = window.location.href;
    if (!gradioURL.includes('?__theme=')) {
        window.location.replace(`${gradioURL}?__theme=${theme}`);
    }
}

function switch_to_txt2img() {
    const tabs = gradioRoot().querySelector('#tabs');
    const buttons = tabs ? tabs.querySelectorAll('button') : null;
    buttons?.[0]?.click();
    return Array.from(arguments);
}

/** @param {number} index */
function switch_to_img2img_tab(index) {
    const tabs = gradioRoot().querySelector('#tabs');
    const buttons = tabs ? tabs.querySelectorAll('button') : null;
    buttons?.[1]?.click();
    const mode = getAppElementById('mode_img2img');
    if (mode) {
        const modeButtons = mode.querySelectorAll('button');
        const button = modeButtons?.[index];
        if (button instanceof HTMLElement) button.click();
    }
}

function switch_to_img2img() {
    switch_to_img2img_tab(0);
    return Array.from(arguments);
}

function switch_to_sketch() {
    switch_to_img2img_tab(1);
    return Array.from(arguments);
}

function switch_to_inpaint() {
    switch_to_img2img_tab(2);
    return Array.from(arguments);
}

function switch_to_inpaint_sketch() {
    switch_to_img2img_tab(3);
    return Array.from(arguments);
}

function switch_to_extras() {
    const tabs = gradioRoot().querySelector('#tabs');
    const buttons = tabs ? tabs.querySelectorAll('button') : null;
    buttons?.[3]?.click();
    return Array.from(arguments);
}

/**
 * @param {string} tabId
 * @returns {number}
 */
function get_tab_index(tabId) {
    const tab = getAppElementById(tabId);
    const buttons = tab ? tab.querySelector('div')?.querySelectorAll('button') : null;
    if (!buttons) return 0;
    const arr = Array.from(buttons);
    for (let i = 0; i < arr.length; i += 1) {
        const btn = arr[i];
        if (btn instanceof HTMLElement && btn.classList.contains('selected')) return i;
    }
    return 0;
}

/**
 * @param {string} tabId
 * @param {IArguments} args
 */
function create_tab_index_args(tabId, args) {
    const res = Array.from(args);
    res[0] = get_tab_index(tabId);
    return res;
}

function get_img2img_tab_index() {
    const res = Array.from(arguments);
    res.splice(-2);
    res[0] = get_tab_index('mode_img2img');
    return res;
}

/**
 * @param {IArguments | unknown[]} args
 * @returns {unknown[]}
 */
function create_submit_args(args) {
    return Array.from(/** @type {any} */ (args));
}

/**
 * @param {string} tabname
 * @param {unknown[]} res
 * @returns {unknown[]}
 */
function normalizeSubmitArgs(tabname, res) {
    try {
        for (let i = 0; i < res.length; i += 1) {
            const value = res[i];
            if (typeof value === 'string') {
                const trimmed = value.trim();
                if (/^-?\d+$/.test(trimmed)) {
                    const numeric = Number.parseInt(trimmed, 10);
                    if (!Number.isNaN(numeric)) res[i] = numeric;
                } else if (/^-?\d*\.\d+$/.test(trimmed)) {
                    const floatValue = Number.parseFloat(trimmed);
                    if (!Number.isNaN(floatValue)) res[i] = floatValue;
                }
            }
        }
    } catch (error) {
        console.warn('normalizeSubmitArgs failed:', error);
    }
    return res;
}

/**
 * @param {string} tabname
 * @param {boolean} showInterrupt
 * @param {boolean} showSkip
 * @param {boolean} showInterrupting
 */
function setSubmitButtonsVisibility(tabname, showInterrupt, showSkip, showInterrupting) {
    const interrupt = getAppElementById(`${tabname}_interrupt`);
    const skip = getAppElementById(`${tabname}_skip`);
    const interrupting = getAppElementById(`${tabname}_interrupting`);
    if (interrupt) interrupt.style.display = showInterrupt ? 'block' : 'none';
    if (skip) skip.style.display = showSkip ? 'block' : 'none';
    if (interrupting) interrupting.style.display = showInterrupting ? 'block' : 'none';
}

/** @param {string} tabname @param {boolean} show */
function showSubmitButtons(tabname, show) {
    setSubmitButtonsVisibility(tabname, !show, !show, false);
}

/** @param {string} tabname */
function showSubmitInterruptingPlaceholder(tabname) {
    setSubmitButtonsVisibility(tabname, false, true, true);
}

/**
 * @param {string} tabname
 * @param {boolean} show
 */
function showRestoreProgressButton(tabname, show) {
    const button = getAppElementById(`${tabname}_restore_progress`);
    if (button) button.style.setProperty('display', show ? 'flex' : 'none', 'important');
}

/**
 * @param {IArguments} args
 * @param {string} galleryContainerId
 * @param {string} galleryId
 * @returns {unknown[]}
 */
function submitWithProgress(args, galleryContainerId, galleryId) {
    const tabname = /** @type {string} */ (galleryContainerId.split('_', 1)[0]);
    showSubmitButtons(tabname, false);

    const id = randomId();
    localSet(`${tabname}_task_id`, id);

    const container = getAppElementById(galleryContainerId);
    const gallery = getAppElementById(galleryId);

    requestProgress(
        id,
        /** @type {HTMLElement} */ (container),
        gallery,
        () => {
            showSubmitButtons(tabname, true);
            localRemove(`${tabname}_task_id`);
            showRestoreProgressButton(tabname, false);
        }
    );

    const res = normalizeSubmitArgs(tabname, create_submit_args(args));
    res[0] = id;
    return res;
}

function submit() {
    return submitWithProgress(arguments, 'txt2img_gallery_container', 'txt2img_gallery');
}

// ---- Strict JSON builders (DOM-based) ----
/** @param {string} id */
function readText(id) {
    const root = getAppElementById(id);
    if (!root) return '';
    const ta = root.querySelector('textarea');
    if (ta && ta instanceof HTMLTextAreaElement) return ta.value ?? '';
    const inp = root.querySelector('input[type=text]');
    if (inp && inp instanceof HTMLInputElement) return inp.value ?? '';
    return '';
}
/** @param {string} id */
function readNumber(id) {
    const root = getAppElementById(id);
    if (!root) return 0;
    const num = root.querySelector('input[type=number]');
    if (num && num instanceof HTMLInputElement) return Number(num.value || 0);
    const rng = root.querySelector('input[type=range]');
    if (rng && rng instanceof HTMLInputElement) return Number(rng.value || 0);
    return 0;
}
/** @param {string} id */
function readFloat(id) { return Number(readNumber(id)); }
/** @param {string} id */
function readInt(id) { return Math.trunc(Number(readNumber(id))); }
/** @param {string} id */
function readCheckbox(id) {
    const root = getAppElementById(id);
    if (!root) return false;
    const cb = root.querySelector('input[type=checkbox]');
    return !!(cb && cb instanceof HTMLInputElement && cb.checked);
}
/** @param {string} id */
function readDropdownValue(id) {
    const root = getAppElementById(id);
    if (!root) return '';
    const sel = root.querySelector('select');
    if (!sel || !(sel instanceof HTMLSelectElement)) return '';
    if (sel.multiple) {
        return Array.from(sel.selectedOptions).map(o => o.value);
    }
    return sel.value;
}
/** @param {string} id */
function readRadioIndex(id) {
    const root = getAppElementById(id);
    if (!root) return 0;
    const buttons = root.querySelectorAll('button');
    let idx = 0;
    buttons.forEach((btn, i) => {
        if (btn instanceof HTMLElement && btn.classList.contains('selected')) idx = i;
    });
    return idx;
}
/** @param {IArguments} _args */
function buildNamedTxt2img(_args) {
    /** @type {Record<string, unknown>} */
    const named = { __strict_version: 1, __source: 'txt2img' };
    /** @type {string[]} */
    const active = [];
    // Core
    named['txt2img_prompt'] = readText('txt2img_prompt'); active.push('txt2img_prompt');
    named['txt2img_neg_prompt'] = readText('txt2img_neg_prompt'); active.push('txt2img_neg_prompt');
    named['txt2img_styles'] = readDropdownValue('txt2img_styles') || []; active.push('txt2img_styles');
    named['txt2img_batch_count'] = readInt('txt2img_batch_count'); active.push('txt2img_batch_count');
    named['txt2img_batch_size'] = readInt('txt2img_batch_size'); active.push('txt2img_batch_size');
    named['txt2img_cfg_scale'] = readFloat('txt2img_cfg_scale'); active.push('txt2img_cfg_scale');
    named['txt2img_distilled_cfg_scale'] = readFloat('txt2img_distilled_cfg_scale'); active.push('txt2img_distilled_cfg_scale');
    named['txt2img_height'] = readInt('txt2img_height'); active.push('txt2img_height');
    named['txt2img_width'] = readInt('txt2img_width'); active.push('txt2img_width');
    const hrEnabled = readCheckbox('txt2img_hr_enable');
    named['txt2img_hr_enable'] = hrEnabled; active.push('txt2img_hr_enable');
    if (hrEnabled) {
        named['txt2img_denoising_strength'] = readFloat('txt2img_denoising_strength'); active.push('txt2img_denoising_strength');
        named['txt2img_hr_scale'] = readFloat('txt2img_hr_scale'); active.push('txt2img_hr_scale');
        named['txt2img_hr_upscaler'] = readDropdownValue('txt2img_hr_upscaler'); active.push('txt2img_hr_upscaler');
        named['txt2img_hires_steps'] = readInt('txt2img_hires_steps'); active.push('txt2img_hires_steps');
        named['txt2img_hr_resize_x'] = readInt('txt2img_hr_resize_x'); active.push('txt2img_hr_resize_x');
        named['txt2img_hr_resize_y'] = readInt('txt2img_hr_resize_y'); active.push('txt2img_hr_resize_y');
        named['hr_checkpoint'] = readDropdownValue('hr_checkpoint'); active.push('hr_checkpoint');
        named['hr_vae_te'] = readDropdownValue('hr_vae_te') || []; active.push('hr_vae_te');
        named['hr_sampler'] = readDropdownValue('hr_sampler'); active.push('hr_sampler');
        named['hr_scheduler'] = readDropdownValue('hr_scheduler'); active.push('hr_scheduler');
        named['txt2img_hr_prompt'] = readText('hires_prompt'); active.push('txt2img_hr_prompt');
        named['txt2img_hr_neg_prompt'] = readText('hires_neg_prompt'); active.push('txt2img_hr_neg_prompt');
        named['txt2img_hr_cfg'] = readFloat('txt2img_hr_cfg'); active.push('txt2img_hr_cfg');
        named['txt2img_hr_distilled_cfg'] = readFloat('txt2img_hr_distilled_cfg'); active.push('txt2img_hr_distilled_cfg');
    }
    named['__active_ids'] = active;
    return named;
}

function submit_named() {
    const res = submitWithProgress(arguments, 'txt2img_gallery_container', 'txt2img_gallery');
    // Put named override into the last argument slot (hidden JSON)
    if (Array.isArray(res) && res.length > 0) {
        res[res.length - 1] = buildNamedTxt2img(arguments);
    }
    return res;
}

function submit_txt2img_upscale() {
    const res = submit(...arguments);
    res[2] = selected_gallery_index();
    return res;
}

function submit_img2img() {
    return submitWithProgress(arguments, 'img2img_gallery_container', 'img2img_gallery');
}

/** @param {IArguments} _args */
function buildNamedImg2img(_args) {
    /** @type {Record<string, unknown>} */
    const named = { __strict_version: 1, __source: 'img2img' };
    /** @type {string[]} */
    const active = [];
    named['img2img_prompt'] = readText('img2img_prompt'); active.push('img2img_prompt');
    named['img2img_neg_prompt'] = readText('img2img_neg_prompt'); active.push('img2img_neg_prompt');
    named['img2img_styles'] = readDropdownValue('img2img_styles') || []; active.push('img2img_styles');
    named['img2img_batch_count'] = readInt('img2img_batch_count'); active.push('img2img_batch_count');
    named['img2img_batch_size'] = readInt('img2img_batch_size'); active.push('img2img_batch_size');
    named['img2img_cfg_scale'] = readFloat('img2img_cfg_scale'); active.push('img2img_cfg_scale');
    named['img2img_distilled_cfg_scale'] = readFloat('img2img_distilled_cfg_scale'); active.push('img2img_distilled_cfg_scale');
    named['img2img_image_cfg_scale'] = readFloat('img2img_image_cfg_scale'); active.push('img2img_image_cfg_scale');
    named['img2img_denoising_strength'] = readFloat('img2img_denoising_strength'); active.push('img2img_denoising_strength');
    named['img2img_selected_scale_tab'] = get_tab_index('img2img_tabs_resize'); active.push('img2img_selected_scale_tab');
    named['img2img_height'] = readInt('img2img_height'); active.push('img2img_height');
    named['img2img_width'] = readInt('img2img_width'); active.push('img2img_width');
    named['img2img_scale_by'] = readFloat('img2img_scale'); active.push('img2img_scale_by');
    named['img2img_resize_mode'] = readRadioIndex('resize_mode'); active.push('img2img_resize_mode');
    const tab = get_tab_index('mode_img2img');
    if ([2,3,4].includes(tab)) {
        named['img2img_mask_blur'] = readInt('img2img_mask_blur'); active.push('img2img_mask_blur');
        named['img2img_mask_alpha'] = readFloat('img2img_mask_alpha'); active.push('img2img_mask_alpha');
        named['img2img_inpainting_fill'] = readRadioIndex('img2img_inpainting_fill'); active.push('img2img_inpainting_fill');
        named['img2img_inpaint_full_res'] = readRadioIndex('img2img_inpaint_full_res') === 1; active.push('img2img_inpaint_full_res');
        named['img2img_inpaint_full_res_padding'] = readInt('img2img_inpaint_full_res_padding'); active.push('img2img_inpaint_full_res_padding');
        named['img2img_inpainting_mask_invert'] = readRadioIndex('img2img_mask_mode'); active.push('img2img_inpainting_mask_invert');
    }
    named['__active_ids'] = active;
    return named;
}

function submit_img2img_named() {
    const res = submitWithProgress(arguments, 'img2img_gallery_container', 'img2img_gallery');
    if (Array.isArray(res) && res.length > 0) {
        res[res.length - 1] = buildNamedImg2img(arguments);
    }
    return res;
}

function submit_extras() {
    return submitWithProgress(arguments, 'extras_gallery_container', 'extras_gallery');
}

/** @param {string} tabname */
function restoreProgress(tabname) {
    showRestoreProgressButton(tabname, false);
    const id = localGet(`${tabname}_task_id`);
    if (typeof id === 'string') {
        showSubmitInterruptingPlaceholder(tabname);
        requestProgress(
            id,
            /** @type {HTMLElement} */ (getAppElementById(`${tabname}_gallery_container`)),
            getAppElementById(`${tabname}_gallery`),
            () => showSubmitButtons(tabname, true),
            undefined,
            0
        );
        return id;
    }
    return null;
}

function restoreProgressTxt2img() {
    return restoreProgress('txt2img');
}

function restoreProgressImg2img() {
    return restoreProgress('img2img');
}

uiWindow.submit = submit;
uiWindow.submit_txt2img_upscale = submit_txt2img_upscale;
uiWindow.restoreProgressTxt2img = restoreProgressTxt2img;
uiWindow.restoreProgressImg2img = restoreProgressImg2img;

/**
 * Configure the width and height elements on `tabname` to accept pasting of resolutions.
 * @param {string} tabname
 */
function setupResolutionPasting(tabname) {
    const width = gradioRoot().querySelector(`#${tabname}_width input[type=number]`);
    const height = gradioRoot().querySelector(`#${tabname}_height input[type=number]`);
    [width, height].forEach((el) => {
        if (!(el instanceof HTMLInputElement)) return;
        el.addEventListener('paste', (event) => {
            const text = event.clipboardData?.getData('text/plain') ?? '';
            const match = text.match(/^\s*(\d+)\D+(\d+)\s*$/);
            if (match && match[1] && match[2] && width instanceof HTMLInputElement && height instanceof HTMLInputElement) {
                width.value = match[1];
                height.value = match[2];
                updateInput(width);
                updateInput(height);
                event.preventDefault();
            }
        });
    });
}

onUiLoaded(() => {
    showRestoreProgressButton('txt2img', Boolean(restoreProgressTxt2img()));
    showRestoreProgressButton('img2img', Boolean(restoreProgressImg2img()));
    setupResolutionPasting('txt2img');
    setupResolutionPasting('img2img');
});

function modelmerger() {
    const id = randomId();
    requestProgress(id, /** @type {HTMLElement} */ (getAppElementById('modelmerger_results_panel')), null, () => {});
    const res = create_submit_args(arguments);
    res[0] = id;
    return res;
}

/** @param {unknown} _ @param {string} promptText @param {string} negativePromptText */
function ask_for_style_name(_, promptText, negativePromptText) {
    const name = prompt('Style name:') ?? '';
    return [name, promptText, negativePromptText];
}

/** @param {string} promptValue @param {string} negativePromptValue */
function confirm_clear_prompt(promptValue, negativePromptValue) {
    if (confirm('Delete prompt?')) {
        promptValue = '';
        negativePromptValue = '';
    }
    return [promptValue, negativePromptValue];
}

/** @type {StableDiffusionOptions} */
var opts = /** @type {any} */ ({});

onAfterUiUpdate(() => {
    if (Object.keys(opts).length !== 0) return;

    const jsonElem = getAppElementById('settings_json');
    if (!jsonElem) return;

    const textarea = jsonElem.querySelector('textarea');
    if (!(textarea instanceof HTMLTextAreaElement)) return;

    try {
        opts = JSON.parse(textarea.value);
    } catch (error) {
        console.error('Failed to parse settings_json:', error);
        return;
    }

    executeCallbacks(optionsAvailableCallbacks, undefined, 'onOptionsAvailable');
    executeCallbacks(optionsChangedCallbacks, undefined, 'onOptionsChanged');

    const valueDescriptor = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value');
    if (valueDescriptor) {
        const originalGet = valueDescriptor.get?.bind(textarea);
        const originalSet = valueDescriptor.set?.bind(textarea);
        Object.defineProperty(textarea, 'value', {
            set(newValue) {
                const oldValue = typeof originalGet === 'function' ? originalGet() : textarea.value;
                if (typeof originalSet === 'function') originalSet(newValue);
                if (oldValue !== newValue) {
                    try {
                        opts = JSON.parse(textarea.value);
                    } catch (error) {
                        console.error('Failed to parse updated settings_json:', error);
                    }
                }
                executeCallbacks(optionsChangedCallbacks, undefined, 'onOptionsChanged');
            },
            get() {
                return typeof originalGet === 'function' ? originalGet() : textarea.value;
            }
        });
    }

    jsonElem.parentElement?.style && (jsonElem.parentElement.style.display = 'none');
});

onOptionsChanged(() => {
    const elem = getAppElementById('sd_checkpoint_hash');
    const hash = typeof opts.sd_checkpoint_hash === 'string' ? opts.sd_checkpoint_hash : '';
    const shortHash = hash.substring(0, 10);
    if (elem && elem.textContent !== shortHash) {
        elem.textContent = shortHash;
        elem.title = hash;
        elem.setAttribute('href', `https://google.com/search?q=${hash}`);
    }
});

/** @type {HTMLTextAreaElement | null} */ let txt2img_textarea = null;
/** @type {HTMLTextAreaElement | null} */ let img2img_textarea = null;

function restart_reload() {
    document.body.style.backgroundColor = 'var(--background-fill-primary)';
    document.body.innerHTML = '<h1 style="font-family:monospace;margin-top:20%;color:lightgray;text-align:center;">Reloading...</h1>';
    const requestPing = () => {
        requestGet('./internal/ping', {}, () => {
            location.reload();
        }, () => {
            setTimeout(requestPing, 500);
        });
    };
    setTimeout(requestPing, 2000);
    return [];
}

/**
 * Trigger an input event for Gradio textboxes after programmatic edits.
 * @param {HTMLInputElement | HTMLTextAreaElement} target
 */
function updateInput(target) {
    const event = new Event('input', { bubbles: true });
    target.dispatchEvent(event);
}

/** @type {string | null} */
let desiredCheckpointName = null;
/** @param {string} name */
function selectCheckpoint(name) {
    desiredCheckpointName = name;
    getAppElementById('change_checkpoint')?.click();
}

/** @type {string | number} */
let desiredVAEName = 0;
/** @param {string | number} vae */
function selectVAE(vae) {
    desiredVAEName = vae;
}

/** @param {number} w @param {number} h @param {number} r */
function currentImg2imgSourceResolution(w, h, r) {
    const img = gradioRoot().querySelector('#mode_img2img > div[style="display: block;"] img, #mode_img2img > div[style="display: block;"] canvas');
    if (img instanceof HTMLImageElement) {
        return [img.naturalWidth || img.width, img.naturalHeight || img.height, r];
    } else if (img instanceof HTMLCanvasElement) {
        return [img.width, img.height, r];
    }
    return [w, h, r];
}

function updateImg2imgResizeToTextAfterChangingImage() {
    setTimeout(() => {
        getAppElementById('img2img_update_resize_to')?.click();
    }, 500);
    return [];
}

/** @param {string} elemId */
function setRandomSeed(elemId) {
    const input = gradioRoot().querySelector(`#${elemId} input`);
    if (!(input instanceof HTMLInputElement)) return [];
    input.value = '-1';
    updateInput(input);
    return [];
}

/** @param {string} tabname */
function switchWidthHeight(tabname) {
    const width = gradioRoot().querySelector(`#${tabname}_width input[type=number]`);
    const height = gradioRoot().querySelector(`#${tabname}_height input[type=number]`);
    if (!(width instanceof HTMLInputElement) || !(height instanceof HTMLInputElement)) return [];
    const tmp = width.value;
    width.value = height.value;
    height.value = tmp;
    updateInput(width);
    updateInput(height);
    return [];
}

/** @type {Record<string, number>} */
const onEditTimers = {};

/**
 * Register a throttled input handler.
 * @param {string} editId
 * @param {HTMLInputElement | HTMLTextAreaElement | null} elem
 * @param {number} afterMs
 * @param {() => void} func
 * @returns {() => void}
 */
function onEdit(editId, elem, afterMs, func) {
    if (!elem) {
        return () => {};
    }
    const edited = () => {
        const existingTimer = onEditTimers[editId];
        if (existingTimer) window.clearTimeout(existingTimer);
        onEditTimers[editId] = window.setTimeout(func, afterMs);
    };
    elem.addEventListener('input', edited);
    return edited;
}

// expose globals for legacy hooks
Object.assign(uiWindow, {
    set_theme,
    all_gallery_buttons,
    selected_gallery_button,
    selected_gallery_index,
    gallery_container_buttons,
    selected_gallery_index_id,
    extract_image_from_gallery,
    switch_to_txt2img,
    switch_to_img2img,
    switch_to_sketch,
    switch_to_inpaint,
    switch_to_inpaint_sketch,
    switch_to_extras,
    get_tab_index,
    create_tab_index_args,
    get_img2img_tab_index,
    create_submit_args,
    normalizeSubmitArgs,
    setSubmitButtonsVisibility,
    showSubmitButtons,
    showSubmitInterruptingPlaceholder,
    showRestoreProgressButton,
    submit_extras,
    modelmerger,
    ask_for_style_name,
    confirm_clear_prompt,
    updateInput,
    selectCheckpoint,
    selectVAE,
    currentImg2imgSourceResolution,
    updateImg2imgResizeToTextAfterChangingImage,
    setRandomSeed,
    switchWidthHeight,
    onEdit,
});
