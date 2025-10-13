/**
 * @typedef {(mutations: MutationRecord[]) => void} UiMutationHandler
 * @typedef {() => void} UiHandler
 */

/**
 * Resolve the Gradio root element, supporting both classic DOM and shadow roots.
 * @returns {Document | ShadowRoot | HTMLElement}
 */
function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    if (elems.length === 0) {
        return document;
    }

    const elem = /** @type {HTMLElement} */ (elems[0]);
    const patched = /** @type {HTMLElement & { getElementById: typeof document.getElementById }} */ (elem);
    patched.getElementById = function(id) {
        return document.getElementById(id);
    };
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

/**
 * Get the currently selected top-level UI tab button (e.g. the button that says "Extras").
 * @returns {HTMLElement | null}
 */
function get_uiCurrentTab() {
    return /** @type {HTMLElement | null} */ (gradioApp().querySelector('#tabs > .tab-nav > button.selected'));
}

/**
 * Get the first currently visible top-level UI tab content (e.g. the div hosting the "txt2img" UI).
 * @returns {HTMLElement | null}
 */
function get_uiCurrentTabContent() {
    return /** @type {HTMLElement | null} */ (gradioApp().querySelector('#tabs > .tabitem[id^=tab_]:not([style*="display: none"])'));
}

/** @type {UiMutationHandler[]} */
const uiUpdateCallbacks = [];
/** @type {UiHandler[]} */
const uiAfterUpdateCallbacks = [];
/** @type {UiHandler[]} */
const uiLoadedCallbacks = [];
/** @type {UiHandler[]} */
const uiTabChangeCallbacks = [];
/** @type {UiHandler[]} */
const optionsChangedCallbacks = [];
/** @type {UiHandler[]} */
const optionsAvailableCallbacks = [];
/** @type {ReturnType<typeof setTimeout> | null} */
let uiAfterUpdateTimeout = null;
/** @type {HTMLElement | null} */
let uiCurrentTab = null;

/**
 * Register callback to be called at each UI update.
 * The callback receives an array of MutationRecords as an argument.
 */
/**
 * @param {unknown} callback
 */
function onUiUpdate(callback) {
    if (typeof callback !== 'function') {
        console.warn('onUiUpdate: ignored non-function callback', callback);
        return;
    }
    uiUpdateCallbacks.push(/** @type {UiMutationHandler} */ (callback));
}

/**
 * Register callback to be called soon after UI updates.
 * The callback receives no arguments.
 *
 * This is preferred over `onUiUpdate` if you don't need
 * access to the MutationRecords, as your function will
 * not be called quite as often.
 */
/**
 * @param {unknown} callback
 */
function onAfterUiUpdate(callback) {
    if (typeof callback !== 'function') {
        console.warn('onAfterUiUpdate: ignored non-function callback', callback);
        return;
    }
    uiAfterUpdateCallbacks.push(/** @type {UiHandler} */ (callback));
}

/**
 * Register callback to be called when the UI is loaded.
 * The callback receives no arguments.
 */
/**
 * @param {unknown} callback
 */
function onUiLoaded(callback) {
    if (typeof callback !== 'function') {
        console.warn('onUiLoaded: ignored non-function callback', callback);
        return;
    }
    uiLoadedCallbacks.push(/** @type {UiHandler} */ (callback));
}

/**
 * Register callback to be called when the UI tab is changed.
 * The callback receives no arguments.
 */
/**
 * @param {unknown} callback
 */
function onUiTabChange(callback) {
    if (typeof callback !== 'function') {
        console.warn('onUiTabChange: ignored non-function callback', callback);
        return;
    }
    uiTabChangeCallbacks.push(/** @type {UiHandler} */ (callback));
}

/**
 * Register callback to be called when the options are changed.
 * The callback receives no arguments.
 * @param callback
 */
/**
 * @param {unknown} callback
 */
function onOptionsChanged(callback) {
    if (typeof callback !== 'function') {
        console.warn('onOptionsChanged: ignored non-function callback', callback);
        return;
    }
    optionsChangedCallbacks.push(/** @type {UiHandler} */ (callback));
}

/**
 * Register callback to be called when the options (in opts global variable) are available.
 * The callback receives no arguments.
 * If you register the callback after the options are available, it's just immediately called.
 */
/**
 * @param {unknown} callback
 */
function onOptionsAvailable(callback) {
    if (typeof callback !== 'function') {
        console.warn('onOptionsAvailable: ignored non-function callback', callback);
        return;
    }
    if (Object.keys(opts).length != 0) {
        try {
            /** @type {UiHandler} */ (callback)();
        } catch (e) {
            console.error('error running callback in onOptionsAvailable:', e);
        }
        return;
    }

    optionsAvailableCallbacks.push(/** @type {UiHandler} */ (callback));
}

/**
 * @param {Function[]} queue
 * @param {unknown} arg
 * @param {string} context
 */
function executeCallbacks(queue, arg, context) {
    if (!Array.isArray(queue) || queue.length === 0) return;
    for (const cb of queue) {
        if (typeof cb !== 'function') {
            console.warn('ignored non-function callback in', context || 'callbacks', cb);
            continue;
        }
        try {
            cb(arg);
        } catch (e) {
            console.error('error running callback in', context || 'callbacks', ':', e);
        }
    }
}

/**
 * Schedule the execution of the callbacks registered with onAfterUiUpdate.
 * The callbacks are executed after a short while, unless another call to this function
 * is made before that time. IOW, the callbacks are executed only once, even
 * when there are multiple mutations observed.
 */
function scheduleAfterUiUpdateCallbacks() {
    if (uiAfterUpdateTimeout !== null) {
        clearTimeout(uiAfterUpdateTimeout);
    }
    uiAfterUpdateTimeout = window.setTimeout(function() {
        executeCallbacks(uiAfterUpdateCallbacks, undefined, 'onAfterUiUpdate');
    }, 200);
}

var executedOnLoaded = false;

document.addEventListener("DOMContentLoaded", function() {
    /** @param {MutationRecord[]} mutations */
    var mutationObserver = new MutationObserver(function(mutations) {
        if (!executedOnLoaded && gradioApp().querySelector('#txt2img_prompt')) {
            executedOnLoaded = true;
            executeCallbacks(uiLoadedCallbacks, undefined, 'onUiLoaded');
        }

        executeCallbacks(uiUpdateCallbacks, mutations, 'onUiUpdate');
        scheduleAfterUiUpdateCallbacks();
        const newTab = get_uiCurrentTab();
        if (newTab && (newTab !== uiCurrentTab)) {
            uiCurrentTab = newTab;
            executeCallbacks(uiTabChangeCallbacks, undefined, 'onUiTabChange');
        }
    });
    mutationObserver.observe(gradioApp(), {childList: true, subtree: true});
});

// Normalize outgoing Gradio payloads so numeric strings become numbers.
// This prevents server-side Slider/Number preprocess errors like
// TypeError: '<' not supported between instances of 'str' and 'int'.
(() => {
    const origFetch = window.fetch;
    // Expose helper for WS wrapper
    /** @type {any} */(window).buildNamedPayload = buildNamedPayload;
    /** @param {unknown} x */
    function coerceNumeric(x) {
        if (typeof x === 'string') {
            const t = x.trim();
            if (/^-?\d+$/.test(t)) {
                const n = Number.parseInt(t, 10);
                if (!Number.isNaN(n)) return n;
            } else if (/^-?\d*\.\d+$/.test(t)) {
                const f = Number.parseFloat(t);
                if (!Number.isNaN(f)) return f;
            }
        }
        return x;
    }

    // Best-effort: enrich request with a named view using gradio_config when available
    /** @param {any} obj */
    function buildNamedPayload(obj) {
        try {
            const fnIndex = typeof obj.fn_index === 'number' ? obj.fn_index : null;
            const data = Array.isArray(obj.data) ? obj.data : null;
            const cfg = /** @type {any} */ (window).gradio_config || null;
            if (!fnIndex || !data || !cfg || !Array.isArray(cfg.components)) return null;
            /** @type {Record<number,string>} */
            const idToElem = {};
            for (const c of cfg.components) {
                const id = c && typeof c.id === 'number' ? c.id : null;
                const elem = c && c.props && typeof c.props.elem_id === 'string' ? c.props.elem_id : null;
                if (id != null && elem) idToElem[id] = elem;
            }
            const deps = /** @type {any[]} */ (cfg.dependencies || cfg.deps || []);
            /** @type {number[]|null} */
            let inputs = null;
            for (const d of deps) {
                const cand = d?.fn_index ?? d?.backend_fn ?? d?.id ?? d?.function_index ?? null;
                if (cand === fnIndex) {
                    // Try common properties that could hold input ids
                    const candidates = /** @type {any[]} */ ([d?.inputs, d?.input_ids, d?.input_component_ids]);
                    for (const arr of candidates) {
                        if (Array.isArray(arr) && arr.every((x) => typeof x === 'number')) {
                            inputs = arr;
                            break;
                        }
                    }
                    if (inputs) break;
                }
            }
            if (!inputs) return null;
            /** @type {Record<string,unknown>} */
            const named = {};
            const L = Math.min(data.length, inputs.length);
            for (let i = 0; i < L; i++) {
                const keyId = /** @type {number} */ (inputs[i] ?? -1);
                const key = idToElem[keyId] || String(keyId);
                named[key] = data[i];
            }
            return named;
        } catch (_) {
            return null;
        }
    }

    window.fetch = function(input, init) {
        try {
            const url = typeof input === 'string' ? input : (input && 'url' in input ? input.url : '');
            let method = (init && init.method) || 'GET';
            try {
                if (method === 'GET' && typeof Request !== 'undefined' && input instanceof Request) {
                    method = input.method || 'GET';
                }
            } catch (_) { /* ignore */ }
            if (method.toUpperCase() === 'POST' && init && typeof init.body === 'string') {
                if (/(\/queue|\/predict|\/internal\/queue)/.test(String(url))) {
                    try {
                        const body = JSON.parse(init.body);
                        if (Array.isArray(body?.data)) {
                            body.data = body.data.map(coerceNumeric);
                            // Attach named mapping when possible
                            const named = buildNamedPayload(body);
                            if (named) body._named = named;
                            init.body = JSON.stringify(body);
                        }
                    } catch (e) {
                        // ignore malformed JSON; fall through to original fetch
                    }
                }
            }
        } catch (e) {
            console.warn('fetch-normalize failed:', e);
        }
        return origFetch.call(this, input, init);
    };
})();

// Normalize numeric strings in WebSocket messages used by Gradio's Queue
(() => {
    const WS = window.WebSocket;
    if (!WS) return;
    const origSend = WS.prototype.send;
    /** @param {unknown} x */
    function coerce(x) {
        if (typeof x === 'string') {
            const t = x.trim();
            if (/^-?\d+$/.test(t)) {
                const n = Number.parseInt(t, 10);
                if (!Number.isNaN(n)) return n;
            } else if (/^-?\d*\.\d+$/.test(t)) {
                const f = Number.parseFloat(t);
                if (!Number.isNaN(f)) return f;
            }
        }
        return x;
    }
    /** @this {WebSocket} @param {any} data */
    WS.prototype.send = function(data) {
        try {
            if (typeof data === 'string') {
                const obj = JSON.parse(data);
                if (obj && Array.isArray(obj.data)) {
                    obj.data = obj.data.map(coerce);
                    // Attach named mapping when possible
                    try {
                        const named = (function(){
                            try { return (/** @type {any} */(window)).buildNamedPayload ? (/** @type {any} */(window)).buildNamedPayload(obj) : null; } catch(_) { return null; }
                        })();
                        if (named) obj._named = named;
                    } catch(_) {}
                    data = JSON.stringify(obj);
                }
            }
        } catch (e) {
            // ignore non-JSON frames
        }
        return origSend.call(this, data);
    };
})();

/**
 * Add keyboard shortcuts:
 * Ctrl+Enter to start/restart a generation
 * Alt/Option+Enter to skip a generation
 * Esc to interrupt a generation
 */
/** @param {KeyboardEvent} e */
document.addEventListener('keydown', function(e) {
    const isEnter = e.key === 'Enter' || e.keyCode === 13;
    const isCtrlKey = e.metaKey || e.ctrlKey;
    const isAltKey = e.altKey;
    const isEsc = e.key === 'Escape';

    const tabContent = get_uiCurrentTabContent();
    if (!tabContent) {
        return;
    }

    const generateButton = /** @type {HTMLButtonElement | null} */ (tabContent.querySelector('button[id$=_generate]'));
    const interruptButton = /** @type {HTMLButtonElement | null} */ (tabContent.querySelector('button[id$=_interrupt]'));
    const skipButton = /** @type {HTMLButtonElement | null} */ (tabContent.querySelector('button[id$=_skip]'));
    if (!generateButton || !interruptButton || !skipButton) {
        return;
    }

    if (isCtrlKey && isEnter) {
        if (interruptButton.style.display === 'block') {
            interruptButton.click();
            /**
             * @param {MutationRecord[]} mutationList
             */
            const callback = (mutationList) => {
                for (const mutation of mutationList) {
                    if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
                        if (interruptButton.style.display === 'none') {
                            generateButton.click();
                            observer.disconnect();
                        }
                    }
                }
            };
            const observer = new MutationObserver(callback);
            observer.observe(interruptButton, {attributes: true});
        } else {
            generateButton.click();
        }
        e.preventDefault();
    }

    if (isAltKey && isEnter) {
        skipButton.click();
        e.preventDefault();
    }

    if (isEsc) {
        const globalPopup = /** @type {HTMLElement | null} */ (document.querySelector('.global-popup'));
        const lightboxModal = /** @type {HTMLElement | null} */ (document.querySelector('#lightboxModal'));
        if (!globalPopup || globalPopup.style.display === 'none') {
            if (document.activeElement === lightboxModal) return;
            if (interruptButton.style.display === 'block') {
                interruptButton.click();
                e.preventDefault();
            }
        }
    }
});

/**
 * checks that a UI element is not in another hidden element or tab content
 * @param {HTMLElement | Document | null} el
 * @returns {boolean}
 */
function uiElementIsVisible(el) {
    if (!el) return false;
    if (el === document) {
        return true;
    }
    if (!(el instanceof HTMLElement)) {
        const parentNode = el.parentNode;
        let next = null;
        if (parentNode instanceof HTMLElement || parentNode instanceof Document) {
            next = parentNode;
        } else if (parentNode instanceof ShadowRoot) {
            next = parentNode.host;
        }
        return uiElementIsVisible(/** @type {HTMLElement | Document | null} */ (next));
    }

    const computedStyle = getComputedStyle(el);
    const isVisible = computedStyle.display !== 'none';

    if (!isVisible) return false;
    const parentNode = el.parentNode;
    let next = null;
    if (parentNode instanceof HTMLElement || parentNode instanceof Document) {
        next = parentNode;
    } else if (parentNode instanceof ShadowRoot) {
        next = parentNode.host;
    }
    return uiElementIsVisible(/** @type {HTMLElement | Document | null} */ (next));
}

/**
 * @param {HTMLElement} el
 * @returns {boolean}
 */
function uiElementInSight(el) {
    const clRect = el.getBoundingClientRect();
    const windowHeight = window.innerHeight;
    const isOnScreen = clRect.bottom > 0 && clRect.top < windowHeight;

    return isOnScreen;
}
