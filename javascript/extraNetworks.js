"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Filtro/ordem/pesquisa de cartões (LoRA/Checkpoints/TI/…); popups de metadata.
 - Safety: JSDoc + guards em todos os seletores; GET/JSON robusto; inserções idempotentes.
*/

/**
 * Shared helpers and state for the Extra Networks front-end.
 * The original script relied on implicit globals and loose DOM typing; this
 * rewrite keeps behaviour intact while making every callsite pass strict
 * TypeScript (via `allowJs` + `checkJs`).
 */

/** @typedef {Document | ShadowRoot | HTMLElement} AppRoot */

/**
 * Record of tab -> callback used when the page requests a filter refresh.
 * @typedef {Record<string, (force?: boolean) => void>} CallbackMap
 */

/**
 * Metadata editor widgets keyed by tab/page identifier.
 * @typedef {{
 *   page: HTMLElement | null;
 *   nameTextarea: HTMLTextAreaElement | HTMLInputElement | null;
 *   button: HTMLElement | null;
 * }} ExtraPageMetadataEditor
 */

/** @type {CallbackMap} */
const extraNetworksApplyFilter = {};
/** @type {CallbackMap} */
const extraNetworksApplySort = {};
/** @type {Record<string, HTMLTextAreaElement | null>} */
const activePromptTextarea = {};
/** @type {Record<string, ExtraPageMetadataEditor>} */
const extraPageUserMetadataEditors = {};

let globalPopup = /** @type {HTMLElement | null} */ (null);
let globalPopupInner = /** @type {HTMLElement | null} */ (null);
/** @type {Record<string, HTMLElement | null>} */
const storedPopupIds = {};

/** @type {Array<() => void>} */
const uiAfterScriptsCallbacks = [];
let uiAfterScriptsTimeout = /** @type {ReturnType<typeof setTimeout> | null} */ (null);
let executedAfterScripts = false;

const re_extranet = /<([^:^>]+:[^:]+):[\d.]+>(.*)/;
const re_extranet_g = /<([^:^>]+:[^:]+):[\d.]+>/g;

const re_extranet_neg = /\(([^:^>]+:[\d.]+)\)/;
const re_extranet_g_neg = /\(([^:^>]+:[\d.]+)\)/g;

/**
 * Returns the active Gradio root element.
 * @returns {AppRoot}
 */
function appRoot() {
    return gradioApp();
}

/**
 * Query a selector and ensure the element is an HTMLElement.
 * @param {AppRoot} root
 * @param {string} selector
 * @returns {HTMLElement | null}
 */
function queryHTMLElement(root, selector) {
    const element = root.querySelector(selector);
    return element instanceof HTMLElement ? element : null;
}

/**
 * Query a selector and ensure the element exposes a text `value` property.
 * @param {AppRoot} root
 * @param {string} selector
 * @returns {HTMLInputElement | HTMLTextAreaElement | null}
 */
function queryTextControl(root, selector) {
    const element = root.querySelector(selector);
    if (element instanceof HTMLInputElement || element instanceof HTMLTextAreaElement) {
        return element;
    }
    return null;
}

/**
 * Safe wrapper around `getElementById` for the Gradio root.
 * @param {string} id
 * @returns {HTMLElement | null}
 */
function getAppElementById(id) {
    const root = appRoot();
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        const result = root.getElementById(id);
        return result instanceof HTMLElement ? result : null;
    }
    const fallback = document.getElementById(id);
    return fallback instanceof HTMLElement ? fallback : null;
}

/**
 * Ensure the argument is an HTMLElement before using DOM APIs.
 * @template {Element} T
 * @param {Element | null} element
 * @param {(node: Element) => node is T} predicate
 * @returns {T | null}
 */
function guardElement(element, predicate) {
    if (element && predicate(element)) {
        return element;
    }
    return null;
}

/**
 * Toggle a CSS snippet under a stable key.
 * @param {string} key
 * @param {string} css
 * @param {boolean} enable
 */
function toggleCss(key, css, enable) {
    let style = document.getElementById(key);
    if (enable && !(style instanceof HTMLStyleElement)) {
        const newStyle = document.createElement('style');
        newStyle.id = key;
        newStyle.type = 'text/css';
        document.head.appendChild(newStyle);
        style = newStyle;
    }

    if (!enable && style) {
        document.head.removeChild(style);
        return;
    }

    if (style instanceof HTMLStyleElement) {
        style.innerHTML = '';
        style.appendChild(document.createTextNode(css));
    }
}

/**
 * Associate the visible prompt textareas with the requested tab.
 * @param {string} tabname
 * @param {string} id
 */
function registerPrompt(tabname, id) {
    const root = appRoot();
    const textarea = root.querySelector(`#${id} > label > textarea`);
    if (!(textarea instanceof HTMLTextAreaElement)) return;

    if (!activePromptTextarea[tabname]) {
        activePromptTextarea[tabname] = textarea;
    }

    textarea.addEventListener('focus', () => {
        activePromptTextarea[tabname] = textarea;
    });
}

/**
 * Attach filtering/sorting wiring for a single Extra Networks tab.
 * @param {string} tabname
 */
function setupExtraNetworksForTab(tabname) {
    const root = appRoot();

    const tabnav = queryHTMLElement(root, `#${tabname}_extra_tabs > div.tab-nav`);
    if (!tabnav) return;

    const controlsDiv = document.createElement('div');
    controlsDiv.classList.add('extra-networks-controls-div');
    tabnav.appendChild(controlsDiv);

    const tabContainer = queryHTMLElement(root, `#${tabname}_extra_tabs`);
    if (!tabContainer) return;

    const tabChildren = tabContainer.querySelectorAll(`:scope > [id^='${tabname}_']`);
    tabChildren.forEach((child) => {
        if (!(child instanceof HTMLElement)) return;

        const tabnameFull = child.id;
        const searchControl = queryTextControl(root, `#${tabnameFull}_extra_search`);
        const sortDirButton = queryHTMLElement(root, `#${tabnameFull}_extra_sort_dir`);
        const refreshButton = queryHTMLElement(root, `#${tabnameFull}_extra_refresh`);

        if (!searchControl || !sortDirButton || !refreshButton) {
            return;
        }

        let currentSortSignature = '';

        /**
         * Apply the filter predicate (search tokens + UI preset).
         * @param {boolean} [force]
         */
        const applyFilter = (force = false) => {
            const searchTerm = searchControl.value.toLowerCase();
            let uiPreset = 3; // default = all

            const radioUi = queryHTMLElement(root, '#forge_ui_preset');
            if (radioUi) {
                const radioInputs = Array.from(radioUi.getElementsByTagName('input'));
                for (let i = 0; i < radioInputs.length; i += 1) {
                    const radioInput = radioInputs[i];
                    if (radioInput && radioInput.checked) {
                        uiPreset = i;
                        break;
                    }
                }
            }

            const tokens = searchTerm.length > 0 ? searchTerm.split(/\s+/).filter(Boolean) : [];
            const isLoraFilterDisabled = Boolean(opts.lora_filter_disabled);

            const cards = root.querySelectorAll(`#${tabname}_extra_tabs div.card`);
            cards.forEach((cardNode) => {
                if (!(cardNode instanceof HTMLElement)) return;

                const searchOnly = cardNode.querySelector('.search_only');
                const textContent = Array.prototype.map
                    .call(cardNode.querySelectorAll('.search_terms, .description'), (t) => t.textContent.toLowerCase())
                    .join(' ');

                let visible = true;
                if (searchOnly && searchTerm.length < 4) {
                    visible = false;
                }

                for (const token of tokens) {
                    if (!textContent.includes(token)) {
                        visible = false;
                        break;
                    }
                }

                const sdVersion = cardNode.getAttribute('data-sort-sdversion');
                if (sdVersion !== null && sdVersion !== 'SdVersion.Unknown' && !isLoraFilterDisabled) {
                    if (uiPreset === 0 && sdVersion !== 'SdVersion.SD1' && sdVersion !== 'SdVersion.SD2') {
                        visible = false;
                    } else if (uiPreset === 1 && sdVersion !== 'SdVersion.SDXL') {
                        visible = false;
                    } else if (uiPreset === 2 && sdVersion !== 'SdVersion.Flux') {
                        visible = false;
                    }
                }

                cardNode.classList.toggle('hidden', !visible);
            });

            applySort(force);
        };

        /**
         * Sort cards according to the active sort control.
         * @param {boolean} [force]
         */
        const applySort = (force = false) => {
            const cards = Array.from(root.querySelectorAll(`#${tabnameFull} div.card`)).filter((card) => card instanceof HTMLElement);
            const parent = queryHTMLElement(root, `#${tabnameFull}_cards`);
            if (!parent) return;

            const sortDir = sortDirButton.dataset.sortdir || 'Ascending';
            const reverse = sortDir === 'Descending';

            const activeSortControl = queryHTMLElement(
                root,
                `#${tabnameFull}_controls .extra-network-control--sort.extra-network-control--enabled`
            );
            const sortKey = activeSortControl?.dataset.sortkey || 'default';
            const sortKeyDataField = `sort${sortKey.charAt(0).toUpperCase()}${sortKey.slice(1)}`;
            const sortSignature = `${sortKey}-${sortDir}-${cards.length}`;

            if (!force && sortSignature === currentSortSignature) {
                return;
            }
            currentSortSignature = sortSignature;

            cards.sort((cardA, cardB) => {
                const a = cardA.dataset[sortKeyDataField] ?? '';
                const b = cardB.dataset[sortKeyDataField] ?? '';

                const aNum = Number(a);
                const bNum = Number(b);
                if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) {
                    return aNum - bNum;
                }
                return a.localeCompare(b);
            });

            if (reverse) {
                cards.reverse();
            }

            parent.innerHTML = '';
            const fragment = document.createDocumentFragment();
            for (const card of cards) {
                fragment.appendChild(card);
            }
            parent.appendChild(fragment);
        };

        searchControl.addEventListener('input', () => applyFilter());
        applySort(true);
        applyFilter(true);

        extraNetworksApplySort[tabnameFull] = (force) => applySort(Boolean(force));
        extraNetworksApplyFilter[tabnameFull] = (force) => applyFilter(Boolean(force));

        const controls = queryHTMLElement(root, `#${tabnameFull}_controls`);
        if (controls) {
            controlsDiv.appendChild(controls);
        }

        if (child.style.display !== 'none') {
            extraNetworksShowControlsForPage(tabname, tabnameFull);
        }
    });

    registerPrompt(tabname, `${tabname}_prompt`);
    registerPrompt(tabname, `${tabname}_neg_prompt`);
}

/**
 * Setup all tabs once the DOM finishes rendering.
 */
function setupExtraNetworks() {
    setupExtraNetworksForTab('txt2img');
    setupExtraNetworksForTab('img2img');
}

/**
 * Move prompts in/out of the extra networks panel.
 * @param {string} tabname
 * @param {string} id
 * @param {boolean} showPrompt
 * @param {boolean} showNegativePrompt
 */
function extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt) {
    const root = appRoot();
    if (!root.querySelector('.toprow-compact-tools')) return;

    const promptContainer = getAppElementById(`${tabname}_prompt_container`);
    const promptRow = getAppElementById(`${tabname}_prompt_row`);
    const negPromptRow = getAppElementById(`${tabname}_neg_prompt_row`);
    if (!promptContainer || !promptRow || !negPromptRow) return;

    const target = id ? getAppElementById(id) : null;

    if (showNegativePrompt && target) {
        target.insertBefore(negPromptRow, target.firstChild);
    } else {
        promptContainer.insertBefore(negPromptRow, promptContainer.firstChild);
    }

    if (showPrompt && target) {
        target.insertBefore(promptRow, target.firstChild);
    } else {
        promptContainer.insertBefore(promptRow, promptContainer.firstChild);
    }

    if (target instanceof HTMLElement) {
        target.classList.toggle('extra-page-prompts-active', showNegativePrompt || showPrompt);
    }
}

/**
 * Show or hide the inline controls for a given tab page.
 * @param {string} tabname
 * @param {string | null} tabnameFull
 */
function extraNetworksShowControlsForPage(tabname, tabnameFull) {
    const root = appRoot();
    const controls = root.querySelectorAll(`#${tabname}_extra_tabs .extra-networks-controls-div > div`);
    controls.forEach((control) => {
        if (!(control instanceof HTMLElement)) return;
        const targetId = tabnameFull ? `${tabnameFull}_controls` : '';
        control.style.display = control.id === targetId ? '' : 'none';
    });
}

/**
 * Reset controls when an unrelated tab is selected.
 * @param {string} tabname
 */
function extraNetworksUnrelatedTabSelected(tabname) {
    extraNetworksMovePromptToTab(tabname, '', false, false);
    extraNetworksShowControlsForPage(tabname, null);
}

/**
 * Called when the user activates an extra network tab.
 * @param {string} tabname
 * @param {string} id
 * @param {boolean} showPrompt
 * @param {boolean} showNegativePrompt
 * @param {string} tabnameFull
 */
function extraNetworksTabSelected(tabname, id, showPrompt, showNegativePrompt, tabnameFull) {
    extraNetworksMovePromptToTab(tabname, id, showPrompt, showNegativePrompt);
    extraNetworksShowControlsForPage(tabname, tabnameFull);
}

/**
 * Force the filter callback for the requested tab.
 * @param {string} tabnameFull
 */
function applyExtraNetworkFilter(tabnameFull) {
    setTimeout(() => {
        const applyFn = extraNetworksApplyFilter[tabnameFull];
        if (applyFn) applyFn(true);
    }, 1);
}

/**
 * Force the sort callback for the requested tab.
 * @param {string} tabnameFull
 */
function applyExtraNetworkSort(tabnameFull) {
    setTimeout(() => {
        const applyFn = extraNetworksApplySort[tabnameFull];
        if (applyFn) applyFn(true);
    }, 1);
}

/**
 * Try to remove an extra network token from a prompt.
 * @param {HTMLTextAreaElement} textarea
 * @param {string} text
 * @param {boolean} isNeg
 * @returns {boolean}
 */
function tryToRemoveExtraNetworkFromPrompt(textarea, text, isNeg) {
    let match = text.match(isNeg ? re_extranet_neg : re_extranet);
    let replaced = false;
    let newTextareaText = textarea.value;
    const extraTextBeforeNet = String(opts.extra_networks_add_text_separator || '');

    if (match) {
        const extraTextAfterNet = match[2] || '';
        const partToSearch = match[1];
        let foundAtPosition = -1;
        const regex = isNeg ? re_extranet_g_neg : re_extranet_g;
        newTextareaText = textarea.value.replaceAll(regex, (found, _, pos) => {
            const innerMatch = found.match(isNeg ? re_extranet_neg : re_extranet);
            if (innerMatch && innerMatch[1] === partToSearch) {
                replaced = true;
                foundAtPosition = pos;
                return '';
            }
            return found;
        });

        if (foundAtPosition >= 0) {
            if (extraTextAfterNet && newTextareaText.substr(foundAtPosition, extraTextAfterNet.length) === extraTextAfterNet) {
                newTextareaText = newTextareaText.slice(0, foundAtPosition) + newTextareaText.slice(foundAtPosition + extraTextAfterNet.length);
            }
            const beforeStart = foundAtPosition - extraTextBeforeNet.length;
            if (extraTextBeforeNet && beforeStart >= 0 && newTextareaText.substr(beforeStart, extraTextBeforeNet.length) === extraTextBeforeNet) {
                newTextareaText = newTextareaText.slice(0, beforeStart) + newTextareaText.slice(beforeStart + extraTextBeforeNet.length);
            }
        }
    } else {
        const pattern = new RegExp(`((?:${extraTextBeforeNet})?${text})`, 'g');
        newTextareaText = textarea.value.replaceAll(pattern, '');
        replaced = newTextareaText !== textarea.value;
    }

    if (replaced) {
        textarea.value = newTextareaText;
        return true;
    }

    return false;
}

/**
 * Append or replace prompt text with the requested token.
 * @param {string} text
 * @param {HTMLTextAreaElement | null} textArea
 * @param {boolean} [isNeg]
 */
function updatePromptArea(text, textArea, isNeg = false) {
    if (!textArea) return;
    if (!tryToRemoveExtraNetworkFromPrompt(textArea, text, isNeg)) {
        textArea.value = `${textArea.value}${opts.extra_networks_add_text_separator || ''}${text}`;
    }
    updateInput(textArea);
}

/**
 * Handle card clicks from the extra networks grid.
 * @param {string} tabname
 * @param {string} textToAdd
 * @param {string} textToAddNegative
 * @param {boolean} allowNegativePrompt
 */
function cardClicked(tabname, textToAdd, textToAddNegative, allowNegativePrompt) {
    const root = appRoot();
    const prompt = queryTextControl(root, `#${tabname}_prompt > label > textarea`);
    const negPrompt = queryTextControl(root, `#${tabname}_neg_prompt > label > textarea`);
    const promptArea = prompt instanceof HTMLTextAreaElement ? prompt : null;
    const negPromptArea = negPrompt instanceof HTMLTextAreaElement ? negPrompt : null;

    if (textToAddNegative.length > 0) {
        updatePromptArea(textToAdd, promptArea, false);
        updatePromptArea(textToAddNegative, negPromptArea, true);
    } else {
        const target = allowNegativePrompt ? activePromptTextarea[tabname] : prompt;
        const resolvedTarget = target instanceof HTMLTextAreaElement ? target : promptArea;
        updatePromptArea(textToAdd, resolvedTarget || null);
    }
}

/**
 * Persist a preview image selection to disk.
 * @param {MouseEvent} event
 * @param {string} tabname
 * @param {string} filename
 */
function saveCardPreview(event, tabname, filename) {
    const root = appRoot();
    const textarea = queryTextControl(root, `#${tabname}_preview_filename  > label > textarea`);
    const button = queryHTMLElement(root, `#${tabname}_save_preview`);
    if (!textarea || !button) return;

    textarea.value = filename;
    updateInput(textarea);
    button.click();

    event.stopPropagation();
    event.preventDefault();
}

/**
 * Update search input when clicking on canned search buttons.
 * @param {string} tabname
 * @param {string} extraNetworksTabname
 * @param {MouseEvent} event
 */
/**
 * Update the search field when a quick filter button is pressed.
 * @param {string} tabname
 * @param {string} extraNetworksTabname
 * @param {MouseEvent} event
 */
function extraNetworksSearchButton(tabname, extraNetworksTabname, event) {
    const root = appRoot();
    const textarea = queryTextControl(root, `#${tabname}_${extraNetworksTabname}_extra_search`);
    if (!textarea) return;

    const button = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    if (!button) return;
    const isSearchAll = button.classList.contains('search-all');
    const text = isSearchAll ? '' : button.textContent.trim();

    textarea.value = text;
    updateInput(textarea);
}

function extraNetworksTreeProcessFileClick(
    /** @type {MouseEvent} */ event,
    /** @type {HTMLElement} */ btn,
    /** @type {string} */ tabname,
    /** @type {string} */ extraNetworksTabname
) {
    void event;
    void btn;
    void tabname;
    void extraNetworksTabname;
}

/**
 * Handle directory interactions inside the tree view.
 * @param {MouseEvent} event
 * @param {HTMLElement} btn
 * @param {string} tabname
 * @param {string} extraNetworksTabname
 */
function extraNetworksTreeProcessDirectoryClick(
    /** @type {MouseEvent} */ event,
    /** @type {HTMLElement} */ btn,
    /** @type {string} */ tabname,
    /** @type {string} */ extraNetworksTabname
) {
    const ul = guardElement(btn.nextElementSibling, (node) => node instanceof HTMLElement);
    if (!ul) return;

    const trueTarget = event.target instanceof HTMLElement ? event.target : null;

    /** @param {HTMLElement} list @param {HTMLElement} trigger */
    const expandOrCollapse = (list, trigger) => {
        if (list.hasAttribute('hidden')) {
            list.removeAttribute('hidden');
            trigger.dataset.expanded = '';
        } else {
            list.setAttribute('hidden', '');
            delete trigger.dataset.expanded;
        }
    };

    /** Remove selection markers from all tree entries. */
    const deselectAll = () => {
        document
            .querySelectorAll('div.tree-list-content')
            .forEach((element) => { if (element instanceof HTMLElement) delete element.dataset.selected; });
    };

    /** @param {HTMLElement} targetButton */
    const selectButton = (targetButton) => {
        deselectAll();
        targetButton.dataset.selected = '';
    };

    /** @param {string} path */
    const updateSearch = (path) => {
        const searchInput = queryTextControl(appRoot(), `#${tabname}_${extraNetworksTabname}_extra_search`);
        if (!searchInput) return;
        searchInput.value = path;
        updateInput(searchInput);
    };

    if (trueTarget && trueTarget.matches('.tree-list-item-action--leading, .tree-list-item-action-chevron')) {
        expandOrCollapse(ul, btn);
        return;
    }

    const isSelected = 'selected' in btn.dataset;
    const isCollapsed = ul.hasAttribute('hidden');

    if (isSelected && !isCollapsed) {
        expandOrCollapse(ul, btn);
        delete btn.dataset.selected;
        updateSearch('');
    } else if (!isSelected && !isCollapsed) {
        expandOrCollapse(ul, btn);
        selectButton(btn);
        if (btn.dataset.path) updateSearch(btn.dataset.path);
    } else {
        selectButton(btn);
        if (btn.dataset.path) updateSearch(btn.dataset.path);
    }
}

/**
 * Tree click routing between files and directories.
 * @param {MouseEvent} event
 * @param {string} tabname
 * @param {string} extraNetworksTabname
 */
function extraNetworksTreeOnClick(event, tabname, extraNetworksTabname) {
    const button = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    if (!button) return;

    const entryType = button.parentElement?.dataset.treeEntryType;
    if (entryType === 'file') {
        extraNetworksTreeProcessFileClick(event, button, tabname, extraNetworksTabname);
    } else {
        extraNetworksTreeProcessDirectoryClick(event, button, tabname, extraNetworksTabname);
    }
}

/**
 * Handle sort mode button clicks.
 * @param {MouseEvent} event
 * @param {string} tabname
 * @param {string} extraNetworksTabname
 */
function extraNetworksControlSortOnClick(event, tabname, extraNetworksTabname) {
    const button = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    const parent = button?.parentElement;
    if (!button || !parent) return;

    parent.querySelectorAll('.extra-network-control--sort').forEach((elem) => {
        if (elem instanceof HTMLElement) elem.classList.remove('extra-network-control--enabled');
    });

    button.classList.add('extra-network-control--enabled');
    applyExtraNetworkSort(`${tabname}_${extraNetworksTabname}`);
}

/**
 * Cycle sort direction (ascending/descending).
 * @param {MouseEvent} event
 * @param {string} tabname
 * @param {string} extraNetworksTabname
 */
function extraNetworksControlSortDirOnClick(event, tabname, extraNetworksTabname) {
    const button = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    if (!button) return;

    const current = button.dataset.sortdir || 'Ascending';
    if (current === 'Ascending') {
        button.dataset.sortdir = 'Descending';
        button.setAttribute('title', 'Sort descending');
    } else {
        button.dataset.sortdir = 'Ascending';
        button.setAttribute('title', 'Sort ascending');
    }
    applyExtraNetworkSort(`${tabname}_${extraNetworksTabname}`);
}

/**
 * Toggle the visibility of the tree view pane.
 * @param {MouseEvent} event
 * @param {string} tabname
 * @param {string} extraNetworksTabname
 */
function extraNetworksControlTreeViewOnClick(event, tabname, extraNetworksTabname) {
    const button = event.currentTarget instanceof HTMLElement ? event.currentTarget : null;
    if (!button) return;

    button.classList.toggle('extra-network-control--enabled');
    const show = !button.classList.contains('extra-network-control--enabled');

    const pane = getAppElementById(`${tabname}_${extraNetworksTabname}_pane`);
    if (pane instanceof HTMLElement) {
        pane.classList.toggle('extra-network-dirs-hidden', show);
    }
}

/** Refresh the active LoRA-like tabs when the backend signals a change. */
function clickLoraRefresh() {
    const targets = [
        'txt2img_lora',
        'txt2img_checkpoints',
        'txt2img_textural_inversion',
        'img2img_lora',
        'img2img_checkpoints',
        'img2img_textural_inversion',
    ];

    targets.forEach((target) => {
        const tabButton = getAppElementById(`${target}-button`);
        if (tabButton && tabButton.getAttribute('aria-selected') === 'true') {
            const applyFn = extraNetworksApplyFilter[target];
            if (applyFn) applyFn(true);
        }
    });
}

/**
 * Trigger backend refresh via the hidden button.
 * @param {MouseEvent} event
 * @param {string} tabname
 * @param {string} extraNetworksTabname
 */
function extraNetworksControlRefreshOnClick(event, tabname, extraNetworksTabname) {
    void event;
    const button = getAppElementById(`${tabname}_${extraNetworksTabname}_extra_refresh_internal`);
    if (button instanceof HTMLElement) {
        button.dispatchEvent(new Event('click'));
    }
}

/** Hide the global popup if present. */
function closePopup() {
    if (globalPopup) {
        globalPopup.style.display = 'none';
    }
}

/**
 * Ensure the reusable popup exists and display arbitrary content inside it.
 * @param {HTMLElement} contents
 */
function popup(contents) {
    const root = appRoot();
    if (!globalPopup) {
        const popupElem = document.createElement('div');
        popupElem.classList.add('global-popup');

        const close = document.createElement('div');
        close.classList.add('global-popup-close');
        close.title = 'Close';
        close.addEventListener('click', closePopup);

        const inner = document.createElement('div');
        inner.classList.add('global-popup-inner');

        popupElem.appendChild(close);
        popupElem.appendChild(inner);

        const main = queryHTMLElement(root, '.main');
        if (!main) return;
        main.appendChild(popupElem);

        globalPopup = popupElem;
        globalPopupInner = inner;
    }

    if (!globalPopupInner) return;

    globalPopupInner.innerHTML = '';
    globalPopupInner.appendChild(contents);
    globalPopup.style.display = 'flex';
}

/**
 * Store popup content by id for reuse.
 * @param {string} id
 */
function popupId(id) {
    if (!storedPopupIds[id]) {
        storedPopupIds[id] = getAppElementById(id);
    }
    const stored = storedPopupIds[id];
    if (stored instanceof HTMLElement) {
        popup(stored);
    }
}

/**
 * Flatten metadata dictionaries for easier rendering.
 * @param {Record<string, unknown>} obj
 * @returns {Record<string, unknown>}
 */
function extraNetworksFlattenMetadata(obj) {
    const result = /** @type {Record<string, unknown>} */ ({});

    for (const key of Object.keys(obj)) {
        if (typeof obj[key] === 'string') {
            try {
                const parsed = JSON.parse(obj[key]);
                if (parsed && typeof parsed === 'object') {
                    obj[key] = parsed;
                }
            } catch (error) {
                console.error(error);
            }
        }
    }

    for (const key of Object.keys(obj)) {
        const value = obj[key];
        if (value && typeof value === 'object') {
            const nested = extraNetworksFlattenMetadata(/** @type {Record<string, unknown>} */ (value));
            for (const nestedKey of Object.keys(nested)) {
                result[`${key}/${nestedKey}`] = nested[nestedKey];
            }
        } else {
            result[key] = value;
        }
    }

    for (const key of Object.keys(result)) {
        if (key.startsWith('modelspec.')) {
            result[key.replaceAll('.', '/')] = result[key];
            delete result[key];
        }
    }

    for (const key of Object.keys(result)) {
        const parts = key.split('/');
        for (let i = 1; i < parts.length; i += 1) {
            const parent = parts.slice(0, i).join('/');
            if (!(parent in result)) {
                result[parent] = '';
            }
        }
    }

    return result;
}

/**
 * Display metadata either as a flattened tree table or raw preformatted text.
 * @param {string} text
 */
function extraNetworksShowMetadata(text) {
    try {
        const parsed = JSON.parse(text);
        if (parsed && typeof parsed === 'object') {
            const flattened = extraNetworksFlattenMetadata(parsed);
            const numericEntries = Object.fromEntries(
                Object.entries(flattened).filter(([, value]) => typeof value === 'number')
            );
            if (Object.keys(numericEntries).length > 0) {
                const table = createVisualizationTable(/** @type {Record<string, number>} */ (numericEntries), 0);
                popup(table);
                return;
            }
        }
    } catch (error) {
        console.error(error);
    }

    const elem = document.createElement('pre');
    elem.classList.add('popup-metadata');
    elem.textContent = text;
    popup(elem);
}

/**
 * Send a GET request with JSON payload and parse the response.
 * @param {string} url
 * @param {Record<string, string>} data
 * @param {(response: any) => void} handler
 * @param {() => void} errorHandler
 */
function requestGet(url, data, handler, errorHandler) {
    const xhr = new XMLHttpRequest();
    const args = Object.entries(data)
        .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`)
        .join('&');
    xhr.open('GET', `${url}?${args}`, true);

    xhr.onreadystatechange = function onReadyStateChange() {
        if (xhr.readyState === 4) {
            if (xhr.status === 200) {
                try {
                    handler(JSON.parse(xhr.responseText));
                } catch (error) {
                    console.error(error);
                    errorHandler();
                }
            } else {
                errorHandler();
            }
        }
    };

    xhr.send(JSON.stringify(data));
}

/**
 * Copy the card path to clipboard.
 * @param {MouseEvent} event
 */
function extraNetworksCopyCardPath(event) {
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (!target) return;

    const text = target.getAttribute('data-clipboard-text');
    if (text) {
        void navigator.clipboard.writeText(text);
    }
    event.stopPropagation();
}

/**
 * Request metadata for a card.
 * @param {MouseEvent} event
 * @param {string} extraPage
 */
function extraNetworksRequestMetadata(event, extraPage) {
    const showError = () => extraNetworksShowMetadata('there was an error getting metadata');
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (!target) return;

    const cardElement = target.closest('[data-name]');
    if (!(cardElement instanceof HTMLElement)) {
        showError();
        return;
    }

    const cardName = cardElement.getAttribute('data-name');
    if (!cardName) {
        showError();
        return;
    }

    requestGet(
        './sd_extra_networks/metadata',
        { page: extraPage, item: cardName },
        (data) => {
            if (data && data.metadata) {
                extraNetworksShowMetadata(data.metadata);
            } else {
                showError();
            }
        },
        showError
    );

    event.stopPropagation();
}

/**
 * Launch the user-metadata editor for a card.
 * @param {MouseEvent} event
 * @param {string} tabname
 * @param {string} extraPage
 */
function extraNetworksEditUserMetadata(event, tabname, extraPage) {
    const target = event.target instanceof HTMLElement ? event.target : null;
    if (!target) return;

    const editorId = `${tabname}_${extraPage}_edit_user_metadata`;
    let editor = extraPageUserMetadataEditors[editorId];

    if (!editor) {
        editor = {
            page: getAppElementById(editorId),
            nameTextarea: queryTextControl(appRoot(), `#${editorId}_name textarea`),
            button: queryHTMLElement(appRoot(), `#${editorId}_button`),
        };
        extraPageUserMetadataEditors[editorId] = editor;
    }

    if (!editor.nameTextarea || !editor.button || !(editor.page instanceof HTMLElement)) return;

    const cardElement = target.closest('[data-name]');
    if (!(cardElement instanceof HTMLElement)) return;

    const cardName = cardElement.getAttribute('data-name');
    if (!cardName) return;

    editor.nameTextarea.value = cardName;
    updateInput(editor.nameTextarea);
    editor.button.click();
    popup(editor.page);

    event.stopPropagation();
}

/**
 * Refresh the HTML of a single card.
 * @param {string} page
 * @param {string} tabname
 * @param {string} name
 */
function extraNetworksRefreshSingleCard(page, tabname, name) {
    requestGet(
        './sd_extra_networks/get-single-card',
        { page, tabname, name },
        (data) => {
            if (!(data && data.html)) return;

            const root = appRoot();
            const card = queryHTMLElement(root, `#${tabname}_${page.replace(/\s+/g, '_')}_cards > .card[data-name="${name}"]`);
            if (!card || !card.parentElement) return;

            const newDiv = document.createElement('div');
            newDiv.innerHTML = data.html;
            const newCard = newDiv.firstElementChild;
            if (!(newCard instanceof HTMLElement)) return;

            newCard.style.display = '';
            card.parentElement.insertBefore(newCard, card);
            card.parentElement.removeChild(card);
        },
        () => {}
    );
}

window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
        closePopup();
    }
});

/** Schedule callbacks that need the extra network HTML to be present. */
function scheduleAfterScriptsCallbacks() {
    if (uiAfterScriptsTimeout) {
        clearTimeout(uiAfterScriptsTimeout);
    }
    uiAfterScriptsTimeout = setTimeout(() => {
        executeCallbacks(uiAfterScriptsCallbacks, undefined, 'uiAfterScripts');
    }, 200);
}

onUiLoaded(() => {
    const observer = new MutationObserver(() => {
        const root = appRoot();
        const existingSearchfields = root.querySelectorAll("[id$='_extra_search']").length;
        const neededSearchfields = root.querySelectorAll("[id$='_extra_tabs'] > .tab-nav > button").length - 2;

        if (!executedAfterScripts && existingSearchfields >= neededSearchfields) {
            observer.disconnect();
            executedAfterScripts = true;
            scheduleAfterScriptsCallbacks();
        }
    });
    observer.observe(appRoot(), { childList: true, subtree: true });
});

uiAfterScriptsCallbacks.push(setupExtraNetworks);
