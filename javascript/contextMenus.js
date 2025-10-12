"use strict";
// @ts-check

/**
 * Context menu infrastructure for WebUI buttons.
 * Historically this file relied on implicit globals and loose DOM access; the
 * version below keeps behaviour equivalent while providing full type safety by
 * leaning on JSDoc annotations and runtime guards.
 */

/**
 * @typedef {{
 *   id: string;
 *   name: string;
 *   func: () => void;
 *   isNew: boolean;
 * }} ContextMenuEntry
 */

/**
 * @typedef {[appendOption: (selector: string, name: string, fn: () => void) => string,
 *            removeOption: (id: string) => void,
 *            ensureListener: () => void]} ContextMenuInitResult
 */

/**
 * Return a stable DOM root that supports the selectors required by the menus.
 * @returns {Document | ShadowRoot | HTMLElement}
 */
function contextRoot() {
    return gradioApp();
}

/**
 * Remove the existing context menu if present.
 */
function dismissContextMenu() {
    const root = contextRoot();
    const existing = root.querySelector('#context-menu');
    if (existing instanceof HTMLElement) {
        existing.remove();
    }
}

/**
 * Generate a unique identifier for menu entries.
 * @returns {string}
 */
function uid() {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
}

/**
 * Create the context menu initialiser triple.
 * @returns {ContextMenuInitResult}
 */
function contextMenuInit() {
    let eventListenerApplied = false;
    /** @type {Map<string, ContextMenuEntry[]>} */
    const menuSpecs = new Map();

    /**
     * Render the context menu at the pointer location.
     * @param {MouseEvent | Touch} pointer
     * @param {ContextMenuEntry[]} menuEntries
     */
    function showContextMenu(pointer, menuEntries) {
        dismissContextMenu();

        const root = contextRoot();
        const baseElement = (typeof uiCurrentTab !== 'undefined' && uiCurrentTab instanceof HTMLElement)
            ? uiCurrentTab
            : document.body;
        const baseStyle = window.getComputedStyle(baseElement);

        const contextMenu = document.createElement('nav');
        contextMenu.id = 'context-menu';
        contextMenu.style.background = baseStyle.background;
        contextMenu.style.color = baseStyle.color;
        contextMenu.style.fontFamily = baseStyle.fontFamily;
        contextMenu.style.top = `${pointer.pageY}px`;
        contextMenu.style.left = `${pointer.pageX}px`;

        const contextMenuList = document.createElement('ul');
        contextMenuList.className = 'context-menu-items';
        contextMenu.append(contextMenuList);

        menuEntries.forEach((entry) => {
            const contextMenuEntry = document.createElement('a');
            contextMenuEntry.textContent = entry.name;
            contextMenuEntry.addEventListener('click', (evt) => {
                evt.preventDefault();
                evt.stopPropagation();
                try {
                    entry.func();
                } finally {
                    dismissContextMenu();
                }
            });
            contextMenuList.append(contextMenuEntry);
        });

        root.appendChild(contextMenu);
    }

    /**
     * Register a context menu option for elements matching the selector.
     * @param {string} targetElementSelector
     * @param {string} entryName
     * @param {() => void} entryFunction
     * @returns {string} unique menu entry id
     */
    function appendContextMenuOption(targetElementSelector, entryName, entryFunction) {
        let entries = menuSpecs.get(targetElementSelector);
        if (!entries) {
            entries = [];
            menuSpecs.set(targetElementSelector, entries);
        }

        const newEntry = {
            id: `${targetElementSelector}_${uid()}`,
            name: entryName,
            func: entryFunction,
            isNew: true,
        };

        entries.push(newEntry);
        return newEntry.id;
    }

    /**
     * Remove a previously registered menu option.
     * @param {string} entryId
     */
    function removeContextMenuOption(entryId) {
        menuSpecs.forEach((entries) => {
            const index = entries.findIndex((entry) => entry.id === entryId);
            if (index !== -1) {
                entries.splice(index, 1);
            }
        });
    }

    /**
     * Attach the global event listener that drives the context menu.
     */
    function addContextMenuEventListener() {
        if (eventListenerApplied) {
            return;
        }

        const root = contextRoot();

        root.addEventListener('click', (event) => {
            if (!event.isTrusted) return;
            dismissContextMenu();
        });

        /**
         * Handle pointer-triggered menu events.
         * @param {MouseEvent | TouchEvent} event
         */
        /** @param {Event} event */
        const handlePointerEvent = (event) => {
            if (!(event instanceof Event)) return;
            if (event.type.startsWith('touch')) {
                if (!(event instanceof TouchEvent)) return;
                if (event.touches.length !== 2) return;
                const touch = event.touches.item(0);
                if (!touch) return;
                processPointer(event, touch);
            } else {
                if (!(event instanceof MouseEvent)) return;
                processPointer(event, event);
            }
        };

        /**
         * Run selector matching and, if applicable, open the menu.
         * @param {MouseEvent | TouchEvent} event
         * @param {MouseEvent | Touch} pointer
         */
        /**
         * @param {Event} event
         * @param {MouseEvent | Touch} pointer
         */
        const processPointer = (event, pointer) => {
            dismissContextMenu();
            const path = event.composedPath();
            const target = path.length > 0 ? path[0] : null;
            menuSpecs.forEach((entries, selector) => {
                if (target instanceof Element && target.matches(selector)) {
                    showContextMenu(pointer, entries);
                    event.preventDefault();
                }
            });
        };

        root.addEventListener('contextmenu', handlePointerEvent);
        root.addEventListener('touchstart', handlePointerEvent);

        eventListenerApplied = true;
    }

    return [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener];
}

const [appendContextMenuOption, removeContextMenuOption, addContextMenuEventListener] = contextMenuInit();

/** @type {ReturnType<typeof setInterval> | null} */
let regen_txt2img = null;
/** @type {ReturnType<typeof setInterval> | null} */
let regen_img2img = null;

(function initDefaultMenus() {
    /** Start example Context Menu Items */
    const generateOnRepeat_txt2img = () => {
        if (regen_txt2img !== null || regen_img2img !== null) return;

        const generate = contextRoot().querySelector('#txt2img_generate');
        const interrupt = contextRoot().querySelector('#txt2img_interrupt');
        if (!(generate instanceof HTMLElement) || !(interrupt instanceof HTMLElement)) return;

        if (!interrupt.offsetParent) {
            generate.click();
        }

        regen_txt2img = setInterval(() => {
            if (interrupt.style.display === 'none') {
                generate.click();
                interrupt.style.display = 'block';
            }
        }, 500);
    };
    appendContextMenuOption('#txt2img_generate', 'Generate forever', generateOnRepeat_txt2img);
    appendContextMenuOption('#txt2img_interrupt', 'Generate forever', generateOnRepeat_txt2img);

    const cancel_regen_txt2img = () => {
        if (regen_txt2img !== null) {
            clearInterval(regen_txt2img);
            regen_txt2img = null;
        }
    };
    appendContextMenuOption('#txt2img_interrupt', 'Cancel generate forever', cancel_regen_txt2img);
    appendContextMenuOption('#txt2img_generate', 'Cancel generate forever', cancel_regen_txt2img);

    const generateOnRepeat_img2img = () => {
        if (regen_txt2img !== null || regen_img2img !== null) return;

        const generate = contextRoot().querySelector('#img2img_generate');
        const interrupt = contextRoot().querySelector('#img2img_interrupt');
        if (!(generate instanceof HTMLElement) || !(interrupt instanceof HTMLElement)) return;

        if (!interrupt.offsetParent) {
            generate.click();
        }

        regen_img2img = setInterval(() => {
            if (interrupt.style.display === 'none') {
                generate.click();
                interrupt.style.display = 'block';
            }
        }, 500);
    };
    appendContextMenuOption('#img2img_generate', 'Generate forever', generateOnRepeat_img2img);
    appendContextMenuOption('#img2img_interrupt', 'Generate forever', generateOnRepeat_img2img);

    const cancel_regen_img2img = () => {
        if (regen_img2img !== null) {
            clearInterval(regen_img2img);
            regen_img2img = null;
        }
    };
    appendContextMenuOption('#img2img_interrupt', 'Cancel generate forever', cancel_regen_img2img);
    appendContextMenuOption('#img2img_generate', 'Cancel generate forever', cancel_regen_img2img);
})();

onAfterUiUpdate(addContextMenuEventListener);
