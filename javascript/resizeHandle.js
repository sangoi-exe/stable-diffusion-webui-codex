"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Draggable grid splitter for two-column rows.
 - Input: Mouse/Touch events; debounced window resize.
 - Safety: Typed ResizeState; min widths; no assumptions on child order beyond 2-col layout.
*/

/** @type {Window & { setupResizeHandle?: (parent: HTMLElement) => void }} */
const resizeWindow = window;


(function() {
    const GRADIO_MIN_WIDTH = 320;
    const PAD = 16;
    const DEBOUNCE_TIME = 100;
    const DOUBLE_TAP_DELAY = 200;

    /**
     * @typedef {HTMLElement & {
     *   resizeHandle?: HTMLDivElement;
     *   minLeftColWidth?: number;
     *   minRightColWidth?: number;
     *   needHideOnMoblie?: boolean;
     *   originalGridTemplateColumns?: string;
     * }} ResizeParent
     */

    /**
     * @typedef {{
     *   tracking: boolean;
     *   parent: ResizeParent | null;
     *   parentWidth: number;
     *   leftCol: HTMLElement | null;
     *   leftColStartWidth: number;
     *   screenX: number;
     *   lastTapTime: number;
     * }} ResizeState
     */

    /** @type {ResizeState} */
    const R = {
        tracking: false,
        parent: null,
        parentWidth: 0,
        leftCol: null,
        leftColStartWidth: 0,
        screenX: 0,
        lastTapTime: 0,
    };

    /** @type {number | undefined} */
    let resizeTimer;
    /** @type {ResizeParent[]} */
    const parents = [];

    /**
     * @param {ResizeParent} element
     * @param {number} width
     */
    function setLeftColGridTemplate(element, width) {
        element.style.gridTemplateColumns = `${width}px 16px 1fr`;
    }

    /**
     * @param {ResizeParent} parent
     * @returns {boolean}
     */
    function displayResizeHandle(parent) {
        if (!parent.needHideOnMoblie) {
            return true;
        }
        if (window.innerWidth < GRADIO_MIN_WIDTH * 2 + PAD * 4) {
            parent.style.display = 'flex';
            if (parent.resizeHandle) parent.resizeHandle.style.display = 'none';
            return false;
        }
        parent.style.display = 'grid';
        if (parent.resizeHandle) parent.resizeHandle.style.display = 'block';
        return true;
    }

    /**
     * @param {ResizeParent} parent
     */
    function afterResize(parent) {
        if (!displayResizeHandle(parent) || parent.style.gridTemplateColumns === parent.originalGridTemplateColumns) {
            return;
        }
        const oldParentWidth = R.parentWidth;
        const newParentWidth = parent.offsetWidth;
        const firstToken = (parent.style.gridTemplateColumns || '').split(' ')[0] || '0';
        const widthL = Number.parseInt(firstToken, 10);
        if (!Number.isFinite(widthL) || !Number.isFinite(oldParentWidth) || !Number.isFinite(newParentWidth)) return;

        const ratio = newParentWidth / (oldParentWidth || 1);
        const newWidthL = Math.max(Math.floor(ratio * widthL), parent.minLeftColWidth ?? GRADIO_MIN_WIDTH);
        setLeftColGridTemplate(parent, newWidthL);
        R.parentWidth = newParentWidth;
    }

    /**
     * @param {ResizeParent} parent
     */
    /**
     * @param {ResizeParent} parent
     */
    function setupResizeHandle(parent) {
        /** @param {MouseEvent | TouchEvent} evt */
        const onDoubleClick = (evt) => {
            evt.preventDefault();
            evt.stopPropagation();
            if (parent.originalGridTemplateColumns) {
                parent.style.gridTemplateColumns = parent.originalGridTemplateColumns;
            }
        };

        const leftCol = parent.firstElementChild instanceof HTMLElement ? parent.firstElementChild : null;
        const rightCol = parent.lastElementChild instanceof HTMLElement ? parent.lastElementChild : null;
        if (!leftCol || !rightCol) return;

        parents.push(parent);
        parent.style.display = 'grid';
        parent.style.gap = '0';

        let leftColTemplate = '';
        const firstChild = parent.children[0];
        if (firstChild instanceof HTMLElement && firstChild.style.flexGrow) {
            leftColTemplate = `${firstChild.style.flexGrow}fr`;
            parent.minLeftColWidth = GRADIO_MIN_WIDTH;
            parent.minRightColWidth = GRADIO_MIN_WIDTH;
            parent.needHideOnMoblie = true;
        } else if (firstChild instanceof HTMLElement && firstChild.style.flexBasis) {
            leftColTemplate = firstChild.style.flexBasis;
            const basis = Number.parseFloat(firstChild.style.flexBasis);
            parent.minLeftColWidth = Number.isFinite(basis) ? basis / 2 : GRADIO_MIN_WIDTH;
            parent.minRightColWidth = 0;
            parent.needHideOnMoblie = false;
        } else {
            leftColTemplate = '1fr';
            parent.minLeftColWidth = GRADIO_MIN_WIDTH;
            parent.minRightColWidth = 0;
            parent.needHideOnMoblie = false;
        }

        const secondChild = parent.children[1];
        const rightBasis = secondChild instanceof HTMLElement && secondChild.style.flexGrow ? secondChild.style.flexGrow : '1';
        const gridTemplateColumns = `${leftColTemplate} ${PAD}px ${rightBasis}fr`;
        parent.style.gridTemplateColumns = gridTemplateColumns;
        parent.originalGridTemplateColumns = gridTemplateColumns;

        const resizeHandle = document.createElement('div');
        resizeHandle.classList.add('resize-handle');
        parent.insertBefore(resizeHandle, rightCol);
        parent.resizeHandle = resizeHandle;

        ['mousedown', 'touchstart'].forEach((eventType) => {
            /** @param {MouseEvent | TouchEvent} evt */
            resizeHandle.addEventListener(eventType, (evt) => {
                if (eventType.startsWith('mouse')) {
                    if (!(evt instanceof MouseEvent) || evt.button !== 0) return;
                } else {
                    if (!(evt instanceof TouchEvent) || evt.changedTouches.length !== 1) return;
                    const currentTime = Date.now();
                    if (R.lastTapTime && currentTime - R.lastTapTime <= DOUBLE_TAP_DELAY) {
                        onDoubleClick(evt);
                        return;
                    }
                    R.lastTapTime = currentTime;
                }

                evt.preventDefault();
                evt.stopPropagation();

                document.body.classList.add('resizing');

                R.tracking = true;
                R.parent = parent;
                R.parentWidth = parent.offsetWidth;
                R.leftCol = leftCol;
                R.leftColStartWidth = leftCol.offsetWidth;
                R.screenX = eventType.startsWith('mouse')
                    ? /** @type {MouseEvent} */ (evt).screenX
                    : /** @type {TouchEvent} */ (evt).changedTouches[0]?.screenX ?? 0;
            });
        });

        resizeHandle.addEventListener('dblclick', onDoubleClick);
        afterResize(parent);
    }

    ['mousemove', 'touchmove'].forEach((eventType) => {
        /** @param {MouseEvent | TouchEvent} evt */
        window.addEventListener(eventType, (evt) => {
            if (!R.tracking || !R.parent || !R.leftCol) return;

            if (eventType.startsWith('mouse')) {
                if (!(evt instanceof MouseEvent) || evt.button !== 0) return;
                evt.preventDefault();
            } else {
                if (!(evt instanceof TouchEvent) || evt.changedTouches.length !== 1) return;
            }
            evt.stopPropagation();

            const screenX = eventType.startsWith('mouse')
                ? /** @type {MouseEvent} */ (evt).screenX
                : /** @type {TouchEvent} */ (evt).changedTouches[0]?.screenX ?? R.screenX;
            const delta = R.screenX - screenX;
            const maxWidth = R.parent.offsetWidth - (R.parent.minRightColWidth ?? 0) - PAD;
            const proposedWidth = Math.max(Math.min(R.leftColStartWidth - delta, maxWidth), R.parent.minLeftColWidth ?? GRADIO_MIN_WIDTH);
            setLeftColGridTemplate(R.parent, proposedWidth);
        });
    });

    ['mouseup', 'touchend'].forEach((eventType) => {
        /** @param {MouseEvent | TouchEvent} evt */
        window.addEventListener(eventType, (evt) => {
            if (!R.tracking) return;
            if (eventType.startsWith('mouse')) {
                if (!(evt instanceof MouseEvent) || evt.button !== 0) return;
            } else {
                if (!(evt instanceof TouchEvent) || evt.changedTouches.length !== 1) return;
            }
            evt.preventDefault();
            evt.stopPropagation();

            R.tracking = false;
            document.body.classList.remove('resizing');
        });
    });

    window.addEventListener('resize', () => {
        if (resizeTimer) window.clearTimeout(resizeTimer);
        resizeTimer = window.setTimeout(() => {
            parents.forEach((parent) => afterResize(parent));
        }, DEBOUNCE_TIME);
    });

    Object.defineProperty(resizeWindow, 'setupResizeHandle', {
        value: setupResizeHandle,
        writable: true,
    });
})();

function setupAllResizeHandles() {
    const win = /** @type {Window & { setupResizeHandle?: (parent: HTMLElement) => void }} */ (window);
    gradioApp().querySelectorAll('.resize-handle-row').forEach((elem) => {
        if (elem instanceof HTMLElement && !elem.querySelector('.resize-handle') && !(elem.children[0] && elem.children[0].classList.contains('hidden'))) {
            if (typeof win.setupResizeHandle === 'function') {
                win.setupResizeHandle(elem);
            }
        }
    });
}

onUiLoaded(setupAllResizeHandles);
