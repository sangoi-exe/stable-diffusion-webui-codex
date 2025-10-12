"use strict";
/*
 DevNotes (2025-10-12)
 - Purpose: Checkbox mirrors accordion open/close state; supports legacy global hook.
 - Safety: JSDoc contract for dynamic properties; idempotent attach on UI load.
*/

/**
 * @typedef {HTMLElement & {
 *   visibleCheckbox?: HTMLInputElement;
 *   onVisibleCheckboxChange?: () => void;
 *   onChecked?: (checked: boolean) => void;
 * }} AccordionElement
 */

/**
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
 * Update the accordion visibility from external controls.
 * @param {string} id
 * @param {boolean} checked
 */
function inputAccordionChecked(id, checked) {
    const accordion = /** @type {(AccordionElement | null)} */ (getAppElementById(id));
    if (!accordion) return;

    const visible = accordion.visibleCheckbox;
    if (visible instanceof HTMLInputElement) {
        visible.checked = checked;
        if (typeof accordion.onVisibleCheckboxChange === 'function') {
            accordion.onVisibleCheckboxChange();
        }
    }
}

/**
 * Enhance a Gradio accordion with synced checkbox controls.
 * @param {HTMLElement} element
 */
function setupAccordion(element) {
    const root = gradioApp();
    const accordion = /** @type {AccordionElement} */ (element);
    const labelWrapCandidate = accordion.querySelector('.label-wrap');
    if (!(labelWrapCandidate instanceof HTMLElement)) return;
    const labelWrap = labelWrapCandidate;
    const gradioCheckbox = root.querySelector(`#${accordion.id}-checkbox input`);
    const gradioCheckboxClass = gradioCheckbox instanceof HTMLElement ? gradioCheckbox.className : '';
    const extra = root.querySelector(`#${accordion.id}-extra`);
    const spanCandidate = labelWrap.querySelector('span');
    if (!(spanCandidate instanceof HTMLElement)) return;
    const span = spanCandidate;

    const isOpen = () => labelWrap.classList.contains('open');

    const observerAccordionOpen = new MutationObserver(() => {
        accordion.classList.toggle('input-accordion-open', isOpen());
        const visible = accordion.visibleCheckbox;
        if (visible instanceof HTMLInputElement && typeof accordion.onVisibleCheckboxChange === 'function') {
            visible.checked = isOpen();
            accordion.onVisibleCheckboxChange();
        }
    });
    observerAccordionOpen.observe(labelWrap, { attributes: true, attributeFilter: ['class'] });

    if (extra) {
        labelWrap.insertBefore(extra, labelWrap.lastElementChild);
    }

    accordion.onChecked = (checked) => {
        if (isOpen() !== checked) labelWrap.click();
    };

    const visibleCheckbox = document.createElement('input');
    visibleCheckbox.type = 'checkbox';
    visibleCheckbox.checked = isOpen();
    visibleCheckbox.id = `${accordion.id}-visible-checkbox`;
    visibleCheckbox.className = `${gradioCheckboxClass} input-accordion-checkbox`.trim();
    span.insertBefore(visibleCheckbox, span.firstChild);

    accordion.visibleCheckbox = visibleCheckbox;
    accordion.onVisibleCheckboxChange = () => {
        if (isOpen() !== visibleCheckbox.checked) labelWrap.click();
        if (gradioCheckbox instanceof HTMLInputElement) {
            gradioCheckbox.checked = visibleCheckbox.checked;
            updateInput(gradioCheckbox);
        }
    };

    visibleCheckbox.addEventListener('click', (event) => {
        event.stopPropagation();
    });
    visibleCheckbox.addEventListener('input', () => {
        if (typeof accordion.onVisibleCheckboxChange === 'function') {
            accordion.onVisibleCheckboxChange();
        }
    });
}

onUiLoaded(() => {
    const root = gradioApp();
    root.querySelectorAll('.input-accordion').forEach((el) => {
        if (el instanceof HTMLElement) {
            setupAccordion(el);
        }
    });
});

(/** @type {any} */ (window)).inputAccordionChecked = inputAccordionChecked;
