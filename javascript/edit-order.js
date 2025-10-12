"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Alt + ←/→ reordena itens separados por vírgula no prompt.
 - Safety: Cálculo de índices via contagem de vírgulas; mantém seleção coerente.
*/

/* alt+left/right moves text in prompt */

/**
 * @param {KeyboardEvent} event
 * @returns {HTMLTextAreaElement | null}
 */
function resolvePromptTextarea(event) {
    const path = event.composedPath ? event.composedPath() : [];
    const candidate = /** @type {unknown} */ (
        (/** @type {Event & { originalTarget?: EventTarget }} */ (event).originalTarget) ||
        (path.length > 0 ? path[0] : event.target)
    );
    return candidate instanceof HTMLTextAreaElement ? candidate : null;
}

/**
 * Handle alt+arrow presses to reorder comma-separated prompt items.
 * @param {KeyboardEvent} event
 */
function keyupEditOrder(event) {
    if (typeof opts !== 'object' || !opts || !opts.keyedit_move) return;

    const target = resolvePromptTextarea(event);
    if (!target) return;
    if (!event.altKey || !event.key) return;

    const isLeft = event.key === 'ArrowLeft';
    const isRight = event.key === 'ArrowRight';
    if (!isLeft && !isRight) return;

    event.preventDefault();

    const selectionStart = target.selectionStart ?? 0;
    const selectionEnd = target.selectionEnd ?? 0;
    const text = target.value || '';
    const items = text.split(',');
    const indexStart = (text.slice(0, selectionStart).match(/,/g) || []).length;
    const indexEnd = (text.slice(0, selectionEnd).match(/,/g) || []).length;
    const range = indexEnd - indexStart + 1;

    /**
     * @param {number} count
     */
    const joinedUntil = (count) => {
        if (count <= 0) return '';
        return items.slice(0, count).join(',');
    };

    if (isLeft && indexStart > 0) {
        const moved = items.splice(indexStart, range);
        items.splice(indexStart - 1, 0, ...moved);
        target.value = items.join(',');
        const before = joinedUntil(indexStart - 1);
        const beforeLength = before.length + (indexStart === 1 ? 0 : 1);
        target.selectionStart = beforeLength;
        const endSlice = joinedUntil(indexEnd);
        target.selectionEnd = endSlice.length;
    } else if (isRight && indexEnd < items.length - 1) {
        const moved = items.splice(indexStart, range);
        items.splice(indexStart + 1, 0, ...moved);
        target.value = items.join(',');
        const before = joinedUntil(indexStart + 1);
        target.selectionStart = before.length + 1;
        const endSlice = joinedUntil(indexEnd + 2);
        target.selectionEnd = endSlice.length;
    }

    updateInput(target);
}

addEventListener('keydown', (event) => {
    if (event instanceof KeyboardEvent) {
        keyupEditOrder(event);
    }
});
