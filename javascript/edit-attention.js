"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Ctrl/Cmd + ↑/↓ ajusta o peso (x:y) da seleção ou bloco entre parênteses.
 - Safety: Seleção protegida com buscas e limites; evita NaN e reformata 1 → remoção.
*/

/**
 * Keyboard shortcuts for adjusting attention weights in the prompt textarea.
 */

/**
 * @typedef {HTMLTextAreaElement & { placeholder?: string }} PromptTextarea
 */

/**
 * @param {FocusEvent | KeyboardEvent} event
 * @returns {PromptTextarea | null}
 */
function resolveTargetTextarea(event) {
    const originalTarget = /** @type {Event & { originalTarget?: EventTarget }} */ (event).originalTarget;
    const path = event.composedPath ? event.composedPath() : [];
    const candidate = /** @type {unknown} */ (originalTarget || (path.length > 0 ? path[0] : event.target));
    if (candidate instanceof HTMLTextAreaElement) {
        return candidate;
    }
    return null;
}

/**
 * @param {string} text
 * @param {number} index
 * @param {string} delimiters
 * @param {number} direction
 * @returns {number}
 */
function seekDelimiter(text, index, delimiters, direction) {
    let cursor = index;
    while (cursor >= 0 && cursor < text.length) {
        const char = text[cursor + (direction < 0 ? -1 : 0)];
        if (char === undefined || delimiters.includes(char)) {
            return cursor;
        }
        cursor += direction;
    }
    return cursor;
}

/**
 * Adjust attention weight based on arrow keypress.
 * @param {KeyboardEvent} event
 */
function keyupEditAttention(event) {
    const target = resolveTargetTextarea(event);
    if (!target) return;
    if (!(event.metaKey || event.ctrlKey)) return;
    if (!event.key) return;

    const isPlus = event.key === 'ArrowUp';
    const isMinus = event.key === 'ArrowDown';
    if (!isPlus && !isMinus) return;

    let selectionStart = target.selectionStart ?? 0;
    let selectionEnd = target.selectionEnd ?? 0;
    let text = target.value || '';

    /**
     * @param {string} OPEN
     * @param {string} CLOSE
     */
    function selectCurrentParenthesisBlock(OPEN, CLOSE) {
        if (selectionStart !== selectionEnd) return false;

        const before = text.substring(0, selectionStart);
        const beforeParen = before.lastIndexOf(OPEN);
        if (beforeParen === -1) return false;

        const beforeClosingParen = before.lastIndexOf(CLOSE);
        if (beforeClosingParen !== -1 && beforeClosingParen > beforeParen) return false;

        const after = text.substring(selectionStart);
        const afterParen = after.indexOf(CLOSE);
        if (afterParen === -1) return false;

        const afterOpeningParen = after.indexOf(OPEN);
        if (afterOpeningParen !== -1 && afterOpeningParen < afterParen) return false;

        const parenContent = text.substring(beforeParen + 1, selectionStart + afterParen);
        if (/.*:-?[\d.]+/s.test(parenContent)) {
            const lastColon = parenContent.lastIndexOf(':');
            selectionStart = beforeParen + 1;
            selectionEnd = selectionStart + lastColon;
        } else {
            selectionStart = beforeParen + 1;
            selectionEnd = selectionStart + parenContent.length;
        }

        if (target) target.setSelectionRange(selectionStart, selectionEnd);
        return true;
    }

    function selectCurrentWord() {
        if (selectionStart !== selectionEnd) return false;
        const whitespaceDelimiters = { Tab: '\t', 'Carriage Return': '\r', 'Line Feed': '\n' };
        let delimiters = String(opts.keyedit_delimiters || ' ');
        const whitespaceKeys = Array.isArray(opts.keyedit_delimiters_whitespace) ? opts.keyedit_delimiters_whitespace : [];
        for (const key of whitespaceKeys) {
            const replacement = whitespaceDelimiters[/** @type {keyof typeof whitespaceDelimiters} */ (key)];
            if (replacement) delimiters += replacement;
        }

        selectionStart = seekDelimiter(text, selectionStart, delimiters, -1);
        selectionEnd = seekDelimiter(text, selectionEnd, delimiters, +1);

        while (selectionStart < selectionEnd && text[selectionStart] === ' ') {
            selectionStart += 1;
        }
        while (selectionEnd > selectionStart && text[selectionEnd - 1] === ' ') {
            selectionEnd -= 1;
        }

        if (target) target.setSelectionRange(selectionStart, selectionEnd);
        return true;
    }

    if (!selectCurrentParenthesisBlock('<', '>') && !selectCurrentParenthesisBlock('(', ')') && !selectCurrentParenthesisBlock('[', ']')) {
        selectCurrentWord();
    }

    event.preventDefault();

    let closeCharacter = ')';
    let delta = Number(opts.keyedit_precision_attention ?? 0.1);
    const startChar = selectionStart > 0 ? text[selectionStart - 1] : '';
    const endChar = text[selectionEnd];

    if (startChar === '<') {
        closeCharacter = '>';
        delta = Number(opts.keyedit_precision_extra ?? delta);
    } else if ((startChar === '(' && endChar === ')') || (startChar === '[' && endChar === ']')) {
        let numParen = 0;
        while (
            text[selectionStart - numParen - 1] === startChar &&
            text[selectionEnd + numParen] === endChar
        ) {
            numParen += 1;
        }

        let weight = 1;
        if (startChar === '[') {
            weight = (1 / 1.1) ** numParen;
        } else {
            weight = 1.1 ** numParen;
        }
        const precision = Number(opts.keyedit_precision_attention ?? 0.1);
        weight = Math.round(weight / precision) * precision;

        text =
            text.slice(0, selectionStart - numParen) +
            '(' +
            text.slice(selectionStart, selectionEnd) +
            ':' +
            weight +
            ')' +
            text.slice(selectionEnd + numParen);
        selectionStart -= numParen - 1;
        selectionEnd -= numParen - 1;
    } else if (startChar !== '(') {
        while (selectionEnd > selectionStart && text[selectionEnd - 1] === ' ') {
            selectionEnd -= 1;
        }

        if (selectionStart === selectionEnd) {
            return;
        }

        text =
            text.slice(0, selectionStart) +
            '(' +
            text.slice(selectionStart, selectionEnd) +
            ':1.0)' +
            text.slice(selectionEnd);

        selectionStart += 1;
        selectionEnd += 1;
    }

    if (text[selectionEnd] !== ':') return;
    const weightLength = text.slice(selectionEnd + 1).indexOf(closeCharacter) + 1;
    const weightString = text.slice(selectionEnd + 1, selectionEnd + weightLength);
    let weight = parseFloat(weightString);
    if (Number.isNaN(weight)) return;

    weight += isPlus ? delta : -delta;
    weight = parseFloat(weight.toPrecision(12));
    if (Number.isInteger(weight)) weight = Number(`${weight}.0`);

    if (closeCharacter === ')' && weight === 1) {
        const endParenPos = text.substring(selectionEnd).indexOf(')');
        if (endParenPos === -1) return;
        text =
            text.slice(0, selectionStart - 1) +
            text.slice(selectionStart, selectionEnd) +
            text.slice(selectionEnd + endParenPos + 1);
        selectionStart -= 1;
        selectionEnd -= 1;
    } else {
        text = text.slice(0, selectionEnd + 1) + weight + text.slice(selectionEnd + weightLength);
    }

    target.value = text;
    target.setSelectionRange(selectionStart, selectionEnd);
    updateInput(target);
}

addEventListener('keydown', (event) => {
    if (event instanceof KeyboardEvent) {
        keyupEditAttention(event);
    }
});
