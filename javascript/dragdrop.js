"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Arrastar/soltar e colar imagens em prompts e campos de imagem.
 - Safety: Checagem de tipos MIME, DataTransfer seguro, fetch → File com try/catch.
*/

/**
 * Drag-and-drop and clipboard image handling for Gradio image widgets.
 */

const ACCEPTED_IMAGE_TYPES = new Set(['image/png', 'image/gif', 'image/jpeg']);

/**
 * @param {FileList | null | undefined} files
 * @returns {files is FileList}
 */
function isValidImageList(files) {
    if (!files || files.length !== 1) return false;
    const file = files.item(0);
    return Boolean(file && ACCEPTED_IMAGE_TYPES.has(file.type));
}

/**
 * Replace the image inside a Gradio image component with the provided file list.
 * @param {HTMLElement} imgWrap
 * @param {FileList} files
 */
function dropReplaceImage(imgWrap, files) {
    if (!isValidImageList(files)) {
        return;
    }

    const tmpFile = files.item(0);
    if (!tmpFile) return;

    const clearButton = imgWrap.querySelector('.modify-upload button + button, .touch-none + div button + button');
    if (clearButton instanceof HTMLElement) {
        clearButton.click();
    }

    const callback = () => {
        const fileInput = imgWrap.querySelector('input[type="file"]');
        if (!(fileInput instanceof HTMLInputElement)) {
            return;
        }

        const dataTransfer = new DataTransfer();
        dataTransfer.items.add(tmpFile);
        fileInput.files = dataTransfer.files;
        fileInput.dispatchEvent(new Event('change'));
    };

    window.requestAnimationFrame(callback);
}

/**
 * Determine whether a drag/drop event carries files.
 * @param {DragEvent} event
 */
function eventHasFiles(event) {
    const transfer = event.dataTransfer;
    if (!transfer) return false;
    if (transfer.files && transfer.files.length > 0) return true;
    if (!transfer.items || transfer.items.length === 0) return false;
    const item = transfer.items[0];
    return Boolean(item && item.kind === 'file');
}

/**
 * @param {string} url
 */
function isURL(url) {
    try {
        const _ = new URL(url);
        void _;
        return true;
    } catch {
        return false;
    }
}

/**
 * @param {Element | null} target
 */
function dragDropTargetIsPrompt(target) {
    if (!target) return false;
    if ((target instanceof HTMLInputElement || target instanceof HTMLTextAreaElement) && target.placeholder.includes('Prompt')) {
        return true;
    }
    const parent = target.parentNode?.parentNode;
    if (parent instanceof Element && typeof parent.className === 'string' && parent.className.includes('prompt')) {
        return true;
    }
    return false;
}

/**
 * @param {Event} event
 * @returns {Element | null}
 */
function getEventElement(event) {
    const path = event.composedPath();
    const candidate = path.length > 0 ? path[0] : null;
    return candidate instanceof Element ? candidate : null;
}

window.document.addEventListener('dragover', (event) => {
    if (!(event instanceof DragEvent)) return;
    if (!eventHasFiles(event)) return;

    const target = getEventElement(event);
    if (!target) return;

    const targetImage = target.closest('[data-testid="image"]');
    if (!dragDropTargetIsPrompt(target) && !targetImage) return;

    event.stopPropagation();
    event.preventDefault();
    const transfer = event.dataTransfer;
    if (transfer) {
        transfer.dropEffect = 'copy';
    }
});

window.document.addEventListener('drop', async (event) => {
    if (!(event instanceof DragEvent)) return;

    const target = getEventElement(event);
    const transfer = event.dataTransfer;
    const url = transfer ? transfer.getData('text/uri-list') || transfer.getData('text/plain') : '';
    if (!eventHasFiles(event) && !isURL(url || '')) return;

    if (dragDropTargetIsPrompt(target)) {
        event.stopPropagation();
        event.preventDefault();

        const isImg2img = get_tab_index('tabs') === 1;
        const promptTargetId = isImg2img ? 'img2img_prompt_image' : 'txt2img_prompt_image';

        const imgParent = getAppElementById(promptTargetId);
        if (!imgParent || !transfer) return;

        const fileInput = imgParent.querySelector('input[type="file"]');
        if (!(fileInput instanceof HTMLInputElement)) return;

        if (eventHasFiles(event)) {
            fileInput.files = transfer.files;
            fileInput.dispatchEvent(new Event('change'));
        } else if (url) {
            try {
                const response = await fetch(url);
                if (!response.ok) {
                    console.error('Error fetching URL:', url, response.status);
                    return;
                }
                const blob = await response.blob();
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(new File([blob], 'image.png'));
                fileInput.files = dataTransfer.files;
                fileInput.dispatchEvent(new Event('change'));
            } catch (error) {
                console.error('Error fetching URL:', url, error);
            }
        }
    }

    if (target instanceof Element) {
        const targetImage = target.closest('[data-testid="image"]');
        if (targetImage instanceof HTMLElement && transfer) {
            event.stopPropagation();
            event.preventDefault();
            dropReplaceImage(targetImage, transfer.files);
        }
    }
});

window.addEventListener('paste', (event) => {
    if (!(event instanceof ClipboardEvent)) return;
    const clipboard = event.clipboardData;
    if (!clipboard) return;

    const files = clipboard.files;
    if (!isValidImageList(files)) {
        return;
    }

    const visibleImageFields = Array.from(gradioApp().querySelectorAll('[data-testid="image"]'))
        .filter((el) => el instanceof HTMLElement && uiElementIsVisible(el))
        .map((el) => /** @type {HTMLElement} */ (el))
        .sort((a, b) => Number(uiElementInSight(b)) - Number(uiElementInSight(a)));

    if (!visibleImageFields.length) {
        return;
    }

    const firstFreeImageField = visibleImageFields.find((el) => !el.querySelector('img'));
    const fallback = visibleImageFields[visibleImageFields.length - 1] || null;
    if (!fallback) return;
    const targetField = firstFreeImageField || fallback;
    dropReplaceImage(targetField, files);
});
/**
 * @param {string} id
 */
function getAppElementById(id) {
    const root = gradioApp();
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        const result = root.getElementById(id);
        if (result instanceof HTMLElement) return result;
    }
    const fallback = document.getElementById(id);
    return fallback instanceof HTMLElement ? fallback : null;
}
