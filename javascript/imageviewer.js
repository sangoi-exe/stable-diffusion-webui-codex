"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Lightbox modal (open/close, next/prev, zoom, tile preview).
 - Safety: All lookups via getAppElementById; idempotent setup; keyboard/gamepad routes call modalImageSwitch.
*/

/**
 * @typedef {Window & {
 *   open?: (url: string) => Window | null;
 *   opts?: typeof opts;
 * }} ViewerWindow
 */

/** @type {ViewerWindow} */
const viewerWindow = window;

/**
 * Best-effort helper to fetch an element by id from the Gradio root.
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
 * @returns {HTMLImageElement | null}
 */
function getModalImage() {
    const element = getAppElementById('modalImage');
    return element instanceof HTMLImageElement ? element : null;
}

/** Close the lightbox modal. */
function closeModal() {
    const modal = getAppElementById('lightboxModal');
    if (modal) modal.style.display = 'none';
}

/**
 * Populate and show the modal.
 * @param {MouseEvent} event
 */
function showModal(event) {
    const target = event.currentTarget instanceof HTMLImageElement
        ? event.currentTarget
        : event.target instanceof HTMLImageElement
            ? event.target
            : null;
    if (!target) return;

    const modalImage = getModalImage();
    const toggleBtn = getAppElementById('modal_toggle_live_preview');
    const modal = getAppElementById('lightboxModal');
    if (!modalImage || !modal) return;

    if (toggleBtn) {
        toggleBtn.innerHTML = opts.js_live_preview_in_modal_lightbox ? '&#x1F5C7;' : '&#x1F5C6;';
    }

    modalImage.src = target.src;
    if (modalImage.style.display === 'none') {
        modal.style.setProperty('background-image', `url(${target.src})`);
    }
    modal.style.display = 'flex';
    modal.focus();

    const tabTxt2Img = getAppElementById('tab_txt2img');
    const tabImg2Img = getAppElementById('tab_img2img');
    const modalSave = getAppElementById('modal_save');
    if (modalSave) {
        const showSave = (tabTxt2Img && tabTxt2Img.style.display !== 'none') ||
            (tabImg2Img && tabImg2Img.style.display !== 'none');
        modalSave.style.display = showSave ? 'inline' : 'none';
    }

    event.stopPropagation();
}

/**
 * @param {number} n
 * @param {number} m
 */
function negmod(n, m) {
    return ((n % m) + m) % m;
}

function updateOnBackgroundChange() {
    const modalImage = getModalImage();
    if (!modalImage || !modalImage.offsetParent) return;

    const currentButton = selected_gallery_button?.();
    const previews = gradioApp().querySelectorAll('.livePreview > img');
    if (opts.js_live_preview_in_modal_lightbox && previews.length > 0) {
        const lastPreview = previews[previews.length - 1];
        if (lastPreview instanceof HTMLImageElement) {
            modalImage.src = lastPreview.src;
        }
    } else if (currentButton && currentButton.children?.length) {
        const childImage = currentButton.children[0];
        if (childImage instanceof HTMLImageElement && modalImage.src !== childImage.src) {
            modalImage.src = childImage.src;
            if (modalImage.style.display === 'none') {
                const modal = getAppElementById('lightboxModal');
                modal?.style.setProperty('background-image', `url(${modalImage.src})`);
            }
        }
    }
}

/**
 * @param {number} offset
 */
function modalImageSwitch(offset) {
    const galleryButtonsRaw = all_gallery_buttons?.();
    const galleryButtons = Array.isArray(galleryButtonsRaw)
        ? galleryButtonsRaw.filter((button) => button instanceof HTMLElement)
        : [];

    if (galleryButtons.length <= 1) return;

    const currentIndex = selected_gallery_index?.() ?? -1;
    if (currentIndex === -1) return;

    const nextButton = /** @type {HTMLElement} */ (galleryButtons[negmod(currentIndex + offset, galleryButtons.length)]);
    nextButton.click();

    const modalImage = getModalImage();
    const modal = getAppElementById('lightboxModal');
    if (!modalImage || !modal) return;

    const firstChild = nextButton.children[0];
    if (firstChild instanceof HTMLImageElement) {
        modalImage.src = firstChild.src;
        if (modalImage.style.display === 'none') {
            modal.style.setProperty('background-image', `url(${modalImage.src})`);
        }
    }

    setTimeout(() => modal.focus(), 10);
}

function saveImage() {
    const tabTxt2Img = getAppElementById('tab_txt2img');
    const tabImg2Img = getAppElementById('tab_img2img');
    const isTxt2Img = tabTxt2Img && tabTxt2Img.style.display !== 'none';
    const isImg2Img = tabImg2Img && tabImg2Img.style.display !== 'none';

    if (isTxt2Img) {
        getAppElementById('save_txt2img')?.click();
    } else if (isImg2Img) {
        getAppElementById('save_img2img')?.click();
    } else {
        console.error('Missing implementation for saving modal of this type');
    }
}

/** @param {MouseEvent | KeyboardEvent} event */
function modalSaveImage(event) {
    saveImage();
    event.stopPropagation();
}

/** @param {MouseEvent | KeyboardEvent} event */
function modalNextImage(event) {
    modalImageSwitch(1);
    event.stopPropagation();
}

/** @param {MouseEvent | KeyboardEvent} event */
function modalPrevImage(event) {
    modalImageSwitch(-1);
    event.stopPropagation();
}

/** @param {KeyboardEvent} event */
function modalKeyHandler(event) {
    switch (event.key) {
        case 's':
            saveImage();
            event.stopPropagation();
            break;
        case 'ArrowLeft':
            modalImageSwitch(-1);
            event.stopPropagation();
            break;
        case 'ArrowRight':
            modalImageSwitch(1);
            event.stopPropagation();
            break;
        case 'Escape':
            closeModal();
            event.stopPropagation();
            break;
    }
}

/**
 * @param {HTMLImageElement} image
 */
function setupImageForLightbox(image) {
    if (image.dataset.modded) return;

    image.dataset.modded = 'true';
    image.style.cursor = 'pointer';
    image.style.userSelect = 'none';

    image.addEventListener(
        'mousedown',
        (evt) => {
            if (evt.button === 1 && typeof viewerWindow.open === 'function' && image.src) {
                viewerWindow.open(image.src);
                evt.preventDefault();
            }
        },
        true
    );

    image.addEventListener(
        'click',
        (evt) => {
            if (!opts.js_modal_lightbox || evt.button !== 0) return;
            modalZoomSet(getModalImage(), Boolean(opts.js_modal_lightbox_initially_zoomed));
            evt.preventDefault();
            showModal(evt);
        },
        true
    );
}

/**
 * @param {HTMLImageElement | null} modalImage
 * @param {boolean} enable
 */
function modalZoomSet(modalImage, enable) {
    if (modalImage) {
        modalImage.classList.toggle('modalImageFullscreen', !!enable);
    }
}

/** @param {MouseEvent} event */
function modalZoomToggle(event) {
    const modalImage = getModalImage();
    modalZoomSet(modalImage, modalImage ? !modalImage.classList.contains('modalImageFullscreen') : false);
    event.stopPropagation();
}

/** @param {MouseEvent} event */
function modalLivePreviewToggle(event) {
    const toggleBtn = getAppElementById('modal_toggle_live_preview');
    opts.js_live_preview_in_modal_lightbox = !opts.js_live_preview_in_modal_lightbox;
    if (toggleBtn) {
        toggleBtn.innerHTML = opts.js_live_preview_in_modal_lightbox ? '&#x1F5C7;' : '&#x1F5C6;';
    }
    event.stopPropagation();
}

/** @param {MouseEvent} event */
function modalTileImageToggle(event) {
    const modalImage = getModalImage();
    const modal = getAppElementById('lightboxModal');
    if (!modalImage || !modal) return;

    const isTiling = modalImage.style.display === 'none';
    if (isTiling) {
        modalImage.style.display = 'block';
        modal.style.setProperty('background-image', 'none');
    } else {
        modalImage.style.display = 'none';
        modal.style.setProperty('background-image', `url(${modalImage.src})`);
    }

    event.stopPropagation();
}

onAfterUiUpdate(() => {
    const previews = gradioApp().querySelectorAll('.gradio-gallery > button > button > img, .gradio-gallery > .livePreview');
    previews.forEach((node) => {
        if (node instanceof HTMLImageElement) {
            setupImageForLightbox(node);
        }
    });
    updateOnBackgroundChange();
});

function buildModal() {
    const modal = document.createElement('div');
    modal.id = 'lightboxModal';
    modal.tabIndex = 0;
    modal.addEventListener('click', () => closeModal());
    modal.addEventListener('keydown', modalKeyHandler, true);

    const modalControls = document.createElement('div');
    modalControls.className = 'modalControls gradio-container';
    modal.append(modalControls);

    const modalZoom = document.createElement('span');
    modalZoom.className = 'modalZoom cursor';
    modalZoom.innerHTML = '&#10529;';
    modalZoom.addEventListener('click', modalZoomToggle, true);
    modalZoom.title = 'Toggle zoomed view';
    modalControls.appendChild(modalZoom);

    const modalTileImage = document.createElement('span');
    modalTileImage.className = 'modalTileImage cursor';
    modalTileImage.innerHTML = '&#8862;';
    modalTileImage.addEventListener('click', modalTileImageToggle, true);
    modalTileImage.title = 'Preview tiling';
    modalControls.appendChild(modalTileImage);

    const modalSave = document.createElement('span');
    modalSave.className = 'modalSave cursor';
    modalSave.id = 'modal_save';
    modalSave.innerHTML = '&#x1F5AB;';
    modalSave.addEventListener('click', modalSaveImage, true);
    modalSave.title = 'Save Image(s)';
    modalControls.appendChild(modalSave);

    const modalToggleLivePreview = document.createElement('span');
    modalToggleLivePreview.className = 'modalToggleLivePreview cursor';
    modalToggleLivePreview.id = 'modal_toggle_live_preview';
    modalToggleLivePreview.innerHTML = '&#x1F5C6;';
    modalToggleLivePreview.addEventListener('click', modalLivePreviewToggle, true);
    modalToggleLivePreview.title = 'Toggle live preview';
    modalControls.appendChild(modalToggleLivePreview);

    const modalClose = document.createElement('span');
    modalClose.className = 'modalClose cursor';
    modalClose.innerHTML = '&times;';
    modalClose.addEventListener('click', () => closeModal(), true);
    modalClose.title = 'Close image viewer';
    modalControls.appendChild(modalClose);

    const modalImage = document.createElement('img');
    modalImage.id = 'modalImage';
    modalImage.tabIndex = 0;
    modalImage.addEventListener('click', () => closeModal());
    modalImage.addEventListener('keydown', modalKeyHandler, true);
    modal.appendChild(modalImage);

    const modalPrev = document.createElement('a');
    modalPrev.className = 'modalPrev';
    modalPrev.innerHTML = '&#10094;';
    modalPrev.tabIndex = 0;
    modalPrev.addEventListener('click', modalPrevImage, true);
    modalPrev.addEventListener('keydown', modalKeyHandler, true);
    modal.appendChild(modalPrev);

    const modalNext = document.createElement('a');
    modalNext.className = 'modalNext';
    modalNext.innerHTML = '&#10095;';
    modalNext.tabIndex = 0;
    modalNext.addEventListener('click', modalNextImage, true);
    modalNext.addEventListener('keydown', modalKeyHandler, true);
    modal.appendChild(modalNext);

    try {
        gradioApp().appendChild(modal);
    } catch (error) {
        console.error('Failed to append modal to gradioApp, falling back to document.body', error);
        document.body.appendChild(modal);
    }
}

document.addEventListener('DOMContentLoaded', buildModal);
