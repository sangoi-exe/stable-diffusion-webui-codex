"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Keep generation params panel in sync with current gallery selection.
 - Safety: Button resolution via helper; keyboard arrows update info.
*/

// attaches listeners to the txt2img and img2img galleries to update displayed generation param text when the image changes

/** @type {HTMLElement | null} */
let txt2img_gallery = null;
/** @type {HTMLElement | null} */
let img2img_gallery = null;
/** @type {HTMLElement | null} */
let modal = null;

/**
 * @param {string} tabName
 * @returns {HTMLElement | null}
 */
function getGallery(tabName) {
    const gallery = gradioApp().querySelector(`#${tabName}_gallery`);
    return gallery instanceof HTMLElement ? gallery : null;
}

/**
 * @param {string} tabName
 * @returns {HTMLButtonElement | null}
 */
function getGenerationInfoButton(tabName) {
    const root = gradioApp();
    let button = null;
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        button = root.getElementById(`${tabName}_generation_info_button`);
    }
    if (!(button instanceof HTMLButtonElement)) {
        button = document.getElementById(`${tabName}_generation_info_button`);
    }
    return button instanceof HTMLButtonElement ? button : null;
}

/**
 * Attach listeners to the gallery for a given tab.
 * @param {string} tabName
 * @returns {HTMLElement | null}
 */
function attachGalleryListeners(tabName) {
    const gallery = getGallery(tabName);
    const button = getGenerationInfoButton(tabName);
    if (!gallery || !button) return gallery;

    gallery.addEventListener('click', () => button.click());
    gallery.addEventListener('keydown', (event) => {
        if (!(event instanceof KeyboardEvent)) return;
        if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
            button.click();
        }
    });
    return gallery;
}

const modalObserver = new MutationObserver((mutations) => {
    mutations.forEach((mutationRecord) => {
        if (!(mutationRecord.target instanceof HTMLElement)) return;
        const selectedTabButton = gradioApp().querySelector('#tabs div button.selected');
        const selectedTab = selectedTabButton instanceof HTMLElement ? selectedTabButton.innerText : '';
        if (
            mutationRecord.target.style.display === 'none' &&
            (selectedTab === 'txt2img' || selectedTab === 'img2img')
        ) {
            const button = getGenerationInfoButton(selectedTab);
            button?.click();
        }
    });
});

onAfterUiUpdate(() => {
    if (!txt2img_gallery) {
        txt2img_gallery = attachGalleryListeners('txt2img');
    }
    if (!img2img_gallery) {
        img2img_gallery = attachGalleryListeners('img2img');
    }
    if (!modal) {
        const root = gradioApp();
        let modalElement = null;
        if ('getElementById' in root && typeof root.getElementById === 'function') {
            modalElement = root.getElementById('lightboxModal');
        }
        if (!(modalElement instanceof HTMLElement)) {
            modalElement = document.getElementById('lightboxModal');
        }
        if (modalElement instanceof HTMLElement) {
            modal = modalElement;
            modalObserver.observe(modalElement, { attributes: true, attributeFilter: ['style'] });
        }
    }
});
