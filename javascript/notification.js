"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Desktop notification + audio cue when new images land in the gallery.
 - Safety: Permission checks; volume clamped; dedup count via Set; requires Notification API.
*/

/**
 * Monitor the gallery and send a browser notification when a new image appears.
 */

/** @type {string | null} */
let lastHeadImg = null;
/** @type {HTMLButtonElement | null} */
let notificationButton = null;

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

onAfterUiUpdate(() => {
    if (!notificationButton) {
        const button = getAppElementById('request_notifications');
        if (button instanceof HTMLButtonElement) {
            notificationButton = button;
            notificationButton.addEventListener('click', () => {
                void Notification.requestPermission();
            }, true);
        }
    }

    const galleryPreviews = Array.from(gradioApp().querySelectorAll('div[id^="tab_"] div[id$="_results"] .thumbnail-item > img'))
        .filter((img) => img instanceof HTMLImageElement);

    if (!galleryPreviews.length) return;

    const headImg = /** @type {HTMLImageElement} */ (galleryPreviews[0]).src;
    if (!headImg || headImg === lastHeadImg) return;
    lastHeadImg = headImg;

    const audioContainer = gradioApp().querySelector('#audio_notification #waveform > div');
    const audio = audioContainer && audioContainer instanceof HTMLElement
        ? audioContainer.shadowRoot?.querySelector('audio')
        : null;
    if (audio instanceof HTMLAudioElement) {
        const volume = Number(opts.notification_volume ?? 100) / 100;
        audio.volume = Number.isFinite(volume) ? Math.min(Math.max(volume, 0), 1) : 1;
        void audio.play();
    }

    if (document.hasFocus()) return;

    const uniqueImages = new Set(galleryPreviews.map((img) => /** @type {HTMLImageElement} */ (img).src));
    const gridAdjustment = opts.return_grid ? 1 : 0;
    const count = uniqueImages.size > 1 ? Math.max(uniqueImages.size - gridAdjustment, 1) : 1;

    if (Notification.permission !== 'granted') return;

    const options = /** @type {NotificationOptions & { image?: string }} */ ({
        body: `Generated ${count} image${count !== 1 ? 's' : ''}`,
        icon: headImg,
    });
    options.image = headImg;

    const notification = new Notification('Stable Diffusion', options);
    notification.onclick = () => {
        window.focus();
        notification.close();
    };
});
