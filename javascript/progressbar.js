"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Queue progress UI + optional live preview with safe wake-lock handling.
 - Contract: requestProgress(id, container, gallery, atEnd, onProgress?, timeout?)
 - Safety: Guards for missing nodes; wakeLock feature-detected; robust JSON parsing.
*/

/**
 * @typedef {{
 *   progress?: number;
 *   eta?: number;
 *   completed?: boolean;
 *   textinfo?: string;
 *   active?: boolean;
 *   queued?: boolean;
 *   id_live_preview?: string;
 *   live_preview?: string;
 * }} ProgressResponse
 */

function rememberGallerySelection() {}
function getGallerySelectedIndex() {}

/**
 * @param {string} url
 * @param {Record<string, unknown>} data
 * @param {(response: any) => void} handler
 * @param {() => void} errorHandler
 */
function request(url, data, handler, errorHandler) {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', url, true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.onreadystatechange = function onReadyStateChange() {
        if (xhr.readyState !== 4) return;
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
    };
    xhr.send(JSON.stringify(data));
}

/** @param {number} value */
function pad2(value) {
    return value < 10 ? `0${value}` : String(value);
}

/** @param {number} seconds */
function formatTime(seconds) {
    if (seconds > 3600) {
        return `${pad2(Math.floor(seconds / 3600))}:${pad2(Math.floor(seconds / 60) % 60)}:${pad2(Math.floor(seconds) % 60)}`;
    }
    if (seconds > 60) {
        return `${pad2(Math.floor(seconds / 60))}:${pad2(Math.floor(seconds) % 60)}`;
    }
    return `${Math.floor(seconds)}s`;
}

/** @type {string} */
let originalAppTitle = document.title;
onUiLoaded(() => {
    originalAppTitle = document.title;
});

/**
 * @param {string} progress
 */
function setTitle(progress) {
    let title = originalAppTitle;
    if (opts.show_progress_in_title && progress) {
        title = `[${progress.trim()}] ${title}`;
    }
    if (document.title !== title) {
        document.title = title;
    }
}

function randomId() {
    return `task(${Math.random().toString(36).slice(2, 7)}${Math.random().toString(36).slice(2, 7)}${Math.random().toString(36).slice(2, 7)})`;
}

/**
 * @param {string} idTask
 * @param {HTMLElement} progressbarContainer
 * @param {HTMLElement | null} gallery
 * @param {() => void} atEnd
 * @param {(response: ProgressResponse) => void} [onProgress]
 * @param {number} [inactivityTimeout]
 */
function requestProgress(idTask, progressbarContainer, gallery, atEnd, onProgress, inactivityTimeout = 40) {
    const dateStart = new Date();
    let wasEverActive = false;
    const parentProgressbar = progressbarContainer.parentElement;
    if (!parentProgressbar) return;

    /** @type {WakeLockSentinel | null} */
    /** @type {WakeLockSentinel | null} */
    let wakeLock = null;

    const requestWakeLock = async () => {
        if (!opts.prevent_screen_sleep_during_generation || wakeLock || !navigator.wakeLock) return;
        try {
            wakeLock = await navigator.wakeLock.request('screen');
        } catch (error) {
            console.error('Wake Lock is not supported.', error);
        }
    };

    const releaseWakeLock = async () => {
        if (!opts.prevent_screen_sleep_during_generation || !wakeLock) return;
        try {
            await wakeLock.release();
        } catch (error) {
            console.error('Wake Lock release failed', error);
        }
        wakeLock = null;
    };

    /** @type {HTMLDivElement | null} */
    let divProgress = document.createElement('div');
    divProgress.className = 'progressDiv';
    divProgress.style.display = opts.show_progressbar ? 'block' : 'none';

    const divInner = document.createElement('div');
    divInner.className = 'progress';
    divProgress.appendChild(divInner);
    parentProgressbar.insertBefore(divProgress, progressbarContainer);

    /** @type {HTMLElement | null} */
    let livePreview = null;

    const removeProgressBar = () => {
        releaseWakeLock();
        if (!divProgress) return;
        setTitle('');
        parentProgressbar.removeChild(divProgress);
        if (gallery && livePreview) gallery.removeChild(livePreview);
        atEnd();
        divProgress = null;
    };

    const refreshProgress = () => {
        const refreshPeriod = Number(opts.live_preview_refresh_period ?? 500) || 500;
        requestWakeLock();
        if (!divProgress) return;
        request(
            './internal/progress',
            { id_task: idTask, live_preview: false },
            /** @param {ProgressResponse} res */ (res) => {
                if (res.completed) {
                    removeProgressBar();
                    return;
                }

                let progressText = '';

                const progressValue = res.progress ?? 0;
                divInner.style.width = `${progressValue * 100.0}%`;
                divInner.style.background = progressValue ? '' : 'transparent';

                if (progressValue > 0) {
                    progressText = `${(progressValue * 100.0).toFixed(0)}%`;
                }

                if (res.eta) {
                    progressText += ` ETA: ${formatTime(res.eta)}`;
                }

                setTitle(progressText);

                if (res.textinfo && !res.textinfo.includes('\n')) {
                    progressText = `${res.textinfo} ${progressText}`;
                }

                divInner.textContent = progressText;

                const elapsedFromStart = (Date.now() - dateStart.getTime()) / 1000;
                if (res.active) wasEverActive = true;

                if ((!res.active && wasEverActive) || (elapsedFromStart > inactivityTimeout && !res.queued && !res.active)) {
                    removeProgressBar();
                    return;
                }

                if (onProgress) onProgress(res);

                window.setTimeout(refreshProgress, refreshPeriod);
            },
            removeProgressBar
        );
    };

    const refreshLivePreview = (liveId = 0) => {
        const refreshPeriod = Number(opts.live_preview_refresh_period ?? 500) || 500;
        request(
            './internal/progress',
            { id_task: idTask, id_live_preview: liveId },
            /** @param {ProgressResponse} res */ (res) => {
                if (!divProgress) return;
                if (res.live_preview && gallery) {
                    const img = new Image();
                    img.onload = () => {
                        if (!livePreview) {
                            livePreview = document.createElement('div');
                            livePreview.className = 'livePreview';
                            gallery.insertBefore(livePreview, gallery.firstElementChild);
                        }
                        livePreview.appendChild(img);
                        while (livePreview.childElementCount > 2) {
                            const first = livePreview.firstElementChild;
                            if (first) livePreview.removeChild(first);
                        }
                    };
                    img.src = res.live_preview;
                }
                const nextLiveId = typeof res.id_live_preview === 'number' ? res.id_live_preview : Number(liveId) || 0;
                window.setTimeout(() => refreshLivePreview(nextLiveId), refreshPeriod);
            },
            removeProgressBar
        );
    };

    refreshProgress();
    if (gallery) refreshLivePreview();
}
