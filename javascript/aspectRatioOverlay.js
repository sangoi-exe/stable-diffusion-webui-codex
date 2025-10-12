/**
 * Interactive aspect-ratio overlay for img2img canvases.
 * Adds strong typing to avoid silent NaN propagation into style math.
 */

/** @type {number} */
let currentWidth = 0;
/** @type {number} */
let currentHeight = 0;
/** @type {number | null} */
let arFrameTimeout = null;

/**
 * Update the overlay rectangle when width/height sliders move.
 * @param {Event} event
 * @param {boolean} isWidth
 * @param {boolean} isHeight
 */
function dimensionChange(event, isWidth, isHeight) {
    const input = event.target instanceof HTMLInputElement ? event.target : null;
    if (!input) {
        return;
    }

    if (isWidth) {
        currentWidth = Number(input.value);
    }
    if (isHeight) {
        currentHeight = Number(input.value);
    }

    const tabImg2img = /** @type {HTMLElement | null} */ (gradioApp().querySelector("#tab_img2img"));
    if (!tabImg2img || tabImg2img.style.display !== "block") {
        return;
    }

    /** @type {HTMLImageElement | null} */
    let targetElement = null;

    const tabIndex = get_tab_index('mode_img2img');
    if (tabIndex === 0) { // img2img
        targetElement = /** @type {HTMLImageElement | null} */ (gradioApp().querySelector('#img2img_image div[class=forge-image-container] img'));
    } else if (tabIndex === 1) { // Sketch
        targetElement = /** @type {HTMLImageElement | null} */ (gradioApp().querySelector('#img2img_sketch div[class=forge-image-container] img'));
    } else if (tabIndex === 2) { // Inpaint
        targetElement = /** @type {HTMLImageElement | null} */ (gradioApp().querySelector('#img2maskimg div[class=forge-image-container] img'));
    } else if (tabIndex === 3) { // Inpaint sketch
        targetElement = /** @type {HTMLImageElement | null} */ (gradioApp().querySelector('#inpaint_sketch div[class=forge-image-container] img'));
    } else if (tabIndex === 4) { // Inpaint upload
        targetElement = /** @type {HTMLImageElement | null} */ (gradioApp().querySelector('#img_inpaint_base div[data-testid=image] img'));
    }

    if (!targetElement || currentWidth <= 0 || currentHeight <= 0) {
        return;
    }

    let arPreviewRect = /** @type {HTMLElement | null} */ (gradioApp().querySelector('#imageARPreview'));
    if (!arPreviewRect) {
        arPreviewRect = document.createElement('div');
        arPreviewRect.id = "imageARPreview";
        gradioApp().appendChild(arPreviewRect);
    }

    const viewportOffset = targetElement.getBoundingClientRect();
    const viewportscale = Math.min(
        targetElement.clientWidth / targetElement.width,
        targetElement.clientHeight / targetElement.height
    );

    const scaledx = targetElement.width * viewportscale;
    const scaledy = targetElement.height * viewportscale;

    const clientRectTop = viewportOffset.top + window.scrollY;
    const clientRectLeft = viewportOffset.left + window.scrollX;
    const clientRectCentreY = clientRectTop + (targetElement.clientHeight / 2);
    const clientRectCentreX = clientRectLeft + (targetElement.clientWidth / 2);

    const arscale = Math.min(scaledx / currentWidth, scaledy / currentHeight);
    const arscaledx = currentWidth * arscale;
    const arscaledy = currentHeight * arscale;

    const arRectTop = clientRectCentreY - (arscaledy / 2);
    const arRectLeft = clientRectCentreX - (arscaledx / 2);

    arPreviewRect.style.top = arRectTop + 'px';
    arPreviewRect.style.left = arRectLeft + 'px';
    arPreviewRect.style.width = arscaledx + 'px';
    arPreviewRect.style.height = arscaledy + 'px';

    if (arFrameTimeout !== null) {
        clearTimeout(arFrameTimeout);
    }
    arFrameTimeout = window.setTimeout(function() {
        arPreviewRect.style.display = 'none';
        arFrameTimeout = null;
    }, 2000);

    arPreviewRect.style.display = 'block';
}

onAfterUiUpdate(function() {
    const arPreviewRect = /** @type {HTMLElement | null} */ (gradioApp().querySelector('#imageARPreview'));
    if (arPreviewRect) {
        arPreviewRect.style.display = 'none';
    }

    const tabImg2img = /** @type {HTMLElement | null} */ (gradioApp().querySelector("#tab_img2img"));
    if (!tabImg2img || tabImg2img.style.display !== "block") {
        return;
    }

    const inputs = gradioApp().querySelectorAll('input');
    inputs.forEach(function(node) {
        const input = /** @type {HTMLInputElement} */ (node);
        const parent = input.parentElement;
        const grandParent = parent?.parentElement?.parentElement ?? null;

        const isWidth = !!parent && (
            (parent.id === "img2img_width" && input.type === "range") ||
            (grandParent?.id === "img2img_width" && input.type === "number")
        );
        const isHeight = !!parent && (
            (parent.id === "img2img_height" && input.type === "range") ||
            (grandParent?.id === "img2img_height" && input.type === "number")
        );

        if ((isWidth || isHeight) && !input.classList.contains('scrollwatch')) {
            input.addEventListener('input', function(ev) {
                dimensionChange(ev, isWidth, isHeight);
            });
            input.classList.add('scrollwatch');
        }

        if (isWidth) {
            currentWidth = Number(input.value);
        }
        if (isHeight) {
            currentHeight = Number(input.value);
        }
    });
});
