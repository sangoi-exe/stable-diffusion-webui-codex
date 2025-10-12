"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Gamepad/VR wheel controls for lightbox navigation.
 - Behaviour: Polls axes X; debounced repeat; wheel maps to prev/next.
 - Safety: Optionality on opts and gamepad; no coupling to UI internals.
*/

/** @type {Array<number | undefined>} */
const activeGamepadIntervals = [];

/**
 * @param {() => boolean} predicate
 * @param {number} timeoutMs
 */
function sleepUntil(predicate, timeoutMs) {
    return new Promise((resolve) => {
        const start = Date.now();
        const timer = window.setInterval(() => {
            if (predicate() || Date.now() - start > timeoutMs) {
                window.clearInterval(timer);
                resolve(undefined);
            }
        }, 20);
    });
}

window.addEventListener('gamepadconnected', (event) => {
    if (!(event instanceof GamepadEvent)) return;
    const index = event.gamepad.index;
    let isWaiting = false;

    const interval = window.setInterval(async () => {
        if (!opts.js_modal_lightbox_gamepad || isWaiting) return;

        const gamepad = navigator.getGamepads()[index];
        const xValue = gamepad?.axes?.[0] ?? 0;

        if (xValue <= -0.3) {
            modalImageSwitch(-1);
            isWaiting = true;
        } else if (xValue >= 0.3) {
            modalImageSwitch(1);
            isWaiting = true;
        }

        if (isWaiting) {
            const repeatDelay = Number(opts.js_modal_lightbox_gamepad_repeat ?? 400);
            await sleepUntil(() => {
                const current = navigator.getGamepads()[index];
                const value = current?.axes?.[0] ?? 0;
                return value < 0.3 && value > -0.3;
            }, repeatDelay);
            isWaiting = false;
        }
    }, 10);

    activeGamepadIntervals[index] = interval;
});

window.addEventListener('gamepaddisconnected', (event) => {
    if (!(event instanceof GamepadEvent)) return;
    const interval = activeGamepadIntervals[event.gamepad.index];
    if (interval !== undefined) {
        window.clearInterval(interval);
        activeGamepadIntervals[event.gamepad.index] = undefined;
    }
});

// Support for pointer wheel on VR-style controllers.
let isWheelScrolling = false;
window.addEventListener('wheel', (event) => {
    if (!opts.js_modal_lightbox_gamepad || isWheelScrolling) return;
    isWheelScrolling = true;

    if (event.deltaX <= -0.6) {
        modalImageSwitch(-1);
    } else if (event.deltaX >= 0.6) {
        modalImageSwitch(1);
    }

    const repeatDelay = Number(opts.js_modal_lightbox_gamepad_repeat ?? 400);
    window.setTimeout(() => {
        isWheelScrolling = false;
    }, repeatDelay);
});
