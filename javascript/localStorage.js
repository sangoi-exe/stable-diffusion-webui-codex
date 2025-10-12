
"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Thin wrappers for localStorage with try/catch so failed quota or privacy modes don’t break flows.
 - API: localSet(key, value), localGet(key, fallback?), localRemove(key).
*/

/**
 * @param {string} key
 * @param {string} value
 */
function localSet(key, value) {
    try {
        localStorage.setItem(key, value);
    } catch (e) {
        console.warn(`Failed to save ${key} to localStorage: ${e}`);
    }
}

/**
 * @param {string} key
 * @param {string | null} [fallback]
 * @returns {string | null}
 */
function localGet(key, fallback = null) {
    try {
        const value = localStorage.getItem(key);
        return value !== null ? value : fallback;
    } catch (e) {
        console.warn(`Failed to load ${key} from localStorage: ${e}`);
    }

    return fallback;
}

/**
 * @param {string} key
 */
function localRemove(key) {
    try {
        localStorage.removeItem(key);
    } catch (e) {
        console.warn(`Failed to remove ${key} from localStorage: ${e}`);
    }
}
