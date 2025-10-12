"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Manage the Extensions page – collect enable/disable/update, status refresh, and index installs.
 - Safety: All DOM reads are guarded; throws when page not ready to avoid partial state writes.
 - Public hooks: extensions_apply, extensions_check, install_extension_from_index,
   config_state_confirm_restore, toggle_all_extensions, toggle_extension.
*/

/**
 * Utilities for managing the extensions page (enable/disable/update toggles).
 */

/**
 * @returns {NodeListOf<HTMLInputElement>}
 */
function getExtensionCheckboxes() {
    return gradioApp().querySelectorAll('#extensions input[type="checkbox"]');
}

/**
 * @param {string} id
 * @returns {HTMLElement | null}
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

/**
 * Collect enabled/disabled/update selections from the checkbox list.
 * @param {unknown} disableList
 * @param {unknown} updateList
 * @param {boolean} disableAll
 * @returns {[string, string, boolean]}
*/
function extensions_apply(disableList, updateList, disableAll) {
    void disableList;
    void updateList;

    /** @type {string[]} */
    const disable = [];
    /** @type {string[]} */
    const update = [];

    const inputs = getExtensionCheckboxes();
    if (inputs.length === 0) {
        throw new Error('Extensions page not yet loaded.');
    }

    inputs.forEach((input) => {
        if (!(input instanceof HTMLInputElement)) return;
        if (input.name.startsWith('enable_') && !input.checked) {
            disable.push(input.name.substring(7));
        }
        if (input.name.startsWith('update_') && input.checked) {
            update.push(input.name.substring(7));
        }
    });

    restart_reload();

    return [JSON.stringify(disable), JSON.stringify(update), Boolean(disableAll)];
}

/**
 * Determine which extensions are disabled and kick off the status refresh.
 * @returns {[string, string]}
 */
function extensions_check() {
    /** @type {string[]} */
    const disable = [];

    getExtensionCheckboxes().forEach((input) => {
        if (input.name.startsWith('enable_') && !input.checked) {
            disable.push(input.name.substring(7));
        }
    });

    gradioApp().querySelectorAll('#extensions .extension_status').forEach((element) => {
        if (element instanceof HTMLElement) {
            element.innerHTML = 'Loading...';
        }
    });

    const id = randomId();
    const progressTarget = getAppElementById('extensions_installed_html');
    if (progressTarget instanceof HTMLElement) {
        requestProgress(id, progressTarget, null, () => {});
    }

    return [id, JSON.stringify(disable)];
}

/**
 * Trigger installation from the extensions index.
 * @param {HTMLInputElement | HTMLButtonElement} button
 * @param {string} url
 */
function install_extension_from_index(button, url) {
    button.disabled = true;
    if ('value' in button) {
        button.value = 'Installing...';
    }

    const textarea = gradioApp().querySelector('#extension_to_install textarea');
    if (textarea instanceof HTMLTextAreaElement) {
        textarea.value = url;
        updateInput(textarea);
    }

    const installButton = gradioApp().querySelector('#install_extension_button');
    if (installButton instanceof HTMLElement) {
        installButton.click();
    }
}

/**
 * Confirm restoration of saved config states.
 * @param {unknown} _
 * @param {string} config_state_name
 * @param {string} config_restore_type
 * @returns {[boolean, string, string]}
 */
function config_state_confirm_restore(_, config_state_name, config_restore_type) {
    if (config_state_name === 'Current') {
        return [false, config_state_name, config_restore_type];
    }

    let restored;
    if (config_restore_type === 'extensions') {
        restored = 'all saved extension versions';
    } else if (config_restore_type === 'webui') {
        restored = 'the webui version';
    } else {
        restored = 'the webui version and all saved extension versions';
    }

    const confirmed = window.confirm(
        `Are you sure you want to restore from this state?\nThis will reset ${restored}.`
    );

    if (confirmed) {
        restart_reload();
        gradioApp().querySelectorAll('#extensions .extension_status').forEach((element) => {
            if (element instanceof HTMLElement) {
                element.innerHTML = 'Loading...';
            }
        });
    }

    return [confirmed, config_state_name, config_restore_type];
}

/**
 * Toggle all extension checkboxes based on the master toggle.
 * @param {Event} event
 */
function toggle_all_extensions(event) {
    const checkbox = event.target instanceof HTMLInputElement ? event.target : null;
    const checked = checkbox ? checkbox.checked : false;

    getExtensionCheckboxes().forEach((input) => {
        if (input.classList.contains('extension_toggle')) {
            input.checked = checked;
        }
    });
}

/**
 * Update the master toggle when an individual extension is toggled.
 */
function toggle_extension() {
    let allExtensionsToggled = true;
    getExtensionCheckboxes().forEach((input) => {
        if (allExtensionsToggled && input.classList.contains('extension_toggle') && !input.checked) {
            allExtensionsToggled = false;
        }
    });

    const masterToggle = gradioApp().querySelector('#extensions .all_extensions_toggle');
    if (masterToggle instanceof HTMLInputElement) {
        masterToggle.checked = allExtensionsToggled;
    }
}
