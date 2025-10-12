// various hints and extra info for the settings tab
/*
 DevNotes (2025-10-12)
 - Purpose: Render comentários antes/depois de itens nas abas de Settings.
 - Safety: Lê opts._comments_* com guards; evita rebind; insere spans com whitespace.
*/

var settingsHintsSetup = false;

/**
 * @param {string} id
 * @returns {HTMLElement | null}
 */
function getSettingElement(id) {
    const root = gradioApp();
    if ('getElementById' in root && typeof root.getElementById === 'function') {
        const element = root.getElementById(id);
        if (element instanceof HTMLElement) return element;
    }
    const fallback = document.getElementById(id);
    return fallback instanceof HTMLElement ? fallback : null;
}

onOptionsChanged(function() {
    if (settingsHintsSetup) return;
    settingsHintsSetup = true;

    const commentsBefore = typeof opts._comments_before === 'object' && opts._comments_before ? opts._comments_before : {};
    const commentsAfter = typeof opts._comments_after === 'object' && opts._comments_after ? opts._comments_after : {};

    gradioApp().querySelectorAll('#settings [id^=setting_]').forEach(function(div) {
        if (!(div instanceof HTMLElement)) return;
        var name = div.id.substring(8);
        var commentBefore = commentsBefore[name];
        var commentAfter = commentsAfter[name];

        if (!commentBefore && !commentAfter) return;

        var span = null;
        if (div.classList.contains('gradio-checkbox')) {
            span = div.querySelector('label span');
        } else if (div.classList.contains('gradio-checkboxgroup')) {
            const firstSpan = div.querySelector('span');
            span = firstSpan ? firstSpan.firstChild : null;
        } else if (div.classList.contains('gradio-radio')) {
            const firstSpan = div.querySelector('span');
            span = firstSpan ? firstSpan.firstChild : null;
        } else {
            var elem = div.querySelector('label span');
            if (elem) span = elem.firstChild;
        }

        if (!span) return;

        var parent = span.parentElement;
        if (!parent) return;

        if (commentBefore) {
            var comment = document.createElement('DIV');
            comment.className = 'settings-comment';
            comment.innerHTML = commentBefore;
            parent.insertBefore(document.createTextNode('\xa0'), span);
            parent.insertBefore(comment, span);
            parent.insertBefore(document.createTextNode('\xa0'), span);
        }
        if (commentAfter) {
            var commentAfterDiv = document.createElement('DIV');
            commentAfterDiv.className = 'settings-comment';
            commentAfterDiv.innerHTML = commentAfter;
            parent.insertBefore(commentAfterDiv, span.nextSibling);
            parent.insertBefore(document.createTextNode('\xa0'), span.nextSibling);
        }
    });
});

function settingsHintsShowQuicksettings() {
    requestGet("./internal/quicksettings-hint", {}, function(data) {
        const rows = Array.isArray(data) ? data : [];
        var table = document.createElement('table');
        table.className = 'popup-table';

        rows.forEach(function(row) {
            if (!row || typeof row !== 'object') return;
            const name = String(/** @type {{ name?: string }} */ (row).name ?? '');
            const label = String(/** @type {{ label?: string }} */ (row).label ?? '');
            var tr = document.createElement('tr');
            var td = document.createElement('td');
            td.textContent = name;
            tr.appendChild(td);

            td = document.createElement('td');
            td.textContent = label;
            tr.appendChild(td);

            table.appendChild(tr);
        });

        popup(table);
    }, function() {});
}
