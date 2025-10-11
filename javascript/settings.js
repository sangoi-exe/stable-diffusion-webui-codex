let settingsExcludeTabsFromShowAll = {
    settings_tab_defaults: 1,
    settings_tab_sysinfo: 1,
    settings_tab_actions: 1,
    settings_tab_licenses: 1,
};

function settingsShowAllTabs() {
    gradioApp().querySelectorAll('#settings > div').forEach(function(elem) {
        if (settingsExcludeTabsFromShowAll[elem.id]) return;

        elem.style.display = "block";
    });
}

function settingsShowOneTab() {
    gradioApp().querySelector('#settings_show_one_page').click();
}

function attachSettingsSearch() {
    var root = gradioApp();
    if (!root) return;

    var edit = root.querySelector('#settings_search');
    var editTextarea = root.querySelector('#settings_search > label > input');
    var buttonShowAllPages = root.getElementById('settings_show_all_pages');
    var settings_tabs = root.querySelector('#settings div');

    if (!edit || !editTextarea || !buttonShowAllPages || !settings_tabs) {
        return;
    }

    if (edit.dataset.bound === 'true') {
        return;
    }
    edit.dataset.bound = 'true';

    onEdit('settingsSearch', editTextarea, 250, function() {
        var searchText = (editTextarea.value || "").trim().toLowerCase();

        root.querySelectorAll('#settings > div[id^=settings_] div[id^=column_settings_] > *').forEach(function(elem) {
            var visible = elem.textContent.trim().toLowerCase().indexOf(searchText) !== -1;
            elem.style.display = visible ? "" : "none";
        });

        if (searchText !== "") {
            settingsShowAllTabs();
        } else {
            settingsShowOneTab();
        }
    });

    if (settings_tabs.firstChild !== edit) {
        settings_tabs.insertBefore(edit, settings_tabs.firstChild || null);
    }
    if (!buttonShowAllPages.dataset.bound) {
        buttonShowAllPages.addEventListener("click", settingsShowAllTabs);
        buttonShowAllPages.dataset.bound = 'true';
    }
    if (settings_tabs.lastChild !== buttonShowAllPages) {
        settings_tabs.appendChild(buttonShowAllPages);
    }
}

onUiLoaded(attachSettingsSearch);
onUiUpdate(attachSettingsSearch);


onOptionsChanged(function() {
    if (gradioApp().querySelector('#settings .settings-category')) return;

    var sectionMap = {};
    gradioApp().querySelectorAll('#settings > div > button').forEach(function(x) {
        sectionMap[x.textContent.trim()] = x;
    });

    opts._categories.forEach(function(x) {
        var section = localization[x[0]] ?? x[0];
        var category = localization[x[1]] ?? x[1];

        var span = document.createElement('SPAN');
        span.textContent = category;
        span.className = 'settings-category';

        var sectionElem = sectionMap[section];
        if (!sectionElem) return;

        sectionElem.parentElement.insertBefore(span, sectionElem);
    });
});
