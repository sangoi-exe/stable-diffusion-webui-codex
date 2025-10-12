"use strict";
// @ts-check
/*
 DevNotes (2025-10-12)
 - Purpose: Render profiling table with expandable rows, grouped by "a/b/c" keys.
 - Input: { records: key->seconds }
 - Safety: Typed row creation; numeric formatting; avoid undefined access on parts/value.
*/

/**
 * @param {HTMLTableElement} table
 * @param {'td' | 'th'} cellName
 * @param {Array<string | number | null | undefined>} items
 * @returns {Array<HTMLTableCellElement | null>}
 */
function createRow(table, cellName, items) {
    const tr = document.createElement('tr');
    /** @type {Array<HTMLTableCellElement | null>} */
    const cells = [];

    items.forEach((value, index) => {
        if (value === undefined || value === null) {
            cells.push(null);
            return;
        }

        const cell = document.createElement(cellName);
        cell.textContent = String(value);
        tr.appendChild(cell);
        cells.push(cell);

        let colspan = 1;
        for (let n = index + 1; n < items.length; n++) {
            if (items[n] !== undefined && items[n] !== null) {
                break;
            }
            colspan += 1;
        }

        if (colspan > 1) {
            cell.colSpan = colspan;
        }
    });

    table.appendChild(tr);
    return cells;
}

/**
 * @param {Record<string, number>} data
 * @param {number} [cutoff]
 * @param {string} [sort]
 */
function createVisualizationTable(data, cutoff = 0, sort = '') {
    const table = document.createElement('table');
    table.className = 'popup-table';

    let keys = Object.keys(data);
    if (sort === 'number') {
        keys = keys.sort((a, b) => (data[b] ?? 0) - (data[a] ?? 0));
    } else {
        keys = keys.sort();
    }

    /** @type {Array<{ key: string; parts: string[]; value: number }>} */
    const items = keys.map((key) => ({ key, parts: key.split('/'), value: Number(data[key] ?? 0) }));
    const maxLength = items.reduce((acc, item) => Math.max(acc, item.parts.length), 0);

    const headerCells = createRow(table, 'th', [cutoff === 0 ? 'key' : 'record', cutoff === 0 ? 'value' : 'seconds']);
    const headerCell = headerCells[0];
    if (headerCell) headerCell.colSpan = maxLength;

    /** @param {string[]} a @param {string[]} b */
    const arraysEqual = (a, b) => a.length === b.length && a.every((value, index) => value === b[index]);

    /**
     * @param {number} level
     * @param {string[]} parent
     * @param {boolean} [hide]
     * @returns {HTMLTableRowElement[]}
     */
    const addLevel = (level, parent, hide = false) => {
        let matching = items.filter((item) => item.parts[level] && !item.parts[level + 1] && arraysEqual(item.parts.slice(0, level), parent));
        if (sort === 'number') {
            matching = matching.sort((a, b) => b.value - a.value);
        } else {
            matching = matching.sort((a, b) => (a.parts[level] || '').localeCompare(b.parts[level] || ''));
        }

        let othersTime = 0;
        /** @type {string[]} */
        const othersList = [];
        /** @type {HTMLTableRowElement[]} */
        const othersRows = [];
        /** @type {HTMLTableRowElement[]} */
        const childrenRows = [];

        matching.forEach((item) => {
            const visible = (cutoff === 0 && !hide) || (!hide && item.value >= cutoff);
            /** @type {Array<string | number>} */
            const cells = [];
            for (let i = 0; i < maxLength; i++) {
                cells.push(item.parts[i] ?? '');
            }
            cells.push(cutoff === 0 ? item.value : item.value.toFixed(3));

            const rowCells = createRow(table, 'td', cells);
            rowCells.slice(0, level).forEach((cell) => {
                if (cell) cell.classList.add('muted');
            });

            const rowElement = rowCells[0]?.parentElement;
            if (!(rowElement instanceof HTMLTableRowElement)) return;

            if (!visible) rowElement.classList.add('hidden');
            if (cutoff === 0 || item.value >= cutoff) {
                childrenRows.push(rowElement);
            } else {
                othersTime += item.value;
                othersList.push(item.parts[level] ?? '');
                othersRows.push(rowElement);
            }

            const childRows = addLevel(level + 1, parent.concat([item.parts[level] ?? '']), true);
            if (childRows.length > 0) {
                const cell = rowCells[level];
                if (cell) {
                    const onclick = () => {
                        cell.classList.remove('link');
                        cell.removeEventListener('click', onclick);
                        childRows.forEach((childRow) => childRow.classList.remove('hidden'));
                    };
                    cell.classList.add('link');
                    cell.addEventListener('click', onclick);
                }
            }
        });

        if (othersTime > 0) {
            /** @type {Array<string | number>} */
            const cells = [];
            for (let i = 0; i < maxLength; i++) {
                cells.push(parent[i] ?? '');
            }
            cells.push(othersTime.toFixed(3));
            cells[level] = 'others';
            const rowCells = createRow(table, 'td', cells);
            rowCells.slice(0, level).forEach((cell) => {
                if (cell) cell.classList.add('muted');
            });

            const cell = rowCells[level];
            const row = cell?.parentElement;
            if (cell && row instanceof HTMLTableRowElement) {
                const onclick = () => {
                    row.classList.add('hidden');
                    cell.classList.remove('link');
                    cell.removeEventListener('click', onclick);
                    othersRows.forEach((r) => r.classList.remove('hidden'));
                };

                cell.title = othersList.join(', ');
                cell.classList.add('link');
                cell.addEventListener('click', onclick);

                if (hide) {
                    row.classList.add('hidden');
                }

                childrenRows.push(row);
            }
        }

        return childrenRows;
    };

    addLevel(0, []);

    return table;
}

/**
 * @param {string} path
 * @param {number} [cutoff]
 */
function showProfile(path, cutoff = 0.05) {
    requestGet(path, {}, (data) => {
        const payload = /** @type {{ records: Record<string, number>; total: number }} */ (data);
        payload.records.total = payload.total;
        const table = createVisualizationTable(payload.records, cutoff, 'number');
        popup(table);
    }, () => {});
}
