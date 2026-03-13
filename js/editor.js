// Data editor modal

let editorCurrentCSV = null;

function openDataEditor() {
    if (!window.rawDataString || window.dataFormat !== 'csv') {
        alert('Editor dát je dostupný len pre CSV formát. Najprv načítajte CSV dáta.');
        return;
    }

    editorCurrentCSV = window.rawDataString;
    const modal = document.getElementById('dataEditorModal');
    modal.classList.add('show');

    // Populate processor select
    populateEditorProcessors();

    // Render the table
    renderEditorTable();
}

function closeDataEditor() {
    document.getElementById('dataEditorModal').classList.remove('show');
}

function editorDownloadData() {
    const data = editorCurrentCSV;
    if (!data || !data.trim()) {
        alert('Žiadne dáta na stiahnutie.');
        return;
    }
    const format = window.dataFormat || 'csv';
    const mimeType = format === 'json' ? 'application/json;charset=utf-8;' : 'text/csv;charset=utf-8;';
    const extension = format === 'json' ? '.json' : '.csv';
    const filename = 'upravene_data' + extension;

    const blob = new Blob([data], { type: mimeType });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

function editorApplyAndClose() {
    if (editorCurrentCSV) {
        // Update the raw data
        window.rawDataString = editorCurrentCSV;

        // Update text area if visible
        const textArea = document.getElementById('dataInput');
        if (textArea) {
            textArea.value = editorCurrentCSV;
        }

        // Re-parse data with updated text
        reparseAfterEdit();
    }
    closeDataEditor();
}

async function reparseAfterEdit() {
    try {
        const format = window.dataFormat;
        const data = window.rawDataString;

        dataLoader = new WasmDataLoaderClass(format);
        const columns = await dataLoader.getAvailableColumns(data);
        window.parsedColumns = columns;

        const targetSelect = document.getElementById('targetColumnSelect');
        const prevTarget = targetSelect.value;
        targetSelect.innerHTML = '';
        columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            targetSelect.appendChild(opt);
        });
        // Restore previous target if still exists
        if (columns.includes(prevTarget)) {
            targetSelect.value = prevTarget;
        }

        buildColumnStatsTable(data, columns, format);

        showStatus('success', `Dáta aktualizované: ${columns.length} stĺpcov.`, 'parseStatus');
    } catch (error) {
        showStatus('error', `Chyba pri aktualizácii: ${error}`, 'parseStatus');
    }
}

function populateEditorProcessors() {
    const select = document.getElementById('editorProcessorSelect');
    select.innerHTML = '<option value="">-- Vyberte procesor --</option>';
    if (availableOptions && availableOptions.processors) {
        availableOptions.processors.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p.name;
            opt.textContent = `${p.name} - ${p.description}`;
            select.appendChild(opt);
        });
    }
    // Clear params
    document.getElementById('editorProcessorParams').innerHTML = '';
}

function updateEditorProcessorParams(processorName) {
    const paramsDiv = document.getElementById('editorProcessorParams');
    paramsDiv.innerHTML = '';

    if (!processorName) return;

    const paramDefs = factory.getProcessorParamDefinitions(processorName);
    if (!paramDefs || paramDefs.length === 0) return;

    paramDefs.forEach(param => {
        const item = document.createElement('div');
        item.className = 'param-item';

        const label = document.createElement('label');
        label.textContent = param.description + ':';
        item.appendChild(label);

        if (param.options && param.options.length > 0) {
            const sel = document.createElement('select');
            sel.id = `editor_param_${param.name}`;
            param.options.forEach(opt => {
                const o = document.createElement('option');
                o.value = opt;
                o.textContent = opt;
                if (opt === param.default_value) o.selected = true;
                sel.appendChild(o);
            });
            item.appendChild(sel);
        } else {
            const input = document.createElement('input');
            input.type = param.param_type === 'number' ? 'number' : 'text';
            input.value = param.default_value;
            input.id = `editor_param_${param.name}`;
            if (param.min !== null && param.min !== undefined) input.min = param.min;
            if (param.max !== null && param.max !== undefined) input.max = param.max;
            item.appendChild(input);
        }

        paramsDiv.appendChild(item);
    });
}

function renderEditorTable() {
    const container = document.getElementById('editorTableContainer');
    if (!editorCurrentCSV) {
        container.innerHTML = '<p style="padding: 30px; color: #6c757d; text-align: center;">Žiadne dáta</p>';
        return;
    }

    try {
        const rawResult = pipeline.getEditableData(editorCurrentCSV, 'csv');
        const result = convertWasmResult(rawResult);
        if (!result || !result.headers || !result.rows) {
            container.innerHTML = '<p style="padding: 30px; color: #dc3545; text-align: center;">Chyba pri spracovaní dát</p>';
            return;
        }
        const headers = result.headers;
        const rows = result.rows;

        // Update column select  
        const colSelect = document.getElementById('editorColumnSelect');
        const prevCol = colSelect.value;
        colSelect.innerHTML = '';
        headers.forEach(h => {
            const opt = document.createElement('option');
            opt.value = h;
            opt.textContent = h;
            colSelect.appendChild(opt);
        });
        if (headers.includes(prevCol)) {
            colSelect.value = prevCol;
        }

        // Build table - klikateľné hlavičky stĺpcov
        let html = '<table class="editor-table"><thead><tr>';
        html += '<th style="width:40px;">#</th>';
        headers.forEach((h, idx) => {
            const escapedH = h.replace(/'/g, "\\'");
            const isSelected = colSelect.value === h;
            html += `<th class="col-header${isSelected ? ' selected-col' : ''}" 
                        onclick="editorSelectColumn('${escapedH}')" 
                        title="Kliknite pre výber stĺpca">${h}</th>`;
        });
        html += '</tr></thead><tbody>';

        const maxShow = Math.min(rows.length, 200);
        const selectedCol = colSelect.value;
        for (let i = 0; i < maxShow; i++) {
            html += '<tr>';
            html += `<td class="row-num">${i + 1}</td>`;
            const row = rows[i];
            headers.forEach((h, colIdx) => {
                const val = row[colIdx] || '';
                const isSelectedCol = h === selectedCol;
                html += `<td class="editable${isSelectedCol ? ' selected-col-cell' : ''}" ondblclick="editorEditCell(this, ${i}, '${h.replace(/'/g, "\\'")}')">
                    ${val}
                </td>`;
            });
            html += '</tr>';
        }

        html += '</tbody></table>';

        if (rows.length > maxShow) {
            html += `<p style="padding: 10px; color: #6c757d; text-align: center; font-style: italic;">
                Zobrazených ${maxShow} z ${rows.length} riadkov
            </p>`;
        }

        container.innerHTML = html;
        updateEditorStatus(`${headers.length} stĺpcov, ${rows.length} riadkov`);
    } catch (error) {
        container.innerHTML = `<p style="padding: 30px; color: #e74c3c; text-align: center;">Chyba: ${error}</p>`;
    }
}

function editorSelectColumn(colName) {
    const colSelect = document.getElementById('editorColumnSelect');
    colSelect.value = colName;
    // Zvýrazni vybraný stĺpec v tabuľke
    document.querySelectorAll('.col-header').forEach(th => {
        th.classList.toggle('selected-col', th.textContent.trim() === colName);
    });
    document.querySelectorAll('.editor-table td.editable').forEach(td => {
        td.classList.remove('selected-col-cell');
    });
    // Zvýrazni bunkám vybraného stĺpca
    const headers = Array.from(document.querySelectorAll('.col-header'));
    const colIdx = headers.findIndex(th => th.textContent.trim() === colName);
    if (colIdx >= 0) {
        document.querySelectorAll('.editor-table tbody tr').forEach(tr => {
            const tds = tr.querySelectorAll('td.editable');
            if (tds[colIdx]) {
                tds[colIdx].classList.add('selected-col-cell');
            }
        });
    }
    // Zobraziť panel procesorov a aktualizovať label
    document.getElementById('editorProcessorPanel').style.display = 'flex';
    document.getElementById('editorSelectedColLabel').textContent = `Stĺpec: ${colName}`;
    updateEditorStatus(`Vybraný stĺpec: ${colName}`);
}


function editorEditCell(td, rowIdx, colName) {
    // Already editing
    if (td.querySelector('input')) return;

    const currentValue = td.textContent.trim();
    const input = document.createElement('input');
    input.type = 'text';
    input.value = currentValue;
    td.textContent = '';
    td.appendChild(input);
    input.focus();
    input.select();

    const finishEdit = async () => {
        const newValue = input.value.trim();
        if (newValue !== currentValue) {
            try {
                const rawResult = pipeline.setCellValue(editorCurrentCSV, rowIdx, colName, newValue);
                const result = convertWasmResult(rawResult);
                editorCurrentCSV = result.csv;
                td.textContent = newValue;
                updateEditorStatus(`Bunka [${rowIdx + 1}, ${colName}] zmenená`);
            } catch (error) {
                td.textContent = currentValue;
                updateEditorStatus(`Chyba: ${error}`);
            }
        } else {
            td.textContent = currentValue;
        }
    };

    input.onblur = finishEdit;
    input.onkeydown = (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            input.blur();
        }
        if (e.key === 'Escape') {
            td.textContent = currentValue;
        }
    };
}

async function editorApplyProcessor() {
    const colName = document.getElementById('editorColumnSelect').value;
    const processorType = document.getElementById('editorProcessorSelect').value;

    if (!colName) {
        updateEditorStatus('Vyberte stĺpec');
        return;
    }
    if (!processorType) {
        updateEditorStatus('Vyberte procesor');
        return;
    }

    try {
        // Collect params - VŽDY zaradiť parametre (aj s default hodnotami)
        const params = [];
        const paramDefs = factory.getProcessorParamDefinitions(processorType);
        if (paramDefs && paramDefs.length > 0) {
            paramDefs.forEach(param => {
                const input = document.getElementById(`editor_param_${param.name}`);
                if (input) {
                    // Zaradiť vždy, aj keď je prázdne (bude sa používať default)
                    params.push([param.name, input.value || param.default_value || '']);
                }
            });
        }

        const rawResult = pipeline.applyProcessorToColumn(
            editorCurrentCSV,
            colName,
            processorType,
            params.length > 0 ? params : null
        );
        const result = convertWasmResult(rawResult);

        editorCurrentCSV = result.csv;
        renderEditorTable();
        updateEditorStatus(`Procesor "${processorType}" aplikovaný na stĺpec "${colName}" (${result.rows_affected} riadkov)`);
    } catch (error) {
        updateEditorStatus(`Chyba: ${error}`);
    }
}

async function editorDeleteColumn() {
    const colName = document.getElementById('editorColumnSelect').value;
    if (!colName) {
        updateEditorStatus('Vyberte stĺpec na vymazanie');
        return;
    }

    if (!confirm(`Naozaj chcete vymazať stĺpec "${colName}"?`)) return;

    try {
        const rawResult = pipeline.deleteColumn(editorCurrentCSV, colName);
        const result = convertWasmResult(rawResult);
        editorCurrentCSV = result.csv;
        renderEditorTable();
        updateEditorStatus(`Stĺpec "${colName}" vymazaný, zostáva ${result.remaining_columns} stĺpcov`);
    } catch (error) {
        updateEditorStatus(`Chyba: ${error}`);
    }
}

async function editorReplaceAll() {
    const colName = document.getElementById('editorColumnSelect').value;
    const searchVal = document.getElementById('editorSearchValue').value;
    const replaceVal = document.getElementById('editorReplaceValue').value;

    if (!colName) {
        updateEditorStatus('Vyberte stĺpec');
        return;
    }
    if (searchVal === '') {
        updateEditorStatus('Zadajte hodnotu na vyhľadanie');
        return;
    }

    try {
        const rawResult = pipeline.replaceAllInColumn(editorCurrentCSV, colName, searchVal, replaceVal);
        const result = convertWasmResult(rawResult);
        editorCurrentCSV = result.csv;
        renderEditorTable();
        updateEditorStatus(`Nahradených ${result.replaced_count} výskytov "${searchVal}" → "${replaceVal}" v stĺpci "${colName}"`);
    } catch (error) {
        updateEditorStatus(`Chyba: ${error}`);
    }
}

function updateEditorStatus(text) {
    document.getElementById('editorStatusText').textContent = text;
}

