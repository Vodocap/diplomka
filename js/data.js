// Data loading, parsing, target selection, inspection

function toggleDataInputMethod() {
    const method = document.querySelector('input[name="dataInputMethod"]:checked').value;
    const textContainer = document.getElementById('textInputContainer');
    const fileContainer = document.getElementById('fileInputContainer');
    
    if (method === 'text') {
        textContainer.style.display = 'block';
        fileContainer.style.display = 'none';
    } else {
        textContainer.style.display = 'none';
        fileContainer.style.display = 'block';
    }
}

function downloadSampleCSV() {
    const csvContent = `age,income,credit_score,debt_ratio,employment_years,education_level,num_accounts,account_balance,loan_amount,monthly_payment,savings,investment_score,property_value,approved
25,50000,700,0.35,3,2,4,12500,15000,450,8000,65,0,1
35,75000,750,0.28,8,3,6,28000,22000,680,25000,78,180000,1
45,60000,650,0.42,15,2,3,8500,28000,820,15000,58,150000,0
22,30000,600,0.48,1,1,2,3500,12000,380,2500,42,0,0
50,90000,800,0.22,20,4,8,45000,35000,920,55000,88,320000,1
30,55000,680,0.38,5,3,5,15000,18000,520,12000,68,0,1
40,80000,720,0.30,12,3,7,32000,25000,750,30000,75,220000,1
28,45000,640,0.44,4,2,3,7500,20000,610,6000,55,0,0
55,95000,790,0.20,25,4,9,58000,40000,980,68000,92,450000,1
32,62000,710,0.33,7,3,6,18500,21000,640,16000,72,125000,1
38,72000,730,0.29,10,3,7,26000,24000,710,28000,76,195000,1
27,48000,660,0.40,3,2,4,9500,17000,550,9500,60,0,0
52,88000,780,0.24,22,4,8,42000,32000,890,52000,86,380000,1
29,51000,670,0.37,4,2,5,11500,19000,570,11000,63,0,1
43,68000,700,0.32,13,3,6,21000,23000,690,22000,74,175000,1
24,42000,630,0.45,2,2,3,6000,14000,480,5000,50,0,0
48,82000,760,0.26,18,3,7,35000,28000,820,38000,82,280000,1
31,58000,690,0.36,6,3,5,14000,20000,600,13500,66,110000,1
36,70000,725,0.31,9,3,6,24000,26000,730,26000,77,200000,1
26,46000,655,0.41,3,2,4,8000,16000,520,7500,58,0,0
53,92000,795,0.21,24,4,9,50000,38000,950,62000,90,410000,1
33,64000,705,0.34,8,3,6,19500,22000,660,18000,70,140000,1
41,76000,740,0.27,14,3,7,30000,27000,780,32000,80,240000,1
23,38000,620,0.47,1,1,2,4500,13000,420,3500,45,0,0
49,85000,770,0.25,19,4,8,40000,33000,870,48000,84,350000,1
34,66000,715,0.30,9,3,6,22000,24000,700,24000,73,160000,1
39,74000,735,0.28,11,3,7,28000,26000,760,29000,78,210000,1
28,49000,665,0.39,4,2,4,10000,18000,560,10500,62,0,0
54,94000,785,0.23,23,4,9,55000,36000,930,65000,89,430000,1
30,56000,685,0.35,5,3,5,13500,19000,590,14500,67,95000,1`;
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    
    link.setAttribute('href', url);
    link.setAttribute('download', 'sample_data.csv');
    link.style.visibility = 'hidden';
    
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

async function getDataString() {
    const method = document.querySelector('input[name="dataInputMethod"]:checked').value;
    if (method === 'text') {
        const data = document.getElementById('dataInput').value;
        if (!data) throw new Error('Vložte dáta');
        return data;
    } else {
        const fileInput = document.getElementById('fileInput');
        const file = fileInput.files[0];
        if (!file) throw new Error('Vyberte súbor');
        return await new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => resolve(e.target.result);
            reader.onerror = () => reject(new Error('Chyba pri čítaní súboru'));
            reader.readAsText(file);
        });
    }
}

async function parseData() {
    try {
        showLoading(true);
        const format = document.getElementById('dataFormatSelect').value;
        const data = await getDataString();

        window.rawDataString = data;
        window.dataFormat = format;

        dataLoader = new WasmDataLoaderClass(format);
        const columns = await dataLoader.getAvailableColumns(data);
        window.parsedColumns = columns;

        // Target selection now via comparison map

        document.getElementById('targetSelectionArea').style.display = 'block';
        
        // Populate target column dropdown
        const targetSelect = document.getElementById('targetColumnSelect');
        targetSelect.innerHTML = '<option value="">-- Vyberte cie\u013eov\u00fa premenn\u00fa --</option>';
        columns.forEach(col => {
            const opt = document.createElement('option');
            opt.value = col;
            opt.textContent = col;
            targetSelect.appendChild(opt);
        });
        // Pre-select last column as default (common convention)
        if (columns.length > 0) {
            targetSelect.value = columns[columns.length - 1];
        }
        
        document.getElementById('inspectDataBtn').disabled = false;
        document.getElementById('editDataBtn').disabled = false;

        // Show feature exploration section and populate selector cards
        document.getElementById('featureExplorationSection').style.display = 'block';
        populateComparisonSelectors();
        populateTargetAnalyzerMethods();
        buildColumnStatsTable(data, columns, format);

        showStatus('success', `Dáta načítané: ${columns.length} stĺpcov nájdených. Porovnajte metódy analýzy a vyberte cieľovú premennú.`, 'parseStatus');
    } catch (error) {
        showStatus('error', `Chyba: ${error}`, 'parseStatus');
    } finally {
        showLoading(false);
    }
}

async function confirmTarget() {
    try {
        showLoading(true);
        const targetColumn = document.getElementById('targetColumnSelect').value;
        if (!targetColumn) {
            showStatus('error', 'Vyberte cieľový stĺpec', 'dataStatus');
            return;
        }

        const data = window.rawDataString;
        const format = window.dataFormat;
        if (!data) {
            showStatus('error', 'Najprv načítajte dáta', 'dataStatus');
            return;
        }

        const result = await pipeline.loadData(data, targetColumn, format);

        const trainPercent = parseInt(document.getElementById('trainTestSplit').value) / 100;
        window.trainTestRatio = trainPercent;

        document.getElementById('inspectDataBtn').disabled = false;

        showStatus('success', `${targetColumn} (Train: ${Math.round(trainPercent * 100)}%, Test: ${Math.round((1 - trainPercent) * 100)}%)`, 'dataStatus');
    } catch (error) {
        showStatus('error', `Chyba: ${error}`, 'dataStatus');
    } finally {
        showLoading(false);
    }
}

async function inspectData() {
    try {
        const result = await pipeline.inspectData(10);
        const data = convertWasmResult(result);
        showDataModal('Nahraté dáta', data);
    } catch (error) {
        console.error('inspectData error:', error);
        alert('Error: ' + error);
    }
}

function showDataModal(title, data) {
    if (!data || !data.feature_names || !data.rows) {
        alert('Chyba: Neplatný formát dát z WASM. Skontrolujte konzolu pre detaily.');
        console.error('Invalid data:', data);
        return;
    }
    
    const modal = document.getElementById('dataModal');
    document.getElementById('modalTitle').textContent = title;
    
    let html = `
        <div class="feature-info">
            <p><strong>Počet riadkov:</strong> ${data.total_rows} (zobrazených prvých ${data.shown_rows})</p>
            <p><strong>Počet stĺpcov:</strong> ${data.total_cols}</p>
            ${data.selected_indices ? `<p><strong>Vybrané indexy:</strong> ${JSON.stringify(data.selected_indices)}</p>` : ''}
            ${data.preprocessing_applied ? `<p><strong>Preprocessing:</strong> Aplikovaný</p>` : ''}
        </div>

        <table class="data-table">
            <thead>
                <tr>
                    <th>#</th>
                    ${data.feature_names.map(name => `<th>${name}</th>`).join('')}
                    <th>${data.target_name}</th>
                </tr>
            </thead>
            <tbody>
                ${data.rows.map((row, idx) => `
                    <tr>
                        <td><strong>${idx + 1}</strong></td>
                        ${row.map(val => `<td>${typeof val === 'number' ? val.toFixed(4) : val}</td>`).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>

        ${data.shown_rows < data.total_rows ? 
            `<p style="margin-top: 15px; color: #6c757d; font-style: italic;">
                ... a ďalších ${data.total_rows - data.shown_rows} riadkov
            </p>` : ''}
    `;

    document.getElementById('modalBody').innerHTML = html;
    modal.classList.add('show');
}

function closeModal() {
    document.getElementById('dataModal').classList.remove('show');
}

