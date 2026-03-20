// Target variable analysis and comparison

async function compareTargetAnalyzers() {
    try {
        const data = window.rawDataString;
        const format = window.dataFormat;
        if (!data) {
            showStatus('error', 'Najprv načítajte dáta', 'compareTargetAnalyzersStatus');
            return;
        }

        // Collect selected analyzers
        const selectedMethods = [];
        document.querySelectorAll('#targetAnalyzerGrid .selector-compare-card.active').forEach(card => {
            const name = card.id.replace('target_card_', '');
            selectedMethods.push(name);
        });

        if (selectedMethods.length === 0) {
            showStatus('error', 'Vyberte aspoň jednu metódu analýzy', 'compareTargetAnalyzersStatus');
            return;
        }

        showLoading(true);
        
        // Odhadujeme koľko to bude trvať
        const dataSize = data.split('\n').length;
        const hasMI = selectedMethods.includes('mutual_information');
        let estimatedTime = '';
        if (hasMI) {
            if (dataSize > 1000) {
                estimatedTime = ' (môže trvať 30-60s, prosím čakajte...)';
            } else if (dataSize > 500) {
                estimatedTime = ' (môže trvať 10-20s, prosím čakajte...)';
            } else if (dataSize > 100) {
                estimatedTime = ' (môže trvať 5-10s, prosím čakajte...)';
            }
        }
        
        showStatus('info', `Porovnávam ${selectedMethods.length} metód${estimatedTime}...`, 'compareTargetAnalyzersStatus');

        const rawResult = await pipeline.compareTargetAnalyzers(data, format, selectedMethods);
        const result = convertWasmResult(rawResult);

        // Store for later use
        window._lastTargetComparison = result;

        // Show target variable map
        buildTargetVariableMap(result);
        document.getElementById('targetComparisonResultsArea').style.display = 'block';

        showStatus('success', `Porovnanie dokončené: ${selectedMethods.length} metód na ${result.total_columns} premenných. Vyberte cieľovú premennú z mapy.`, 'compareTargetAnalyzersStatus');
    } catch (error) {
        showStatus('error', `Chyba porovnania: ${error}`, 'compareTargetAnalyzersStatus');
    } finally {
        showLoading(false);
    }
}

function buildTargetVariableMap(result) {
    const mapDiv = document.getElementById('targetVariableMap');
    if (!result || !result.columns || !result.analyzer_results) {
        mapDiv.innerHTML = '<p style="color:#e74c3c;">Chyba: Neplatný výsledok porovnania</p>';
        return;
    }

    const columns = result.columns;
    const analyzerResults = result.analyzer_results;
    const totalMethods = analyzerResults.length;

    // Count how many methods ranked each column in top N (top 5)
    const topN = Math.min(5, columns.length);
    const columnVotes = {};
    columns.forEach(col => { columnVotes[col] = 0; });

    analyzerResults.forEach(ar => {
        if (!ar.ranking || ar.ranking.length === 0) return;
        const topCols = ar.ranking.slice(0, topN);
        topCols.forEach(entry => {
            if (columnVotes[entry.column_name] !== undefined) {
                columnVotes[entry.column_name]++;
            }
        });
    });

    let html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:8px;">';
    columns.forEach(colName => {
        const votes = columnVotes[colName];
        const votePct = totalMethods > 0 ? (votes / totalMethods * 100) : 0;
        
        const bg = votePct === 100 ? '#c8e6c9' 
            : votePct >= 67 ? '#fff9c4'
            : votePct >= 33 ? '#ffe0b2'
            : '#ffcdd2';
        
        const dots = Array.from({length: totalMethods}, (_, i) => 
            i < votes ? '\u25cf' : '\u25cb'
        ).join('');

        html += `<div onclick="selectTargetFromMap('${colName}')" style="
            padding:12px; background:${bg}; border:2px solid #dee2e6; 
            cursor:pointer; transition:transform 0.1s; font-size:13px;
        " onmouseover="this.style.transform='scale(1.02)'" onmouseout="this.style.transform='scale(1)'">
            <div style="font-weight:bold; color:#cc0000; margin-bottom:4px;">${colName}</div>
            <div style="font-size:11px; color:#495057; display:flex; justify-content:space-between; align-items:center;">
                <span>${votes}/${totalMethods}</span>
                <span style="font-weight:bold;">${dots}</span>
            </div>
        </div>`;
    });
    html += '</div>';

    // Scatter plot if SMC and MI are both selected
    const smcResult = analyzerResults.find(ar => ar.method === 'smc');
    const miResult = analyzerResults.find(ar => ar.method === 'mutual_information');
    
    if (smcResult && miResult && smcResult.ranking && miResult.ranking) {
        html += '<h4 style="color:#495057;margin:25px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;">Scatter plot: SMC vs Mutual Information</h4>';
        html += '<p style="font-size:12px;color:#6c757d;margin-bottom:10px;">X-os: SMC hodnota | Y-os: Suma MI hodnoty. Prejdite myšou nad bodmi pre názov premennej.</p>';
        
        // Build data arrays
        const plotData = [];
        columns.forEach(colName => {
            const smcEntry = smcResult.ranking.find(r => r.column_name === colName);
            const miEntry = miResult.ranking.find(r => r.column_name === colName);
            if (smcEntry && miEntry) {
                plotData.push({
                    name: colName,
                    smc: smcEntry.score,
                    mi: miEntry.score
                });
            }
        });

        if (plotData.length > 0) {
            // Find min/max for scaling
            const smcValues = plotData.map(d => d.smc);
            const miValues = plotData.map(d => d.mi);
            const smcMin = Math.min(...smcValues);
            const smcMax = Math.max(...smcValues);
            const miMin = Math.min(...miValues);
            const miMax = Math.max(...miValues);

            const width = 600;
            const height = 400;
            const margin = {top: 20, right: 20, bottom: 50, left: 60};
            const plotWidth = width - margin.left - margin.right;
            const plotHeight = height - margin.top - margin.bottom;

            // Scale functions
            const smcRange = smcMax - smcMin || 1;
            const miRange = miMax - miMin || 1;
            const scaleX = (val) => margin.left + ((val - smcMin) / smcRange) * plotWidth;
            const scaleY = (val) => margin.top + plotHeight - ((val - miMin) / miRange) * plotHeight;

            // Start SVG
            html += `<div style="background:white;border:1px solid #dee2e6;padding:15px;display:inline-block;">`;
            html += `<svg width="${width}" height="${height}" style="font-family:Segoe UI,sans-serif;">`;
            
            // Grid lines
            html += `<g opacity="0.1">`;
            for (let i = 0; i <= 5; i++) {
                const x = margin.left + (plotWidth * i / 5);
                const y = margin.top + (plotHeight * i / 5);
                html += `<line x1="${x}" y1="${margin.top}" x2="${x}" y2="${margin.top + plotHeight}" stroke="#000" stroke-width="1"/>`;
                html += `<line x1="${margin.left}" y1="${y}" x2="${margin.left + plotWidth}" y2="${y}" stroke="#000" stroke-width="1"/>`;
            }
            html += `</g>`;

            // Axes
            html += `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="#000" stroke-width="2"/>`;
            html += `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#000" stroke-width="2"/>`;

            // Axis labels
            html += `<text x="${margin.left + plotWidth/2}" y="${height - 10}" text-anchor="middle" font-size="12" fill="#495057">SMC</text>`;
            html += `<text x="15" y="${margin.top + plotHeight/2}" text-anchor="middle" font-size="12" fill="#495057" transform="rotate(-90, 15, ${margin.top + plotHeight/2})">Σ MI</text>`;

            // Tick labels
            for (let i = 0; i <= 5; i++) {
                const smcVal = smcMin + (smcRange * i / 5);
                const miVal = miMin + (miRange * i / 5);
                const x = margin.left + (plotWidth * i / 5);
                const y = margin.top + plotHeight - (plotHeight * i / 5);
                html += `<text x="${x}" y="${margin.top + plotHeight + 20}" text-anchor="middle" font-size="10" fill="#6c757d">${smcVal.toFixed(3)}</text>`;
                html += `<text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" font-size="10" fill="#6c757d">${miVal.toFixed(3)}</text>`;
            }

            // Data points
            plotData.forEach((d, idx) => {
                const cx = scaleX(d.smc);
                const cy = scaleY(d.mi);
                const pointId = `point_${idx}`;
                const tooltipId = `tooltip_${idx}`;
                
                // Smart tooltip positioning
                const tooltipWidth = d.name.length * 7 + 20;
                const tooltipX = cx + 10 + tooltipWidth > width ? cx - tooltipWidth - 10 : cx + 10;
                const tooltipY = cy - 30 < margin.top ? cy + 10 : cy - 30;
                
                html += `<circle id="${pointId}" cx="${cx}" cy="${cy}" r="5" fill="#cc0000" stroke="white" stroke-width="2" opacity="0.7" style="cursor:pointer;transition:all 0.2s;" onclick="selectTargetFromMap('${d.name}')" onmouseover="document.getElementById('${pointId}').setAttribute('r', '8'); document.getElementById('${pointId}').setAttribute('opacity', '1'); document.getElementById('${tooltipId}').style.display='block';" onmouseout="document.getElementById('${pointId}').setAttribute('r', '5'); document.getElementById('${pointId}').setAttribute('opacity', '0.7'); document.getElementById('${tooltipId}').style.display='none';" />`;
                
                // Tooltip
                html += `<g id="${tooltipId}" style="display:none;pointer-events:none;">`;
                html += `<rect x="${tooltipX}" y="${tooltipY}" width="${tooltipWidth}" height="25" fill="white" stroke="#cc0000" stroke-width="2" rx="3"/>`;
                html += `<text x="${tooltipX + 10}" y="${tooltipY + 16}" font-size="11" font-weight="bold" fill="#cc0000">${d.name}</text>`;
                html += `</g>`;
            });

            html += `</svg></div>`;
            html += '<div style="margin-top:10px;font-size:12px;color:#6c757d;">Kliknite na bod pre výber premennej. Vyššie hodnoty na oboch osiach = lepší kandidát na cieľovú premennú.</div>';
        }
    }

    // Legend
    html += '<div style="margin-top:15px;font-size:12px;display:flex;gap:10px;flex-wrap:wrap;">';
    html += '<span style="background:#c8e6c9;padding:4px 10px;border:1px solid #aaa;">Všetky metódy</span>';
    html += '<span style="background:#fff9c4;padding:4px 10px;border:1px solid #aaa;">Väčšina metód (≥67%)</span>';
    html += '<span style="background:#ffe0b2;padding:4px 10px;border:1px solid #aaa;">Niektoré metódy (≥33%)</span>';
    html += '<span style="background:#ffcdd2;padding:4px 10px;border:1px solid #aaa;">Menej metód</span>';
    html += '</div>';

    mapDiv.innerHTML = html;
}

function selectTargetFromMap(colName) {
    const targetInput = document.getElementById('targetColumnSelect');
    targetInput.value = colName;
    
    // Visual feedback
    document.querySelectorAll('#targetVariableMap > div > div').forEach(el => {
        el.style.border = '2px solid #dee2e6';
    });
    event.target.closest('div[onclick]').style.border = '4px solid #cc0000';
    
    showStatus('info', `Vybratá premenná: ${colName}`, 'compareTargetAnalyzersStatus');
}

// Legacy compatibility
function selectTargetFromAnalysis(colName, suggestedType) {
    selectTargetFromMap(colName);
}

function populateTargetAnalyzerMethods() {
    try {
        const rawAnalyzers = pipeline.getAvailableTargetAnalyzers();
        const analyzers = convertWasmResult(rawAnalyzers);
        if (!analyzers || !analyzers.length) return;

        const grid = document.getElementById('targetAnalyzerGrid');
        grid.innerHTML = '';

        analyzers.forEach(ana => {
            const card = document.createElement('div');
            card.className = 'selector-compare-card';
            card.id = `target_card_${ana.name}`;

            card.innerHTML = `
                <div class="card-header">
                    <input type="checkbox" id="target_ana_${ana.name}" onchange="toggleTargetCard('${ana.name}')">
                    <label for="target_ana_${ana.name}">${ana.metric_name}</label>
                </div>
                <div class="card-desc">${ana.description}</div>
                <div class="card-tags" style="font-size:11px;color:#6c757d;">${ana.metric_explanation}</div>
                <button onclick="showAnalyzerDetails('${ana.name}'); event.stopPropagation();" 
                        style="margin-top:8px; padding:6px 12px; font-size:11px; background:#6c757d; color:white; border:none; cursor:pointer; border-radius:4px;">
                    Zobraziť detaily (matica, kandidáti)
                </button>
            `;
            grid.appendChild(card);
        });

        window._targetAnalyzers = analyzers;
    } catch (e) {
        console.warn('Could not load target analyzers:', e);
    }
}

function toggleTargetCard(name) {
    const cb = document.getElementById(`target_ana_${name}`);
    const card = document.getElementById(`target_card_${name}`);
    card.classList.toggle('active', cb.checked);
}

function toggleAllAnalyzers() {
    const cards = document.querySelectorAll('#targetAnalyzerGrid .selector-compare-card');
    const allChecked = Array.from(cards).every(c => c.classList.contains('active'));
    cards.forEach(card => {
        const cb = card.querySelector('input[type="checkbox"]');
        cb.checked = !allChecked;
        card.classList.toggle('active', !allChecked);
    });
}

async function showAnalyzerDetails(method) {
    try {
        const data = window.rawDataString;
        const format = window.dataFormat;
        if (!data) {
            alert('Najprv načítajte dáta');
            return;
        }

        showLoading(true);
        const rawResult = await pipeline.analyzeTargetWith(data, format, method);
        const result = convertWasmResult(rawResult);

        // Create modal to show details
        const modal = document.getElementById('analyzerDetailsModal');
        const content = document.getElementById('analyzerDetailsContent');
        const titleEl = document.getElementById('analyzerDetailsTitle');
        
        titleEl.textContent = `${result.metric_name} - ${result.method_description}`;
        content.innerHTML = result.html;

        modal.classList.add('show');
    } catch (error) {
        alert(`Chyba pri načítaní detailov: ${error}`);
    } finally {
        showLoading(false);
    }
}

function closeAnalyzerDetailsModal() {
    document.getElementById('analyzerDetailsModal').classList.remove('show');
}

function buildColumnStatsTable(data, columns, format) {
    const container = document.getElementById('columnStatsContainer');
    if (!container) return;
    if (format !== 'csv') {
        container.innerHTML = '<p style="color:#6c757d;">Štatistický prehľad je dostupný len pre CSV formát.</p>';
        return;
    }
    const lines = data.split('\n').filter(l => l.trim());
    if (lines.length < 2) return;

    const headers = lines[0].split(',').map(h => h.trim());
    const rows = lines.slice(1).map(l => l.split(',').map(v => v.trim()));

    let html = '<table class="column-stats"><thead><tr>';
    html += '<th>Premenná</th><th>Typ</th><th>Unikátnych</th><th>Min</th><th>Max</th><th>Priemer</th>';
    html += '</tr></thead><tbody>';

    columns.forEach((col, colIdx) => {
        // Find column index in headers (might not match columns if target excluded)
        const hIdx = headers.indexOf(col);
        if (hIdx === -1) return;

        const values = rows.map(r => r[hIdx]).filter(v => v !== undefined && v !== '');
        const numValues = values.map(Number).filter(v => !isNaN(v));
        const isNumeric = numValues.length > values.length * 0.8;
        const uniqueCount = new Set(values).size;

        let min = '-', max = '-', mean = '-';
        if (isNumeric && numValues.length > 0) {
            min = Math.min(...numValues).toFixed(2);
            max = Math.max(...numValues).toFixed(2);
            mean = (numValues.reduce((a,b) => a+b, 0) / numValues.length).toFixed(2);
        }

        const ratio = uniqueCount / values.length;
        let typeLabel, suggestion;
        if (!isNumeric) {
            typeLabel = 'Kategorický';
            suggestion = uniqueCount <= 10 ? 'Klasifikácia' : 'Kategorický';
        } else if (uniqueCount <= 10 || ratio < 0.05) {
            typeLabel = 'Diskrétny';
            suggestion = 'Klasifikácia';
        } else {
            typeLabel = 'Spojitý';
            suggestion = 'Regresia';
        }

        html += `<tr data-column="${col}" onclick="selectTargetFromStats('${col}')">`;
        html += `<td style="text-align:left;font-weight:bold;">${col}</td>`;
        html += `<td>${typeLabel}</td>`;
        html += `<td>${uniqueCount}</td>`;
        html += `<td>${min}</td>`;
        html += `<td>${max}</td>`;
        html += `<td>${mean}</td>`;
        html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
}

function selectTargetFromStats(colName) {
    const targetSelect = document.getElementById('targetColumnSelect');
    targetSelect.value = colName;
    highlightStatsRow(colName);
}

function highlightStatsRow(colName) {
    document.querySelectorAll('.column-stats tbody tr').forEach(tr => {
        tr.classList.toggle('selected-row', tr.dataset.column === colName);
    });
}

// ===== Feature exploration / selector comparison =====

