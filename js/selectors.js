// Feature selector comparison, training, PDF export, synergy

function populateComparisonSelectors() {
    if (!availableOptions || !availableOptions.selectors) return;
    const grid = document.getElementById('selectorCompareGrid');
    grid.innerHTML = '';

    availableOptions.selectors.forEach(sel => {
        
        const card = document.createElement('div');
        card.className = 'selector-compare-card';
        card.id = `compare_card_${sel.name}`;

        const tags = sel.supported_types.map(t =>
            `<span class="card-tag">${t}</span>`
        ).join('');
        const warnTag = sel.requires_binning
            ? '<span class="card-tag warn">Vyžaduje Binner</span>'
            : '';

        // Build param inputs based on selector type
        let paramsHtml = '';
        const params = factory.getSelectorParams(sel.name);
        if (params && params.length > 0) {
            params.forEach(p => {
                let defaultVal = '';
                let label = p;
                let description = '';
                let inputType = 'number';
                let selectOptions = null;

                // Common params
                if (p === 'num_features') {
                    defaultVal = '5';
                    label = 'Počet features';
                    description = 'Koľko features chcete vybrať z celkového počtu.';
                } else if (p === 'threshold') {
                    defaultVal = sel.name === 'correlation' ? '0.95' : '0.01';
                    label = 'Prahová hodnota';
                    description = sel.name === 'correlation'
                        ? 'Features s koreláciou nad touto hodnotou budú odstránené.'
                        : 'Minimálna variancia – features pod touto hodnotou budú vylúčené.';
                }

                if (selectOptions) {
                    const opts = selectOptions.map(o => `<option value="${o}" ${o === defaultVal ? 'selected' : ''}>${o}</option>`).join('');
                    paramsHtml += `<div class="form-group" style="margin-bottom:8px;">
                        <label>${label} (${p}):</label>
                        <select id="compare_param_${sel.name}_${p}">${opts}</select>
                        ${description ? `<div class="param-desc">${description}</div>` : ''}
                    </div>`;
                } else {
                    paramsHtml += `<div class="form-group" style="margin-bottom:8px;">
                        <label>${label} (${p}):</label>
                        <input type="${inputType}" step="any" value="${defaultVal}" id="compare_param_${sel.name}_${p}">
                        ${description ? `<div class="param-desc">${description}</div>` : ''}
                    </div>`;
                }
            });
        }

        card.innerHTML = `
            <div class="card-header">
                <input type="checkbox" id="compare_sel_${sel.name}" onchange="toggleCompareCard('${sel.name}')">
                <label for="compare_sel_${sel.name}">${sel.name}</label>
            </div>
            <div class="card-desc">${sel.description}</div>
            <div class="card-tags">${tags}${warnTag}</div>
            <div class="card-params">${paramsHtml}</div>
        `;
        grid.appendChild(card);
    });
}

function toggleCompareCard(name) {
    const cb = document.getElementById(`compare_sel_${name}`);
    const card = document.getElementById(`compare_card_${name}`);
    card.classList.toggle('active', cb.checked);
}

function toggleAllSelectors() {
    const cards = document.querySelectorAll('#selectorCompareGrid .selector-compare-card');
    const allChecked = Array.from(cards).every(c => c.classList.contains('active'));
    cards.forEach(card => {
        const cb = card.querySelector('input[type="checkbox"]');
        cb.checked = !allChecked;
        card.classList.toggle('active', !allChecked);
    });
}

// Detekcia redundancie medzi vybranými featurmi
let previouslySelected = new Set();

function attachRedundancyChecks() {
    const checkboxes = document.querySelectorAll('#userFeatureMap input[type=checkbox]');
    if (checkboxes.length === 0) return;

    // Inicializuj previouslySelected s aktuálnym stavom
    previouslySelected.clear();
    document.querySelectorAll('#userFeatureMap input[type=checkbox]:checked').forEach(cb => {
        previouslySelected.add(parseInt(cb.dataset.featureIdx));
    });

    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', async function() {
            // Zbierz vybrané indices
            const selectedIndices = [];
            const currentlySelected = new Set();
            document.querySelectorAll('#userFeatureMap input[type=checkbox]:checked').forEach(cb => {
                const idx = parseInt(cb.dataset.featureIdx);
                selectedIndices.push(idx);
                currentlySelected.add(idx);
            });

            if (selectedIndices.length === 0) {
                // Skryj upozornenie ak nič nie je vybrané
                const warningArea = document.getElementById('redundancyWarningArea');
                if (warningArea) warningArea.style.display = 'none';
                previouslySelected.clear();
                return;
            }

            // Zisti, ktoré features boli pridané (nie default)
            const newlyAdded = [];
            for (let idx of currentlySelected) {
                if (!previouslySelected.has(idx)) {
                    newlyAdded.push(idx);
                }
            }

            // Ak nebola pridaná žiadna nová feature, len aktualizuj stav a skonči
            if (newlyAdded.length === 0) {
                previouslySelected = new Set(currentlySelected);
                const warningArea = document.getElementById('redundancyWarningArea');
                if (warningArea) warningArea.style.display = 'none';
                return;
            }

            // Zisti cieľový stĺpec index zo result dát
            let targetColIndex = -1;
            if (window._lastComparisonResult && window._lastComparisonResult.target_index !== undefined) {
                targetColIndex = window._lastComparisonResult.target_index;
            }

            // Zavolaj API pre detekciu redundancie s novo pridanými features
            try {
                // Pošli prvú novo pridanú feature ako fokus, alebo -1 ak žiadna
                const focusFeature = newlyAdded.length > 0 ? newlyAdded[0] : -1;
                
                const rawResult = await pipeline.checkFeatureRedundancy(
                    window.rawDataString, 
                    window.dataFormat, 
                    selectedIndices, 
                    targetColIndex,
                    focusFeature
                );
                const result = convertWasmResult(rawResult);
                displayRedundancyWarnings(result);
                
                // Aktualizuj stav
                previouslySelected = new Set(currentlySelected);
            } catch (error) {
                console.warn('Redundancy check failed:', error);
            }
        });
    });
}

function displayRedundancyWarnings(report) {
    let warningArea = document.getElementById('redundancyWarningArea');
    
    if (!warningArea) {
        // Vytvor warning area ak neexistuje
        const mapContainer = document.getElementById('userFeatureMap');
        if (!mapContainer) return;
        
        warningArea = document.createElement('div');
        warningArea.id = 'redundancyWarningArea';
        mapContainer.parentNode.insertBefore(warningArea, mapContainer);
    }

    if (!report.has_issues) {
        // Zobraz OK správu alebo skry warning area
        if (report.summary && report.summary.startsWith('OK:')) {
            let html = `<div style="margin-top:12px; padding:12px; background:#e8f5e9; border-left:4px solid #4caf50; border-radius:4px;">`;
            html += `<strong style="color:#2e7d32;">✓ ${report.summary}</strong>`;
            html += `</div>`;
            warningArea.innerHTML = html;
            warningArea.style.display = 'block';
            
            // Skry po 3 sekundách
            setTimeout(() => {
                if (warningArea) warningArea.style.display = 'none';
            }, 3000);
        } else {
            warningArea.style.display = 'none';
        }
        return;
    }

    // Zobraz upozornenia
    let html = `<div style="margin-top:12px; padding:12px; background:#fff5e6; border-left:4px solid #ff9800; border-radius:4px;">`;
    html += `<strong style="color:#e65100;">${report.summary || 'Upozornenie na vysokú multikolinearitu:'}</strong><br>`;
    
    report.warnings.forEach(warning => {
        html += `<div style="margin-top:8px; font-size:12px; color:#495057;">`;
        
        if (warning.warning_type === 'high_correlation') {
            // Korelačné upozornenie - lineárna závislosť
            html += `⚠️ <strong>${warning.feature1_name}</strong> ↔ <strong>${warning.feature2_name}</strong>: `;
            html += `lineárna korelácia <strong>${(warning.correlation * 100).toFixed(1)}%</strong>`;
        } else if (warning.warning_type === 'high_mutual_information') {
            // NMI upozornenie - zdielaná informácia (zachytáva lineárne + nelineárne závislosti)
            html += `⚠️ <strong>${warning.feature1_name}</strong> ↔ <strong>${warning.feature2_name}</strong>: `;
            html += `zdieľaná informácia <strong>${(warning.correlation * 100).toFixed(1)}%</strong>`;
            html += ` <span style="font-size:10px;color:#999;">(NMI - zachytáva aj nelineárne závislosti)</span>`;
        }
        html += `</div>`;
    });

    
    html += `</div>`;
    warningArea.innerHTML = html;
    warningArea.style.display = 'block';
}

async function compareSelectors() {
    try {
        const targetColumn = document.getElementById('targetColumnSelect').value;
        if (!targetColumn) {
            showStatus('error', 'Najprv vyberte cieľovú premennú', 'compareSelectorStatus');
            return;
        }
        if (!pipeline) {
            showStatus('error', 'Pipeline nie je inicializovaný', 'compareSelectorStatus');
            return;
        }
        const data = window.rawDataString;
        const format = window.dataFormat;
        if (!data) {
            showStatus('error', 'Najprv načítajte dáta', 'compareSelectorStatus');
            return;
        }

        // Collect selected selectors and their params
        const selectorConfigs = [];
        document.querySelectorAll('#selectorCompareGrid .selector-compare-card.active').forEach(card => {
            const name = card.id.replace('compare_card_', '');
            const params = [];
            card.querySelectorAll('.card-params input, .card-params select').forEach(el => {
                const paramName = el.id.replace(`compare_param_${name}_`, '');
                if (el.value) {
                    params.push([paramName, el.value]);
                }
            });
            selectorConfigs.push({ name, params });
        });

        if (selectorConfigs.length === 0) {
            showStatus('error', 'Vyberte aspoň jeden selektor na porovnanie', 'compareSelectorStatus');
            return;
        }

        showLoading(true);
        showStatus('info', `Porovnávam ${selectorConfigs.length} selektorov...`, 'compareSelectorStatus');

        const rawResult = await pipeline.compareSelectors(data, targetColumn, format, selectorConfigs);
        const result = convertWasmResult(rawResult);

        // Store results for training
        window._lastComparisonResult = result;


        // Open comparison in modal
        const modal = document.getElementById('dataModal');
        document.getElementById('modalTitle').textContent = `Porovnanie selektorov (cieľ: ${targetColumn})`;
        document.getElementById('modalBody').innerHTML = result.comparison_html;
        modal.classList.add('show');

        // Attach event handlers for interactive elements in comparison HTML
        const trainBtn = document.getElementById('trainFromComparisonBtn');
        if (trainBtn) trainBtn.onclick = trainFromComparison;
        const selectAllBtn = document.getElementById('featureMapSelectAll');
        if (selectAllBtn) selectAllBtn.onclick = () => document.querySelectorAll('#userFeatureMap input[type=checkbox]').forEach(c => c.checked = true);
        const deselectAllBtn = document.getElementById('featureMapDeselectAll');
        if (deselectAllBtn) deselectAllBtn.onclick = () => document.querySelectorAll('#userFeatureMap input[type=checkbox]').forEach(c => c.checked = false);

        // Attach redundancy check to all feature checkboxes
        attachRedundancyChecks();

        showStatus('success', `Porovnanie dokončené: ${selectorConfigs.length} selektorov na ${result.total_features} príznakoch. Teraz môžete natrénovať model s vybranými features.`, 'compareSelectorStatus');
    } catch (error) {
        showStatus('error', `Chyba porovnania: ${error}`, 'compareSelectorStatus');
    } finally {
        showLoading(false);
    }
}

// Train model with features selected by all compared selectors + user selection
async function trainFromComparison() {
    const comparison = window._lastComparisonResult;
    if (!comparison || !comparison.selectors) {
        alert('Najprv spustite porovnanie selektorov');
        return;
    }
    if (!pipeline || !window.pipelineBuilt) {
        alert('Najprv vytvorte a nakonfigurujte pipeline (model)');
        return;
    }

    const trainRatio = window.trainTestRatio || 0.8;
    const resultsDiv = document.getElementById('comparisonTrainingResults');
    const statusSpan = document.getElementById('comparisonTrainStatus');

    // Collect user-selected features from checkboxes
    const userIndices = [];
    document.querySelectorAll('#userFeatureMap input[type=checkbox]:checked').forEach(cb => {
        userIndices.push(parseInt(cb.dataset.featureIdx));
    });
    userIndices.sort((a, b) => a - b);

    statusSpan.textContent = 'Trénujem varianty...';
    resultsDiv.innerHTML = '<p style="color:#6c757d;font-style:italic;">Prebieha tréning...</p>';

    const allResults = [];
    const featureNames = comparison.feature_names || [];

    try {
        // 1. Each selector's selection
        const selectors = Array.isArray(comparison.selectors) ? comparison.selectors : [];
        for (const sel of selectors) {
            const selData = (sel instanceof Map) ? Object.fromEntries(sel) : sel;
            if (selData.error) {
                allResults.push({ name: selData.selector_name || selData.config_name, error: selData.error, indices: [], count: 0 });
                continue;
            }
            const indices = selData.selected_indices || [];
            if (indices.length === 0) {
                allResults.push({ name: selData.selector_name, error: 'Nevybral žiadne features', indices: [], count: 0 });
                continue;
            }
            try {
                const t0 = performance.now();
                const trainRaw = await pipeline.trainWithFeatureIndices(trainRatio, indices);
                const trainTime = performance.now() - t0;
                const trainResult = convertWasmResult(trainRaw);
                allResults.push({ name: selData.selector_name, result: trainResult, indices, count: indices.length, timeMs: trainTime });
            } catch (e) {
                allResults.push({ name: selData.selector_name, error: String(e), indices, count: indices.length });
            }
        }

        // 2. Baseline - all features, no selection
        try {
            const t0 = performance.now();
            const baseRaw = await pipeline.trainWithSplit(trainRatio);
            const baseTime = performance.now() - t0;
            const baseResult = convertWasmResult(baseRaw);
            allResults.push({ name: 'Bez selekcie (všetky features)', result: baseResult, indices: null, count: comparison.total_features, timeMs: baseTime });
        } catch (e) {
            allResults.push({ name: 'Bez selekcie (všetky features)', error: String(e), indices: null, count: comparison.total_features });
        }

        // 3. User's custom selection
        if (userIndices.length > 0) {
            try {
                const t0 = performance.now();
                const userRaw = await pipeline.trainWithFeatureIndices(trainRatio, userIndices);
                const userTime = performance.now() - t0;
                const userResult = convertWasmResult(userRaw);
                allResults.push({ name: 'Vlastný výber používateľa', result: userResult, indices: userIndices, count: userIndices.length, timeMs: userTime, isUser: true });
            } catch (e) {
                allResults.push({ name: 'Vlastný výber používateľa', error: String(e), indices: userIndices, count: userIndices.length, isUser: true });
            }
        }

        // Render results
        renderTrainingComparison(allResults, comparison.total_features, featureNames, resultsDiv);
        statusSpan.textContent = `Hotovo – ${allResults.length} variantov natrénovaných`;
    } catch (error) {
        statusSpan.textContent = `Chyba: ${error}`;
        resultsDiv.innerHTML = `<p style="color:#e74c3c;">Chyba pri trénovaní: ${error}</p>`;
    }
}

// Legacy: Train model with features selected by all compared selectors + baseline
async function trainAllSelectors() {
    trainFromComparison();
}

// Global storage for comparison results
let globalComparisonResults = {
    allResults: [],
    totalFeatures: 0,
    featureNames: [],
    isClassification: false
};

// Render training comparison table
function renderTrainingComparison(allResults, totalFeatures, featureNames, targetDiv) {
    const resultsDiv = targetDiv || document.getElementById('selectorTrainingResults');
    
    // Store globally for embedded comparison
    globalComparisonResults = {
        allResults: allResults,
        totalFeatures: totalFeatures,
        featureNames: featureNames,
        isClassification: false // will be set below
    };
    
    // Determine if classification or regression from first successful result
    const firstSuccess = allResults.find(r => r.result);
    if (!firstSuccess) {
        // All failed - show errors
        let errHtml = '<h3 style="margin-top:20px;margin-bottom:10px;color:#e74c3c;">Všetky pokusy o trénovanie zlyhali</h3>';
        errHtml += '<ul style="color:#e74c3c;">';
        allResults.forEach(r => {
            errHtml += `<li><strong>${r.name}</strong>: ${r.error || 'Neznáma chyba'}</li>`;
        });
        errHtml += '</ul>';
        resultsDiv.innerHTML = errHtml;
        return;
    }
    const isClassification = firstSuccess.result.evaluation_mode === 'classification';
    globalComparisonResults.isClassification = isClassification;

    let html = '<h3 style="margin-top:20px;margin-bottom:10px;color:#cc0000;">Porovnanie výsledkov tréningu</h3>';
    html += '<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:12px;">';
    html += '<thead><tr>';
    html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:left;font-size:11px;">Metóda</th>';
    html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;">Features</th>';
    
    if (isClassification) {
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Accuracy">ACC</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="F1 Score">F1</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Precision">PREC</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Recall/Sensitivity">REC</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Specificity">SPEC</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="False Positives">FP</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="False Negatives">FN</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Matthews Correlation Coeff">MCC</th>';
    } else {
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="R-squared">R²</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Root Mean Squared Error">RMSE</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Mean Absolute Error">MAE</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Mean Absolute Percentage Error">MAPE</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Median Absolute Error">MedAE</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="Pearsonova korelácia predikcie">CORR</th>';
    }
    html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;">Čas (ms)</th>';
    html += '</tr></thead><tbody>';

    // Find best values for highlighting
    let bestMetric = -Infinity;
    allResults.forEach(r => {
        if (!r.result) return;
        const val = isClassification ? r.result.accuracy : r.result.r2_score;
        if (val > bestMetric) bestMetric = val;
    });

    allResults.forEach((r, idx) => {
        const isBaseline = r.indices === null;
        const isUser = r.isUser === true;
        const isEmbedded = r.isEmbedded === true;
        const isBest = r.result && ((isClassification ? r.result.accuracy : r.result.r2_score) === bestMetric);
        const bg = isBest ? 'background:#ffe4e1;' : isEmbedded ? 'background:#fff9e6;' : isUser ? 'background:#fff0f0;' : (idx % 2 === 0 ? 'background:#f8f9fa;' : '');
        
        html += `<tr style="${bg}">`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;font-weight:${isBaseline ? 'normal' : 'bold'};font-size:12px;${isUser ? 'color:#cc0000;' : ''}${isEmbedded ? 'color:#ff8c00;' : ''}">${r.name}${isBest ? ' (best)' : ''}</td>`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.count}/${totalFeatures}</td>`;

        if (r.error) {
            const colSpan = isClassification ? 8 : 6;
            html += `<td colspan="${colSpan}" style="padding:6px;border:1px solid #dee2e6;color:#e74c3c;text-align:center;font-size:11px;">${r.error}</td>`;
        } else if (r.result) {
            if (isClassification) {
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;color:#cc0000;font-size:12px;">${(r.result.accuracy * 100).toFixed(1)}%</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.f1_score.toFixed(3)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.precision.toFixed(3)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.recall.toFixed(3)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.specificity.toFixed(3)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${Math.round(r.result.false_positives)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${Math.round(r.result.false_negatives)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.mcc.toFixed(3)}</td>`;
            } else {
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;color:#cc0000;font-size:12px;">${r.result.r2_score.toFixed(4)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.rmse.toFixed(3)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.mae.toFixed(3)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.mape ? r.result.mape.toFixed(2) + '%' : 'N/A'}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.median_absolute_error.toFixed(3)}</td>`;
                html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:12px;">${r.result.pearson_correlation.toFixed(3)}</td>`;
            }
            const timeStr = r.timeMs ? `${r.timeMs.toFixed(0)}` : '-';
            html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;color:#6c757d;font-size:12px;">${timeStr}</td>`;
        }
        html += '</tr>';
    });

    html += '</tbody></table></div>';
    
    // Add legend for abbreviations
    html += '<div style="margin-top:12px;padding:12px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;font-size:11px;color:#495057;">';
    if (isClassification) {
        html += '<strong>Metriky klasifikácie:</strong><br>';
        html += '• ACC = Accuracy | F1 = F1 Score | PREC = Precision | REC = Recall/Sensitivity | SPEC = Specificity<br>';
        html += '• FP = počet falošne pozitívnych prípadov | FN = počet falošne negatívnych prípadov<br>';
        html += '• MCC = Matthews Correlation Coefficient (vhodnejší ako accuracy pre imbalancované dáta)<br>';
    } else {
        html += '<strong>Metriky regresie:</strong><br>';
        html += '• R² = Koeficient determinácie (max 1.0, môže byť záporné pre zlé modely, vyššia = lepšia) | RMSE = Root Mean Squared Error<br>';
        html += '• MAE = Mean Absolute Error | MAPE = Mean Absolute Percentage Error<br>';
        html += '• MedAE = Median Absolute Error | CORR = Pearsonova korelácia (sklon linearity)<br>';
    }
    html += '</div>';

    // Show which features each selector picked
    html += '<details style="margin-top:15px;border:1px solid #dee2e6;">';
    html += '<summary style="padding:12px;background:#f8f9fa;cursor:pointer;font-weight:600;color:#cc0000;">Vybrané features jednotlivými variantmi</summary>';
    html += '<div style="padding:15px;">';
    allResults.forEach(r => {
        if (r.indices === null) return; // skip baseline
        const names = r.indices && featureNames.length > 0 
            ? r.indices.map(i => featureNames[i] || `[${i}]`).join(', ')
            : (r.indices || []).join(', ');
        const style = r.isUser ? 'color:#cc0000;' : '';
        html += `<p style="${style}"><strong>${r.name}</strong> (${r.count} features): ${names || 'N/A'}</p>`;
    });
    html += '</div></details>';

    // Add button to compare with embedded methods
    html += '<div style="margin-top:20px;padding:15px;background:#fff0f0;border:1px solid #cc0000;border-radius:4px;">';
    html += '<h4 style="margin:0 0 8px 0;color:#cc0000;">Porovnanie s Embedded metódami</h4>';
    html += '<p style="margin:0 0 12px 0;font-size:12px;color:#495057;">Embedded metódy vyberajú features priamo počas trénovania modelu (koeficienty, feature importance). Porovnajte ich výsledky s filter metódami použitými vyššie.</p>';
    html += '<button id="compareEmbeddedBtn" style="background:#cc0000;color:white;border:none;padding:10px 20px;cursor:pointer;font-size:14px;font-weight:bold;border-radius:4px;">Porovnať s Embedded metódami</button>';
    html += ' <span id="embeddedComparisonStatus" style="margin-left:12px;font-size:0.9em;color:#6c757d;"></span>';
    html += '<div id="embeddedComparisonResults" style="margin-top:15px;"></div>';
    html += '</div>';

    // Matrix R² comparison - only for regression
    if (!isClassification) {
        html += '<div style="margin-top:20px;padding:15px;background:#fff0f0;border:1px solid #cc0000;border-radius:4px;">';
        html += '<h4 style="margin:0 0 8px 0;color:#cc0000;">Porovnanie Model R² vs Maticové R²</h4>';
        html += '<p style="margin:0 0 12px 0;font-size:12px;color:#495057;">';
        html += 'Maticové R² = r<sup>T</sup><sub>yX</sub> R<sup>-1</sup> r<sub>yX</sub> — teoretický koeficient determinácie vypočítaný iba z korelačnej matice <b>bez trénovania modelu</b>.<br>';
        html += 'Porovnanie s modelovým R² odhaľuje, či sú v dátach nelineárne vzťahy alebo interakcie.';
        html += '</p>';
        html += '<button id="compareMatrixR2Btn" style="background:#cc0000;color:white;border:none;padding:10px 20px;cursor:pointer;font-size:14px;font-weight:bold;border-radius:4px;">Porovnať R² (Model vs Matica)</button>';
        html += ' <span id="matrixR2Status" style="margin-left:12px;font-size:0.9em;color:#6c757d;"></span>';
        html += '<div id="matrixR2Results" style="margin-top:15px;"></div>';
        html += '</div>';
    }

    // ── PDF Export button ──
    html += '<div style="margin-top:20px;padding:15px;background:#fff0f0;border:1px solid #cc0000;border-radius:4px;">';
    html += '<h4 style="margin:0 0 8px 0;color:#cc0000;">Exportovať výsledky do PDF</h4>';
    html += '<p style="margin:0 0 12px 0;font-size:12px;color:#495057;">Uložte si kompletný prehľad výsledkov trénovania, scatter plot SMC vs MI, a matice korelácií / MI do PDF súboru.</p>';
    html += '<button id="exportPdfBtn" style="background:#cc0000;color:white;border:none;padding:10px 20px;cursor:pointer;font-size:14px;font-weight:bold;border-radius:4px;">Stiahnuť PDF report</button>';
    html += ' <span id="pdfExportStatus" style="margin-left:12px;font-size:0.9em;color:#6c757d;"></span>';
    html += '</div>';

    // Synergy analysis section
    html += '<div style="margin-top:20px;padding:15px;background:#fff0f0;border:1px solid #cc0000;border-radius:4px;">';
    html += '<h4 style="margin:0 0 8px 0;color:#cc0000;">Analýza synergií medzi features (Synergická MI)</h4>';
    html += '<p style="margin:0 0 12px 0;font-size:12px;color:#495057;">Vypočíta Joint Mutual Information MI((F1, F2); target) pre dvojice features.<br>Ak je Synergická MI &gt; MI(F1;target) + MI(F2;target), dvojica má <strong>synergiu</strong> — spolu nesú VIAC informácie o cieľovej premennej ako samostatne.</p>';
    html += '<div style="display:flex;gap:10px;flex-wrap:wrap;">';
    html += '<button id="synergyWithSelectedBtn" style="background:#cc0000;color:white;border:none;padding:10px 16px;cursor:pointer;font-size:13px;font-weight:bold;border-radius:4px;">Synergie nevybraných s vybranými</button>';
    html += '<button id="synergyAmongUnselectedBtn" style="background:#cc0000;color:white;border:none;padding:10px 16px;cursor:pointer;font-size:13px;font-weight:bold;border-radius:4px;">Synergie medzi nevybranými</button>';
    html += '</div>';
    html += '<span id="synergyStatus" style="display:block;margin-top:8px;font-size:0.9em;color:#6c757d;"></span>';
    html += '<div id="synergyResults" style="margin-top:15px;"></div>';
    html += '</div>';

    resultsDiv.innerHTML = html;

    // Attach event handler for embedded methods comparison
    const embeddedBtn = document.getElementById('compareEmbeddedBtn');
    if (embeddedBtn) {
        embeddedBtn.onclick = async () => {
            await compareWithEmbeddedMethods(allResults, totalFeatures, featureNames, isClassification);
        };
    }

    // Attach event handler for Matrix R² comparison (only for regression)
    if (!isClassification) {
        const matrixR2Btn = document.getElementById('compareMatrixR2Btn');
        if (matrixR2Btn) {
            matrixR2Btn.onclick = async () => {
                await compareMatrixR2(allResults, totalFeatures, featureNames);
            };
        }
    }

    // Attach PDF export button
    const pdfBtn = document.getElementById('exportPdfBtn');
    if (pdfBtn) {
        pdfBtn.onclick = async () => {
            await generatePdfReport(allResults, totalFeatures, featureNames, isClassification);
        };
    }

    // Attach synergy analysis buttons
    const synergyWithBtn = document.getElementById('synergyWithSelectedBtn');
    if (synergyWithBtn) {
        synergyWithBtn.onclick = async () => {
            await runSynergyAnalysis('with_selected');
        };
    }
    const synergyAmongBtn = document.getElementById('synergyAmongUnselectedBtn');
    if (synergyAmongBtn) {
        synergyAmongBtn.onclick = async () => {
            await runSynergyAnalysis('among_unselected');
        };
    }
}

// ─── PDF Report Generation ───
async function generatePdfReport(allResults, totalFeatures, featureNames, isClassification) {
    const statusSpan = document.getElementById('pdfExportStatus');
    statusSpan.textContent = 'Generujem PDF report...';
    statusSpan.style.color = '#3498db';

    try {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF({ orientation: 'landscape', unit: 'mm', format: 'a4' });
        const pageWidth = pdf.internal.pageSize.getWidth();
        const pageHeight = pdf.internal.pageSize.getHeight();
        const margin = 15;
        const usableWidth = pageWidth - 2 * margin;

        // ── Page 1: Title + Training Comparison Table ──
        pdf.setFillColor(204, 0, 0);
        pdf.rect(0, 0, pageWidth, 25, 'F');
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(18);
        pdf.setFont('helvetica', 'bold');
        pdf.text('ML Pipeline - Report výsledkov', pageWidth / 2, 16, { align: 'center' });

        pdf.setTextColor(80, 80, 80);
        pdf.setFontSize(10);
        pdf.setFont('helvetica', 'normal');
        const now = new Date();
        pdf.text(`Dátum: ${now.toLocaleDateString('sk-SK')} ${now.toLocaleTimeString('sk-SK')}`, margin, 35);
        pdf.text(`Počet features: ${totalFeatures}`, margin, 41);
        const targetCol = window._lastComparisonResult ? window._lastComparisonResult.target_column : '-';
        pdf.text(`Cieľová premenná: ${targetCol}`, margin, 47);
        pdf.text(`Typ úlohy: ${isClassification ? 'Klasifikácia' : 'Regresia'}`, margin, 53);

        // Build table data
        const successResults = allResults.filter(r => r.result);
        const failedResults = allResults.filter(r => r.error);

        let tableHeaders, tableBody;
        if (isClassification) {
            tableHeaders = ['Metóda', 'Features', 'ACC', 'F1', 'PREC', 'REC', 'SPEC', 'FP', 'FN', 'MCC', 'Čas(ms)'];
            tableBody = allResults.map(r => {
                const row = [r.name, `${r.count}/${totalFeatures}`];
                if (r.error) {
                    row.push({ content: r.error, colSpan: 9, styles: { textColor: [231, 76, 60], fontSize: 7 } });
                } else if (r.result) {
                    row.push((r.result.accuracy * 100).toFixed(1) + '%');
                    row.push(r.result.f1_score.toFixed(3));
                    row.push(r.result.precision.toFixed(3));
                    row.push(r.result.recall.toFixed(3));
                    row.push(r.result.specificity.toFixed(3));
                    row.push(Math.round(r.result.false_positives).toString());
                    row.push(Math.round(r.result.false_negatives).toString());
                    row.push(r.result.mcc.toFixed(3));
                    row.push(r.timeMs ? r.timeMs.toFixed(0) : '-');
                }
                return row;
            });
        } else {
            tableHeaders = ['Metóda', 'Features', 'R²', 'RMSE', 'MAE', 'MAPE', 'MedAE', 'CORR', 'Čas(ms)'];
            tableBody = allResults.map(r => {
                const row = [r.name, `${r.count}/${totalFeatures}`];
                if (r.error) {
                    row.push({ content: r.error, colSpan: 7, styles: { textColor: [231, 76, 60], fontSize: 7 } });
                } else if (r.result) {
                    row.push(r.result.r2_score.toFixed(4));
                    row.push(r.result.rmse.toFixed(3));
                    row.push(r.result.mae.toFixed(3));
                    row.push(r.result.mape ? r.result.mape.toFixed(2) + '%' : 'N/A');
                    row.push(r.result.median_absolute_error.toFixed(3));
                    row.push(r.result.pearson_correlation.toFixed(3));
                    row.push(r.timeMs ? r.timeMs.toFixed(0) : '-');
                }
                return row;
            });
        }

        // Find best result
        let bestIdx = -1, bestVal = -Infinity;
        allResults.forEach((r, i) => {
            if (!r.result) return;
            const val = isClassification ? r.result.accuracy : r.result.r2_score;
            if (val > bestVal) { bestVal = val; bestIdx = i; }
        });

        // Draw table manually
        let y = 60;
        const colCount = tableHeaders.length;
        const colWidths = tableHeaders.map((h, i) => {
            if (i === 0) return usableWidth * 0.22;
            return (usableWidth * 0.78) / (colCount - 1);
        });

        // Header row
        pdf.setFillColor(204, 0, 0);
        let x = margin;
        const headerH = 8;
        tableHeaders.forEach((h, i) => {
            pdf.rect(x, y, colWidths[i], headerH, 'F');
            pdf.setTextColor(255, 255, 255);
            pdf.setFontSize(7);
            pdf.setFont('helvetica', 'bold');
            pdf.text(h, x + 2, y + 5.5);
            x += colWidths[i];
        });
        y += headerH;

        // Data rows
        tableBody.forEach((row, rowIdx) => {
            if (y > pageHeight - 15) {
                pdf.addPage();
                y = margin;
            }
            const rowH = 7;
            x = margin;
            const isBest = rowIdx === bestIdx;
            row.forEach((cell, colIdx) => {
                const cellContent = typeof cell === 'object' ? cell.content : String(cell);
                const cs = typeof cell === 'object' && cell.colSpan ? cell.colSpan : 1;
                let w = 0;
                for (let c = 0; c < cs; c++) w += colWidths[colIdx + c];

                if (isBest) pdf.setFillColor(255, 228, 225);
                else if (rowIdx % 2 === 0) pdf.setFillColor(248, 249, 250);
                else pdf.setFillColor(255, 255, 255);
                pdf.rect(x, y, w, rowH, 'F');

                // Border
                pdf.setDrawColor(222, 226, 230);
                pdf.rect(x, y, w, rowH, 'S');

                if (typeof cell === 'object' && cell.styles && cell.styles.textColor) {
                    pdf.setTextColor(...cell.styles.textColor);
                } else if (colIdx === 0) {
                    pdf.setTextColor(51, 51, 51);
                } else if (colIdx === 2 && isBest) {
                    pdf.setTextColor(204, 0, 0);
                } else {
                    pdf.setTextColor(80, 80, 80);
                }
                pdf.setFontSize(colIdx === 0 ? 7 : 7);
                pdf.setFont('helvetica', colIdx === 0 || isBest ? 'bold' : 'normal');
                pdf.text(cellContent, x + 1.5, y + 5, { maxWidth: w - 3 });
                x += w;
            });
            y += rowH;
        });

        // Legend
        y += 5;
        if (y > pageHeight - 30) { pdf.addPage(); y = margin; }
        pdf.setFontSize(8);
        pdf.setTextColor(80, 80, 80);
        pdf.setFont('helvetica', 'normal');
        if (isClassification) {
            pdf.text('ACC=Accuracy | F1=F1 Score | PREC=Precision | REC=Recall | SPEC=Specificity | FP=False Positives | FN=False Negatives | MCC=Matthews Corr. Coeff.', margin, y);
        } else {
            pdf.text('R²=Koef. determinácie | RMSE=Root Mean Squared Error | MAE=Mean Abs Error | MAPE=Mean Abs % Error | MedAE=Median Abs Error | CORR=Pearson korelácia', margin, y);
        }

        // Feature list
        y += 8;
        if (y > pageHeight - 20) { pdf.addPage(); y = margin; }
        pdf.setFontSize(9);
        pdf.setFont('helvetica', 'bold');
        pdf.setTextColor(204, 0, 0);
        pdf.text('Vybrané features jednotlivými metódami:', margin, y);
        y += 5;
        pdf.setFont('helvetica', 'normal');
        pdf.setFontSize(7);
        pdf.setTextColor(80, 80, 80);
        allResults.forEach(r => {
            if (r.indices === null) return;
            if (y > pageHeight - 10) { pdf.addPage(); y = margin; }
            const names = r.indices && featureNames.length > 0
                ? r.indices.map(i => featureNames[i] || `[${i}]`).join(', ')
                : (r.indices || []).join(', ');
            const line = `${r.name} (${r.count}): ${names || 'N/A'}`;
            const lines = pdf.splitTextToSize(line, usableWidth);
            lines.forEach(l => {
                if (y > pageHeight - 10) { pdf.addPage(); y = margin; }
                pdf.text(l, margin, y);
                y += 4;
            });
        });

        statusSpan.textContent = 'Zachytávam grafy...';
        await new Promise(r => setTimeout(r, 50));

        // ── Page 2: SMC vs MI Scatter Plot ──
        const scatterSvg = document.querySelector('#targetVariableMap svg');
        if (scatterSvg) {
            pdf.addPage();
            pdf.setFillColor(204, 0, 0);
            pdf.rect(0, 0, pageWidth, 18, 'F');
            pdf.setTextColor(255, 255, 255);
            pdf.setFontSize(14);
            pdf.setFont('helvetica', 'bold');
            pdf.text('Scatter plot: SMC vs Mutual Information', pageWidth / 2, 12, { align: 'center' });

            try {
                const svgContainer = scatterSvg.parentElement;
                const canvas = await html2canvas(svgContainer, {
                    backgroundColor: '#ffffff',
                    scale: 2,
                    logging: false,
                    useCORS: true
                });
                const imgData = canvas.toDataURL('image/png');
                const imgW = usableWidth;
                const imgH = (canvas.height / canvas.width) * imgW;
                const maxImgH = pageHeight - 35;
                const finalH = Math.min(imgH, maxImgH);
                const finalW = (finalH / imgH) * imgW;
                pdf.addImage(imgData, 'PNG', margin + (usableWidth - finalW) / 2, 25, finalW, finalH);
            } catch (e) {
                pdf.setTextColor(150, 150, 150);
                pdf.setFontSize(10);
                pdf.text('(Scatter plot sa nepodarilo zachytiť)', margin, 35);
            }
        }

        // ── Page 3: Target Variable Map ──
        const targetMapDiv = document.getElementById('targetVariableMap');
        if (targetMapDiv && targetMapDiv.children.length > 0) {
            pdf.addPage();
            pdf.setFillColor(204, 0, 0);
            pdf.rect(0, 0, pageWidth, 18, 'F');
            pdf.setTextColor(255, 255, 255);
            pdf.setFontSize(14);
            pdf.setFont('helvetica', 'bold');
            pdf.text('Mapa premenných - cieľová premenná', pageWidth / 2, 12, { align: 'center' });

            try {
                const canvas = await html2canvas(targetMapDiv, {
                    backgroundColor: '#ffffff',
                    scale: 2,
                    logging: false,
                    useCORS: true,
                    width: Math.min(targetMapDiv.scrollWidth, 1200)
                });
                const imgData = canvas.toDataURL('image/png');
                const imgW = usableWidth;
                const imgH = (canvas.height / canvas.width) * imgW;
                const maxImgH = pageHeight - 35;
                const finalH = Math.min(imgH, maxImgH);
                const finalW = (finalH / imgH) * imgW;
                pdf.addImage(imgData, 'PNG', margin + (usableWidth - finalW) / 2, 25, finalW, finalH);
            } catch (e) {
                pdf.setTextColor(150, 150, 150);
                pdf.setFontSize(10);
                pdf.text('(Mapu premenných sa nepodarilo zachytiť)', margin, 35);
            }
        }

        // ── Page 4+5: Correlation & MI Matrices ──
        const appendHeatmapMatrixPages = async () => {
            for (const matrixMode of ['correlation', 'mi']) {
                pdf.addPage();
                pdf.setFillColor(204, 0, 0);
                pdf.rect(0, 0, pageWidth, 18, 'F');
                pdf.setTextColor(255, 255, 255);
                pdf.setFontSize(14);
                pdf.setFont('helvetica', 'bold');
                const matrixTitle = matrixMode === 'correlation'
                    ? 'Matica korelácií (Pearson)'
                    : 'Matica Mutual Information (KSG)';
                pdf.text(matrixTitle, pageWidth / 2, 12, { align: 'center' });

                try {
                    const heatmapCanvas = renderHeatmapToCanvas(matrixMode);
                    const imgData = heatmapCanvas.toDataURL('image/png');
                    const imgW = usableWidth;
                    const imgH = (heatmapCanvas.height / heatmapCanvas.width) * imgW;
                    const maxImgH = pageHeight - 35;
                    const finalH = Math.min(imgH, maxImgH);
                    const finalW = (finalH / imgH) * imgW;
                    pdf.addImage(imgData, 'PNG', margin + (usableWidth - finalW) / 2, 25, finalW, finalH);
                } catch (e) {
                    pdf.setTextColor(150, 150, 150);
                    pdf.setFontSize(10);
                    pdf.text(`(Maticu ${matrixMode} sa nepodarilo vygenerovať: ${e})`, margin, 35);
                }
            }
        };

        // Generate heatmap images if data is available
        if (_heatmapData) {
            await appendHeatmapMatrixPages();
        } else {
            // Try to compute heatmap data on the fly
            try {
                statusSpan.textContent = 'Počítam matice korelácií a MI...';
                await new Promise(r => setTimeout(r, 50));
                const rawResult = pipeline.getFeatureMatrices(window.rawDataString, window.dataFormat);
                _heatmapData = convertWasmResult(rawResult);

                await appendHeatmapMatrixPages();
            } catch (e) {
                console.warn('Could not compute matrices for PDF:', e);
            }
        }

        // ── Optional: Matrix R² vs Model R² section (regression only) ──
        if (!isClassification) {
            try {
                const matrixR2Results = document.getElementById('matrixR2Results');
                if (matrixR2Results) {
                    // If user did not run comparison yet, generate it before export.
                    if (!matrixR2Results.textContent || !matrixR2Results.textContent.trim()) {
                        statusSpan.textContent = 'Počítam sekciu Model R² vs Maticové R²...';
                        await compareMatrixR2(allResults, totalFeatures, featureNames);
                        await new Promise(r => setTimeout(r, 80));
                    }

                    const matrixR2Section = matrixR2Results.parentElement;
                    if (matrixR2Section && matrixR2Results.textContent && matrixR2Results.textContent.trim()) {
                        pdf.addPage();
                        pdf.setFillColor(204, 0, 0);
                        pdf.rect(0, 0, pageWidth, 18, 'F');
                        pdf.setTextColor(255, 255, 255);
                        pdf.setFontSize(14);
                        pdf.setFont('helvetica', 'bold');
                        pdf.text('Porovnanie Model R² vs Maticové R²', pageWidth / 2, 12, { align: 'center' });

                        try {
                            const canvas = await html2canvas(matrixR2Section, {
                                backgroundColor: '#ffffff',
                                scale: 2,
                                logging: false,
                                useCORS: true,
                                width: Math.min(matrixR2Section.scrollWidth, 1400)
                            });
                            const imgData = canvas.toDataURL('image/png');
                            const imgW = usableWidth;
                            const imgH = (canvas.height / canvas.width) * imgW;
                            const maxImgH = pageHeight - 35;
                            const finalH = Math.min(imgH, maxImgH);
                            const finalW = (finalH / imgH) * imgW;
                            pdf.addImage(imgData, 'PNG', margin + (usableWidth - finalW) / 2, 25, finalW, finalH);
                        } catch (e) {
                            pdf.setTextColor(150, 150, 150);
                            pdf.setFontSize(10);
                            pdf.text('(Sekciu Model R² vs Maticové R² sa nepodarilo zachytiť)', margin, 35);
                        }
                    }
                }
            } catch (e) {
                console.warn('Could not include Matrix R² section in PDF:', e);
            }
        }

        // Save PDF + text report for copy/paste into Word
        const dateStamp = now.toISOString().slice(0,10);
        const filename = `ml-pipeline-report-${dateStamp}.pdf`;
        const txtFilename = `ml-pipeline-report-${dateStamp}.txt`;
        pdf.save(filename);

        const textReport = buildPlainTextReport(allResults, totalFeatures, featureNames, isClassification, now);
        downloadTextFile(textReport, txtFilename);

        statusSpan.textContent = `PDF a TXT uložené: ${filename}, ${txtFilename}`;
        statusSpan.style.color = '#27ae60';
    } catch (err) {
        console.error('PDF generation error:', err);
        statusSpan.textContent = `Chyba pri generovaní PDF: ${err}`;
        statusSpan.style.color = '#e74c3c';
    }
}

function buildPlainTextReport(allResults, totalFeatures, featureNames, isClassification, now) {
    const lines = [];
    const targetCol = window._lastComparisonResult ? window._lastComparisonResult.target_column : '-';
    const taskType = isClassification ? 'Klasifikacia' : 'Regresia';

    lines.push('ML Pipeline - textovy report');
    lines.push('====================================');
    lines.push(`Datum: ${now.toLocaleDateString('sk-SK')} ${now.toLocaleTimeString('sk-SK')}`);
    lines.push(`Cielova premenna: ${targetCol}`);
    lines.push(`Typ ulohy: ${taskType}`);
    lines.push(`Pocet features: ${totalFeatures}`);
    lines.push('');

    lines.push('Vysledky trenovania');
    lines.push('------------------------------------');

    if (isClassification) {
        lines.push('Metoda | Features | ACC | F1 | PREC | REC | SPEC | FP | FN | MCC | Cas(ms)');
    } else {
        lines.push('Metoda | Features | R2 | RMSE | MAE | MAPE | MedAE | CORR | Cas(ms)');
    }

    allResults.forEach(r => {
        const base = `${r.name} | ${r.count}/${totalFeatures}`;
        if (r.error) {
            lines.push(`${base} | CHYBA: ${r.error}`);
            return;
        }
        if (!r.result) {
            lines.push(`${base} | N/A`);
            return;
        }

        if (isClassification) {
            lines.push([
                base,
                `${(r.result.accuracy * 100).toFixed(1)}%`,
                r.result.f1_score.toFixed(3),
                r.result.precision.toFixed(3),
                r.result.recall.toFixed(3),
                r.result.specificity.toFixed(3),
                String(Math.round(r.result.false_positives)),
                String(Math.round(r.result.false_negatives)),
                r.result.mcc.toFixed(3),
                r.timeMs ? r.timeMs.toFixed(0) : '-'
            ].join(' | '));
        } else {
            const mapeText = (r.result.mape !== undefined && r.result.mape !== null)
                ? `${r.result.mape.toFixed(2)}%`
                : 'N/A';
            lines.push([
                base,
                r.result.r2_score.toFixed(4),
                r.result.rmse.toFixed(3),
                r.result.mae.toFixed(3),
                mapeText,
                r.result.median_absolute_error.toFixed(3),
                r.result.pearson_correlation.toFixed(3),
                r.timeMs ? r.timeMs.toFixed(0) : '-'
            ].join(' | '));
        }
    });

    lines.push('');
    lines.push('Vybrane features jednotlivymi metodami');
    lines.push('------------------------------------');

    allResults.forEach(r => {
        if (r.indices === null) {
            lines.push(`${r.name} (${r.count}): bez selekcie (vsetky features)`);
            return;
        }
        const names = r.indices && featureNames.length > 0
            ? r.indices.map(i => featureNames[i] || `[${i}]`).join(', ')
            : (r.indices || []).join(', ');
        lines.push(`${r.name} (${r.count}): ${names || 'N/A'}`);
    });

    if (!isClassification) {
        const matrixR2Section = document.getElementById('matrixR2Results');
        if (matrixR2Section && matrixR2Section.textContent && matrixR2Section.textContent.trim()) {
            lines.push('');
            lines.push('Sekcia Model R2 vs Maticove R2');
            lines.push('------------------------------------');
            lines.push(matrixR2Section.textContent.replace(/\s+/g, ' ').trim());
        }
    }

    lines.push('');
    lines.push('Koniec reportu');
    return lines.join('\n');
}

function downloadTextFile(content, filename) {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

// Render heatmap matrix to an offscreen canvas for PDF export
function renderHeatmapToCanvas(mode) {
    if (!_heatmapData) throw new Error('Heatmap data not available');
    const matrix = mode === 'correlation' ? _heatmapData.correlation_matrix : _heatmapData.mi_matrix;
    const names = _heatmapData.feature_names;
    const n = names.length;

    // Sizing
    let cellSize = n <= 10 ? 40 : n <= 15 ? 32 : n <= 22 ? 26 : n <= 35 ? 20 : 16;
    const headerSize = Math.max(60, Math.min(120, n * 3 + 30));
    const canvasW = headerSize + n * cellSize + 20;
    const canvasH = headerSize + n * cellSize + 20;

    const canvas = document.createElement('canvas');
    canvas.width = canvasW * 2;
    canvas.height = canvasH * 2;
    const ctx = canvas.getContext('2d');
    ctx.scale(2, 2);
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvasW, canvasH);

    // Normalize MI values
    let maxVal = 1.0;
    if (mode === 'mi') {
        maxVal = 0;
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                if (i !== j && matrix[i][j] > maxVal) maxVal = matrix[i][j];
        if (maxVal === 0) maxVal = 1;
    }

    // Draw cells
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = matrix[i][j];
            const x = headerSize + j * cellSize;
            const y = headerSize + i * cellSize;

            ctx.fillStyle = heatmapColor(mode === 'mi' ? val / maxVal : val, mode);
            ctx.fillRect(x, y, cellSize, cellSize);

            // Border
            ctx.strokeStyle = '#e0e0e0';
            ctx.lineWidth = 0.5;
            ctx.strokeRect(x, y, cellSize, cellSize);

            // Value text
            if (n <= 20) {
                const fontSize = cellSize <= 20 ? 6 : cellSize <= 26 ? 7 : 8;
                ctx.fillStyle = textColor(mode === 'mi' ? val / maxVal : val, mode);
                ctx.font = `${fontSize}px sans-serif`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(val.toFixed(2), x + cellSize / 2, y + cellSize / 2);
            }
        }
    }

    // Draw labels
    const labelFontSize = n <= 10 ? 9 : n <= 15 ? 8 : n <= 22 ? 7 : 6;
    ctx.fillStyle = '#333';
    ctx.font = `${labelFontSize}px sans-serif`;

    // Left labels (row headers)
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < n; i++) {
        const label = names[i].length > 15 ? names[i].substring(0, 14) + '…' : names[i];
        ctx.fillText(label, headerSize - 4, headerSize + i * cellSize + cellSize / 2);
    }

    // Top labels (column headers, rotated)
    ctx.save();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let j = 0; j < n; j++) {
        const label = names[j].length > 15 ? names[j].substring(0, 14) + '…' : names[j];
        const x = headerSize + j * cellSize + cellSize / 2;
        const y = headerSize - 4;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(-Math.PI / 4);
        ctx.fillStyle = '#333';
        ctx.fillText(label, 0, 0);
        ctx.restore();
    }
    ctx.restore();

    // Color scale legend at bottom
    const legendY = headerSize + n * cellSize + 8;
    const legendW = Math.min(200, canvasW - 2 * 20);
    const legendH = 10;
    const legendX = (canvasW - legendW) / 2;

    for (let px = 0; px < legendW; px++) {
        const t = px / legendW;
        if (mode === 'correlation') {
            ctx.fillStyle = heatmapColor(t * 2 - 1, mode);
        } else {
            ctx.fillStyle = heatmapColor(t, mode);
        }
        ctx.fillRect(legendX + px, legendY, 1, legendH);
    }
    ctx.strokeStyle = '#999';
    ctx.strokeRect(legendX, legendY, legendW, legendH);

    ctx.fillStyle = '#333';
    ctx.font = '8px sans-serif';
    ctx.textAlign = 'center';
    if (mode === 'correlation') {
        ctx.fillText('-1', legendX, legendY + legendH + 10);
        ctx.fillText('0', legendX + legendW / 2, legendY + legendH + 10);
        ctx.fillText('+1', legendX + legendW, legendY + legendH + 10);
    } else {
        ctx.fillText('0', legendX, legendY + legendH + 10);
        ctx.fillText(maxVal.toFixed(2), legendX + legendW, legendY + legendH + 10);
    }

    return canvas;
}

// ─── Synergy analysis (Synergická MI) ───
async function runSynergyAnalysis(mode) {
    const statusSpan = document.getElementById('synergyStatus');
    const resultsDiv = document.getElementById('synergyResults');
    const comparison = window._lastComparisonResult;

    if (!comparison) {
        statusSpan.textContent = 'Najprv spustite porovnanie selektorov.';
        statusSpan.style.color = '#e74c3c';
        return;
    }

    // Zbieraj vybrané indexy z checkboxov
    const selectedIndices = [];
    document.querySelectorAll('#userFeatureMap input[type=checkbox]:checked').forEach(cb => {
        selectedIndices.push(parseInt(cb.dataset.featureIdx));
    });

    if (mode === 'with_selected' && selectedIndices.length === 0) {
        statusSpan.textContent = 'Najprv vyberte aspoň jednu feature v mape vyššie.';
        statusSpan.style.color = '#e74c3c';
        return;
    }

    try {
        statusSpan.textContent = 'Počítam Joint Mutual Information pre dvojice... (môže trvať niekoľko sekúnd)';
        statusSpan.style.color = '#3498db';

        const rawResult = await pipeline.computeSynergyAnalysis(
            window.rawDataString, window.dataFormat,
            comparison.target_column, selectedIndices, mode
        );
        const result = convertWasmResult(rawResult);
        renderSynergyResults(result, resultsDiv, mode === 'with_selected' ? new Set(selectedIndices) : new Set());
        statusSpan.textContent = '';
    } catch (err) {
        statusSpan.textContent = `Chyba: ${err}`;
        statusSpan.style.color = '#e74c3c';
    }
}

function renderSynergyResults(result, div, selectedSet = new Set()) {
    const pairs = result.pairs || [];
    const mode = result.mode;
    const withSelected = mode === 'with_selected';

    const modeLabel = withSelected
        ? 'Synergie nevybraných features s vybranými'
        : 'Synergie medzi nevybranými features';

    let html = `<h4 style="color:#cc0000;margin:0 0 10px;">${modeLabel}</h4>`;
    html += '<p style="font-size:11px;color:#6c757d;margin-bottom:10px;">';
    html += 'Synergická MI = MI((F1, F2); target) — koľko informácie dvojica spoločne nesie o cieľovej premennej.<br>';
    html += 'Prínos Synergickej MI = Synergická MI − (MI(F1;target) + MI(F2;target)). <strong style="color:#27ae60;">Kladná hodnota = dvojica má synergiu</strong> (spolu nesú VIAC informácie ako samostatne).</p>';

    if (withSelected && selectedSet.size > 0) {
        html += '<div style="margin-bottom:10px;padding:7px 12px;background:#e8f4fd;border-left:4px solid #2980b9;border-radius:3px;font-size:12px;">';
        html += '<strong style="color:#2980b9;">&#9654; V ľavom stĺpci sú vždy vybrané premenné, v pravom sú tie ktoré neboli vybrané</strong>';
        html += '</div>';
    }

    if (pairs.length === 0) {
        html += '<p style="color:#6c757d;font-style:italic;">Nenašli sa žiadne dvojice na analýzu.</p>';
        div.innerHTML = html;
        return;
    }

    // Ak sme v režime with_selected, zaruč že vybraná feature je vždy vľavo
    const normalizedPairs = pairs.map(p => {
        if (withSelected && selectedSet.size > 0) {
            const idx1Selected = selectedSet.has(p.idx1);
            const idx2Selected = selectedSet.has(p.idx2);
            if (!idx1Selected && idx2Selected) {
                // Otočiť: selected ide vľavo
                return { ...p,
                    idx1: p.idx2, name1: p.name2, mi1: p.mi2,
                    idx2: p.idx1, name2: p.name1, mi2: p.mi1,
                };
            }
        }
        return p;
    });

    // Rozdeľ na synergické (synergy > 0) a ostatné
    const synergisticPairs = normalizedPairs.filter(p => p.synergy > 0.001);
    const otherPairs      = normalizedPairs.filter(p => p.synergy <= 0.001);

    // Top 3 sú prvé 3 synergické (zoradené zostupne podľa synergy)
    const top3Indices = new Set(synergisticPairs.slice(0, 3).map((_, i) => i));

    const rankLabels  = ['1.', '2.', '3.'];
    const top3Styles  = [
        'background:#fff8c0;border-left:3px solid #e6c200;',
        'background:#fff8c0;border-left:3px solid #e6c200;',
        'background:#fff8c0;border-left:3px solid #e6c200;',
    ];

    html += '<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;font-size:12px;">';
    html += '<thead><tr>';
    html += '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;width:28px;">#</th>';
    const f1Header = withSelected
        ? '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#1a6fa0;color:white;text-align:left;">Feature 1</th>'
        : '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:left;">Feature 1</th>';
    html += f1Header;
    html += '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;" title="MI(F1; target)">MI₁</th>';
    html += '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:left;">Feature 2</th>';
    html += '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;" title="MI(F2; target)">MI₂</th>';
    html += '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;" title="MI(F1;target) + MI(F2;target)">MI₁+MI₂</th>';
    html += '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-weight:bold;" title="MI((F1,F2); target)">Synergická MI</th>';
    html += '<th style="padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;" title="Synergická MI − (MI₁+MI₂)">Prínos Synergickej MI</th>';
    html += '</tr></thead><tbody>';

    // Najprv synergické páry, potom top 15 ostatných
    const displayPairs = [...synergisticPairs, ...otherPairs.slice(0, 15)];

    displayPairs.forEach((p, idx) => {
        const hasSynergy = p.synergy > 0.001;
        const isTop3     = hasSynergy && top3Indices.has(idx);
        const rank        = isTop3 ? idx : -1;

        let rowStyle;
        if (isTop3) {
            rowStyle = top3Styles[rank];
        } else if (hasSynergy) {
            rowStyle = 'background:#c8e6c9;';
        } else {
            rowStyle = idx % 2 === 0 ? 'background:#f8f9fa;' : '';
        }

        const synergyStyle = hasSynergy ? 'font-weight:bold;color:#1e8449;' : 'color:#e74c3c;';
        const rankBadge    = isTop3 ? `${rankLabels[rank]} ` : '';

        html += `<tr style="${rowStyle}">`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-size:14px;">${rankBadge || (hasSynergy ? '✓' : '')}</td>`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;"<strong style="color:#cc0000;">F${p.idx1}</strong>: ${p.name1}</td>`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;">${p.mi1.toFixed(4)}</td>`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;"><strong style="color:#cc0000;">F${p.idx2}</strong>: ${p.name2}</td>`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;">${p.mi2.toFixed(4)}</td>`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;">${p.individual_sum.toFixed(4)}</td>`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;">${p.joint_mi.toFixed(4)}</td>`;
        html += `<td style="padding:6px;border:1px solid #dee2e6;text-align:center;${synergyStyle}">${hasSynergy ? '+' : ''}${p.synergy.toFixed(4)}${hasSynergy ? ' ✓' : ''}</td>`;
        html += '</tr>';
    });

    html += '</tbody></table></div>';

    // Sumár
    if (synergisticPairs.length > 0) {
        html += '<div style="margin-top:12px;padding:10px;background:#c8e6c9;border-left:4px solid #27ae60;font-size:12px;">';
        html += `<strong>Nájdených ${synergisticPairs.length} synergických dvojíc</strong> (Synergická MI &gt; MI₁ + MI₂).<br>`;
        html += 'Tieto features spolu nesú VIAC informácie o cieľovej premennej ako súčet ich individuálnych MI. ';
        html += 'Zvážte ich spoločné zaradenie do modelu.<br>';
        const top3 = synergisticPairs.slice(0, 3);
        html += '<strong>Top 3 páry:</strong> ';
        html += top3.map((p, i) => `${rankLabels[i]} <strong>${p.name1}</strong> + <strong>${p.name2}</strong> (Prínos MI +${p.synergy.toFixed(4)})`).join(', ');
        html += '</div>';
    } else {
        html += '<div style="margin-top:12px;padding:10px;background:#fff3cd;border-left:4px solid #ffc107;font-size:12px;">';
        html += '<strong>Info:</strong> Žiadna analyzovaná dvojica nevykazuje synergiu (Synergická MI ≤ MI₁ + MI₂ pre všetky páry).';
        html += '</div>';
    }

    div.innerHTML = html;
}

// ─── Matrix R² vs Model R² comparison ───
async function compareMatrixR2(allResults, totalFeatures, featureNames) {
    const statusSpan = document.getElementById('matrixR2Status');
    const resultsDiv = document.getElementById('matrixR2Results');
    const btn = document.getElementById('compareMatrixR2Btn');

    try {
        btn.disabled = true;
        statusSpan.textContent = 'Počítam maticové R² pre všetky varianty...';
        statusSpan.style.color = '#3498db';

        // For each result that has feature indices and an R² score, compute matrix R²
        let comparisons = [];
        for (const r of allResults) {
            if (!r.result || r.result.r2_score === undefined) continue;

            // Determine feature indices for this variant
            let featureIndices;
            if (r.indices === null) {
                // Baseline: all features
                featureIndices = Array.from({length: totalFeatures}, (_, i) => i);
            } else {
                featureIndices = r.indices;
            }

            if (!featureIndices || featureIndices.length === 0) continue;

            try {
                const matrixResult = convertWasmResult(pipeline.computeMatrixR2(featureIndices));
                comparisons.push({
                    name: r.name,
                    modelR2: r.result.r2_score,
                    matrixR2: matrixResult.matrix_r2,
                    numFeatures: featureIndices.length,
                    featureCorrelations: matrixResult.feature_correlations || [],
                    isEmbedded: r.isEmbedded || false,
                    isUser: r.isUser || false,
                });
            } catch (e) {
                console.warn(`Matrix R² pre "${r.name}" zlyhalo:`, e);
            }
        }

        if (comparisons.length === 0) {
            resultsDiv.innerHTML = '<p style="color:#e74c3c;">Nemožno vypočítať maticové R² — žiadne úspešné výsledky.</p>';
            return;
        }

        // Render comparison table
        let html = '<table style="width:100%;border-collapse:collapse;font-size:12px;margin-top:10px;">';
        html += '<thead><tr>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:left;font-size:11px;">Metóda</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;">Features</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="R² z natrénovaného modelu">Model R²</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;" title="R² = rᵀᵧX R⁻¹ rᵧX z korelačnej matice">Maticové R²</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;">Rozdiel</th>';
        html += '<th style="padding:10px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:11px;">Diagnostika</th>';
        html += '</tr></thead><tbody>';

        comparisons.forEach((c, idx) => {
            const diff = c.modelR2 - c.matrixR2;
            const absDiff = Math.abs(diff);

            // Determine scenario
            let scenario, scenarioColor;
            if (absDiff < 0.05) {
                scenario = 'Lineárny vzťah';
                scenarioColor = '#28a745';
            } else if (diff > 0.05) {
                scenario = 'Nelineárne vzorce';
                scenarioColor = '#ff8c00';
            } else {
                scenario = 'Možný overfitting';
                scenarioColor = '#e74c3c';
            }

            const bg = idx % 2 === 0 ? 'background:#f8f9fa;' : '';
            const nameStyle = c.isUser ? 'color:#cc0000;' : c.isEmbedded ? 'color:#ff8c00;' : '';
            
            html += `<tr style="${bg}">`;
            html += `<td style="padding:8px 6px;border:1px solid #dee2e6;font-weight:bold;${nameStyle}">${c.name}</td>`;
            html += `<td style="padding:8px 6px;border:1px solid #dee2e6;text-align:center;">${c.numFeatures}/${totalFeatures}</td>`;
            html += `<td style="padding:8px 6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;color:#2c3e50;">${c.modelR2.toFixed(4)}</td>`;
            html += `<td style="padding:8px 6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;color:#3498db;">${c.matrixR2.toFixed(4)}</td>`;
            html += `<td style="padding:8px 6px;border:1px solid #dee2e6;text-align:center;color:${diff >= 0 ? '#28a745' : '#e74c3c'};">${diff >= 0 ? '+' : ''}${diff.toFixed(4)}</td>`;
            html += `<td style="padding:8px 6px;border:1px solid #dee2e6;text-align:center;color:${scenarioColor};font-weight:bold;">${scenario}</td>`;
            html += '</tr>';
        });
        html += '</tbody></table>';

        // Legend / explanation
        html += '<div style="margin-top:15px;padding:12px;background:#f8f9fa;border:1px solid #dee2e6;border-radius:4px;font-size:11px;color:#495057;">';
        html += '<strong>Vysvetlenie diagnostiky:</strong><br>';
        html += '<span style="color:#28a745;font-weight:bold;">Lineárny vzťah</span> (|rozdiel| &lt; 0.05): Model R² ≈ Maticové R². Vzťah medzi features a targetom je prevažne lineárny. Lineárna predikcia je pre tieto dáta vhodná.<br>';
        html += '<span style="color:#ff8c00;font-weight:bold;">Nelineárne vzorce</span> (Model R² &gt; Maticové R² + 0.05): Model zachytáva nelineárne vzory alebo interakcie, ktoré čistá korelačná matica nevidí. Model je kvalitnejší než by naznačovala lineárna teória.<br>';
        html += '<span style="color:#e74c3c;font-weight:bold;">Možný overfitting</span> (Model R² &lt; Maticové R² − 0.05): Model nedosahuje ani teoretické lineárne R². Možné príčiny: zle nastavený model, preučenie (overfitting), alebo zašumené testovacie dáta.<br>';
        html += '<br><strong>Vzorec:</strong> Maticové R² = r<sup>T</sup><sub>yX</sub> · R<sup>−1</sup> · r<sub>yX</sub>, kde r<sub>yX</sub> je vektor korelácií Y s features a R je korelačná matica medzi features.';
        html += '</div>';

        // Detailed feature correlations (collapsible)
        html += '<details style="margin-top:12px;border:1px solid #dee2e6;">';
        html += '<summary style="padding:10px;background:#f8f9fa;cursor:pointer;font-weight:600;color:#2c3e50;">Korelácie features s targetom (detail)</summary>';
        html += '<div style="padding:12px;">';

        // Show feature correlations for baseline (all features)
        const baseline = comparisons.find(c => c.numFeatures === totalFeatures);
        if (baseline && baseline.featureCorrelations.length > 0) {
            // Sort by absolute correlation descending
            const sorted = [...baseline.featureCorrelations].sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
            
            html += '<table style="width:100%;border-collapse:collapse;font-size:11px;">';
            html += '<thead><tr>';
            html += '<th style="padding:6px;border:1px solid #dee2e6;background:#e9ecef;text-align:left;">Feature</th>';
            html += '<th style="padding:6px;border:1px solid #dee2e6;background:#e9ecef;text-align:center;">r (korelácia)</th>';
            html += '<th style="padding:6px;border:1px solid #dee2e6;background:#e9ecef;text-align:center;">r² (variancia)</th>';
            html += '<th style="padding:6px;border:1px solid #dee2e6;background:#e9ecef;text-align:center;">Smer</th>';
            html += '</tr></thead><tbody>';

            sorted.forEach(fc => {
                const absCorr = Math.abs(fc.correlation);
                const barWidth = Math.round(absCorr * 100);
                const barColor = fc.correlation >= 0 ? '#3498db' : '#e74c3c';
                const direction = fc.correlation >= 0 ? '↗ pozitívna' : '↘ negatívna';
                
                html += '<tr>';
                html += `<td style="padding:5px 6px;border:1px solid #dee2e6;font-weight:bold;">${fc.name}</td>`;
                html += `<td style="padding:5px 6px;border:1px solid #dee2e6;text-align:center;">${fc.correlation.toFixed(4)}</td>`;
                html += `<td style="padding:5px 6px;border:1px solid #dee2e6;text-align:center;">`;
                html += `<div style="display:flex;align-items:center;gap:4px;">`;
                html += `<div style="background:${barColor};height:8px;width:${barWidth}%;border-radius:3px;min-width:2px;"></div>`;
                html += `<span>${fc.r_squared.toFixed(4)}</span>`;
                html += `</div></td>`;
                html += `<td style="padding:5px 6px;border:1px solid #dee2e6;text-align:center;color:${fc.correlation >= 0 ? '#3498db' : '#e74c3c'};font-size:10px;">${direction}</td>`;
                html += '</tr>';
            });
            html += '</tbody></table>';
        }
        html += '</div></details>';

        resultsDiv.innerHTML = html;
        statusSpan.textContent = 'Hotovo';
        statusSpan.style.color = '#28a745';
    } catch (e) {
        statusSpan.textContent = 'Chyba: ' + e.message;
        statusSpan.style.color = '#e74c3c';
        console.error('Matrix R² comparison error:', e);
    } finally {
        btn.disabled = false;
    }
}

async function compareWithEmbeddedMethods(allResults, totalFeatures, featureNames, isClassification) {
    const statusSpan = document.getElementById('embeddedComparisonStatus');
    const resultsDiv = document.getElementById('embeddedComparisonResults');
    const btn = document.getElementById('compareEmbeddedBtn');

    try {
        btn.disabled = true;
        statusSpan.textContent = 'Analyzujem features pomocou embedded metódy...';
        statusSpan.style.color = '#28a745';

        // Calculate average number of features from filter methods (exclude baseline and user selection)
        const filterResults = allResults.filter(r => r.indices && !r.isUser);
        const avgFeatures = filterResults.length > 0
            ? Math.round(filterResults.reduce((sum, r) => sum + r.count, 0) / filterResults.length)
            : 5; // default

        // Get training split ratio from UI
        const trainRatio = parseFloat(document.getElementById('trainTestSplit').value) / 100;

        // Call embedded feature ranking (pass isClassification from pipeline evaluation mode)
        const rankingResult = await pipeline.getEmbeddedFeatureRanking(trainRatio, avgFeatures, isClassification);
        const ranking = convertWasmResult(rankingResult);

        statusSpan.textContent = `Embedded metóda vybrala ${ranking.selected_indices.length} features, trénovanie...`;

        // Train model with embedded-selected features
        const t0 = performance.now();
        const embeddedTrainResult = await pipeline.trainWithFeatureIndices(
            trainRatio,
            ranking.selected_indices
        );
        const trainTime = performance.now() - t0;
        const embeddedResult = convertWasmResult(embeddedTrainResult);

        statusSpan.textContent = 'Hotovo!';
        statusSpan.style.color = '#28a745';
        btn.disabled = false;

        // Add embedded result to global results
        const embeddedEntry = {
            name: ranking.method,
            result: embeddedResult,
            indices: ranking.selected_indices,
            count: ranking.selected_indices.length,
            timeMs: trainTime,
            isEmbedded: true
        };

        // Check if embedded result already exists, replace it
        const existingIdx = globalComparisonResults.allResults.findIndex(r => r.isEmbedded);
        if (existingIdx >= 0) {
            globalComparisonResults.allResults[existingIdx] = embeddedEntry;
        } else {
            globalComparisonResults.allResults.push(embeddedEntry);
        }

        // Re-render the entire comparison table with embedded results included
        const comparisonResultsDiv = document.getElementById('comparisonTrainingResults');
        renderTrainingComparison(
            globalComparisonResults.allResults,
            globalComparisonResults.totalFeatures,
            globalComparisonResults.featureNames,
            comparisonResultsDiv
        );

        // Show success message in embedded section
        resultsDiv.innerHTML = '<p style="color:#28a745;font-weight:bold;">✓ Embedded metóda pridaná do tabuľky porovnania vyššie</p>';

    } catch (error) {
        statusSpan.textContent = `Chyba: ${error}`;
        statusSpan.style.color = '#e74c3c';
        resultsDiv.innerHTML = `<p style="color:#e74c3c;">Chyba pri embedded porovnaní: ${error}</p>`;
        btn.disabled = false;
        console.error('Embedded comparison error:', error);
    }
}

