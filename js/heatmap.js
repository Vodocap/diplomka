// Fullscreen heatmap matrix modal

async function openHeatmapModal() {
    const data = window.rawDataString;
    const format = window.dataFormat;
    if (!data) { alert('Najprv načítajte dáta'); return; }
    if (!pipeline) { alert('Pipeline nie je inicializovaný'); return; }

    const modal = document.getElementById('heatmapModal');
    modal.classList.add('show');
    document.getElementById('heatmapContainer').innerHTML = '<p style="color:#6c757d;padding:40px;text-align:center;">⏳ Počítam matice... (môže trvať niekoľko sekúnd)</p>';

    try {
        showLoading(true);
        const rawResult = pipeline.getFeatureMatrices(data, format);
        _heatmapData = convertWasmResult(rawResult);
        _heatmapMode = 'correlation';
        _heatmapShowValues = _heatmapData.num_features <= 15;
        _heatmapSorted = false;
        _heatmapThreshold = 0.0;
        _heatmapMaxVal = 1.0;
        
        document.getElementById('heatmapBtnCorr').classList.add('active');
        document.getElementById('heatmapBtnMI').classList.remove('active');
        document.getElementById('heatmapBtnValues').classList.toggle('active', _heatmapShowValues);
        document.getElementById('heatmapBtnSort').classList.remove('active');
        document.getElementById('heatmapThresholdSlider').value = 0;
        document.getElementById('heatmapThresholdValue').textContent = '0.000';
        document.getElementById('heatmapInfo').textContent = `${_heatmapData.num_features} príznakov × ${_heatmapData.num_rows} riadkov`;

        renderHeatmap();
    } catch (err) {
        document.getElementById('heatmapContainer').innerHTML = `<p style="color:red;padding:40px;">Chyba: ${err}</p>`;
    } finally {
        showLoading(false);
    }
}

function closeHeatmapModal() {
    document.getElementById('heatmapModal').classList.remove('show');
}

function switchHeatmapMatrix(mode) {
    _heatmapMode = mode;
    document.getElementById('heatmapBtnCorr').classList.toggle('active', mode === 'correlation');
    document.getElementById('heatmapBtnMI').classList.toggle('active', mode === 'mi');
    document.getElementById('heatmapTitle').textContent = mode === 'correlation' ? 'Matica korelácií (Pearson)' : 'Matica Mutual Information (KSG)';

    // Recompute max value for new mode so slider covers 0–100 % of data range
    if (mode === 'mi' && _heatmapData) {
        const matrix = _heatmapData.mi_matrix;
        const n = _heatmapData.feature_names.length;
        let maxV = 0;
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                if (i !== j && matrix[i][j] > maxV) maxV = matrix[i][j];
        _heatmapMaxVal = maxV > 0 ? maxV : 1.0;
    } else {
        _heatmapMaxVal = 1.0;
    }

    // Reset threshold to 0
    _heatmapThreshold = 0.0;
    document.getElementById('heatmapThresholdSlider').value = 0;
    document.getElementById('heatmapThresholdValue').textContent = '0.000';

    renderHeatmap();
}

function toggleHeatmapValues() {
    _heatmapShowValues = !_heatmapShowValues;
    document.getElementById('heatmapBtnValues').classList.toggle('active', _heatmapShowValues);
    // Re-render to apply proper dynamic font sizes
    renderHeatmap();
}

function updateHeatmapThreshold(val) {
    // Slider is 0–100 (%), map to actual value range of current matrix
    _heatmapThreshold = (parseFloat(val) / 100) * _heatmapMaxVal;
    document.getElementById('heatmapThresholdValue').textContent = _heatmapThreshold.toFixed(3);
    renderHeatmap();
}

function toggleHeatmapSort() {
    _heatmapSorted = !_heatmapSorted;
    document.getElementById('heatmapBtnSort').classList.toggle('active', _heatmapSorted);
    renderHeatmap();
}

function renderHeatmap() {
    if (!_heatmapData) return;
    const container = document.getElementById('heatmapContainer');
    const tooltip = document.getElementById('heatmapTooltip');

    const matrix = _heatmapMode === 'correlation' ? _heatmapData.correlation_matrix : _heatmapData.mi_matrix;
    const names = [..._heatmapData.feature_names];
    const n = names.length;

    // Dynamic sizing based on number of features
    let cellSize, fontSize, headerHeight, headerFontSize, maxHeaderWidth;
    if (n <= 10) {
        cellSize = 38; fontSize = 10; headerHeight = 110; headerFontSize = 10; maxHeaderWidth = 90;
    } else if (n <= 15) {
        cellSize = 32; fontSize = 9; headerHeight = 100; headerFontSize = 9; maxHeaderWidth = 80;
    } else if (n <= 22) {
        cellSize = 26; fontSize = 7; headerHeight = 90; headerFontSize = 8; maxHeaderWidth = 70;
    } else if (n <= 35) {
        cellSize = 20; fontSize = 6; headerHeight = 80; headerFontSize = 7; maxHeaderWidth = 60;
    } else {
        cellSize = 16; fontSize = 5; headerHeight = 70; headerFontSize = 6; maxHeaderWidth = 50;
    }

    // Determine ordering
    let order = Array.from({length: n}, (_, i) => i);
    if (_heatmapSorted && n > 2) {
        const avgAbs = order.map(i => {
            let sum = 0;
            for (let j = 0; j < n; j++) if (j !== i) sum += Math.abs(matrix[i][j]);
            return sum / (n - 1);
        });
        order.sort((a, b) => avgAbs[b] - avgAbs[a]);
    }

    // Normalize MI for coloring
    let maxVal = 1.0;
    if (_heatmapMode === 'mi') {
        maxVal = 0;
        for (let i = 0; i < n; i++)
            for (let j = 0; j < n; j++)
                if (i !== j && matrix[i][j] > maxVal) maxVal = matrix[i][j];
        if (maxVal === 0) maxVal = 1;
    }

    // Truncate long names for headers
    const maxLabelLen = n <= 15 ? 18 : (n <= 22 ? 14 : 10);
    const truncName = (name) => name.length > maxLabelLen ? name.substring(0, maxLabelLen - 1) + '…' : name;

    // Build table with dynamic sizing
    let html = `<table class="heatmap-table" style="font-size:${fontSize}px;"><thead><tr>`;
    html += `<th class="corner" style="max-width:${maxHeaderWidth}px;"></th>`;
    for (const idx of order) {
        html += `<th class="col-header" title="${names[idx]}" style="height:${headerHeight}px;font-size:${headerFontSize}px;max-width:${cellSize}px;">${truncName(names[idx])}</th>`;
    }
    html += '</tr></thead><tbody>';

    const threshold = _heatmapThreshold;

    for (const ri of order) {
        html += '<tr>';
        html += `<th class="row-header" title="${names[ri]}" style="font-size:${headerFontSize}px;max-width:${maxHeaderWidth}px;">${truncName(names[ri])}</th>`;
        for (const ci of order) {
            const raw = matrix[ri][ci];
            const norm = _heatmapMode === 'mi' ? raw / maxVal : raw;
            const absVal = Math.abs(raw);
            const isDiag = ri === ci;
            const belowThreshold = !isDiag && threshold > 0 && absVal < threshold;

            let bg, tc;
            if (belowThreshold) {
                bg = '#f5f5f5'; tc = '#d0d0d0';
            } else if (isDiag && _heatmapMode === 'correlation') {
                bg = '#e0e0e0'; tc = '#999';
            } else {
                bg = heatmapColor(norm, _heatmapMode);
                tc = textColor(Math.abs(norm), _heatmapMode);
            }
            const valClass = _heatmapShowValues ? ' show-values' : '';
            const filtClass = belowThreshold ? ' filtered-out' : '';
            const displayVal = _heatmapMode === 'correlation' ? raw.toFixed(2) : raw.toFixed(3);
            html += `<td class="${valClass}${filtClass}" style="background:${bg};color:${tc};min-width:${cellSize}px;height:${cellSize}px;font-size:${_heatmapShowValues ? fontSize : 0}px;" data-row="${ri}" data-col="${ci}" data-val="${raw}">${displayVal}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;

    // Legend
    const legend = document.getElementById('heatmapLegend');
    if (_heatmapMode === 'correlation') {
        legend.innerHTML = `
            <span>-1.0</span>
            <div class="lg-bar" style="background:linear-gradient(to right, rgb(255,76,51), rgb(255,178,153), white, rgb(153,204,255), rgb(0,128,255));"></div>
            <span>+1.0</span>
            <span style="margin-left:15px;color:#999;">Pearson korelácia</span>`;
    } else {
        legend.innerHTML = `
            <span>0</span>
            <div class="lg-bar" style="background:linear-gradient(to right, white, rgb(180,140,220), rgb(80,40,180));"></div>
            <span>${maxVal.toFixed(3)}</span>
            <span style="margin-left:15px;color:#999;">Mutual Information (nats) · MI=0 pre nepočítané páry (|corr|&lt;0.3)</span>`;
    }

    // Tooltip on hover
    container.onmousemove = (e) => {
        const td = e.target.closest('td');
        if (!td || td.dataset.row === undefined) { tooltip.style.display = 'none'; return; }
        const ri = parseInt(td.dataset.row);
        const ci = parseInt(td.dataset.col);
        const val = parseFloat(td.dataset.val);
        const label = _heatmapMode === 'correlation' ? 'Korelácia' : 'MI';
        tooltip.innerHTML = `<strong>${names[ri]}</strong> × <strong>${names[ci]}</strong><br>${label}: <strong>${val.toFixed(4)}</strong>`;
        tooltip.style.display = 'block';
        tooltip.style.left = (e.clientX + 14) + 'px';
        tooltip.style.top = (e.clientY - 10) + 'px';
    };
    container.onmouseleave = () => { tooltip.style.display = 'none'; };
}
