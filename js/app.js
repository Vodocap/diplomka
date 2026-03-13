// App init, pipeline builder, event setup

async function initApp() {
    
    try {
        showLoading(true);
        
        let wasmModule;
        try {
            const wasmUrl = new URL('./pkg/wasm.js', document.baseURI).href;
            wasmModule = await import(wasmUrl);
        } catch (importError) {
            throw importError;
        }
        
        const init = wasmModule.default || wasmModule.init;
        const WasmFactory = wasmModule.WasmFactory;
        const WasmMLPipeline = wasmModule.WasmMLPipeline;
        WasmDataLoaderClass = wasmModule.WasmDataLoader;

        if (!init || !WasmFactory) {
            throw new Error('WASM module exports not found');
        }

        await init();

        factory = new WasmFactory();
        pipeline = new WasmMLPipeline();
        availableOptions = factory.getAvailableOptions();
        
        if (!availableOptions || !availableOptions.models || availableOptions.models.length === 0) {
            throw new Error('Žiadne modely neboli načítané z WASM Factory.');
        }

        populateOptions();
        setupEventListeners();
        showStatus('info', 'Aplikácia pripravená!', 'parseStatus');
    } catch (error) {
        console.error('Init error:', error);
        const message = (error && error.message) ? error.message : String(error);
        showStatus('error', `Chyba pri načítaní WASM: ${message}`, 'parseStatus');
    } finally {
        showLoading(false);
    }
}

function populateOptions() {
    if (!availableOptions) return;

    const modelSelect = document.getElementById('modelSelect');
    availableOptions.models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.name;
        opt.textContent = `${m.name} - ${m.description}`;
        modelSelect.appendChild(opt);
    });

    displayInfo();
}

function displayInfo() {
    const modelsDiv = document.getElementById('modelsInfo');
    availableOptions.models.forEach(m => {
        const badge = document.createElement('span');
        badge.className = 'option-badge';
        badge.textContent = m.name;
        badge.title = m.description;
        modelsDiv.appendChild(badge);
    });

    const processorsDiv = document.getElementById('processorsInfo');
    availableOptions.processors.forEach(p => {
        const badge = document.createElement('span');
        badge.className = 'option-badge';
        badge.textContent = p.name;
        processorsDiv.appendChild(badge);
    });

    const selectorsDiv = document.getElementById('selectorsInfo');
    availableOptions.selectors.forEach(s => {
        const badge = document.createElement('span');
        badge.className = 'option-badge';
        badge.textContent = s.name;
        selectorsDiv.appendChild(badge);
    });
}

function setupEventListeners() {
    document.getElementById('buildPipelineBtn').onclick = buildPipeline;
    document.getElementById('parseDataBtn').onclick = parseData;
    document.getElementById('confirmTargetBtn').onclick = confirmTarget;
    document.getElementById('inspectDataBtn').onclick = inspectData;
    document.getElementById('editDataBtn').onclick = openDataEditor;
    document.getElementById('compareTargetAnalyzersBtn').onclick = compareTargetAnalyzers;
    document.getElementById('compareSelectorBtn').onclick = compareSelectors;
    document.getElementById('showMatrixBtn').onclick = openHeatmapModal;
    document.getElementById('showMatrixBtnTarget').onclick = openHeatmapModal;
    document.getElementById('selectAllSelectorsBtn').onclick = toggleAllSelectors;
    document.getElementById('trainAllSelectorsBtn').onclick = trainAllSelectors;
    
    // Data input method radio buttons
    document.getElementById('dataInputText').onchange = toggleDataInputMethod;
    document.getElementById('dataInputFile').onchange = toggleDataInputMethod;
    
    // Train/Test split slider
    document.getElementById('trainTestSplit').oninput = function(e) {
        const trainPercent = parseInt(e.target.value);
        const testPercent = 100 - trainPercent;
        document.getElementById('splitValue').textContent = `${trainPercent}%`;
        document.getElementById('trainPercent').textContent = `${trainPercent}%`;
        document.getElementById('testPercent').textContent = `${testPercent}%`;
    };
    
    // File input change listener
    document.getElementById('fileInput').onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            const fileInfo = document.getElementById('fileInfo');
            fileInfo.textContent = `Vybraný súbor: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
            
            // Auto-detect format from file extension
            const ext = file.name.split('.').pop().toLowerCase();
            if (ext === 'csv') {
                document.getElementById('dataFormatSelect').value = 'csv';
            } else if (ext === 'json') {
                document.getElementById('dataFormatSelect').value = 'json';
            }
        }
    };

    document.getElementById('modelSelect').onchange = function() {
        updateModelParams(this.value);
    };

    // Update editor processor params when processor changes
    document.getElementById('editorProcessorSelect').onchange = function() {
        updateEditorProcessorParams(this.value);
    };
}

function hideAllParamFields() {
    document.getElementById('modelParamsContainer').style.display = 'none';
    document.getElementById('knnKGroup').style.display = 'none';
    document.getElementById('treeMaxDepthGroup').style.display = 'none';
    document.getElementById('rfNEstimatorsGroup').style.display = 'none';
    document.getElementById('rfMaxDepthGroup').style.display = 'none';
    document.getElementById('rfMinSamplesLeafGroup').style.display = 'none';
    document.getElementById('svmCGroup').style.display = 'none';
    document.getElementById('svmEpsGroup').style.display = 'none';
    document.getElementById('svmKernelGroup').style.display = 'none';
    document.getElementById('svmGammaGroup').style.display = 'none';
    document.getElementById('gbtNEstimatorsGroup').style.display = 'none';
    document.getElementById('gbtMaxDepthGroup').style.display = 'none';
    document.getElementById('gbtLearningRateGroup').style.display = 'none';
    document.getElementById('polynomDegreeGroup').style.display = 'none';
}

function updateModelParams(modelName) {
    hideAllParamFields();
    
    if (!modelName) return;

    const params = factory.getModelParams(modelName);
    
    if (params.length > 0) {
        document.getElementById('modelParamsContainer').style.display = 'block';
        
        params.forEach(param => {
            if (param === 'k') {
                document.getElementById('knnKGroup').style.display = 'block';
            } else if (param === 'max_depth' && modelName === 'tree') {
                document.getElementById('treeMaxDepthGroup').style.display = 'block';
            } else if (param === 'n_estimators' && modelName === 'rf') {
                document.getElementById('rfNEstimatorsGroup').style.display = 'block';
            } else if (param === 'max_depth' && modelName === 'rf') {
                document.getElementById('rfMaxDepthGroup').style.display = 'block';
            } else if (param === 'min_samples_leaf') {
                document.getElementById('rfMinSamplesLeafGroup').style.display = 'block';
            } else if (param === 'c') {
                document.getElementById('svmCGroup').style.display = 'block';
            } else if (param === 'eps') {
                document.getElementById('svmEpsGroup').style.display = 'block';
            } else if (param === 'kernel') {
                document.getElementById('svmKernelGroup').style.display = 'block';
            } else if (param === 'gamma') {
                document.getElementById('svmGammaGroup').style.display = 'block';
            } else if (param === 'n_estimators' && modelName === 'gbt') {
                document.getElementById('gbtNEstimatorsGroup').style.display = 'block';
            } else if (param === 'max_depth' && modelName === 'gbt') {
                document.getElementById('gbtMaxDepthGroup').style.display = 'block';
            } else if (param === 'learning_rate') {
                document.getElementById('gbtLearningRateGroup').style.display = 'block';
            } else if (param === 'degree') {
                document.getElementById('polynomDegreeGroup').style.display = 'block';
            }
        });
    }
}

async function buildPipeline() {
    try {
        showLoading(true);
        
        const model = document.getElementById('modelSelect').value;
        const evalMode = document.getElementById('evalModeSelect').value;

        if (!model) throw new Error('Vyberte model');

        // Validácia evaluation_mode vs model
        if (evalMode) {
            if (model === 'linreg' && evalMode === 'classification') {
                showStatus('error', 'Upozornenie: Lineárna regresia (linreg) vyžaduje evaluation_mode = "regression", nie "classification".', 'pipelineStatus');
                if (!confirm('Vybrali ste nekompatibilný evaluation_mode.\n\nModel: Lineárna Regresia (linreg)\nEvaluation Mode: classification\n\nMal by byť: regression\n\nChcete pokračovať?')) {
                    return;
                }
            }
            if (model === 'logreg' && evalMode === 'regression') {
                showStatus('error', 'Upozornenie: Logistická regresia (logreg) vyžaduje evaluation_mode = "classification", nie "regression".', 'pipelineStatus');
                if (!confirm('Vybrali ste nekompatibilný evaluation_mode.\n\nModel: Logistická Regresia (logreg)\nEvaluation Mode: regression\n\nMal by byť: classification\n\nChcete pokračovať?')) {
                    return;
                }
            }
        }

        // Zozbierať parametre
        const modelParams = [];

        const knnK = document.getElementById('knnK').value;
        if (knnK && model === 'knn') {
            modelParams.push(['k', knnK]);
        }

        const treeMaxDepth = document.getElementById('treeMaxDepth').value;
        if (treeMaxDepth && model === 'tree') {
            modelParams.push(['max_depth', treeMaxDepth]);
        }

        // Random Forest params
        if (model === 'rf') {
            const v = document.getElementById('rfNEstimators').value;
            if (v) modelParams.push(['n_estimators', v]);
            const d = document.getElementById('rfMaxDepth').value;
            if (d) modelParams.push(['max_depth', d]);
            const l = document.getElementById('rfMinSamplesLeaf').value;
            if (l) modelParams.push(['min_samples_leaf', l]);
        }

        // SVM params
        if (model === 'svm') {
            const c = document.getElementById('svmC').value;
            if (c) modelParams.push(['c', c]);
            const eps = document.getElementById('svmEps').value;
            if (eps) modelParams.push(['eps', eps]);
            const kernel = document.getElementById('svmKernel').value;
            if (kernel) modelParams.push(['kernel', kernel]);
            const gamma = document.getElementById('svmGamma').value;
            if (gamma) modelParams.push(['gamma', gamma]);
        }

        // Gradient Boosting params
        if (model === 'gbt') {
            const n = document.getElementById('gbtNEstimators').value;
            if (n) modelParams.push(['n_estimators', n]);
            const d = document.getElementById('gbtMaxDepth').value;
            if (d) modelParams.push(['max_depth', d]);
            const lr = document.getElementById('gbtLearningRate').value;
            if (lr) modelParams.push(['learning_rate', lr]);
        }

        // Polynomálna regresia params
        if (model === 'polynom') {
            const deg = document.getElementById('polynomDegree').value;
            if (deg) modelParams.push(['degree', deg]);
        }

        const config = {
            model: model,
            processors: [],
            processor: null,
            selector: null,
            evaluation_mode: evalMode || null,
            model_params: modelParams.length > 0 ? modelParams : null,
            selector_params: null,
            processor_params: null
        };

        const result = await pipeline.buildFromConfig(config);
        
        document.getElementById('currentPipelineInfo').style.display = 'block';
        document.getElementById('currentModel').textContent = result.model_name;
        document.getElementById('currentProcessor').textContent = 'Dáta upravené manuálne';
        document.getElementById('currentEvalMode').textContent = result.evaluation_mode;
        
        window.pipelineBuilt = true;
        showStatus('success', 'Pipeline vytvorený!', 'pipelineStatus');
    } catch (error) {
        console.error('Build pipeline error:', error);
        showStatus('error', `Chyba: ${error}`, 'pipelineStatus');
    } finally {
        showLoading(false);
    }
}

function setupBasicListeners() {
    // Escape key closes modals
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            const hm = document.getElementById('heatmapModal');
            if (hm && hm.classList.contains('show')) { closeHeatmapModal(); return; }
            const ad = document.getElementById('analyzerDetailsModal');
            if (ad && ad.classList.contains('show')) { closeAnalyzerDetailsModal(); return; }
            const de = document.getElementById('dataEditorModal');
            if (de && de.classList.contains('show')) { closeDataEditor(); return; }
            const dm = document.getElementById('dataModal');
            if (dm && dm.classList.contains('show')) { closeModal(); return; }
        }
    });

    // Data input method radio buttons - these must work even if WASM fails
    const radioText = document.getElementById('dataInputText');
    const radioFile = document.getElementById('dataInputFile');
    if (radioText) radioText.onchange = toggleDataInputMethod;
    if (radioFile) radioFile.onchange = toggleDataInputMethod;

    // File input change listener
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.onchange = function(e) {
            const file = e.target.files[0];
            if (file) {
                const fileInfo = document.getElementById('fileInfo');
                fileInfo.textContent = `Vybraný súbor: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`;
                const ext = file.name.split('.').pop().toLowerCase();
                if (ext === 'csv') {
                    document.getElementById('dataFormatSelect').value = 'csv';
                } else if (ext === 'json') {
                    document.getElementById('dataFormatSelect').value = 'json';
                }
            }
        };
    }

    // Train/Test split slider
    const splitSlider = document.getElementById('trainTestSplit');
    if (splitSlider) {
        splitSlider.oninput = function(e) {
            const trainPercent = parseInt(e.target.value);
            const testPercent = 100 - trainPercent;
            document.getElementById('splitValue').textContent = `${trainPercent}%`;
            document.getElementById('trainPercent').textContent = `${trainPercent}%`;
            document.getElementById('testPercent').textContent = `${testPercent}%`;
        };
    }
}
