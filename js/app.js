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

    // Populate evaluation mode select dynamically
    const evalSelect = document.getElementById('evalModeSelect');
    if (availableOptions.evaluation_modes) {
        availableOptions.evaluation_modes.forEach(em => {
            const opt = document.createElement('option');
            opt.value = em.name;
            opt.textContent = em.description;
            evalSelect.appendChild(opt);
        });
    }

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
    document.getElementById('selectAllAnalyzersBtn').onclick = toggleAllAnalyzers;
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

function updateModelParams(modelName) {
    const container = document.getElementById('modelParamsContainer');
    const fields = document.getElementById('modelParamsFields');
    fields.innerHTML = '';

    if (!modelName) {
        container.style.display = 'none';
        return;
    }

    const paramDefs = factory.getModelParamDefinitions(modelName);

    if (paramDefs && paramDefs.length > 0) {
        container.style.display = 'block';

        paramDefs.forEach(p => {
            const group = document.createElement('div');
            group.className = 'form-group';

            const label = document.createElement('label');
            label.textContent = p.description + ':';
            group.appendChild(label);

            if (p.param_type === 'select' && p.options) {
                const select = document.createElement('select');
                select.id = 'model_param_' + p.name;
                p.options.forEach(opt => {
                    const option = document.createElement('option');
                    option.value = opt;
                    option.textContent = opt;
                    if (opt === p.default_value) option.selected = true;
                    select.appendChild(option);
                });
                group.appendChild(select);
            } else {
                const input = document.createElement('input');
                input.type = 'number';
                input.id = 'model_param_' + p.name;
                input.value = p.default_value;
                input.placeholder = p.default_value;
                if (p.min != null) input.min = p.min;
                if (p.max != null) input.max = p.max;
                if (p.default_value.includes('.')) input.step = 'any';
                group.appendChild(input);
            }

            fields.appendChild(group);
        });
    } else {
        container.style.display = 'none';
    }
}

async function buildPipeline() {
    try {
        showLoading(true);
        
        const model = document.getElementById('modelSelect').value;
        const evalMode = document.getElementById('evalModeSelect').value;

        if (!model) throw new Error('Vyberte model');

        // Validácia evaluation_mode vs model (dynamicky podľa model_type z factory)
        if (evalMode) {
            const modelInfo = availableOptions.models.find(m => m.name === model);
            if (modelInfo && modelInfo.model_type !== 'both' && modelInfo.model_type !== evalMode) {
                showStatus('error', `Upozornenie: Model ${model} vyžaduje evaluation_mode = "${modelInfo.model_type}", nie "${evalMode}".`, 'pipelineStatus');
                if (!confirm(`Vybrali ste nekompatibilný evaluation_mode.\n\nModel: ${modelInfo.description}\nEvaluation Mode: ${evalMode}\n\nMal by byť: ${modelInfo.model_type}\n\nChcete pokračovať?`)) {
                    return;
                }
            }
        }

        // Zozbierať parametre dynamicky z vygenerovaných polí
        const modelParams = [];
        const paramFields = document.getElementById('modelParamsFields');
        if (paramFields) {
            paramFields.querySelectorAll('input, select').forEach(el => {
                const paramName = el.id.replace('model_param_', '');
                if (el.value) {
                    modelParams.push([paramName, el.value]);
                }
            });
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
