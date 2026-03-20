/* tslint:disable */
/* eslint-disable */

/**
 * WASM fasada pre nacitavanie dat.
 * Obaluje DataLoader trait a vystavuje metody pre JavaScript cez wasm_bindgen.
 */
export class WasmDataLoader {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Automaticky detekuje formát
     */
    static createAuto(data: string): WasmDataLoader;
    /**
     * Získa dostupné stĺpce z dát
     */
    getAvailableColumns(data: string): any;
    /**
     * Načíta dáta (vracia info o dátach)
     */
    loadData(data: string, target_column: string): any;
    constructor(format: string);
    /**
     * Validuje formát dát
     */
    validateFormat(data: string): void;
}

/**
 * WASM fasada pre enumeraciu dostupnych moznosti (modely, procesory, selektory).
 * Frontend vola getAvailableOptions() pre dynamicke naplnenie UI komponentov.
 */
export class WasmFactory {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Získa všetky dostupné možnosti pre frontend
     */
    getAvailableOptions(): any;
    /**
     * Získa definície metrík pre daný evaluation mode
     */
    getEvaluationMetrics(eval_mode: string): any;
    /**
     * Získa detailné definície parametrov pre model
     */
    getModelParamDefinitions(model_name: string): any;
    /**
     * Získa podporované parametre pre model
     */
    getModelParams(model_name: string): any;
    /**
     * Získa detailné definície parametrov pre procesor
     */
    getProcessorParamDefinitions(processor_name: string): any;
    /**
     * Získa detailné definície parametrov pre selector
     */
    getSelectorParamDefinitions(selector_name: string): any;
    /**
     * Získa podporované parametre pre selector
     */
    getSelectorParams(selector_name: string): any;
    constructor();
}

/**
 * Hlavna WASM fasada pre ML pipeline - obaluje cely workflow (load data, build, train, predict, evaluate).
 * Obsahuje cache pre analyzovane data, matice korelacie/MI/SMC a split indexy.
 */
export class WasmMLPipeline {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Analyzuje stĺpce dát a odporúča najlepšiu cieľovú premennú
     * Pre každý stĺpec vypočíta priemernú absolútnu koreláciu s ostatnými stĺpcami
     */
    analyzeTargetCandidates(data: string, format: string): any;
    /**
     * Analyzuje cieľovú premennú pomocou zvoleného analyzátora (s cachovaním dát)
     */
    analyzeTargetWith(data: string, format: string, method: string): any;
    /**
     * Aplikuje procesor na konkrétny stĺpec v CSV dátach a vráti upravené CSV
     */
    applyProcessorToColumn(data: string, column_name: string, processor_type: string, params_json: any): any;
    /**
     * Vytvorí pipeline z JSON konfigurácie
     */
    buildFromConfig(config_json: any): any;
    /**
     * Skontroluje redundanciu vybraných features na základe korelácie a MI (vylučuje target column)
     * Ak je focus_feature zadaný (>= 0), kontroluje len páry s touto feature
     */
    checkFeatureRedundancy(data: string, format: string, selected_indices: Uint32Array, target_col_index: number, focus_feature: number): any;
    /**
     * Porovná viaceré feature selektory na dátach BEZ potreby pipeline.
     * Umožňuje používateľovi preskúmať feature selection ešte pred vytvorením pipeline.
     */
    compareSelectors(data: string, target_column: string, format: string, selectors_json: any): any;
    compareTargetAnalyzers(data: string, format: string, methods: string[]): any;
    /**
     * Vypočíta maticové R² = rᵀᵧX R⁻¹ rᵧX pre dané feature indices.
     * Toto je teoretický R² z korelačnej matice BEZ trénovania modelu.
     * Porovnanie s model R² odhaľuje nelinearitu v dátach.
     */
    computeMatrixR2(indices_js: any): any;
    /**
     * Vypočíta Synergická MI pre dvojice features s cieľovou premennou.
     * Synergická MI = MI((X1, X2); Y) — koľko informácie dvojica spoločne nesie o Y.
     * Prínos Synergickej MI = Synergická MI - (MI(X1;Y) + MI(X2;Y)). Kladná = dvojica má synergiu.
     *
     * mode: "with_selected" — páry (nevybraná, vybraná)
     *       "among_unselected" — páry (nevybraná, nevybraná)
     */
    computeSynergyAnalysis(data: string, format: string, target_col: string, selected_indices: Uint32Array, mode: string): any;
    /**
     * Vymaže stĺpec z CSV dát
     */
    deleteColumn(data: string, column_name: string): any;
    /**
     * Evaluácia (split dáta)
     */
    evaluate(_train_ratio: number): any;
    /**
     * Získa zoznam dostupných procesorov
     */
    static getAvailableProcessors(): any;
    /**
     * Vráti zoznam dostupných analyzátorov cieľovej premennej
     */
    getAvailableTargetAnalyzers(): any;
    /**
     * Vráti všetky dáta vo formáte vhodnom pre editor (všetky riadky, všetky stĺpce)
     */
    getEditableData(data: string, format: string): any;
    /**
     * Získaj feature importance/ranking z embedded metódy (Ridge L2, Tree importance)
     */
    getEmbeddedFeatureRanking(train_ratio: number, top_k: number, is_classification: boolean): any;
    /**
     * Vráti surovú korelačnú a MI maticu spolu s názvami stĺpcov (pre JS heatmap vizualizáciu)
     */
    getFeatureMatrices(data: string, format: string): any;
    /**
     * Get feature selection details with names and scores
     */
    getFeatureSelectionInfo(): any;
    /**
     * Info o pipeline
     */
    getInfo(): any;
    /**
     * Získa parametre pre daný procesor
     */
    static getProcessorParams(processor_type: string): any;
    /**
     * Get detailed selection info stub (selection is done externally via compareSelectors)
     */
    getSelectionDetails(): any;
    /**
     * Inspect uploaded data - returns first N rows with feature names
     */
    inspectData(max_rows: number): any;
    /**
     * Inspect processed data - returns first N rows after preprocessing
     */
    inspectProcessedData(max_rows: number): any;
    /**
     * Načíta a pripraví dáta
     */
    loadData(data: string, target_column: string, format: string): any;
    constructor();
    /**
     * Predikcia
     */
    predict(input: any): any;
    /**
     * Nahradí všetky výskyty hodnoty v stĺpci (Replace All)
     */
    replaceAllInColumn(data: string, column_name: string, search_value: string, replace_value: string): any;
    /**
     * Zmení hodnotu konkrétnej bunky v CSV
     */
    setCellValue(data: string, row_idx: number, column_name: string, new_value: string): any;
    /**
     * Trénuje model
     */
    train(): any;
    /**
     * Trénuje model s konkrétnymi feature indices (z porovnania selektorov)
     */
    trainWithFeatureIndices(train_ratio: number, indices_js: any): any;
    /**
     * Trénuje model s train/test splitom a vracia evaluačné metriky
     */
    trainWithSplit(train_ratio: number): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmmlpipeline_free: (a: number, b: number) => void;
    readonly wasmmlpipeline_analyzeTargetCandidates: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmmlpipeline_analyzeTargetWith: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmmlpipeline_applyProcessorToColumn: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: any) => [number, number, number];
    readonly wasmmlpipeline_buildFromConfig: (a: number, b: any) => [number, number, number];
    readonly wasmmlpipeline_checkFeatureRedundancy: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly wasmmlpipeline_compareSelectors: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: any) => [number, number, number];
    readonly wasmmlpipeline_compareTargetAnalyzers: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmmlpipeline_computeMatrixR2: (a: number, b: any) => [number, number, number];
    readonly wasmmlpipeline_computeSynergyAnalysis: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number];
    readonly wasmmlpipeline_deleteColumn: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmmlpipeline_evaluate: (a: number, b: number) => [number, number, number];
    readonly wasmmlpipeline_getAvailableProcessors: () => any;
    readonly wasmmlpipeline_getAvailableTargetAnalyzers: (a: number) => [number, number, number];
    readonly wasmmlpipeline_getEditableData: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmmlpipeline_getEmbeddedFeatureRanking: (a: number, b: number, c: number, d: number) => [number, number, number];
    readonly wasmmlpipeline_getFeatureMatrices: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmmlpipeline_getFeatureSelectionInfo: (a: number) => [number, number, number];
    readonly wasmmlpipeline_getInfo: (a: number) => [number, number, number];
    readonly wasmmlpipeline_getProcessorParams: (a: number, b: number) => any;
    readonly wasmmlpipeline_getSelectionDetails: (a: number) => [number, number, number];
    readonly wasmmlpipeline_inspectData: (a: number, b: number) => [number, number, number];
    readonly wasmmlpipeline_inspectProcessedData: (a: number, b: number) => [number, number, number];
    readonly wasmmlpipeline_loadData: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmmlpipeline_new: () => number;
    readonly wasmmlpipeline_predict: (a: number, b: any) => [number, number, number];
    readonly wasmmlpipeline_replaceAllInColumn: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => [number, number, number];
    readonly wasmmlpipeline_setCellValue: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => [number, number, number];
    readonly wasmmlpipeline_train: (a: number) => [number, number, number];
    readonly wasmmlpipeline_trainWithFeatureIndices: (a: number, b: number, c: any) => [number, number, number];
    readonly wasmmlpipeline_trainWithSplit: (a: number, b: number) => [number, number, number];
    readonly __wbg_wasmdataloader_free: (a: number, b: number) => void;
    readonly __wbg_wasmfactory_free: (a: number, b: number) => void;
    readonly wasmdataloader_createAuto: (a: number, b: number) => [number, number, number];
    readonly wasmdataloader_getAvailableColumns: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmdataloader_loadData: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmdataloader_new: (a: number, b: number) => [number, number, number];
    readonly wasmdataloader_validateFormat: (a: number, b: number, c: number) => [number, number];
    readonly wasmfactory_getAvailableOptions: (a: number) => any;
    readonly wasmfactory_getEvaluationMetrics: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getModelParamDefinitions: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getModelParams: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getProcessorParamDefinitions: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getSelectorParamDefinitions: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getSelectorParams: (a: number, b: number, c: number) => any;
    readonly wasmfactory_new: () => number;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
