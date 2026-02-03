/* tslint:disable */
/* eslint-disable */

export class CsvLoader {
    free(): void;
    [Symbol.dispose](): void;
    get_headers(): any;
    get_training_data(target_header: string): any;
    len(): number;
    load_csv(csv_text: string): void;
    load_csv_async(csv_text: string): Promise<void>;
    constructor();
}

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

export class WasmFactory {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Získa všetky dostupné možnosti pre frontend
     */
    getAvailableOptions(): any;
    /**
     * Získa kompatibilné procesory pre model
     */
    getCompatibleProcessors(_model_name: string): any;
    /**
     * Získa kompatibilné selektory pre model
     */
    getCompatibleSelectors(_model_name: string): any;
    /**
     * Získa podporované parametre pre model
     */
    getModelParams(model_name: string): any;
    /**
     * Získa detaily o presete (model, processor, selector)
     */
    getPresetDetails(preset_name: string): any;
    /**
     * Získa podporované parametre pre selector
     */
    getSelectorParams(selector_name: string): any;
    constructor();
}

export class WasmMLPipeline {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Vytvorí pipeline z JSON konfigurácie
     */
    buildFromConfig(config_json: any): any;
    /**
     * Vytvorí pipeline z presetu
     */
    buildFromPreset(preset_name: string, model: string, model_params: any, selector_params: any): any;
    /**
     * Evaluácia (split dáta)
     */
    evaluate(_train_ratio: number): any;
    /**
     * Získa zoznam dostupných procesorov
     */
    static getAvailableProcessors(): any;
    /**
     * Info o pipeline
     */
    getInfo(): any;
    /**
     * Získa parametre pre daný procesor
     */
    static getProcessorParams(processor_type: string): any;
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
     * Trénuje model
     */
    train(): any;
    /**
     * Trénuje model s train/test splitom a vracia evaluačné metriky
     */
    trainWithSplit(train_ratio: number): any;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmfactory_free: (a: number, b: number) => void;
    readonly wasmfactory_getAvailableOptions: (a: number) => any;
    readonly wasmfactory_getCompatibleProcessors: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getCompatibleSelectors: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getModelParams: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getPresetDetails: (a: number, b: number, c: number) => any;
    readonly wasmfactory_getSelectorParams: (a: number, b: number, c: number) => any;
    readonly wasmfactory_new: () => number;
    readonly __wbg_csvloader_free: (a: number, b: number) => void;
    readonly __wbg_wasmdataloader_free: (a: number, b: number) => void;
    readonly csvloader_get_headers: (a: number) => any;
    readonly csvloader_get_training_data: (a: number, b: number, c: number) => [number, number, number];
    readonly csvloader_len: (a: number) => number;
    readonly csvloader_load_csv: (a: number, b: number, c: number) => [number, number];
    readonly csvloader_load_csv_async: (a: number, b: number, c: number) => any;
    readonly csvloader_new: () => number;
    readonly wasmdataloader_createAuto: (a: number, b: number) => [number, number, number];
    readonly wasmdataloader_getAvailableColumns: (a: number, b: number, c: number) => [number, number, number];
    readonly wasmdataloader_loadData: (a: number, b: number, c: number, d: number, e: number) => [number, number, number];
    readonly wasmdataloader_new: (a: number, b: number) => [number, number, number];
    readonly wasmdataloader_validateFormat: (a: number, b: number, c: number) => [number, number];
    readonly __wbg_wasmmlpipeline_free: (a: number, b: number) => void;
    readonly wasmmlpipeline_buildFromConfig: (a: number, b: any) => [number, number, number];
    readonly wasmmlpipeline_buildFromPreset: (a: number, b: number, c: number, d: number, e: number, f: any, g: any) => [number, number, number];
    readonly wasmmlpipeline_evaluate: (a: number, b: number) => [number, number, number];
    readonly wasmmlpipeline_getAvailableProcessors: () => any;
    readonly wasmmlpipeline_getInfo: (a: number) => [number, number, number];
    readonly wasmmlpipeline_getProcessorParams: (a: number, b: number) => any;
    readonly wasmmlpipeline_loadData: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => [number, number, number];
    readonly wasmmlpipeline_new: () => number;
    readonly wasmmlpipeline_predict: (a: number, b: any) => [number, number, number];
    readonly wasmmlpipeline_train: (a: number) => [number, number, number];
    readonly wasmmlpipeline_trainWithSplit: (a: number, b: number) => [number, number, number];
    readonly wasm_bindgen__closure__destroy__h062d55d9c7b8c652: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h516edae654ab5801: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__h8f27a5fbd1cc09f3: (a: number, b: number, c: any) => void;
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
