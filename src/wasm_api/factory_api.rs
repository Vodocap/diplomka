use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use crate::models::factory::ModelFactory;
use crate::processing::factory::ProcessorFactory;
use crate::processing::ProcessorParam;
use crate::feature_selection_strategies::factory::FeatureSelectorFactory;
use crate::data_loading::factory::DataLoaderFactory;
use crate::pipeline::director::MLPipelineDirector;

#[derive(Serialize, Deserialize)]
pub struct AvailableOptions {
    pub models: Vec<ModelInfo>,
    pub processors: Vec<ProcessorInfo>,
    pub selectors: Vec<SelectorInfo>,
    pub data_formats: Vec<FormatInfo>,
    pub presets: Vec<PresetInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub description: String,
    pub model_type: String,
}

#[derive(Serialize, Deserialize)]
pub struct ProcessorInfo {
    pub name: String,
    pub description: String,
}

#[derive(Serialize, Deserialize)]
pub struct SelectorInfo {
    pub name: String,
    pub description: String,
    pub supported_types: Vec<String>,
    pub requires_binning: bool,
}

#[derive(Serialize, Deserialize)]
pub struct FormatInfo {
    pub name: String,
    pub description: String,
}

#[derive(Serialize, Deserialize)]
pub struct PresetInfo {
    pub name: String,
    pub description: String,
    pub model_type: String,
}

#[wasm_bindgen]
pub struct WasmFactory;

#[wasm_bindgen]
impl WasmFactory {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmFactory {
        console_error_panic_hook::set_once();
        WasmFactory
    }

    /// Získa všetky dostupné možnosti pre frontend
    #[wasm_bindgen(js_name = getAvailableOptions)]
    pub fn get_available_options(&self) -> JsValue {
        let models: Vec<ModelInfo> = ModelFactory::available_models()
            .iter()
            .map(|name| ModelInfo {
                name: name.to_string(),
                description: ModelFactory::get_model_description(name)
                    .unwrap_or("").to_string(),
                model_type: ModelFactory::get_model_type(name)
                    .unwrap_or("unknown").to_string(),
            })
            .collect();

        let processors: Vec<ProcessorInfo> = ProcessorFactory::available()
            .iter()
            .map(|name| ProcessorInfo {
                name: name.to_string(),
                description: ProcessorFactory::get_description(name)
                    .unwrap_or("").to_string(),
            })
            .collect();

        let selectors: Vec<SelectorInfo> = FeatureSelectorFactory::available()
            .iter()
            .map(|name| SelectorInfo {
                name: name.to_string(),
                description: FeatureSelectorFactory::get_description(name)
                    .unwrap_or("").to_string(),
                supported_types: FeatureSelectorFactory::get_supported_types(name)
                    .iter()
                    .map(|s| s.to_string())
                    .collect(),
                requires_binning: FeatureSelectorFactory::requires_binning(name),
            })
            .collect();

        let data_formats: Vec<FormatInfo> = DataLoaderFactory::available_formats()
            .iter()
            .map(|name| FormatInfo {
                name: name.to_string(),
                description: DataLoaderFactory::get_format_description(name)
                    .unwrap_or("").to_string(),
            })
            .collect();

        let presets: Vec<PresetInfo> = MLPipelineDirector::available_presets()
            .iter()
            .map(|preset| PresetInfo {
                name: preset.name.to_string(),
                description: preset.description.to_string(),
                model_type: preset.model_type.to_string(),
            })
            .collect();

        let options = AvailableOptions {
            models,
            processors,
            selectors,
            data_formats,
            presets,
        };

        serde_wasm_bindgen::to_value(&options).unwrap()
    }

    /// Získa kompatibilné procesory pre model
    #[wasm_bindgen(js_name = getCompatibleProcessors)]
    pub fn get_compatible_processors(&self, _model_name: &str) -> JsValue {
        use crate::pipeline::compatibility::CompatibilityRegistry;
        let registry = CompatibilityRegistry::instance().lock().unwrap();
        let processors = registry.get_compatible_processors(_model_name);
        serde_wasm_bindgen::to_value(&processors).unwrap()
    }

    /// Získa kompatibilné selektory pre model
    #[wasm_bindgen(js_name = getCompatibleSelectors)]
    pub fn get_compatible_selectors(&self, _model_name: &str) -> JsValue {
        use crate::pipeline::compatibility::CompatibilityRegistry;
        let registry = CompatibilityRegistry::instance().lock().unwrap();
        let selectors = registry.get_compatible_selectors(_model_name);
        serde_wasm_bindgen::to_value(&selectors).unwrap()
    }

    /// Získa podporované parametre pre model
    #[wasm_bindgen(js_name = getModelParams)]
    pub fn get_model_params(&self, model_name: &str) -> JsValue {
        use crate::models::factory::ModelFactory;
        let params = ModelFactory::get_supported_params(model_name);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Získa podporované parametre pre selector
    #[wasm_bindgen(js_name = getSelectorParams)]
    pub fn get_selector_params(&self, selector_name: &str) -> JsValue {
        use crate::feature_selection_strategies::factory::FeatureSelectorFactory;
        let params = FeatureSelectorFactory::get_supported_params(selector_name);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Získa detailné definície parametrov pre procesor
    #[wasm_bindgen(js_name = getProcessorParamDefinitions)]
    pub fn get_processor_param_definitions(&self, processor_name: &str) -> JsValue {
        let params = ProcessorFactory::get_param_definitions(processor_name);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Získa detaily o presete (model, processor, selector)
    #[wasm_bindgen(js_name = getPresetDetails)]
    pub fn get_preset_details(&self, preset_name: &str) -> JsValue {
        use crate::pipeline::director::MLPipelineDirector;
        let presets = MLPipelineDirector::available_presets();
        
        for preset in presets {
            if preset.name == preset_name {
                return serde_wasm_bindgen::to_value(&preset).unwrap();
            }
        }
        
        JsValue::NULL
    }
}
