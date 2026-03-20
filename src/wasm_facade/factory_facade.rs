use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use crate::models::model_factory::ModelFactory;
use crate::processing::processor_factory::ProcessorFactory;
use crate::feature_selection_strategies::feature_selector_factory::FeatureSelectorFactory;
use crate::data_loading::data_loader_factory::DataLoaderFactory;

/// Suhrnna konfiguracia vsetkych dostupnych moznosti (modely, procesory, selektory, formaty).
/// Serializuje sa do JS pre dynamicke generovanie UI.
#[derive(Serialize, Deserialize)]
pub struct AvailableOptions
{
    pub models: Vec<ModelInfo>,
    pub processors: Vec<ProcessorInfo>,
    pub selectors: Vec<SelectorInfo>,
    pub data_formats: Vec<FormatInfo>,
    pub evaluation_modes: Vec<EvalModeInfo>,
}

#[derive(Serialize, Deserialize)]
pub struct ModelInfo
{
    pub name: String,
    pub description: String,
    pub model_type: String,
}

#[derive(Serialize, Deserialize)]
pub struct ProcessorInfo
{
    pub name: String,
    pub description: String,
}

#[derive(Serialize, Deserialize)]
pub struct SelectorInfo
{
    pub name: String,
    pub description: String,
    pub supported_types: Vec<String>,
    pub requires_binning: bool,
}

#[derive(Serialize, Deserialize)]
pub struct FormatInfo
{
    pub name: String,
    pub description: String,
}

#[derive(Serialize, Deserialize)]
pub struct EvalModeInfo
{
    pub name: String,
    pub description: String,
}

#[derive(Serialize, Deserialize)]
pub struct MetricDefinition
{
    pub key: String,
    pub short_name: String,
    pub full_name: String,
    pub format_type: String, // "percent", "decimal3", "decimal4", "integer"
}

/// WASM fasada pre enumeraciu dostupnych moznosti (modely, procesory, selektory).
/// Frontend vola getAvailableOptions() pre dynamicke naplnenie UI komponentov.
#[wasm_bindgen]
pub struct WasmFactory;

#[wasm_bindgen]
impl WasmFactory
{
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmFactory
    {
        console_error_panic_hook::set_once();
        WasmFactory
    }

    /// Získa všetky dostupné možnosti pre frontend
    #[wasm_bindgen(js_name = getAvailableOptions)]
    pub fn get_available_options(&self) -> JsValue
    {
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

        let evaluation_modes = vec![
            EvalModeInfo {
                name: "classification".to_string(),
                description: "Classification - predikcia diskrétnych tried".to_string(),
            },
            EvalModeInfo {
                name: "regression".to_string(),
                description: "Regression - predikcia spojitých hodnôt".to_string(),
            },
        ];

        let options = AvailableOptions {
            models,
            processors,
            selectors,
            data_formats,
            evaluation_modes,
        };

        serde_wasm_bindgen::to_value(&options).unwrap()
    }

    /// Získa podporované parametre pre model
    #[wasm_bindgen(js_name = getModelParams)]
    pub fn get_model_params(&self, model_name: &str) -> JsValue
    {
        use crate::models::model_factory::ModelFactory;
        let params = ModelFactory::get_supported_params(model_name);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Získa detailné definície parametrov pre model
    #[wasm_bindgen(js_name = getModelParamDefinitions)]
    pub fn get_model_param_definitions(&self, model_name: &str) -> JsValue
    {
        use crate::models::model_factory::ModelFactory;
        let params = ModelFactory::get_param_definitions(model_name);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Získa podporované parametre pre selector
    #[wasm_bindgen(js_name = getSelectorParams)]
    pub fn get_selector_params(&self, selector_name: &str) -> JsValue
    {
        use crate::feature_selection_strategies::feature_selector_factory::FeatureSelectorFactory;
        let params = FeatureSelectorFactory::get_supported_params(selector_name);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Získa detailné definície parametrov pre selector
    #[wasm_bindgen(js_name = getSelectorParamDefinitions)]
    pub fn get_selector_param_definitions(&self, selector_name: &str) -> JsValue
    {
        use crate::feature_selection_strategies::feature_selector_factory::FeatureSelectorFactory;
        let params = FeatureSelectorFactory::get_param_definitions(selector_name);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Získa detailné definície parametrov pre procesor
    #[wasm_bindgen(js_name = getProcessorParamDefinitions)]
    pub fn get_processor_param_definitions(&self, processor_name: &str) -> JsValue
    {
        let params = ProcessorFactory::get_param_definitions(processor_name);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Získa definície metrík pre daný evaluation mode
    #[wasm_bindgen(js_name = getEvaluationMetrics)]
    pub fn get_evaluation_metrics(&self, eval_mode: &str) -> JsValue
    {
        let metrics = match eval_mode
        {
            "classification" => vec![
                MetricDefinition { key: "accuracy".into(), short_name: "ACC".into(), full_name: "Accuracy".into(), format_type: "percent".into() },
                MetricDefinition { key: "f1_score".into(), short_name: "F1".into(), full_name: "F1 Score".into(), format_type: "decimal3".into() },
                MetricDefinition { key: "precision".into(), short_name: "PREC".into(), full_name: "Precision".into(), format_type: "decimal3".into() },
                MetricDefinition { key: "recall".into(), short_name: "REC".into(), full_name: "Recall/Sensitivity".into(), format_type: "decimal3".into() },
                MetricDefinition { key: "specificity".into(), short_name: "SPEC".into(), full_name: "Specificity".into(), format_type: "decimal3".into() },
                MetricDefinition { key: "false_positives".into(), short_name: "FP".into(), full_name: "False Positives".into(), format_type: "integer".into() },
                MetricDefinition { key: "false_negatives".into(), short_name: "FN".into(), full_name: "False Negatives".into(), format_type: "integer".into() },
                MetricDefinition { key: "mcc".into(), short_name: "MCC".into(), full_name: "Matthews Correlation Coefficient".into(), format_type: "decimal3".into() },
            ],
            _ => vec![
                MetricDefinition { key: "r2_score".into(), short_name: "R²".into(), full_name: "Coefficient of determination".into(), format_type: "decimal4".into() },
                MetricDefinition { key: "rmse".into(), short_name: "RMSE".into(), full_name: "Root Mean Squared Error".into(), format_type: "decimal3".into() },
                MetricDefinition { key: "mae".into(), short_name: "MAE".into(), full_name: "Mean Absolute Error".into(), format_type: "decimal3".into() },
                MetricDefinition { key: "mape".into(), short_name: "MAPE".into(), full_name: "Mean Absolute Percentage Error".into(), format_type: "percent".into() },
                MetricDefinition { key: "median_absolute_error".into(), short_name: "MedAE".into(), full_name: "Median Absolute Error".into(), format_type: "decimal3".into() },
                MetricDefinition { key: "pearson_correlation".into(), short_name: "CORR".into(), full_name: "Pearsonova korelácia predikcie".into(), format_type: "decimal3".into() },
            ],
        };
        serde_wasm_bindgen::to_value(&metrics).unwrap()
    }
}
