use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[cfg(not(target_arch = "wasm32"))]
use crate::pipeline::{MLPipeline, MLPipelineBuilder, MLPipelineDirector};
#[cfg(not(target_arch = "wasm32"))]
use crate::data_loading::{DataLoader, DataLoaderFactory};
#[cfg(not(target_arch = "wasm32"))]
use smartcore::linalg::basic::matrix::DenseMatrix;

#[derive(Serialize, Deserialize)]
pub struct PipelineConfig {
    pub model: String,
    pub processor: Option<String>,
    pub selector: Option<String>,
    pub evaluation_mode: Option<String>,
    pub model_params: Option<Vec<(String, String)>>,
    pub selector_params: Option<Vec<(String, String)>>,
}

#[derive(Serialize, Deserialize)]
pub struct PipelineInfoResult {
    pub model_name: String,
    pub model_type: String,
    pub processor: Option<String>,
    pub selector: Option<String>,
    pub evaluation_mode: String,
}

#[derive(Serialize, Deserialize)]
pub struct TrainingResult {
    pub success: bool,
    pub message: String,
    pub samples_trained: usize,
}

#[derive(Serialize, Deserialize)]
pub struct EvaluationResult {
    pub metrics: std::collections::HashMap<String, f64>,
    pub model_name: String,
    pub evaluation_type: String,
}

#[wasm_bindgen]
pub struct WasmMLPipeline {
    #[cfg(not(target_arch = "wasm32"))]
    pipeline: Option<MLPipeline>,
    
    #[cfg(not(target_arch = "wasm32"))]
    data_cache: Option<(DenseMatrix<f64>, Vec<f64>)>,
}

#[wasm_bindgen]
impl WasmMLPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmMLPipeline {
        console_error_panic_hook::set_once();
        
        WasmMLPipeline {
            #[cfg(not(target_arch = "wasm32"))]
            pipeline: None,
            #[cfg(not(target_arch = "wasm32"))]
            data_cache: None,
        }
    }

    /// Vytvorí pipeline z JSON konfigurácie
    #[wasm_bindgen(js_name = buildFromConfig)]
    pub fn build_from_config(&mut self, config_json: JsValue) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let config: PipelineConfig = serde_wasm_bindgen::from_value(config_json)
                .map_err(|e| JsValue::from_str(&format!("Config parse error: {:?}", e)))?;

            let mut builder = MLPipelineBuilder::new()
                .model(&config.model);

            if let Some(proc) = config.processor {
                builder = builder.processor(&proc);
            }

            if let Some(sel) = config.selector {
                builder = builder.feature_selector(&sel);
            }

            if let Some(mode) = config.evaluation_mode {
                builder = builder.evaluation_mode(&mode);
            }

            if let Some(params) = config.model_params {
                for (key, value) in params {
                    builder = builder.model_param(&key, &value);
                }
            }

            if let Some(params) = config.selector_params {
                for (key, value) in params {
                    builder = builder.selector_param(&key, &value);
                }
            }

            let pipeline = builder.build()
                .map_err(|e| JsValue::from_str(&e))?;

            let info = pipeline.info();
            let result = PipelineInfoResult {
                model_name: info.model_name,
                model_type: info.model_type,
                processor: info.processor,
                selector: info.selector,
                evaluation_mode: info.evaluation_mode,
            };

            self.pipeline = Some(pipeline);

            Ok(serde_wasm_bindgen::to_value(&result).unwrap())
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Vytvorí pipeline z presetu
    #[wasm_bindgen(js_name = buildFromPreset)]
    pub fn build_from_preset(&mut self, preset_name: &str, model: &str) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let pipeline = match preset_name {
                "basic_classification" => MLPipelineDirector::build_basic_classification(model),
                "basic_regression" => MLPipelineDirector::build_basic_regression(model),
                "knn_classifier" => MLPipelineDirector::build_knn_classifier(5),
                "decision_tree" => MLPipelineDirector::build_decision_tree_classifier(),
                _ => return Err(JsValue::from_str(&format!("Neznámy preset: {}", preset_name))),
            }.map_err(|e| JsValue::from_str(&e))?;

            let info = pipeline.info();
            let result = PipelineInfoResult {
                model_name: info.model_name,
                model_type: info.model_type,
                processor: info.processor,
                selector: info.selector,
                evaluation_mode: info.evaluation_mode,
            };

            self.pipeline = Some(pipeline);

            Ok(serde_wasm_bindgen::to_value(&result).unwrap())
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Načíta a pripraví dáta
    #[wasm_bindgen(js_name = loadData)]
    pub fn load_data(
        &mut self,
        data: &str,
        target_column: &str,
        format: &str,
    ) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut loader = DataLoaderFactory::create(format)
                .map_err(|e| JsValue::from_str(&e))?;

            let loaded = loader.load_from_string(data, target_column)
                .map_err(|e| JsValue::from_str(&e))?;

            let num_samples = loaded.num_samples();
            
            self.data_cache = Some((loaded.x_data, loaded.y_data));

            let result = serde_json::json!({
                "success": true,
                "samples": num_samples,
                "message": format!("Načítaných {} vzoriek", num_samples)
            });

            Ok(serde_wasm_bindgen::to_value(&result).unwrap())
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Trénuje model
    #[wasm_bindgen(js_name = train)]
    pub fn train(&mut self) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.pipeline.is_none() {
                return Err(JsValue::from_str("Pipeline nie je vytvorený"));
            }

            if self.data_cache.is_none() {
                return Err(JsValue::from_str("Dáta nie sú načítané"));
            }

            let (x_data, y_data) = self.data_cache.as_ref().unwrap();
            let num_samples = x_data.shape().0;

            self.pipeline.as_mut().unwrap()
                .train(x_data.clone(), y_data.clone())
                .map_err(|e| JsValue::from_str(&e))?;

            let result = TrainingResult {
                success: true,
                message: format!("Model úspešne natrénovaný na {} vzorkách", num_samples),
                samples_trained: num_samples,
            };

            Ok(serde_wasm_bindgen::to_value(&result).unwrap())
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Predikcia
    #[wasm_bindgen(js_name = predict)]
    pub fn predict(&self, input: JsValue) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.pipeline.is_none() {
                return Err(JsValue::from_str("Pipeline nie je vytvorený"));
            }

            let input_vec: Vec<f64> = serde_wasm_bindgen::from_value(input)
                .map_err(|e| JsValue::from_str(&format!("Input parse error: {:?}", e)))?;

            let prediction = self.pipeline.as_ref().unwrap()
                .predict(input_vec)
                .map_err(|e| JsValue::from_str(&e))?;

            Ok(serde_wasm_bindgen::to_value(&prediction).unwrap())
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Evaluácia (split dáta)
    #[wasm_bindgen(js_name = evaluate)]
    pub fn evaluate(&self, train_ratio: f64) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.pipeline.is_none() {
                return Err(JsValue::from_str("Pipeline nie je vytvorený"));
            }

            if self.data_cache.is_none() {
                return Err(JsValue::from_str("Dáta nie sú načítané"));
            }

            let (x_data, y_data) = self.data_cache.as_ref().unwrap();
            let total_samples = x_data.shape().0;
            let train_size = (total_samples as f64 * train_ratio) as usize;

            // Jednoduchý split (v produkcii by sme použili shuffle)
            let x_train_rows: Vec<Vec<f64>> = (0..train_size)
                .map(|i| (0..x_data.shape().1).map(|j| *x_data.get(i, j)).collect())
                .collect();
            let x_test_rows: Vec<Vec<f64>> = (train_size..total_samples)
                .map(|i| (0..x_data.shape().1).map(|j| *x_data.get(i, j)).collect())
                .collect();

            let x_train = DenseMatrix::from_2d_vec(&x_train_rows);
            let x_test = DenseMatrix::from_2d_vec(&x_test_rows);
            let y_train = y_data[..train_size].to_vec();
            let y_test = y_data[train_size..].to_vec();

            // Tréning a evaluácia
            let mut pipeline_clone = self.pipeline.as_ref().unwrap();
            // Note: toto nebude fungovať lebo pipeline nie je Clone
            // V reálnej implementácii by sme potrebovali inú stratégiu
            
            Err(JsValue::from_str("Evaluate metóda vyžaduje refaktoring - pipeline nie je Clone"))
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Info o pipeline
    #[wasm_bindgen(js_name = getInfo)]
    pub fn get_info(&self) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref pipeline) = self.pipeline {
                let info = pipeline.info();
                let result = PipelineInfoResult {
                    model_name: info.model_name,
                    model_type: info.model_type,
                    processor: info.processor,
                    selector: info.selector,
                    evaluation_mode: info.evaluation_mode,
                };
                Ok(serde_wasm_bindgen::to_value(&result).unwrap())
            } else {
                Err(JsValue::from_str("Pipeline nie je vytvorený"))
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }
}
