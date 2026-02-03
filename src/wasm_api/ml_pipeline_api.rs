use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use crate::pipeline::{MLPipeline, MLPipelineBuilder, MLPipelineDirector};
use crate::data_loading::DataLoaderFactory;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};

#[derive(Serialize, Deserialize)]
pub struct PipelineConfig {
    pub model: String,
    #[serde(default)]
    pub processors: Vec<String>,  // Nové: zoznam procesorov
    pub processor: Option<String>,  // Späť kompatibilita
    pub selector: Option<String>,
    pub evaluation_mode: Option<String>,
    pub model_params: Option<Vec<(String, String)>>,
    pub selector_params: Option<Vec<(String, String)>>,
    pub processor_params: Option<Vec<(String, String)>>,  // Nové: parametre pre procesory
}

#[derive(Serialize, Deserialize)]
pub struct PipelineInfoResult {
    pub model_name: String,
    pub model_type: String,
    pub processors: Vec<String>,  // Nové: zoznam procesorov
    pub processor: Option<String>,  // Späť kompatibilita
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
pub struct TrainingWithEvaluationResult {
    pub success: bool,
    pub message: String,
    pub samples_trained: usize,
    pub evaluation_mode: String,
    // Classification metrics
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub f1_score: f64,
    // Regression metrics
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub r2_score: f64,
    // Feature selection info
    pub selected_features_indices: Option<Vec<usize>>,
    pub total_features_before: usize,
    pub total_features_after: usize,
}

#[derive(Serialize, Deserialize)]
pub struct EvaluationResult {
    pub metrics: std::collections::HashMap<String, f64>,
    pub model_name: String,
    pub evaluation_type: String,
}

#[wasm_bindgen]
pub struct WasmMLPipeline {
    pipeline: Option<MLPipeline>,
    data_cache: Option<(DenseMatrix<f64>, Vec<f64>)>,
    feature_names: Vec<String>,
}

#[wasm_bindgen]
impl WasmMLPipeline {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmMLPipeline {
        console_error_panic_hook::set_once();
        WasmMLPipeline {
            pipeline: None,
            data_cache: None,
            feature_names: Vec::new(),
        }
    }

    /// Vytvorí pipeline z JSON konfigurácie
    #[wasm_bindgen(js_name = buildFromConfig)]
    pub fn build_from_config(&mut self, config_json: JsValue) -> Result<JsValue, JsValue> {
        let config: PipelineConfig = serde_wasm_bindgen::from_value(config_json)
            .map_err(|e| JsValue::from_str(&format!("Config parse error: {:?}", e)))?;

        let mut builder = MLPipelineBuilder::new()
            .model(&config.model);

        // Podpora pre viacero procesorov
        if !config.processors.is_empty() {
            builder = builder.processors(config.processors.clone());
        } else if let Some(proc) = config.processor {
            // Späť kompatibilita - jeden procesor
            builder = builder.add_processor(&proc);
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

        // TODO: Processor params - aplikujú sa na procesory v chain
        // if let Some(params) = config.processor_params {
        //     for (key, value) in params {
        //         builder = builder.processor_param(&key, &value);
        //     }
        // }

        let pipeline = builder.build()
            .map_err(|e| JsValue::from_str(&e))?;

        let info = pipeline.info();
        let result = PipelineInfoResult {
            model_name: info.model_name,
            model_type: info.model_type,
            processors: if !config.processors.is_empty() {
                config.processors.clone()
            } else if let Some(ref p) = info.processor {
                vec![p.clone()]
            } else {
                vec![]
            },
            processor: info.processor,
            selector: info.selector,
            evaluation_mode: info.evaluation_mode,
        };

        self.pipeline = Some(pipeline);

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Vytvorí pipeline z presetu
    #[wasm_bindgen(js_name = buildFromPreset)]
    pub fn build_from_preset(
        &mut self, 
        preset_name: &str, 
        model: &str,
        model_params: JsValue,
        selector_params: JsValue
    ) -> Result<JsValue, JsValue> {
        
        // Parsovať parametre
        let model_params_vec: Option<Vec<(String, String)>> = if !model_params.is_null() && !model_params.is_undefined() {
            let parsed = serde_wasm_bindgen::from_value(model_params);
            parsed.ok()
        } else {
            None
        };

        let selector_params_vec: Option<Vec<(String, String)>> = if !selector_params.is_null() && !selector_params.is_undefined() {
            let parsed = serde_wasm_bindgen::from_value(selector_params);
            parsed.ok()
        } else {
            None
        };


        // Získať defaultné hodnoty z presetu a vytvoriť builder
        let mut builder = MLPipelineBuilder::new();

        match preset_name {
            "basic_classification" => {
                builder = builder
                    .model(model)
                    .add_processor("scaler")
                    .feature_selector("variance")
                    .selector_param("threshold", "0.05")
                    .evaluation_mode("classification");
            },
            "basic_regression" => {
                builder = builder
                    .model(model)
                    .add_processor("scaler")
                    .feature_selector("correlation")
                    .evaluation_mode("regression");
            },
            "knn_classifier" => {
                builder = builder
                    .model("knn")
                    .model_param("k", "5")
                    .add_processor("scaler")
                    .feature_selector("variance")
                    .selector_param("threshold", "0.05")
                    .evaluation_mode("classification");
            },
            "knn_regressor" => {
                builder = builder
                    .model("knn")
                    .model_param("k", "5")
                    .add_processor("scaler")
                    .feature_selector("correlation")
                    .evaluation_mode("regression");
            },
            "decision_tree" => {
                builder = builder
                    .model("tree")
                    .feature_selector("information_gain")
                    .selector_param("num_features", "10")
                    .evaluation_mode("classification");
            },
            "advanced_classification" => {
                builder = builder
                    .model(model)
                    .add_processor("scaler")
                    .feature_selector("chi_square")
                    .selector_param("num_features", "10")
                    .evaluation_mode("classification");
            },
            "advanced_regression" => {
                builder = builder
                    .model(model)
                    .add_processor("scaler")
                    .feature_selector("mutual_information")
                    .selector_param("num_features", "10")
                    .evaluation_mode("regression");
            },
            "minimal" => {
                builder = builder
                    .model(model)
                    .evaluation_mode("classification");
            },
            _ => return Err(JsValue::from_str(&format!("Neznámy preset: {}", preset_name))),
        }

        // Užívateľské parametre prepíšu defaulty
        if let Some(params) = model_params_vec {
            for (key, value) in params {
                builder = builder.model_param(&key, &value);
            }
        }

        if let Some(params) = selector_params_vec {
            for (key, value) in params {
                builder = builder.selector_param(&key, &value);
            }
        }

        // Postaviť pipeline
        let pipeline = builder.build()
            .map_err(|e| {
                web_sys::console::error_1(&format!("Build error: {}", e).into());
                JsValue::from_str(&e)
            })?;

        let info = pipeline.info();
        
        // Získať processors zo buildera
        let processor_list = if let Some(ref p) = info.processor {
            vec![p.clone()]
        } else {
            vec![]
        };
        
        let result = PipelineInfoResult {
            model_name: info.model_name,
            model_type: info.model_type,
            processors: processor_list.clone(),
            processor: info.processor,
            selector: info.selector,
            evaluation_mode: info.evaluation_mode,
        };

        self.pipeline = Some(pipeline);

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Načíta a pripraví dáta
    #[wasm_bindgen(js_name = loadData)]
    pub fn load_data(
        &mut self,
        data: &str,
        target_column: &str,
        format: &str,
    ) -> Result<JsValue, JsValue> {
        let mut loader = DataLoaderFactory::create(format)
            .map_err(|e| JsValue::from_str(&e))?;

        let loaded = loader.load_from_string(data, target_column)
            .map_err(|e| JsValue::from_str(&e))?;

        let num_samples = loaded.num_samples();
        let num_features = loaded.x_data.shape().1;
        
        // Extract feature names from the first line if CSV
        self.feature_names = if format == "csv" {
            let first_line = data.lines().next().unwrap_or("");
            let headers: Vec<&str> = first_line.split(',').collect();
            headers.iter()
                .filter(|h| h.trim() != target_column)
                .map(|h| h.trim().to_string())
                .collect()
        } else {
            // Default feature names
            (0..num_features).map(|i| format!("feature{}", i + 1)).collect()
        };
        
        self.data_cache = Some((loaded.x_data, loaded.y_data));

        let result = serde_json::json!({
            "success": true,
            "samples": num_samples,
            "features": num_features,
            "feature_names": self.feature_names,
            "message": format!("Načítaných {} vzoriek s {} features", num_samples, num_features)
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Trénuje model
    #[wasm_bindgen(js_name = train)]
    pub fn train(&mut self) -> Result<JsValue, JsValue> {
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

    /// Trénuje model s train/test splitom a vracia evaluačné metriky
    #[wasm_bindgen(js_name = trainWithSplit)]
    pub fn train_with_split(&mut self, train_ratio: f64) -> Result<JsValue, JsValue> {
        
        if self.pipeline.is_none() {
            return Err(JsValue::from_str("Pipeline nie je vytvorený"));
        }

        if self.data_cache.is_none() {
            return Err(JsValue::from_str("Dáta nie sú načítané"));
        }

        let (x_data, y_data) = self.data_cache.as_ref().unwrap();
        let num_samples = x_data.shape().0;
        let train_size = (num_samples as f64 * train_ratio) as usize;

        // Split data - convert slices to DenseMatrix and Vec
        let x_train_slice = x_data.slice(0..train_size, 0..x_data.shape().1);
        let y_train = y_data[0..train_size].to_vec();
        let x_test_slice = x_data.slice(train_size..num_samples, 0..x_data.shape().1);
        let y_test = y_data[train_size..num_samples].to_vec();

        // Convert slices to concrete DenseMatrix
        let x_train_data: Vec<Vec<f64>> = (0..x_train_slice.shape().0)
            .map(|i| (0..x_train_slice.shape().1)
                .map(|j| *x_train_slice.get((i, j)))
                .collect())
            .collect();
        let x_train = DenseMatrix::from_2d_vec(&x_train_data)
            .map_err(|e| JsValue::from_str(&format!("Chyba pri vytváraní train matice: {:?}", e)))?;

        let x_test_data: Vec<Vec<f64>> = (0..x_test_slice.shape().0)
            .map(|i| (0..x_test_slice.shape().1)
                .map(|j| *x_test_slice.get((i, j)))
                .collect())
            .collect();
        let x_test = DenseMatrix::from_2d_vec(&x_test_data)
            .map_err(|e| JsValue::from_str(&format!("Chyba pri vytváraní test matice: {:?}", e)))?;

        // Train model - train() sa postará o preprocessing a feature selection
        self.pipeline.as_mut().unwrap()
            .train(x_train, y_train.clone())
            .map_err(|e| JsValue::from_str(&e))?;
        
        // Get feature selection info AFTER training (keď už sú indexy cached)
        let (features_before, features_after) = if let Some(ref indices) = self.pipeline.as_ref().unwrap().selected_indices {
            (x_data.shape().1, indices.len())
        } else {
            (x_data.shape().1, x_data.shape().1)
        };

        // Predict on test set (preprocessing sa aplikuje automaticky v predict())
        
        let mut predictions = Vec::new();
        for row_idx in 0..x_test.shape().0 {
            let row: Vec<f64> = (0..x_test.shape().1)
                .map(|col_idx| *x_test.get((row_idx, col_idx)))
                .collect();
            
            
            // predict() už interné volá prepare_data
            let pred_result = self.pipeline.as_mut().unwrap()
                .predict(row)
                .map_err(|e| {
                    JsValue::from_str(&format!("Prediction error on row {}: {}", row_idx, e))
                })?;
            
            
            // Predikcia vracia Vec, zoberieme prvý prvok
            let pred: f64 = pred_result.first().copied().unwrap_or(0.0);
            predictions.push(pred);
        }

        // Calculate metrics based on evaluation mode
        let eval_mode = self.pipeline.as_ref().unwrap().info().evaluation_mode;
        
        // Get selected indices for result (if any were cached during training)
        let selected_indices_for_result = self.pipeline.as_ref().unwrap().selected_indices.clone();
        
        let mut result = TrainingWithEvaluationResult {
            success: true,
            message: format!("Model natrénovaný na {} vzorkách, testovaný na {}", train_size, num_samples - train_size),
            samples_trained: train_size,
            evaluation_mode: eval_mode.clone(),
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mae: 0.0,
            r2_score: 0.0,
            selected_features_indices: selected_indices_for_result,
            total_features_before: features_before,
            total_features_after: features_after,
        };


        if eval_mode == "classification" {
            // Classification metrics
            let mut correct = 0;
            let mut true_positives = 0.0;
            let mut false_positives = 0.0;
            let mut false_negatives = 0.0;
            
            for (i, &pred) in predictions.iter().enumerate() {
                let actual: f64 = y_test[i];
                if (pred - actual).abs() < 0.5 {
                    correct += 1;
                    if actual > 0.5 {
                        true_positives += 1.0;
                    }
                } else {
                    if pred > 0.5 {
                        false_positives += 1.0;
                    } else {
                        false_negatives += 1.0;
                    }
                }
            }

            result.accuracy = correct as f64 / predictions.len() as f64;
            result.precision = if true_positives + false_positives > 0.0 {
                true_positives / (true_positives + false_positives)
            } else {
                0.0
            };
            result.recall = if true_positives + false_negatives > 0.0 {
                true_positives / (true_positives + false_negatives)
            } else {
                0.0
            };
            result.f1_score = if result.precision + result.recall > 0.0 {
                2.0 * result.precision * result.recall / (result.precision + result.recall)
            } else {
                0.0
            };
        } else {
            // Regression metrics
            let n = predictions.len() as f64;
            let mut sum_squared_error = 0.0;
            let mut sum_abs_error = 0.0;
            let y_mean = y_test.iter().sum::<f64>() / n;
            let mut sum_squared_total = 0.0;

            for (i, &pred) in predictions.iter().enumerate() {
                let actual: f64 = y_test[i];
                let error: f64 = pred - actual;
                sum_squared_error += error * error;
                sum_abs_error += error.abs();
                sum_squared_total += (actual - y_mean).powi(2);
            }

            result.mse = sum_squared_error / n;
            result.rmse = result.mse.sqrt();
            result.mae = sum_abs_error / n;
            result.r2_score = if sum_squared_total > 0.0 {
                1.0 - (sum_squared_error / sum_squared_total)
            } else {
                0.0
            };
        }

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Predikcia
    #[wasm_bindgen(js_name = predict)]
    pub fn predict(&self, input: JsValue) -> Result<JsValue, JsValue> {
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

    /// Evaluácia (split dáta)
    #[wasm_bindgen(js_name = evaluate)]
    pub fn evaluate(&self, _train_ratio: f64) -> Result<JsValue, JsValue> {
        if self.pipeline.is_none() {
            return Err(JsValue::from_str("Pipeline nie je vytvorený"));
        }

        if self.data_cache.is_none() {
            return Err(JsValue::from_str("Dáta nie sú načítané"));
        }

        // TODO: Implementácia evaluácie vyžaduje refaktoring pipeline (Clone trait)
        Err(JsValue::from_str("Evaluate metóda nie je zatiaľ implementovaná"))
    }

    /// Info o pipeline
    #[wasm_bindgen(js_name = getInfo)]
    pub fn get_info(&self) -> Result<JsValue, JsValue> {
        if let Some(ref pipeline) = self.pipeline {
            let info = pipeline.info();
            let processor_list = if let Some(ref p) = info.processor {
                vec![p.clone()]
            } else {
                vec![]
            };
            let result = PipelineInfoResult {
                model_name: info.model_name,
                model_type: info.model_type,
                processors: processor_list,
                processor: info.processor,
                selector: info.selector,
                evaluation_mode: info.evaluation_mode,
            };
            Ok(serde_wasm_bindgen::to_value(&result).unwrap())
        } else {
            Err(JsValue::from_str("Pipeline nie je vytvorený"))
        }
    }
    
    /// Získa zoznam dostupných procesorov
    #[wasm_bindgen(js_name = getAvailableProcessors)]
    pub fn get_available_processors() -> JsValue {
        use crate::processing::ProcessorFactory;
        let processors = ProcessorFactory::available();
        serde_wasm_bindgen::to_value(&processors).unwrap()
    }
    
    /// Získa parametre pre daný procesor
    #[wasm_bindgen(js_name = getProcessorParams)]
    pub fn get_processor_params(processor_type: &str) -> JsValue {
        use crate::processing::ProcessorFactory;
        let params = ProcessorFactory::get_processor_params(processor_type);
        serde_wasm_bindgen::to_value(&params).unwrap()
    }

    /// Inspect uploaded data - returns first N rows with feature names
    #[wasm_bindgen(js_name = inspectData)]
    pub fn inspect_data(&self, max_rows: usize) -> Result<JsValue, JsValue> {
        if let Some((ref x, ref y)) = self.data_cache {
            let (rows, cols) = x.shape();
            let rows_to_show = rows.min(max_rows);
            
            let mut data = Vec::new();
            for row in 0..rows_to_show {
                let mut row_data = Vec::new();
                for col in 0..cols {
                    row_data.push(*x.get((row, col)));
                }
                row_data.push(y[row]); // Add target
                data.push(row_data);
            }
            
            let feature_names = if self.feature_names.is_empty() {
                (0..cols).map(|i| format!("feature{}", i + 1)).collect::<Vec<_>>()
            } else {
                self.feature_names.clone()
            };
            
            let result = serde_json::json!({
                "rows": data,
                "total_rows": rows,
                "total_cols": cols,
                "shown_rows": rows_to_show,
                "feature_names": feature_names,
                "target_name": "target"
            });
            
            Ok(serde_wasm_bindgen::to_value(&result).unwrap())
        } else {
            Err(JsValue::from_str("No data loaded"))
        }
    }

    /// Inspect processed data - returns first N rows after preprocessing
    #[wasm_bindgen(js_name = inspectProcessedData)]
    pub fn inspect_processed_data(&self, max_rows: usize) -> Result<JsValue, JsValue> {
        if let Some(ref pipeline) = self.pipeline {
            if let Some((ref x, ref y)) = self.data_cache {
                // Apply feature selection first
                let selected_x = if pipeline.selector.is_some() && pipeline.selected_indices.is_some() {
                    let indices = pipeline.selected_indices.as_ref().unwrap();
                    let (rows, _) = x.shape();
                    let cols = indices.len();
                    let mut data = vec![vec![0.0; cols]; rows];
                    
                    for (new_col, &old_col) in indices.iter().enumerate() {
                        for row in 0..rows {
                            data[row][new_col] = *x.get((row, old_col));
                        }
                    }
                    DenseMatrix::from_2d_vec(&data).unwrap()
                } else {
                    x.clone()
                };

                // Apply preprocessing
                let processed = pipeline.preprocess(&selected_x);
                
                let (rows, cols) = processed.shape();
                let rows_to_show = rows.min(max_rows);
                
                let mut data = Vec::new();
                for row in 0..rows_to_show {
                    let mut row_data = Vec::new();
                    for col in 0..cols {
                        row_data.push(*processed.get((row, col)));
                    }
                    row_data.push(y[row]); // Add target
                    data.push(row_data);
                }
                
                // Build feature names based on selection
                let feature_names: Vec<String> = if let Some(ref indices) = pipeline.selected_indices {
                    if self.feature_names.is_empty() {
                        indices.iter().map(|&i| format!("feature{}", i + 1)).collect()
                    } else {
                        indices.iter().map(|&i| self.feature_names.get(i).cloned()
                            .unwrap_or_else(|| format!("feature{}", i + 1))).collect()
                    }
                } else {
                    if self.feature_names.is_empty() {
                        (0..cols).map(|i| format!("feature{}", i + 1)).collect()
                    } else {
                        self.feature_names.clone()
                    }
                };
                
                let result = serde_json::json!({
                    "rows": data,
                    "total_rows": rows,
                    "total_cols": cols,
                    "shown_rows": rows_to_show,
                    "feature_names": feature_names,
                    "target_name": "target",
                    "selected_indices": pipeline.selected_indices,
                    "preprocessing_applied": pipeline.processor.is_some()
                });
                
                Ok(serde_wasm_bindgen::to_value(&result).unwrap())
            } else {
                Err(JsValue::from_str("No data loaded"))
            }
        } else {
            Err(JsValue::from_str("Pipeline not built"))
        }
    }

    /// Get feature selection details with names and scores
    #[wasm_bindgen(js_name = getFeatureSelectionInfo)]
    pub fn get_feature_selection_info(&self) -> Result<JsValue, JsValue> {
        if let Some(ref pipeline) = self.pipeline {
            if let Some(ref selector) = pipeline.selector {
                if let Some(ref indices) = pipeline.selected_indices {
                    let total_features = if let Some((ref x, _)) = self.data_cache {
                        x.shape().1
                    } else {
                        self.feature_names.len()
                    };
                    
                    // Get feature scores if available
                    let mut feature_scores_map = std::collections::HashMap::new();
                    let metric_name = selector.get_metric_name();
                    
                    if let Some((ref x, ref y)) = self.data_cache {
                        if let Some(scores) = selector.get_feature_scores(x, y) {
                            for (idx, score) in scores {
                                feature_scores_map.insert(idx, score);
                            }
                        }
                    }
                    
                    let kept_features: Vec<_> = indices.iter()
                        .map(|&i| {
                            if self.feature_names.is_empty() {
                                format!("feature{}", i + 1)
                            } else {
                                self.feature_names.get(i).cloned()
                                    .unwrap_or_else(|| format!("feature{}", i + 1))
                            }
                        })
                        .collect();
                    
                    let kept_scores: Vec<Option<f64>> = indices.iter()
                        .map(|&i| feature_scores_map.get(&i).copied())
                        .collect();
                    
                    let dropped_indices: Vec<usize> = (0..total_features)
                        .filter(|i| !indices.contains(i))
                        .collect();
                    
                    let dropped_features: Vec<_> = dropped_indices.iter()
                        .map(|&i| {
                            if self.feature_names.is_empty() {
                                format!("feature{}", i + 1)
                            } else {
                                self.feature_names.get(i).cloned()
                                    .unwrap_or_else(|| format!("feature{}", i + 1))
                            }
                        })
                        .collect();
                    
                    let dropped_scores: Vec<Option<f64>> = dropped_indices.iter()
                        .map(|&i| feature_scores_map.get(&i).copied())
                        .collect();
                    
                    let result = serde_json::json!({
                        "selector_name": selector.get_name(),
                        "metric_name": metric_name,
                        "total_features_before": total_features,
                        "total_features_after": indices.len(),
                        "kept_indices": indices,
                        "kept_features": kept_features,
                        "kept_scores": kept_scores,
                        "dropped_indices": dropped_indices,
                        "dropped_features": dropped_features,
                        "dropped_scores": dropped_scores
                    });
                    
                    Ok(serde_wasm_bindgen::to_value(&result).unwrap())
                } else {
                    Err(JsValue::from_str("Feature selection not performed yet (train first)"))
                }
            } else {
                Ok(serde_wasm_bindgen::to_value(&serde_json::json!({
                    "selector_name": "None",
                    "message": "No feature selector configured"
                })).unwrap())
            }
        } else {
            Err(JsValue::from_str("Pipeline not built"))
        }
    }
}
