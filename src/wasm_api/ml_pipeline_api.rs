use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use crate::pipeline::{MLPipeline, MLPipelineBuilder};
use crate::data_loading::DataLoaderFactory;
use crate::feature_selection_strategies::factory::FeatureSelectorFactory;
use crate::target_analysis::TargetAnalyzerFactory;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};

#[derive(Serialize, Deserialize)]
pub struct SelectorCompareConfig {
    pub name: String,
    #[serde(default)]
    pub params: Vec<(String, String)>,
}

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

// EvaluationResult removed - not used

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
            "minimal" => {
                builder = builder
                    .model(model)
                    .evaluation_mode("classification");
            },
            
            // === POKROČILÉ KLASIFIKAČNÉ ===
            "advanced_classification" => {
                builder = builder
                    .model(model)
                    .add_processor("scaler")
                    .feature_selector("chi_square")
                    .selector_param("num_features", "10")
                    .evaluation_mode("classification");
            },
            "logreg_minmax_chisquare" => {
                builder = builder
                    .model("logreg")
                    .add_processor("minmax_scaler")
                    .feature_selector("chi_square")
                    .selector_param("num_features", "10")
                    .evaluation_mode("classification");
            },
            "tree_binner_infogain" => {
                builder = builder
                    .model("tree")
                    .add_processor("binner")  // použije default 10 bins
                    .feature_selector("information_gain")
                    .selector_param("num_features", "10")
                    .evaluation_mode("classification");
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
            "knn_robust_mi" => {
                builder = builder
                    .model("knn")
                    .model_param("k", "5")
                    .add_processor("robust_scaler")
                    .feature_selector("mutual_information")
                    .selector_param("num_features", "10")
                    .evaluation_mode("classification");
            },
            
            // === POKROČILÉ REGRESNÉ ===
            "advanced_regression" => {
                builder = builder
                    .model(model)
                    .add_processor("scaler")
                    .feature_selector("mutual_information")
                    .selector_param("num_features", "10")
                    .evaluation_mode("regression");
            },
            "knn_regressor" => {
                builder = builder
                    .model("knn")
                    .model_param("k", "5")
                    .add_processor("scaler")
                    .feature_selector("correlation")
                    .evaluation_mode("regression");
            },
            "linreg_minmax_correlation" => {
                builder = builder
                    .model("linreg")
                    .add_processor("minmax_scaler")
                    .feature_selector("correlation")
                    .evaluation_mode("regression");
            },
            "linreg_robust_mi" => {
                builder = builder
                    .model("linreg")
                    .add_processor("robust_scaler")
                    .feature_selector("mutual_information")
                    .selector_param("num_features", "10")
                    .evaluation_mode("regression");
            },
            
            // === S TRANSFORMÁCIAMI ===
            "linreg_log_correlation" => {
                builder = builder
                    .model("linreg")
                    .add_processor("log_transformer")  // použije default offset=1
                    .feature_selector("correlation")
                    .evaluation_mode("regression");
            },
            "knn_power_variance" => {
                builder = builder
                    .model("knn")
                    .model_param("k", "5")
                    .add_processor("power_transformer")
                    .feature_selector("variance")
                    .selector_param("threshold", "0.01")
                    .evaluation_mode("regression");
            },
            
            // === S OUTLIER HANDLINGOM ===
            "linreg_outlier_correlation" => {
                builder = builder
                    .model("linreg")
                    .add_processor("outlier_clipper")
                    .feature_selector("correlation")
                    .evaluation_mode("regression");
            },
            "knn_outlier_variance" => {
                builder = builder
                    .model("knn")
                    .model_param("k", "5")
                    .add_processor("outlier_clipper")
                    .feature_selector("variance")
                    .selector_param("threshold", "0.01")
                    .evaluation_mode("regression");
            },
            
            // === BEZ FEATURE SELECTION ===
            "logreg_scaler_only" => {
                builder = builder
                    .model("logreg")
                    .add_processor("scaler")
                    .evaluation_mode("classification");
            },
            "linreg_minmax_only" => {
                builder = builder
                    .model("linreg")
                    .add_processor("minmax_scaler")
                    .evaluation_mode("regression");
            },
            "knn_robust_only" => {
                builder = builder
                    .model("knn")
                    .model_param("k", "5")
                    .add_processor("robust_scaler")
                    .evaluation_mode("classification");
            },
            
            // === S LABEL ENCODING ===
            "tree_labelenc_chisquare" => {
                builder = builder
                    .model("tree")
                    .add_processor("label_encoder")
                    .feature_selector("chi_square")
                    .selector_param("num_features", "10")
                    .evaluation_mode("classification");
            },
            "logreg_labelenc_variance" => {
                builder = builder
                    .model("logreg")
                    .add_processor("label_encoder")
                    .feature_selector("variance")
                    .selector_param("threshold", "0.01")
                    .evaluation_mode("classification");
            },
            
            // === DECISION TREE ===
            "decision_tree" => {
                builder = builder
                    .model("tree")
                    .feature_selector("information_gain")
                    .selector_param("num_features", "10")
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

        // Reset training state to ensure clean baseline (no leftover indices)
        self.pipeline.as_mut().unwrap().reset_training_state();

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

    /// Trénuje model s konkrétnymi feature indices (z porovnania selektorov)
    #[wasm_bindgen(js_name = trainWithFeatureIndices)]
    pub fn train_with_feature_indices(&mut self, train_ratio: f64, indices_js: JsValue) -> Result<JsValue, JsValue> {
        if self.pipeline.is_none() {
            return Err(JsValue::from_str("Pipeline nie je vytvorený"));
        }

        if self.data_cache.is_none() {
            return Err(JsValue::from_str("Dáta nie sú načítané"));
        }

        let indices: Vec<usize> = serde_wasm_bindgen::from_value(indices_js)
            .map_err(|e| JsValue::from_str(&format!("Indices parse error: {:?}", e)))?;

        if indices.is_empty() {
            return Err(JsValue::from_str("Zoznam feature indices je prázdny"));
        }

        let (x_data, y_data) = self.data_cache.as_ref().unwrap();
        let num_samples = x_data.shape().0;
        let num_features = x_data.shape().1;
        let train_size = (num_samples as f64 * train_ratio) as usize;

        // Validate indices
        for &idx in &indices {
            if idx >= num_features {
                return Err(JsValue::from_str(&format!(
                    "Feature index {} je mimo rozsah (max {})", idx, num_features - 1
                )));
            }
        }

        // Reset pipeline training state and set new indices
        let pipeline = self.pipeline.as_mut().unwrap();
        pipeline.reset_training_state();
        pipeline.set_selected_indices(indices.clone());

        // Split data
        let x_train_slice = x_data.slice(0..train_size, 0..num_features);
        let y_train = y_data[0..train_size].to_vec();
        let x_test_slice = x_data.slice(train_size..num_samples, 0..num_features);
        let y_test = y_data[train_size..num_samples].to_vec();

        // Convert slices to DenseMatrix
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

        // Train with the set indices - train() will use select_features_cached
        self.pipeline.as_mut().unwrap()
            .train(x_train, y_train.clone())
            .map_err(|e| JsValue::from_str(&e))?;

        // Predict on test set
        let mut predictions = Vec::new();
        for row_idx in 0..x_test.shape().0 {
            let row: Vec<f64> = (0..x_test.shape().1)
                .map(|col_idx| *x_test.get((row_idx, col_idx)))
                .collect();
            
            let pred_result = self.pipeline.as_mut().unwrap()
                .predict(row)
                .map_err(|e| {
                    JsValue::from_str(&format!("Prediction error on row {}: {}", row_idx, e))
                })?;
            
            let pred: f64 = pred_result.first().copied().unwrap_or(0.0);
            predictions.push(pred);
        }

        // Calculate metrics
        let eval_mode = self.pipeline.as_ref().unwrap().info().evaluation_mode;
        
        let mut result = TrainingWithEvaluationResult {
            success: true,
            message: format!("Model natrénovaný na {} vzorkách s {} features (z {}), testovaný na {}", 
                train_size, indices.len(), num_features, num_samples - train_size),
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
            selected_features_indices: Some(indices),
            total_features_before: num_features,
            total_features_after: self.pipeline.as_ref().unwrap().selected_indices.as_ref().map_or(num_features, |i| i.len()),
        };

        if eval_mode == "classification" {
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
    
    /// Get detailed selection information (e.g., correlation matrix for correlation selector)
    #[wasm_bindgen(js_name = getSelectionDetails)]
    pub fn get_selection_details(&self) -> Result<JsValue, JsValue> {
        if let Some(ref pipeline) = self.pipeline {
            if let Some(ref selector) = pipeline.selector {
                let html = selector.get_selection_details();
                Ok(JsValue::from_str(&html))
            } else {
                Ok(JsValue::from_str(""))
            }
        } else {
            Err(JsValue::from_str("Pipeline not built"))
        }
    }

    /// Porovná viaceré feature selektory na dátach BEZ potreby pipeline.
    /// Umožňuje používateľovi preskúmať feature selection ešte pred vytvorením pipeline.
    #[wasm_bindgen(js_name = compareSelectors)]
    pub fn compare_selectors(
        &self,
        data: &str,
        target_column: &str,
        format: &str,
        selectors_json: JsValue,
    ) -> Result<JsValue, JsValue> {
        let selector_configs: Vec<SelectorCompareConfig> = serde_wasm_bindgen::from_value(selectors_json)
            .map_err(|e| JsValue::from_str(&format!("Invalid selector configs: {:?}", e)))?;

        if selector_configs.is_empty() {
            return Err(JsValue::from_str("Vyberte aspoň jeden selektor na porovnanie"));
        }

        // Load data
        let mut loader = DataLoaderFactory::create(format)
            .map_err(|e| JsValue::from_str(&e))?;
        let loaded = loader.load_from_string(data, target_column)
            .map_err(|e| JsValue::from_str(&e))?;

        let x = &loaded.x_data;
        let y = &loaded.y_data;
        let (n_rows, n_features) = x.shape();

        // Extract feature names
        let feature_names: Vec<String> = if format == "csv" {
            let first_line = data.lines().next().unwrap_or("");
            first_line.split(',')
                .map(|h| h.trim().to_string())
                .filter(|h| h != target_column)
                .collect()
        } else {
            (0..n_features).map(|i| format!("feature{}", i + 1)).collect()
        };

        // Run each selector
        let mut selector_results: Vec<serde_json::Value> = Vec::new();

        for config in &selector_configs {
            match FeatureSelectorFactory::create(&config.name) {
                Ok(mut selector) => {
                    // Set params
                    for (key, value) in &config.params {
                        let _ = selector.set_param(key, value);
                    }
                    // Run selection (catch panics gracefully)
                    let selected = selector.get_selected_indices(x, y);
                    let scores = selector.get_feature_scores(x, y);
                    let details_html = selector.get_selection_details();
                    let metric_name = selector.get_metric_name().to_string();

                    // Build per-feature scores map
                    let mut score_map: Vec<Option<f64>> = vec![None; n_features];
                    if let Some(ref s) = scores {
                        for &(idx, val) in s {
                            if idx < n_features {
                                score_map[idx] = Some(val);
                            }
                        }
                    }

                    selector_results.push(serde_json::json!({
                        "selector_name": selector.get_name(),
                        "config_name": config.name,
                        "selected_indices": selected,
                        "total_features": n_features,
                        "selected_count": selected.len(),
                        "metric_name": metric_name,
                        "score_map": score_map,
                        "details_html": details_html,
                        "error": null
                    }));
                },
                Err(e) => {
                    selector_results.push(serde_json::json!({
                        "selector_name": config.name,
                        "config_name": config.name,
                        "error": format!("Chyba: {}", e),
                        "selected_indices": [],
                        "total_features": n_features,
                        "selected_count": 0,
                        "metric_name": "",
                        "score_map": [],
                        "details_html": ""
                    }));
                }
            }
        }

        // Build comparison HTML
        let comparison_html = Self::build_comparison_html(&selector_results, &feature_names, n_features);

        let result = serde_json::json!({
            "selectors": selector_results,
            "comparison_html": comparison_html,
            "total_features": n_features,
            "total_rows": n_rows,
            "feature_names": feature_names,
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Analyzuje stĺpce dát a odporúča najlepšiu cieľovú premennú
    /// Pre každý stĺpec vypočíta priemernú absolútnu koreláciu s ostatnými stĺpcami
    #[wasm_bindgen(js_name = analyzeTargetCandidates)]
    pub fn analyze_target_candidates(&self, data: &str, format: &str) -> Result<JsValue, JsValue> {
        if format != "csv" {
            return Err(JsValue::from_str("Analýza je zatiaľ podporovaná len pre CSV formát"));
        }
        
        let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() < 3 {
            return Err(JsValue::from_str("Nedostatok dát (minimum 2 riadky + hlavička)"));
        }
        
        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let num_cols = headers.len();
        if num_cols < 2 {
            return Err(JsValue::from_str("Potrebné sú aspoň 2 stĺpce"));
        }
        
        // Parse all data rows
        let mut all_data: Vec<Vec<f64>> = Vec::new();
        let mut skipped = 0usize;
        for line in &lines[1..] {
            let vals: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if vals.len() != num_cols { skipped += 1; continue; }
            match vals.iter().map(|v| v.parse::<f64>()).collect::<Result<Vec<f64>, _>>() {
                Ok(row) => all_data.push(row),
                Err(_) => { skipped += 1; }
            }
        }
        
        if all_data.len() < 2 {
            return Err(JsValue::from_str("Nedostatok numerických riadkov pre analýzu"));
        }
        let n = all_data.len();
        
        // Full correlation matrix
        let mut corr_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        for i in 0..num_cols {
            let col_i: Vec<f64> = all_data.iter().map(|r| r[i]).collect();
            corr_matrix[i][i] = 1.0;
            for j in (i+1)..num_cols {
                let col_j: Vec<f64> = all_data.iter().map(|r| r[j]).collect();
                let c = Self::pearson_corr(&col_i, &col_j);
                corr_matrix[i][j] = c;
                corr_matrix[j][i] = c;
            }
        }
        
        // Analyze each column as potential target
        let mut candidates: Vec<serde_json::Value> = Vec::new();
        for col_idx in 0..num_cols {
            let col_values: Vec<f64> = all_data.iter().map(|r| r[col_idx]).collect();
            
            // Unique values
            let mut uniq = std::collections::HashSet::new();
            for &v in &col_values { uniq.insert(v.to_bits()); }
            let unique_count = uniq.len();
            
            // Variance
            let mean = col_values.iter().sum::<f64>() / n as f64;
            let variance = col_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
            
            // Average abs correlation with other columns
            let mut total = 0.0f64;
            let mut max_c = 0.0f64;
            for j in 0..num_cols {
                if j == col_idx { continue; }
                let ac = corr_matrix[col_idx][j].abs();
                total += ac;
                if ac > max_c { max_c = ac; }
            }
            let avg = if num_cols > 1 { total / (num_cols - 1) as f64 } else { 0.0 };
            
            // Determine classification vs regression
            let is_cat = unique_count <= 10 || (unique_count as f64 / n as f64) < 0.05;
            let stype = if is_cat { "classification" } else { "regression" };
            let score = avg * 100.0;
            
            candidates.push(serde_json::json!({
                "column_index": col_idx,
                "column_name": headers[col_idx],
                "unique_values": unique_count,
                "variance": (variance * 10000.0).round() / 10000.0,
                "avg_correlation": (avg * 10000.0).round() / 10000.0,
                "max_correlation": (max_c * 10000.0).round() / 10000.0,
                "suggested_type": stype,
                "target_score": (score * 10.0).round() / 10.0
            }));
        }
        
        // Sort by target_score descending
        candidates.sort_by(|a, b| {
            let sa = a["target_score"].as_f64().unwrap_or(0.0);
            let sb = b["target_score"].as_f64().unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap()
        });
        
        // Build HTML
        let mut html = String::new();
        html.push_str("<div style='font-family:Segoe UI,sans-serif;'>");
        html.push_str(&format!("<p style='color:#6c757d;margin-bottom:15px;'>Analýza <b>{}</b> riadkov × <b>{}</b> stĺpcov", n, num_cols));
        if skipped > 0 { html.push_str(&format!(" ({} ne-numerických preskočených)", skipped)); }
        html.push_str("</p>");
        
        // Ranking table
        html.push_str("<table style='width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px;'>");
        html.push_str("<thead><tr>");
        for h in &["#", "Stĺpec", "Unikátnych hodnôt", "Priem. |korelácia|", "Max |korelácia|", "Odporúčaný typ", "Skóre prediktability"] {
            html.push_str(&format!("<th style='padding:10px 8px;border:1px solid #dee2e6;background:#3498db;color:white;text-align:center;font-size:12px;'>{}</th>", h));
        }
        html.push_str("</tr></thead><tbody>");
        
        for (rank, cand) in candidates.iter().enumerate() {
            let bg = match rank {
                0 => "background:#d1ecf1;",
                1 | 2 => "background:#e8f4f8;",
                _ => if rank % 2 == 0 { "background:#f8f9fa;" } else { "" },
            };
            let type_icon = if cand["suggested_type"].as_str() == Some("classification") { "Klasifikácia" } else { "Regresia" };
            
            html.push_str(&format!("<tr style='{}'>", bg));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>", rank + 1));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;font-weight:bold;'>{}</td>", cand["column_name"].as_str().unwrap_or("")));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand["unique_values"]));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand["avg_correlation"]));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand["max_correlation"]));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", type_icon));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;font-weight:bold;color:#2980b9;font-size:1.1em;'>{}</td>", cand["target_score"]));
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");
        
        // Correlation matrix (only if <= 15 columns)
        if num_cols <= 15 {
            html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Korelačná matica všetkých stĺpcov</h4>");
            html.push_str("<div style='overflow-x:auto;'><table style='border-collapse:collapse;font-size:11px;'>");
            html.push_str("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;'></th>");
            for h in &headers {
                html.push_str(&format!("<th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{}</th>", h));
            }
            html.push_str("</tr>");
            for i in 0..num_cols {
                html.push_str(&format!("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;white-space:nowrap;'>{}</th>", &headers[i]));
                for j in 0..num_cols {
                    let c = corr_matrix[i][j];
                    let ac = c.abs();
                    let bg_color = if i == j {
                        "#e0e0e0".to_string()
                    } else if ac > 0.7 {
                        format!("rgba(52,152,219,{})", 0.3 + ac * 0.5)
                    } else if ac > 0.4 {
                        format!("rgba(46,204,113,{})", 0.2 + ac * 0.3)
                    } else {
                        format!("rgba(200,200,200,{})", 0.1 + ac * 0.2)
                    };
                    html.push_str(&format!("<td style='padding:6px;border:1px solid #ddd;text-align:center;background:{};font-size:10px;'>{:.3}</td>", bg_color, c));
                }
                html.push_str("</tr>");
            }
            html.push_str("</table></div>");
            html.push_str("<div style='margin-top:8px;font-size:11px;display:flex;gap:10px;flex-wrap:wrap;'>");
            html.push_str("<span style='background:rgba(52,152,219,0.7);padding:2px 8px;color:white;'>Silná korelácia (|r| &gt; 0.7)</span>");
            html.push_str("<span style='background:rgba(46,204,113,0.4);padding:2px 8px;'>Stredná korelácia (|r| &gt; 0.4)</span>");
            html.push_str("<span style='background:rgba(200,200,200,0.3);padding:2px 8px;'>Slabá korelácia</span>");
            html.push_str("</div>");
        }
        
        // Recommendation box
        let best_name = candidates[0]["column_name"].as_str().unwrap_or("");
        let best_score = candidates[0]["target_score"].as_f64().unwrap_or(0.0);
        let best_type = if candidates[0]["suggested_type"].as_str() == Some("classification") { "klasifikácia" } else { "regresia" };
        html.push_str(&format!(
            "<div style='margin-top:15px;padding:15px;background:#d1ecf1;border-left:4px solid #3498db;'>\
            <strong>Odporúčanie:</strong> Najlepšia cieľová premenná je <strong>{}</strong> \
            (skóre prediktability: <strong>{:.1}</strong>, odporúčaný typ: <strong>{}</strong>)\
            </div>", best_name, best_score, best_type
        ));
        html.push_str("</div>");
        
        let recommended_target = candidates[0]["column_name"].as_str().unwrap_or("").to_string();
        let recommended_type = candidates[0]["suggested_type"].as_str().unwrap_or("classification").to_string();
        
        let result = serde_json::json!({
            "candidates": candidates,
            "recommended_target": recommended_target,
            "recommended_type": recommended_type,
            "html": html,
            "total_rows": n,
            "total_cols": num_cols,
            "skipped_rows": skipped
        });
        
        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Vráti zoznam dostupných analyzátorov cieľovej premennej
    #[wasm_bindgen(js_name = getAvailableTargetAnalyzers)]
    pub fn get_available_target_analyzers(&self) -> Result<JsValue, JsValue> {
        let analyzers: Vec<serde_json::Value> = TargetAnalyzerFactory::available()
            .into_iter()
            .map(|(name, desc)| {
                let analyzer = TargetAnalyzerFactory::create(name).unwrap();
                serde_json::json!({
                    "name": name,
                    "description": desc,
                    "metric_name": analyzer.get_metric_name(),
                    "metric_explanation": analyzer.get_metric_explanation(),
                })
            })
            .collect();
        Ok(serde_wasm_bindgen::to_value(&analyzers).unwrap())
    }

    /// Analyzuje cieľovú premennú pomocou zvoleného analyzátora
    #[wasm_bindgen(js_name = analyzeTargetWith)]
    pub fn analyze_target_with(&self, data: &str, format: &str, method: &str) -> Result<JsValue, JsValue> {
        if format != "csv" {
            return Err(JsValue::from_str("Analýza je zatiaľ podporovaná len pre CSV formát"));
        }

        let analyzer = TargetAnalyzerFactory::create(method)
            .map_err(|e| JsValue::from_str(&e))?;

        let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() < 3 {
            return Err(JsValue::from_str("Nedostatok dát (minimum 2 riadky + hlavička)"));
        }

        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let num_cols = headers.len();
        if num_cols < 2 {
            return Err(JsValue::from_str("Potrebné sú aspoň 2 stĺpce"));
        }

        // Parse data rows
        let mut all_data: Vec<Vec<f64>> = Vec::new();
        let mut skipped = 0usize;
        for line in &lines[1..] {
            let vals: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if vals.len() != num_cols { skipped += 1; continue; }
            match vals.iter().map(|v| v.parse::<f64>()).collect::<Result<Vec<f64>, _>>() {
                Ok(row) => all_data.push(row),
                Err(_) => { skipped += 1; }
            }
        }

        if all_data.len() < 2 {
            return Err(JsValue::from_str("Nedostatok numerických riadkov pre analýzu"));
        }
        let n = all_data.len();

        // Convert rows to columns
        let columns: Vec<Vec<f64>> = (0..num_cols)
            .map(|col_idx| all_data.iter().map(|row| row[col_idx]).collect())
            .collect();

        // Run analyzer
        let candidates = analyzer.analyze(&columns, &headers);

        // Build result HTML
        let mut html = String::new();
        html.push_str("<div style='font-family:Segoe UI,sans-serif;'>");
        html.push_str(&format!("<p style='color:#6c757d;margin-bottom:10px;'>Analýza <b>{}</b> riadkov x <b>{}</b> stĺpcov", n, num_cols));
        if skipped > 0 { html.push_str(&format!(" ({} ne-numerických preskočených)", skipped)); }
        html.push_str("</p>");

        // Metric explanation box
        html.push_str(&format!(
            "<div style='margin-bottom:15px;padding:12px;background:#eaf2f8;border-left:4px solid #3498db;font-size:0.9em;'>\
            <strong>Metrika: {}</strong><br>\
            <span style='color:#495057;'>{}</span>\
            </div>",
            analyzer.get_metric_name(),
            analyzer.get_metric_explanation()
        ));

        // Ranking table
        html.push_str("<table style='width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px;'>");
        html.push_str("<thead><tr>");
        for h in &["#", "Stĺpec", "Unikátnych hodnôt", "Variancia", analyzer.get_metric_name(), "Odporúčaný typ", "Skóre"] {
            html.push_str(&format!("<th style='padding:10px 8px;border:1px solid #dee2e6;background:#3498db;color:white;text-align:center;font-size:12px;'>{}</th>", h));
        }
        html.push_str("</tr></thead><tbody>");

        for (rank, cand) in candidates.iter().enumerate() {
            let bg = match rank {
                0 => "background:#d1ecf1;",
                1 | 2 => "background:#e8f4f8;",
                _ => if rank % 2 == 0 { "background:#f8f9fa;" } else { "" },
            };
            let type_label = if cand.suggested_type == "classification" { "Klasifikácia" } else { "Regresia" };

            // Main metric value from extra_metrics (first one)
            let metric_val = cand.extra_metrics.first().map(|(_, v)| format!("{:.4}", v)).unwrap_or_default();

            html.push_str(&format!("<tr style='{}cursor:pointer;' onclick=\"window.selectTargetFromAnalysis && window.selectTargetFromAnalysis('{}', '{}')\">", bg, cand.column_name, cand.suggested_type));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>", rank + 1));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;font-weight:bold;'>{}</td>", cand.column_name));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand.unique_values));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand.variance));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", metric_val));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", type_label));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;font-weight:bold;color:#2980b9;font-size:1.1em;'>{:.1}</td>", cand.score));
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");

        // Details HTML (matrix etc.)
        let details_html = analyzer.get_details_html(&columns, &headers, &candidates);
        if !details_html.is_empty() {
            html.push_str("<details style='margin-bottom:15px;border:1px solid #dee2e6;'>");
            html.push_str(&format!(
                "<summary style='padding:12px;background:#f8f9fa;cursor:pointer;font-weight:600;color:#3498db;'>Detailná vizualizácia - {}</summary>",
                analyzer.get_description()
            ));
            html.push_str(&format!("<div style='padding:15px;overflow:auto;max-height:500px;'>{}</div>", details_html));
            html.push_str("</details>");
        }

        // Recommendation box
        if let Some(best) = candidates.first() {
            let best_type = if best.suggested_type == "classification" { "klasifikácia" } else { "regresia" };
            html.push_str(&format!(
                "<div style='margin-top:15px;padding:15px;background:#d1ecf1;border-left:4px solid #3498db;'>\
                <strong>Odporúčanie ({}):</strong> Najlepšia cieľová premenná je <strong>{}</strong> \
                (skóre: <strong>{:.1}</strong>, odporúčaný typ: <strong>{}</strong>)\
                </div>",
                analyzer.get_description(), best.column_name, best.score, best_type
            ));
        }
        html.push_str("</div>");

        // Structured result
        let cand_json: Vec<serde_json::Value> = candidates.iter().map(|c| {
            serde_json::json!({
                "column_index": c.column_index,
                "column_name": c.column_name,
                "score": c.score,
                "unique_values": c.unique_values,
                "variance": c.variance,
                "suggested_type": c.suggested_type,
                "extra_metrics": c.extra_metrics,
            })
        }).collect();

        let recommended_target = candidates.first().map(|c| c.column_name.clone()).unwrap_or_default();
        let recommended_type = candidates.first().map(|c| c.suggested_type.clone()).unwrap_or_else(|| "classification".to_string());

        let result = serde_json::json!({
            "method": method,
            "method_description": analyzer.get_description(),
            "metric_name": analyzer.get_metric_name(),
            "metric_explanation": analyzer.get_metric_explanation(),
            "candidates": cand_json,
            "recommended_target": recommended_target,
            "recommended_type": recommended_type,
            "html": html,
            "total_rows": n,
            "total_cols": num_cols,
            "skipped_rows": skipped,
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
}

// Helper methods (not exported to WASM)
impl WasmMLPipeline {
    fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 { return 0.0; }
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }
        let den = (den_x * den_y).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }

    /// Vytvori porovnávaciu HTML tabulku pre viaceré selektory
    fn build_comparison_html(
        selector_results: &[serde_json::Value],
        feature_names: &[String],
        total_features: usize,
    ) -> String {
        let mut html = String::new();
        html.push_str("<div style='font-family:Segoe UI,sans-serif;'>");

        // Summary cards
        html.push_str("<div style='display:flex;gap:15px;flex-wrap:wrap;margin-bottom:20px;'>");
        for r in selector_results {
            let name = r["selector_name"].as_str().unwrap_or("");
            let err = r["error"].as_str();
            let count = r["selected_count"].as_u64().unwrap_or(0);
            let border_color = if err.is_some() { "#e74c3c" } else { "#3498db" };
            html.push_str(&format!(
                "<div style='flex:1;min-width:180px;padding:15px;border:2px solid {};background:white;'>\
                <div style='font-weight:bold;color:{};margin-bottom:8px;'>{}</div>",
                border_color, border_color, name
            ));
            if let Some(e) = err {
                html.push_str(&format!("<div style='color:#e74c3c;font-size:0.9em;'>{}</div>", e));
            } else {
                let metric = r["metric_name"].as_str().unwrap_or("");
                html.push_str(&format!(
                    "<div style='font-size:1.5em;font-weight:bold;color:#2c3e50;'>{}/{}</div>\
                    <div style='font-size:0.85em;color:#6c757d;'>vybraných features</div>\
                    <div style='font-size:0.8em;color:#6c757d;margin-top:4px;'>Metrika: {}</div>",
                    count, total_features, metric
                ));
            }
            html.push_str("</div>");
        }
        html.push_str("</div>");

        // Comparison matrix table
        let valid_selectors: Vec<&serde_json::Value> = selector_results.iter()
            .filter(|r| r["error"].is_null())
            .collect();

        if !valid_selectors.is_empty() {
            html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Porovnávacia matica</h4>");
            html.push_str("<div style='overflow-x:auto;'>");
            html.push_str("<table style='width:100%;border-collapse:collapse;font-size:13px;'>");

            // Header
            html.push_str("<thead><tr>");
            html.push_str("<th style='padding:10px;border:1px solid #dee2e6;background:#3498db;color:white;text-align:left;min-width:120px;position:sticky;left:0;z-index:1;'>Príznak</th>");
            for r in &valid_selectors {
                let name = r["selector_name"].as_str().unwrap_or("");
                let count = r["selected_count"].as_u64().unwrap_or(0);
                html.push_str(&format!(
                    "<th style='padding:10px;border:1px solid #dee2e6;background:#3498db;color:white;text-align:center;min-width:100px;'>{}<br><small>({}/{})</small></th>",
                    name, count, total_features
                ));
            }
            html.push_str("<th style='padding:10px;border:1px solid #dee2e6;background:#2c3e50;color:white;text-align:center;'>Zhoda</th>");
            html.push_str("</tr></thead><tbody>");

            // Feature rows
            for (idx, fname) in feature_names.iter().enumerate() {
                let mut selected_by = 0usize;
                let total_valid = valid_selectors.len();

                let mut cells = String::new();
                for r in &valid_selectors {
                    let sel_arr = r["selected_indices"].as_array();
                    let is_selected = sel_arr.map_or(false, |a|
                        a.iter().any(|v| v.as_u64() == Some(idx as u64))
                    );
                    if is_selected { selected_by += 1; }

                    // Score
                    let score_text = r["score_map"].as_array()
                        .and_then(|arr| arr.get(idx))
                        .and_then(|v| v.as_f64())
                        .map(|v| format!("<br><small style='color:#6c757d;'>{:.4}</small>", v))
                        .unwrap_or_default();

                    let (bg, icon) = if is_selected {
                        ("rgba(52,152,219,0.15)", "[+]")
                    } else {
                        ("rgba(189,195,199,0.15)", "[-]")
                    };
                    cells.push_str(&format!(
                        "<td style='padding:8px;border:1px solid #dee2e6;text-align:center;background:{};'><strong>{}</strong>{}</td>",
                        bg, icon, score_text
                    ));
                }

                // Consensus column
                let consensus_pct = if total_valid > 0 { (selected_by as f64 / total_valid as f64 * 100.0) as u64 } else { 0 };
                let consensus_bg = if consensus_pct == 100 { "#c8e6c9" }
                    else if consensus_pct >= 50 { "#fff9c4" }
                    else if consensus_pct > 0 { "#ffe0b2" }
                    else { "#ffcdd2" };

                let row_bg = if idx % 2 == 0 { "" } else { "background:#fafafa;" };
                html.push_str(&format!("<tr style='{}'>", row_bg));
                html.push_str(&format!(
                    "<td style='padding:8px;border:1px solid #dee2e6;font-weight:bold;position:sticky;left:0;background:white;z-index:1;'>{} <span style='color:#6c757d;font-size:0.8em;'>[{}]</span></td>",
                    fname, idx
                ));
                html.push_str(&cells);
                html.push_str(&format!(
                    "<td style='padding:8px;border:1px solid #dee2e6;text-align:center;background:{};font-weight:bold;'>{}/{} ({}%)</td>",
                    consensus_bg, selected_by, total_valid, consensus_pct
                ));
                html.push_str("</tr>");
            }
            html.push_str("</tbody></table></div>");

            // Legend
            html.push_str("<div style='margin-top:10px;font-size:12px;display:flex;gap:15px;flex-wrap:wrap;'>");
            html.push_str("<span style='background:rgba(52,152,219,0.15);padding:3px 10px;'>[+] Vybraný</span>");
            html.push_str("<span style='background:rgba(189,195,199,0.15);padding:3px 10px;'>[-] Nevybraný</span>");
            html.push_str("<span style='background:#c8e6c9;padding:3px 10px;'>100% zhoda</span>");
            html.push_str("<span style='background:#fff9c4;padding:3px 10px;'>&ge;50% zhoda</span>");
            html.push_str("<span style='background:#ffcdd2;padding:3px 10px;'>0% zhoda</span>");
            html.push_str("<div style='color:#6c757d;padding:3px 10px;'><strong>Čísla v hranatých zátvorkách [0], [1], [2]...</strong> označujú poradie features v datasete</div>");
            html.push_str("</div>");

            // Per-selector details in collapsible sections
            html.push_str("<h4 style='color:#495057;margin:25px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Detaily jednotlivých selektorov</h4>");
            
            // Feature mapping table (shown once before all selectors)
            html.push_str("<div style='margin-bottom:15px;padding:12px;background:#f8f9fa;border:1px solid #dee2e6;'>");
            html.push_str("<strong style='color:#495057;margin-bottom:8px;display:block;'>Mapa features (index → názov stĺpca):</strong>");
            html.push_str("<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;font-size:12px;'>");
            for (idx, fname) in feature_names.iter().enumerate() {
                html.push_str(&format!(
                    "<div style='padding:4px 8px;background:white;border:1px solid #dee2e6;'><strong style='color:#3498db;'>F{}</strong>: {}</div>",
                    idx, fname
                ));
            }
            html.push_str("</div></div>");
            
            for r in &valid_selectors {
                let name = r["selector_name"].as_str().unwrap_or("");
                let details = r["details_html"].as_str().unwrap_or("");
                if !details.is_empty() {
                    html.push_str(&format!(
                        "<details style='margin-bottom:10px;border:1px solid #dee2e6;'>\
                        <summary style='padding:12px;background:#f8f9fa;cursor:pointer;font-weight:600;color:#3498db;'>{} - Detailná analýza</summary>\
                        <div style='padding:15px;overflow:auto;max-height:500px;'>{}</div>\
                        </details>",
                        name, details
                    ));
                }
            }
        }

        html.push_str("</div>");
        html
    }
}
