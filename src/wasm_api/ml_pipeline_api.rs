use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};
use crate::pipeline::{MLPipeline, MLPipelineBuilder};
use crate::data_loading::DataLoaderFactory;
use crate::feature_selection_strategies::factory::FeatureSelectorFactory;
use crate::target_analysis::TargetAnalyzerFactory;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};
use std::cell::RefCell;
use statrs::function::gamma::digamma;

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
pub struct RedundancyWarning {
    pub feature1_idx: usize,
    pub feature1_name: String,
    pub feature2_idx: usize,
    pub feature2_name: String,
    pub correlation: f64,
    pub warning_type: String,  // "high_correlation" alebo "high_redundancy"
    pub message: String,
}

#[derive(Serialize, Deserialize)]
pub struct RedundancyReport {
    pub warnings: Vec<RedundancyWarning>,
    pub has_issues: bool,
    pub summary: String,
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
    pub specificity: f64,
    pub false_positive_rate: f64,
    pub false_negative_rate: f64,
    pub mcc: f64,
    pub true_positives: f64,
    pub true_negatives: f64,
    pub false_positives: f64,
    pub false_negatives: f64,
    // Regression metrics
    pub mse: f64,
    pub rmse: f64,
    pub mae: f64,
    pub mape: f64,
    pub median_absolute_error: f64,
    pub pearson_correlation: f64,
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
    /// Cache pre analyzované dáta aby sme ich nemuseli parsovať viackrát
    analysis_cache: RefCell<Option<(usize, usize, Vec<Vec<f64>>, Vec<String>)>>,  // (num_rows, num_cols, columns, headers)
    /// Centralizovaný cache pre matice - Key: (num_rows, num_cols), Value: (corr_matrix, mi_matrix, smc_matrix)
    matrices_cache: RefCell<Option<(Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>)>>,  // (correlation, MI, SMC)
    /// Cache pre split indexy - zabezpečuje rovnaký split pri všetkých porovnaniach
    split_cache: RefCell<Option<(f64, Vec<usize>, Vec<usize>)>>,  // (train_ratio, train_indices, test_indices)
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
            analysis_cache: RefCell::new(None),
            matrices_cache: RefCell::new(None),
            split_cache: RefCell::new(None),
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
        
        // Vymaž split cache pri načítaní nových dát
        *self.split_cache.borrow_mut() = None;
        // Vymaž matrices cache pri načítaní nových dát
        *self.matrices_cache.borrow_mut() = None;
        // Vymaž analysis cache pri načítaní nových dát
        *self.analysis_cache.borrow_mut() = None;

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

    /// Helper: Zabezpečí že split indexy sú cache-ované pre daný train_ratio
    /// Ak cache neexistuje alebo sa zmenil train_ratio, vytvorí nový deterministický split
    /// Vracia (train_indices, test_indices)
    fn ensure_split_indices(&self, train_ratio: f64) -> Result<(Vec<usize>, Vec<usize>), JsValue> {
        if self.data_cache.is_none() {
            return Err(JsValue::from_str("Dáta nie sú načítané"));
        }

        let (x_data, _) = self.data_cache.as_ref().unwrap();
        let num_samples = x_data.shape().0;
        let train_size = (num_samples as f64 * train_ratio) as usize;

        // Skontroluj či už máme cached split pre tento train_ratio
        {
            let cache = self.split_cache.borrow();
            if let Some((cached_ratio, ref train_idx, ref test_idx)) = *cache {
                // Porovnaj s malou toleranciou pre floating point
                if (cached_ratio - train_ratio).abs() < 1e-9 {
                    return Ok((train_idx.clone(), test_idx.clone()));
                }
            }
        }

        // Vytvor nový deterministický split
        let train_indices: Vec<usize> = (0..train_size).collect();
        let test_indices: Vec<usize> = (train_size..num_samples).collect();

        // Ulož do cache
        *self.split_cache.borrow_mut() = Some((train_ratio, train_indices.clone(), test_indices.clone()));

        Ok((train_indices, test_indices))
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

        // Zabezpeč cached split indexy (deterministický split pre všetky porovnania)
        let (train_indices, test_indices) = self.ensure_split_indices(train_ratio)?;

        let (x_data, y_data) = self.data_cache.as_ref().unwrap();
        let num_samples = x_data.shape().0;
        let num_features = x_data.shape().1;

        // Reset training state to ensure clean baseline (no leftover indices)
        self.pipeline.as_mut().unwrap().reset_training_state();

        // Split data pomocou cached indexov
        // Vytvor train data
        let x_train_data: Vec<Vec<f64>> = train_indices.iter()
            .map(|&i| (0..num_features).map(|j| *x_data.get((i, j))).collect())
            .collect();
        let y_train: Vec<f64> = train_indices.iter().map(|&i| y_data[i]).collect();
        
        // Vytvor test data
        let x_test_data: Vec<Vec<f64>> = test_indices.iter()
            .map(|&i| (0..num_features).map(|j| *x_data.get((i, j))).collect())
            .collect();
        let y_test: Vec<f64> = test_indices.iter().map(|&i| y_data[i]).collect();

        // Convert to DenseMatrix
        let x_train = DenseMatrix::from_2d_vec(&x_train_data)
            .map_err(|e| JsValue::from_str(&format!("Chyba pri vytváraní train matice: {:?}", e)))?;

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
            message: format!("Model natrénovaný na {} vzorkách, testovaný na {}", 
                train_indices.len(), test_indices.len()),
            samples_trained: train_indices.len(),
            evaluation_mode: eval_mode.clone(),
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            specificity: 0.0,
            false_positive_rate: 0.0,
            false_negative_rate: 0.0,
            mcc: 0.0,
            true_positives: 0.0,
            true_negatives: 0.0,
            false_positives: 0.0,
            false_negatives: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mae: 0.0,
            mape: 0.0,
            median_absolute_error: 0.0,
            pearson_correlation: 0.0,
            r2_score: 0.0,
            selected_features_indices: selected_indices_for_result,
            total_features_before: features_before,
            total_features_after: features_after,
        };


        if eval_mode == "classification" {
            // Classification metrics - confusion matrix
            let mut correct = 0;
            let mut tp = 0.0;  // True Positives
            let mut tn = 0.0;  // True Negatives
            let mut fp = 0.0;  // False Positives
            let mut fn_ = 0.0; // False Negatives
            
            for (i, &pred) in predictions.iter().enumerate() {
                let actual: f64 = y_test[i];
                let p_val = if pred > 0.5 { 1.0 } else { 0.0 };
                let a_val = actual.round();
                
                if (p_val - a_val).abs() < 0.1 {
                    correct += 1;
                }
                
                if p_val == 1.0 && a_val == 1.0 { tp += 1.0; }
                else if p_val == 0.0 && a_val == 0.0 { tn += 1.0; }
                else if p_val == 1.0 && a_val == 0.0 { fp += 1.0; }
                else if p_val == 0.0 && a_val == 1.0 { fn_ += 1.0; }
            }

            result.true_positives = tp;
            result.true_negatives = tn;
            result.false_positives = fp;
            result.false_negatives = fn_;
            
            result.accuracy = correct as f64 / predictions.len() as f64;
            result.precision = if tp + fp > 0.0 {
                tp / (tp + fp)
            } else {
                0.0
            };
            result.recall = if tp + fn_ > 0.0 {
                tp / (tp + fn_)
            } else {
                0.0
            };
            result.f1_score = if result.precision + result.recall > 0.0 {
                2.0 * result.precision * result.recall / (result.precision + result.recall)
            } else {
                0.0
            };
            
            // Specificity (TNR - True Negative Rate)
            result.specificity = if tn + fp > 0.0 {
                tn / (tn + fp)
            } else {
                0.0
            };
            
            // False Positive Rate (FPR)
            result.false_positive_rate = if fp + tn > 0.0 {
                fp / (fp + tn)
            } else {
                0.0
            };
            
            // False Negative Rate (FNR)
            result.false_negative_rate = if fn_ + tp > 0.0 {
                fn_ / (fn_ + tp)
            } else {
                0.0
            };
            
            // Matthews Correlation Coefficient (MCC)
            let mcc_denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
            result.mcc = if mcc_denom > 0.0 {
                (tp * tn - fp * fn_) / mcc_denom
            } else {
                0.0
            };
        } else {
            // Regression metrics
            let n = predictions.len() as f64;
            let mut sum_squared_error = 0.0;
            let mut sum_abs_error = 0.0;
            let mut sum_abs_pct_error = 0.0;
            let y_mean = y_test.iter().sum::<f64>() / n;
            let mut sum_squared_total = 0.0;
            let mut abs_errors = Vec::new();
            let mut correlation_sum = 0.0;
            let mut y_sum = 0.0;
            let mut pred_sum = 0.0;
            let mut y_sq_sum = 0.0;
            let mut pred_sq_sum = 0.0;

            for (i, &pred) in predictions.iter().enumerate() {
                let actual: f64 = y_test[i];
                let error: f64 = pred - actual;
                sum_squared_error += error * error;
                sum_abs_error += error.abs();
                abs_errors.push(error.abs());
                sum_squared_total += (actual - y_mean).powi(2);
                
                // MAPE
                if actual.abs() > 1e-10 {
                    sum_abs_pct_error += (error.abs() / actual.abs()) * 100.0;
                }
                
                // Pearson correlation
                y_sum += actual;
                pred_sum += pred;
                y_sq_sum += actual * actual;
                pred_sq_sum += pred * pred;
                correlation_sum += actual * pred;
            }

            result.mse = sum_squared_error / n;
            result.rmse = result.mse.sqrt();
            result.mae = sum_abs_error / n;
            result.mape = sum_abs_pct_error / n;
            result.r2_score = if sum_squared_total > 0.0 {
                1.0 - (sum_squared_error / sum_squared_total)
            } else {
                0.0
            };
            
            // Median Absolute Error
            if !abs_errors.is_empty() {
                abs_errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let mid = abs_errors.len() / 2;
                result.median_absolute_error = if abs_errors.len() % 2 == 0 {
                    (abs_errors[mid - 1] + abs_errors[mid]) / 2.0
                } else {
                    abs_errors[mid]
                };
            }
            
            // Pearson Correlation Coefficient
            let mean_y = y_sum / n;
            let mean_pred = pred_sum / n;
            let num = correlation_sum - n * mean_y * mean_pred;
            let den = ((y_sq_sum - n * mean_y * mean_y) * (pred_sq_sum - n * mean_pred * mean_pred)).sqrt();
            result.pearson_correlation = if den > 0.0 { num / den } else { 0.0 };
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

        // Zabezpeč cached split indexy (rovnaký split ako pri ostatných porovnaniach)
        let (train_indices, test_indices) = self.ensure_split_indices(train_ratio)?;

        let (x_data, y_data) = self.data_cache.as_ref().unwrap();
        let num_samples = x_data.shape().0;
        let num_features = x_data.shape().1;

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

        // Split data pomocou cached indexov
        let x_train_data: Vec<Vec<f64>> = train_indices.iter()
            .map(|&i| (0..num_features).map(|j| *x_data.get((i, j))).collect())
            .collect();
        let y_train: Vec<f64> = train_indices.iter().map(|&i| y_data[i]).collect();
        
        let x_test_data: Vec<Vec<f64>> = test_indices.iter()
            .map(|&i| (0..num_features).map(|j| *x_data.get((i, j))).collect())
            .collect();
        let y_test: Vec<f64> = test_indices.iter().map(|&i| y_data[i]).collect();

        // Convert to DenseMatrix
        // Convert to DenseMatrix
        let x_train = DenseMatrix::from_2d_vec(&x_train_data)
            .map_err(|e| JsValue::from_str(&format!("Chyba pri vytváraní train matice: {:?}", e)))?;

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
                train_indices.len(), indices.len(), num_features, test_indices.len()),
            samples_trained: train_indices.len(),
            evaluation_mode: eval_mode.clone(),
            accuracy: 0.0,
            precision: 0.0,
            recall: 0.0,
            f1_score: 0.0,
            specificity: 0.0,
            false_positive_rate: 0.0,
            false_negative_rate: 0.0,
            mcc: 0.0,
            true_positives: 0.0,
            true_negatives: 0.0,
            false_positives: 0.0,
            false_negatives: 0.0,
            mse: 0.0,
            rmse: 0.0,
            mae: 0.0,
            mape: 0.0,
            median_absolute_error: 0.0,
            pearson_correlation: 0.0,
            r2_score: 0.0,
            selected_features_indices: Some(indices),
            total_features_before: num_features,
            total_features_after: self.pipeline.as_ref().unwrap().selected_indices.as_ref().map_or(num_features, |i| i.len()),
        };

        if eval_mode == "classification" {
            let mut correct = 0;
            let mut tp = 0.0;  // True Positives
            let mut tn = 0.0;  // True Negatives
            let mut fp = 0.0;  // False Positives
            let mut fn_ = 0.0; // False Negatives
            
            for (i, &pred) in predictions.iter().enumerate() {
                let actual: f64 = y_test[i];
                let p_val = if pred > 0.5 { 1.0 } else { 0.0 };
                let a_val = actual.round();
                
                if (p_val - a_val).abs() < 0.1 {
                    correct += 1;
                }
                
                if p_val == 1.0 && a_val == 1.0 { tp += 1.0; }
                else if p_val == 0.0 && a_val == 0.0 { tn += 1.0; }
                else if p_val == 1.0 && a_val == 0.0 { fp += 1.0; }
                else if p_val == 0.0 && a_val == 1.0 { fn_ += 1.0; }
            }

            result.true_positives = tp;
            result.true_negatives = tn;
            result.false_positives = fp;
            result.false_negatives = fn_;
            
            result.accuracy = correct as f64 / predictions.len() as f64;
            result.precision = if tp + fp > 0.0 {
                tp / (tp + fp)
            } else {
                0.0
            };
            result.recall = if tp + fn_ > 0.0 {
                tp / (tp + fn_)
            } else {
                0.0
            };
            result.f1_score = if result.precision + result.recall > 0.0 {
                2.0 * result.precision * result.recall / (result.precision + result.recall)
            } else {
                0.0
            };
            
            // Specificity (TNR - True Negative Rate)
            result.specificity = if tn + fp > 0.0 {
                tn / (tn + fp)
            } else {
                0.0
            };
            
            // False Positive Rate (FPR)
            result.false_positive_rate = if fp + tn > 0.0 {
                fp / (fp + tn)
            } else {
                0.0
            };
            
            // False Negative Rate (FNR)
            result.false_negative_rate = if fn_ + tp > 0.0 {
                fn_ / (fn_ + tp)
            } else {
                0.0
            };
            
            // Matthews Correlation Coefficient (MCC)
            let mcc_denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
            result.mcc = if mcc_denom > 0.0 {
                (tp * tn - fp * fn_) / mcc_denom
            } else {
                0.0
            };
        } else {
            let n = predictions.len() as f64;
            let mut sum_squared_error = 0.0;
            let mut sum_abs_error = 0.0;
            let mut sum_abs_pct_error = 0.0;
            let y_mean = y_test.iter().sum::<f64>() / n;
            let mut sum_squared_total = 0.0;
            let mut abs_errors = Vec::new();
            let mut correlation_sum = 0.0;
            let mut y_sum = 0.0;
            let mut pred_sum = 0.0;
            let mut y_sq_sum = 0.0;
            let mut pred_sq_sum = 0.0;

            for (i, &pred) in predictions.iter().enumerate() {
                let actual: f64 = y_test[i];
                let error: f64 = pred - actual;
                sum_squared_error += error * error;
                sum_abs_error += error.abs();
                abs_errors.push(error.abs());
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

    /// Získaj feature importance/ranking z embedded metódy (Ridge L2, Tree importance)
    #[wasm_bindgen(js_name = getEmbeddedFeatureRanking)]
    pub fn get_embedded_feature_ranking(&mut self, train_ratio: f64, top_k: usize) -> Result<JsValue, JsValue> {
        if self.data_cache.is_none() {
            return Err(JsValue::from_str("Dáta nie sú načítané"));
        }

        // Zabezpeč cached split indexy (rovnaký split ako pri ostatných porovnaniach)
        let (train_indices, _test_indices) = self.ensure_split_indices(train_ratio)?;

        let (x_data, y_data) = self.data_cache.as_ref().unwrap();
        let num_features = x_data.shape().1;

        if train_indices.is_empty() {
            return Err(JsValue::from_str("Train set je prázdny"));
        }

        // Split data pomocou cached indexov
        let x_train_data: Vec<Vec<f64>> = train_indices.iter()
            .map(|&i| (0..num_features).map(|j| *x_data.get((i, j))).collect())
            .collect();
        let y_train: Vec<f64> = train_indices.iter().map(|&i| y_data[i]).collect();

        let x_train_matrix = DenseMatrix::from_2d_vec(&x_train_data)
            .map_err(|e| JsValue::from_str(&format!("Chyba pri vytváraní train matice: {:?}", e)))?;

        // Detekuj typ úlohy (klasifikácia vs regresia) na základe unikátnych hodnôt y
        let unique_y: std::collections::HashSet<i32> = y_train.iter()
            .map(|&v| v as i32)
            .collect();
        let is_classification = unique_y.len() < 20; // heuristika

        // Použij embedded selector factory
        use crate::embedded::EmbeddedSelectorFactory;
        let mut selector = EmbeddedSelectorFactory::create_for_task(is_classification);
        
        // Natrénuj a získaj ranking
        let ranked = selector.fit_and_rank(&x_train_matrix, &y_train)
            .map_err(|e| JsValue::from_str(&format!("Embedded selector error: {}", e)))?;
        
        let method_name = selector.get_name();

        // Vyber top_k
        let effective_k = top_k.min(num_features);
        let selected_indices: Vec<usize> = ranked.iter()
            .take(effective_k)
            .map(|(idx, _)| *idx)
            .collect();

        let selected_scores: Vec<f64> = ranked.iter()
            .take(effective_k)
            .map(|(_, score)| *score)
            .collect();

        // Všetky scores v pôvodnom poradí
        let mut all_scores = vec![0.0; num_features];
        for (idx, score) in ranked.iter() {
            all_scores[*idx] = *score;
        }

        let result = serde_json::json!({
            "method": method_name,
            "selected_indices": selected_indices,
            "selected_scores": selected_scores,
            "all_scores": all_scores,
            "is_classification": is_classification,
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Helper: Výpočet Pearsonovej korelácie (deprecated - používa sa v embedded moduloch)
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 || x.len() != y.len() { return 0.0; }
        
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

        // Extract feature names (KEEP original order and ALL features)
        let mut feature_names: Vec<String> = if format == "csv" {
            let first_line = data.lines().next().unwrap_or("");
            first_line.split(',')
                .map(|h| h.trim().to_string())
                .collect()
        } else {
            (0..n_features).map(|i| format!("feature{}", i + 1)).collect()
        };
        
        // Ensure feature_names matches n_features (shouldn't differ, but be safe)
        while feature_names.len() < n_features {
            feature_names.push(format!("feature{}", feature_names.len() + 1));
        }

        // Parse and cache columns for matrix computation
        let (columns, _, _, _) = self.parse_csv_data_cached(data, true)?;
        
        // Pre-compute and cache correlation/MI/SMC matrices
        // This will be reused by check_feature_redundancy and other functions
        let _ = self.compute_and_cache_matrices(&columns);

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
        let target_index = feature_names.iter().position(|f| f == target_column).unwrap_or(n_features + 1);
        let comparison_html = Self::build_comparison_html(&selector_results, &feature_names, n_features, target_index);

        let result = serde_json::json!({
            "selectors": selector_results,
            "comparison_html": comparison_html,
            "total_features": n_features,
            "total_rows": n_rows,
            "feature_names": feature_names,
            "target_column": target_column,
            "target_index": target_index,
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
            
            // Score_j = Σ r²_jk (sum of squared correlations)
            let mut sum_r2 = 0.0f64;
            let mut max_c = 0.0f64;
            for j in 0..num_cols {
                if j == col_idx { continue; }
                let c = corr_matrix[col_idx][j];
                sum_r2 += c * c;
                if c.abs() > max_c { max_c = c.abs(); }
            }
            
            // Determine classification vs regression
            let is_cat = unique_count <= 10 || (unique_count as f64 / n as f64) < 0.05;
            let stype = if is_cat { "classification" } else { "regression" };
            
            candidates.push(serde_json::json!({
                "column_index": col_idx,
                "column_name": headers[col_idx],
                "unique_values": unique_count,
                "variance": (variance * 10000.0).round() / 10000.0,
                "sum_r2": (sum_r2 * 10000.0).round() / 10000.0,
                "max_correlation": (max_c * 10000.0).round() / 10000.0,
                "suggested_type": stype,
                "target_score": (sum_r2 * 10000.0).round() / 10000.0
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
        for h in &["#", "Premenná", "Unikátnych hodnôt", "Σr²", "Max |korelácia|"] {
            html.push_str(&format!("<th style='padding:10px 8px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:12px;'>{}</th>", h));
        }
        html.push_str("</tr></thead><tbody>");
        
        for (rank, cand) in candidates.iter().enumerate() {
            let bg = match rank {
                0 => "background:#d1ecf1;",
                1 | 2 => "background:#e8f4f8;",
                _ => if rank % 2 == 0 { "background:#f8f9fa;" } else { "" },
            };
            let _type_icon = if cand["suggested_type"].as_str() == Some("classification") { "Klasifikácia" } else { "Regresia" };
            
            html.push_str(&format!("<tr style='{}'>", bg));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>", rank + 1));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;font-weight:bold;'>{}</td>", cand["column_name"].as_str().unwrap_or("")));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand["unique_values"]));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand["sum_r2"]));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand["max_correlation"]));
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
        let _best_name = candidates[0]["column_name"].as_str().unwrap_or("");
        let _best_score = candidates[0]["target_score"].as_f64().unwrap_or(0.0);
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

    /// Parseuje CSV dáta a vracia (columns, headers, počet riadkov, počet skipnutých).
    /// Výsledok sa cachuje. Automaticky sa resetuje keď sa zmenia rozmery dát.
    fn parse_csv_data_cached(&self, data: &str, should_cache: bool) -> Result<(Vec<Vec<f64>>, Vec<String>, usize, usize), JsValue> {
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
            match vals.iter().map(|v| {
                let trimmed = v.trim();
                if trimmed.is_empty() { Ok(0.0) } else { trimmed.parse::<f64>().or_else(|_| trimmed.replace(',', ".").parse::<f64>()) }
            }).collect::<Result<Vec<f64>, _>>() {
                Ok(row) => all_data.push(row),
                Err(_) => { skipped += 1; }
            }
        }

        if all_data.len() < 2 {
            return Err(JsValue::from_str("Nedostatok numerických riadkov pre analýzu"));
        }

        let num_rows = all_data.len();

        // Convert rows to columns
        let columns: Vec<Vec<f64>> = (0..num_cols)
            .map(|col_idx| all_data.iter().map(|row| row[col_idx]).collect())
            .collect();

        // Cache the result if requested
        if should_cache {
            *self.analysis_cache.borrow_mut() = Some((num_rows, num_cols, columns.clone(), headers.clone()));
        }

        Ok((columns, headers, num_rows, skipped))
    }

    /// Vypočíta a cachuje korelačnú, MI a SMC matice pre budúcu opätovnú použitie
    fn compute_and_cache_matrices(&self, columns: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Skontroluj či sú matice už cachované
        if let Some((corr, mi, smc)) = self.matrices_cache.borrow().as_ref() {
            return (corr.clone(), mi.clone(), smc.clone());
        }

        let num_cols = columns.len();
        if num_cols == 0 {
            return (vec![], vec![], vec![]);
        }

        let n = columns[0].len();

        // Pearsonova korelácia
        let pearson_corr = |x: &[f64], y: &[f64]| -> f64 {
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
        };

        // Korelačná matica (symetrická)
        let mut corr_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        for i in 0..num_cols {
            corr_matrix[i][i] = 1.0;
            for j in (i+1)..num_cols {
                let c = pearson_corr(&columns[i], &columns[j]);
                corr_matrix[i][j] = c;
                corr_matrix[j][i] = c;
            }
        }

        // MI matica - počítaj len pre páry s vyššou koreláciou (optimalizácia)
        // Pre ostatné páry nechaj 0.0 (lazy evaluation)
        let mut mi_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        let k_neighbors = if n < 10 { 2 } else if n < 50 { 3 } else { 5 };
        
        for i in 0..num_cols {
            mi_matrix[i][i] = 0.0;  // MI so sebou samým je 0
            for j in (i+1)..num_cols {
                // Počítaj MI len ak je korelácia aspoň 0.3 (inak to nie je zaujímavé)
                if corr_matrix[i][j].abs() > 0.3 {
                    let mi = Self::estimate_mi_ksg(&columns[i], &columns[j], k_neighbors);
                    mi_matrix[i][j] = mi;
                    mi_matrix[j][i] = mi;
                }
            }
        }

        // SMC matica (placeholder - zatiaľ len nuly)
        let smc_matrix = vec![vec![0.0f64; num_cols]; num_cols];

        // Cache matice
        *self.matrices_cache.borrow_mut() = Some((corr_matrix.clone(), mi_matrix.clone(), smc_matrix.clone()));

        (corr_matrix, mi_matrix, smc_matrix)
    }

    /// Analyzuje cieľovú premennú pomocou zvoleného analyzátora (s cachovaním dát)
    #[wasm_bindgen(js_name = analyzeTargetWith)]
    pub fn analyze_target_with(&self, data: &str, format: &str, method: &str) -> Result<JsValue, JsValue> {
        if format != "csv" {
            return Err(JsValue::from_str("Analýza je zatiaľ podporovaná len pre CSV formát"));
        }

        let analyzer = TargetAnalyzerFactory::create(method)
            .map_err(|e| JsValue::from_str(&e))?;

        // Use cached data if possible
        let (columns, headers, n, skipped) = self.parse_csv_data_cached(data, true)?;

        // Pre-compute and cache matrices for later use
        self.compute_and_cache_matrices(&columns);

        // Run analyzer
        let candidates = analyzer.analyze(&columns, &headers);

        // Build result HTML
        let mut html = String::new();
        html.push_str("<div style='font-family:Segoe UI,sans-serif;'>");
        html.push_str(&format!("<p style='color:#6c757d;margin-bottom:10px;'>Analýza <b>{}</b> riadkov x <b>{}</b> stĺpcov", n, headers.len()));
        if skipped > 0 { html.push_str(&format!(" ({} ne-numerických preskočených)", skipped)); }
        html.push_str("</p>");

        // Metric explanation box
        html.push_str(&format!(
            "<div style='margin-bottom:15px;padding:12px;background:#ffe4e1;border-left:4px solid #cc0000;font-size:0.9em;'>\
            <strong>Metrika: {}</strong><br>\
            <span style='color:#495057;'>{}</span>\
            </div>",
            analyzer.get_metric_name(),
            analyzer.get_metric_explanation()
        ));

        // Ranking table
        html.push_str("<table style='width:100%;border-collapse:collapse;font-size:13px;margin-bottom:20px;'>");
        html.push_str("<thead><tr>");
        for h in &["#", "Premenná", "Unikátnych hodnôt", analyzer.get_metric_name()] {
            html.push_str(&format!("<th style='padding:10px 8px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;font-size:12px;'>{}</th>", h));
        }
        html.push_str("</tr></thead><tbody>");

        for (rank, cand) in candidates.iter().enumerate() {
            let bg = match rank {
                0 => "background:#d1ecf1;",
                1 | 2 => "background:#e8f4f8;",
                _ => if rank % 2 == 0 { "background:#f8f9fa;" } else { "" },
            };

            // Main metric value from extra_metrics (first one)
            let metric_val = cand.extra_metrics.first().map(|(_, v)| format!("{:.4}", v)).unwrap_or_default();

            html.push_str(&format!("<tr style='{}cursor:pointer;' onclick=\"window.selectTargetFromAnalysis && window.selectTargetFromAnalysis('{}', '{}')\">", bg, cand.column_name, cand.suggested_type));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>", rank + 1));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;font-weight:bold;'>{}</td>", cand.column_name));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", cand.unique_values));
            html.push_str(&format!("<td style='padding:8px;border:1px solid #dee2e6;text-align:center;'>{}</td>", metric_val));
            html.push_str("</tr>");
        }
        html.push_str("</tbody></table>");

        // Details HTML (matrix etc.)
        let details_html = analyzer.get_details_html(&columns, &headers, &candidates);
        if !details_html.is_empty() {
            html.push_str("<details style='margin-bottom:15px;border:1px solid #dee2e6;'>");
            html.push_str(&format!(
                "<summary style='padding:12px;background:#f8f9fa;cursor:pointer;font-weight:600;color:#cc0000;'>Detailná vizualizácia - {}</summary>",
                analyzer.get_description()
            ));
            html.push_str(&format!("<div style='padding:15px;overflow:auto;max-height:500px;'>{}</div>", details_html));
            html.push_str("</details>");
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
            "total_cols": headers.len(),
            "skipped_rows": skipped,
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Výpočet entropie premennej pomocou histogramu
    fn estimate_entropy(x: &[f64], bins: usize) -> f64 {
        if x.is_empty() { return 0.0; }
        
        let min_val = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        if (max_val - min_val).abs() < 1e-10 { return 0.0; }
        
        let bin_width = (max_val - min_val) / bins as f64;
        let mut counts = vec![0usize; bins];
        
        for &val in x {
            let bin_idx = ((val - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            counts[bin_idx] += 1;
        }
        
        let n = x.len() as f64;
        let mut entropy = 0.0;
        for &count in &counts {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// KSG odhad mutual information medzi dvoma spojitými premennými
    fn estimate_mi_ksg(x_col: &[f64], y: &[f64], k: usize) -> f64 {
        let n = x_col.len();
        if n <= k { 
            return 0.0; 
        }

        let mut nx_vec = vec![0usize; n];
        let mut ny_vec = vec![0usize; n];

        for i in 0..n {
            let mut distances = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i == j { continue; }
                let dx = (x_col[i] - x_col[j]).abs();
                let dy = (y[i] - y[j]).abs();
                distances.push(dx.max(dy));
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let epsilon = distances[k - 1];

            let mut nx = 0usize;
            let mut ny = 0usize;
            for j in 0..n {
                if i == j { continue; }
                if (x_col[i] - x_col[j]).abs() < epsilon { nx += 1; }
                if (y[i] - y[j]).abs() < epsilon { ny += 1; }
            }
            nx_vec[i] = nx;
            ny_vec[i] = ny;
        }

        let psi_k = digamma(k as f64);
        let psi_n = digamma(n as f64);
        let mut mean_psi = 0.0;
        for i in 0..n {
            mean_psi += digamma((nx_vec[i] + 1) as f64) + digamma((ny_vec[i] + 1) as f64);
        }
        mean_psi /= n as f64;
        (psi_k - mean_psi + psi_n).max(0.0)
    }

    /// Skontroluje redundanciu vybraných features na základe korelácie a MI (vylučuje target column)
    /// Ak je focus_feature zadaný (>= 0), kontroluje len páry s touto feature
    #[wasm_bindgen(js_name = checkFeatureRedundancy)]
    pub fn check_feature_redundancy(&self, data: &str, format: &str, selected_indices: Vec<usize>, target_col_index: i32, focus_feature: i32) -> Result<JsValue, JsValue> {
        if format != "csv" {
            return Err(JsValue::from_str("Analýza je dostupná len pre CSV formát"));
        }

        if selected_indices.is_empty() {
            let report = RedundancyReport {
                warnings: vec![],
                has_issues: false,
                summary: "Žiadne features vybrané.".to_string(),
            };
            return Ok(serde_wasm_bindgen::to_value(&report).unwrap());
        }

        // Parseuj a cachuj dáta
        let (columns, headers, _, _) = self.parse_csv_data_cached(data, true)?;
        
        if selected_indices.iter().any(|&i| i >= columns.len()) {
            return Err(JsValue::from_str("Neplatný feature index"));
        }

        // Vypočítaj a cachuj matice (ak už nie sú cachované)
        let (corr_matrix, mi_matrix, _smc_matrix) = self.compute_and_cache_matrices(&columns);

        let mut warnings = Vec::new();
        
        let focus_idx = if focus_feature >= 0 { Some(focus_feature as usize) } else { None };

        // Skontroluj korelácie a MI LEN medzi vybranými featurmi
        // Vylúč cieľový stĺpec
        // Ak je focus_feature zadaný, kontroluj len páry s touto feature
        for i in 0..selected_indices.len() {
            for j in (i+1)..selected_indices.len() {
                let idx1 = selected_indices[i];
                let idx2 = selected_indices[j];
                
                // Preskočiť ak nie sú platné alebo sú cieľový stĺpec
                if idx1 as i32 == target_col_index || idx2 as i32 == target_col_index {
                    continue;
                }
                
                // Ak je focus_feature zadaný, kontroluj len páry kde je jeden z nich focus
                if let Some(focus) = focus_idx {
                    if idx1 != focus && idx2 != focus {
                        continue;
                    }
                }
                
                // Kontrola korelácie (z cachovanej matice)
                let corr = corr_matrix[idx1][idx2].abs();
                
                // Kontrola Normalized Mutual Information
                // Skús použiť cachovanú hodnotu, ak nie je, vypočítaj ju (lazy evaluation)
                let mi_raw = if mi_matrix[idx1][idx2] > 0.0 {
                    mi_matrix[idx1][idx2]
                } else {
                    let n_samples = columns[idx1].len();
                    let k_neighbors = if n_samples < 10 { 2 } else if n_samples < 50 { 3 } else { 5 };
                    Self::estimate_mi_ksg(&columns[idx1], &columns[idx2], k_neighbors)
                };
                
                // Normalizuj MI pomocou entropií → NMI v rozsahu [0, 1]
                let h1 = Self::estimate_entropy(&columns[idx1], 20);
                let h2 = Self::estimate_entropy(&columns[idx2], 20);
                let nmi = if h1 > 0.0 && h2 > 0.0 {
                    (2.0 * mi_raw / (h1 + h2)).min(1.0)
                } else {
                    0.0
                };
                
                // Upozornenie ak je korelácia > 0.90 (veľmi vysoká lineárna závislosť)
                if corr > 0.90 {
                    warnings.push(RedundancyWarning {
                        feature1_idx: idx1,
                        feature1_name: headers[idx1].clone(),
                        feature2_idx: idx2,
                        feature2_name: headers[idx2].clone(),
                        correlation: corr,
                        warning_type: "high_correlation".to_string(),
                        message: format!(
                            "Premenné '{}' a '{}' sú veľmi silné lineárne korelované ({:.1}%). Jedna z nich je pravdepodobne redundantná.",
                            headers[idx1], headers[idx2], corr * 100.0
                        ),
                    });
                }
                
                // Upozornenie ak je NMI > 0.75 (vysoká zdieľaná informácia)
                // NMI je v rozsahu [0,1] podobne ako korelácia
                if nmi > 0.75 {
                    warnings.push(RedundancyWarning {
                        feature1_idx: idx1,
                        feature1_name: headers[idx1].clone(),
                        feature2_idx: idx2,
                        feature2_name: headers[idx2].clone(),
                        correlation: nmi,
                        warning_type: "high_mutual_information".to_string(),
                        message: format!(
                            "Premenné '{}' a '{}' zdieľajú vysokú informáciu (NMI={:.1}%). Môžu obsahovať duplicitné informácie.",
                            headers[idx1], headers[idx2], nmi * 100.0
                        ),
                    });
                }
            }
        }

        let has_issues = !warnings.is_empty();
        let summary = if has_issues {
            if let Some(focus) = focus_idx {
                if focus < headers.len() {
                    format!("Upozornenie: '{}' má vysokú redundanciu s {} inými premennými!", headers[focus], warnings.len())
                } else {
                    format!("Upozornenie: Nájdené {} páry vysoce korelovaných premenných!", warnings.len())
                }
            } else {
                format!("Upozornenie: Nájdené {} páry vysoce korelovaných premenných v tvojom výbere!", warnings.len())
            }
        } else {
            if focus_idx.is_some() {
                "OK: Novo pridaná feature je nezávislá od ostatných vybraných.".to_string()
            } else {
                "OK: Vybrané features sú dostatočne nezávislé.".to_string()
            }
        };

        let report = RedundancyReport {
            warnings,
            has_issues,
            summary,
        };

        Ok(serde_wasm_bindgen::to_value(&report).unwrap())
    }


    #[wasm_bindgen(js_name = compareTargetAnalyzers)]
    pub fn compare_target_analyzers(&self, data: &str, format: &str, methods: Vec<String>) -> Result<JsValue, JsValue> {
        if format != "csv" {
            return Err(JsValue::from_str("Analýza je zatiaľ podporovaná len pre CSV formát"));
        }

        // Use cached data
        let (columns, headers, n, skipped) = self.parse_csv_data_cached(data, true)?;
        let num_cols = headers.len();

        // Pre-compute and cache matrices for later use
        self.compute_and_cache_matrices(&columns);

        // Run all analyzers
        let mut analyzer_results = vec![];
        for method in &methods {
            let analyzer = match TargetAnalyzerFactory::create(method) {
                Ok(a) => a,
                Err(e) => {
                    analyzer_results.push(serde_json::json!({
                        "method": method,
                        "error": e,
                        "ranking": []
                    }));
                    continue;
                }
            };

            let candidates = analyzer.analyze(&columns, &headers);
            
            let ranking: Vec<serde_json::Value> = candidates.iter().map(|c| {
                serde_json::json!({
                    "column_name": c.column_name,
                    "score": c.score,
                    "unique_values": c.unique_values,
                    "variance": c.variance,
                    "suggested_type": c.suggested_type,
                })
            }).collect();

            analyzer_results.push(serde_json::json!({
                "method": method,
                "method_name": analyzer.get_description(),
                "metric_name": analyzer.get_metric_name(),
                "ranking": ranking
            }));
        }

        let result = serde_json::json!({
            "total_columns": num_cols,
            "total_rows": n,
            "skipped_rows": skipped,
            "columns": headers,
            "analyzer_results": analyzer_results,
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }


    // ======================================================================
    // Data Editor WASM API methods
    // ======================================================================

    /// Vráti všetky dáta vo formáte vhodnom pre editor (všetky riadky, všetky stĺpce)
    #[wasm_bindgen(js_name = getEditableData)]
    pub fn get_editable_data(&self, data: &str, format: &str) -> Result<JsValue, JsValue> {
        if format != "csv" {
            return Err(JsValue::from_str("Editor dát je podporovaný len pre CSV formát"));
        }

        let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return Err(JsValue::from_str("Prázdne dáta"));
        }

        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let mut rows: Vec<Vec<String>> = Vec::new();

        for line in &lines[1..] {
            let vals: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            rows.push(vals);
        }

        let result = serde_json::json!({
            "headers": headers,
            "rows": rows,
            "total_rows": rows.len(),
            "total_cols": headers.len()
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Aplikuje procesor na konkrétny stĺpec v CSV dátach a vráti upravené CSV
    #[wasm_bindgen(js_name = applyProcessorToColumn)]
    pub fn apply_processor_to_column(
        &self,
        data: &str,
        column_name: &str,
        processor_type: &str,
        params_json: JsValue,
    ) -> Result<JsValue, JsValue> {
        use crate::processing::ProcessorFactory;

        let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() < 2 {
            return Err(JsValue::from_str("Nedostatok dát"));
        }

        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let col_idx = headers.iter().position(|h| h == column_name)
            .ok_or_else(|| JsValue::from_str(&format!("Stĺpec '{}' nebol nájdený", column_name)))?;

        // Parse rows
        let mut all_values: Vec<Vec<String>> = Vec::new();
        for line in &lines[1..] {
            let mut vals: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            while vals.len() < headers.len() {
                vals.push(String::new());
            }
            all_values.push(vals);
        }

        let n = all_values.len();
        if n == 0 {
            return Err(JsValue::from_str("Žiadne hodnoty v stĺpci"));
        }

        // === Textové procesory (pracujú priamo na reťazcoch) ===
        if processor_type == "comma_to_dot" {
            for row in all_values.iter_mut() {
                if col_idx < row.len() {
                    row[col_idx] = row[col_idx].replace(',', ".");
                }
            }
            return self.rebuild_csv_result(&headers, &all_values, column_name, processor_type, n);
        }

        if processor_type == "thousands_separator_remover" {
            for row in all_values.iter_mut() {
                if col_idx < row.len() {
                    // Odstráni čiarky, ktoré slúžia ako oddeľovače tisícov
                    // Vzor: 1,000 alebo 1,000,000
                    row[col_idx] = row[col_idx].replace(',', "");
                }
            }
            return self.rebuild_csv_result(&headers, &all_values, column_name, processor_type, n);
        }

        // === Time converter - parsuje HH:MM:SS reťazce ===
        if processor_type == "time_converter" {
            // Parsujeme parametre
            let mut input_format = "seconds".to_string();
            let mut output_unit = "seconds".to_string();
            if !params_json.is_null() && !params_json.is_undefined() {
                let params: Vec<(String, String)> = serde_wasm_bindgen::from_value(params_json)
                    .unwrap_or_default();
                for (key, value) in &params {
                    match key.as_str() {
                        "input_format" => input_format = value.clone(),
                        "output_unit" => output_unit = value.clone(),
                        _ => {}
                    }
                }
            }

            for row in all_values.iter_mut() {
                if col_idx >= row.len() { continue; }
                let trimmed = row[col_idx].trim().to_string();
                if trimmed.is_empty() { continue; }

                // Parsuj hodnotu na sekundy podľa vstupného formátu
                let seconds = match input_format.as_str() {
                    "hh:mm:ss" | "hhmmss" => {
                        Self::parse_time_string(&trimmed)
                    },
                    "hh:mm" | "hhmm" => {
                        let parts: Vec<&str> = trimmed.split(':').collect();
                        if parts.len() == 2 {
                            let h: f64 = parts[0].parse().unwrap_or(0.0);
                            let m: f64 = parts[1].parse().unwrap_or(0.0);
                            Some(h * 3600.0 + m * 60.0)
                        } else {
                            trimmed.parse::<f64>().ok().map(|v| v * 3600.0 / 100.0)
                        }
                    },
                    "mm:ss" | "mmss" => {
                        let parts: Vec<&str> = trimmed.split(':').collect();
                        if parts.len() == 2 {
                            let m: f64 = parts[0].parse().unwrap_or(0.0);
                            let s: f64 = parts[1].parse().unwrap_or(0.0);
                            Some(m * 60.0 + s)
                        } else {
                            trimmed.parse::<f64>().ok().map(|v| v * 60.0 / 100.0)
                        }
                    },
                    "minutes" | "m" => trimmed.parse::<f64>().ok().map(|v| v * 60.0),
                    "hours" | "h" => trimmed.parse::<f64>().ok().map(|v| v * 3600.0),
                    _ => trimmed.parse::<f64>().ok(), // seconds
                };

                if let Some(sec) = seconds {
                    let result = match output_unit.as_str() {
                        "minutes" | "m" => sec / 60.0,
                        "hours" | "h" => sec / 3600.0,
                        _ => sec, // seconds
                    };
                    row[col_idx] = Self::format_number(result);
                }
            }
            return self.rebuild_csv_result(&headers, &all_values, column_name, processor_type, n);
        }

        // === Enkódovacie procesory (string-level) ===
        let encoding_processors = ["onehot_encoder", "label_encoder", "ordinal_encoder", "frequency_encoder", "target_encoder"];
        if encoding_processors.contains(&processor_type) {
            // Zozbieraj textové hodnoty stĺpca
            let col_strings: Vec<String> = all_values.iter()
                .map(|row| row.get(col_idx).cloned().unwrap_or_default().trim().to_string())
                .collect();

            match processor_type {
                "onehot_encoder" => {
                    // Zisti unikátne hodnoty (v poradí výskytu)
                    let mut unique_vals: Vec<String> = Vec::new();
                    for v in &col_strings {
                        if !unique_vals.contains(v) {
                            unique_vals.push(v.clone());
                        }
                    }

                    // Vytvor nové hlavičky: nahraď pôvodný stĺpec novými
                    let mut new_headers: Vec<String> = Vec::new();
                    let mut ohe_start = 0;
                    for (i, h) in headers.iter().enumerate() {
                        if i == col_idx {
                            ohe_start = new_headers.len();
                            for uv in &unique_vals {
                                let col_name = if uv.is_empty() {
                                    format!("{}_empty", column_name)
                                } else {
                                    format!("{}_{}", column_name, uv)
                                };
                                new_headers.push(col_name);
                            }
                        } else {
                            new_headers.push(h.clone());
                        }
                    }

                    // Prepíš riadky
                    let mut new_all_values: Vec<Vec<String>> = Vec::new();
                    for (i, row) in all_values.iter().enumerate() {
                        let mut new_row: Vec<String> = Vec::new();
                        for (j, val) in row.iter().enumerate() {
                            if j == col_idx {
                                for uv in &unique_vals {
                                    if col_strings[i] == *uv {
                                        new_row.push("1".to_string());
                                    } else {
                                        new_row.push("0".to_string());
                                    }
                                }
                            } else {
                                new_row.push(val.clone());
                            }
                        }
                        new_all_values.push(new_row);
                    }

                    return self.rebuild_csv_result(&new_headers, &new_all_values, column_name, processor_type, n);
                },
                "label_encoder" => {
                    // Priraď každej unikátnej hodnote číslo 0, 1, 2...
                    let mut label_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                    let mut counter = 0usize;
                    for v in &col_strings {
                        if !label_map.contains_key(v) {
                            label_map.insert(v.clone(), counter);
                            counter += 1;
                        }
                    }
                    for (i, row) in all_values.iter_mut().enumerate() {
                        if col_idx < row.len() {
                            let label = label_map.get(&col_strings[i]).unwrap();
                            row[col_idx] = label.to_string();
                        }
                    }
                },
                "ordinal_encoder" => {
                    // Získaj sort_mode z parametrov
                    let mut sort_mode = "alphabetical".to_string();
                    if !params_json.is_null() && !params_json.is_undefined() {
                        let params: Vec<(String, String)> = serde_wasm_bindgen::from_value(params_json)
                            .unwrap_or_default();
                        for (key, value) in &params {
                            if key == "sort_mode" {
                                sort_mode = value.clone();
                            }
                        }
                    }

                    let mut unique_vals: Vec<String> = Vec::new();
                    for v in &col_strings {
                        if !unique_vals.contains(v) {
                            unique_vals.push(v.clone());
                        }
                    }
                    if sort_mode == "alphabetical" {
                        unique_vals.sort();
                    }
                    // Priraď poradie
                    let ordinal_map: std::collections::HashMap<String, usize> = unique_vals.iter()
                        .enumerate().map(|(i, v)| (v.clone(), i)).collect();
                    for (i, row) in all_values.iter_mut().enumerate() {
                        if col_idx < row.len() {
                            let ord = ordinal_map.get(&col_strings[i]).unwrap();
                            row[col_idx] = ord.to_string();
                        }
                    }
                },
                "frequency_encoder" => {
                    let total = col_strings.len() as f64;
                    let mut freq_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                    for v in &col_strings {
                        *freq_map.entry(v.clone()).or_insert(0) += 1;
                    }
                    for (i, row) in all_values.iter_mut().enumerate() {
                        if col_idx < row.len() {
                            let count = *freq_map.get(&col_strings[i]).unwrap() as f64;
                            row[col_idx] = Self::format_number(count / total);
                        }
                    }
                },
                "target_encoder" => {
                    // Target encoder bez target stĺpca → fallback na frequency encoder
                    let total = col_strings.len() as f64;
                    let mut freq_map: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
                    for v in &col_strings {
                        *freq_map.entry(v.clone()).or_insert(0) += 1;
                    }
                    for (i, row) in all_values.iter_mut().enumerate() {
                        if col_idx < row.len() {
                            let count = *freq_map.get(&col_strings[i]).unwrap() as f64;
                            row[col_idx] = Self::format_number(count / total);
                        }
                    }
                },
                _ => {}
            }

            return self.rebuild_csv_result(&headers, &all_values, column_name, processor_type, n);
        }

        // === Numerické procesory (DenseMatrix) ===
        // Extrahuj stĺpec ako f64
        let mut col_values: Vec<f64> = Vec::with_capacity(n);
        let mut nan_indices: Vec<usize> = Vec::new();

        for (i, row) in all_values.iter().enumerate() {
            let trimmed = row.get(col_idx).map(|v| v.trim()).unwrap_or("");
            if trimmed.is_empty() {
                col_values.push(0.0); // placeholder
                nan_indices.push(i);
            } else {
                match trimmed.parse::<f64>() {
                    Ok(v) => col_values.push(v),
                    Err(_) => {
                        col_values.push(0.0);
                        nan_indices.push(i);
                    }
                }
            }
        }

        // Pre fit/transform vylúčime NaN riadky (ponecháme len platné)
        let valid_values: Vec<f64> = col_values.iter().enumerate()
            .filter(|(i, _)| !nan_indices.contains(i))
            .map(|(_, v)| *v)
            .collect();

        if valid_values.is_empty() {
            return Err(JsValue::from_str("Žiadne platné numerické hodnoty v stĺpci"));
        }

        // Vytvor maticu len z platných hodnôt pre fit
        let valid_matrix_data: Vec<Vec<f64>> = valid_values.iter().map(|v| vec![*v]).collect();
        let valid_matrix = DenseMatrix::from_2d_vec(&valid_matrix_data)
            .map_err(|e| JsValue::from_str(&format!("Chyba pri vytváraní matice: {:?}", e)))?;

        // Vytvor procesor BEZ SelectiveProcessor wrappera (raw)!
        let mut processor = ProcessorFactory::create_raw(processor_type)
            .map_err(|e| JsValue::from_str(&e))?;

        // Aplikuj parametre
        if !params_json.is_null() && !params_json.is_undefined() {
            let params: Vec<(String, String)> = serde_wasm_bindgen::from_value(params_json)
                .unwrap_or_default();
            for (key, value) in params {
                let _ = processor.set_param(&key, &value);
            }
        }

        // Fit na platné hodnoty
        processor.fit(&valid_matrix);

        // Transform celý stĺpec (vrátane pôvodne NaN)
        let full_matrix_data: Vec<Vec<f64>> = col_values.iter().map(|v| vec![*v]).collect();
        let full_matrix = DenseMatrix::from_2d_vec(&full_matrix_data)
            .map_err(|e| JsValue::from_str(&format!("Chyba: {:?}", e)))?;
        let result_matrix = processor.transform(&full_matrix);

        // Aktualizuj hodnoty v stĺpci
        for (i, row) in all_values.iter_mut().enumerate() {
            if col_idx < row.len() {
                if nan_indices.contains(&i) {
                    // Ponechaj prázdne bunky prázdne
                    row[col_idx] = String::new();
                } else {
                    let new_val = *result_matrix.get((i, 0));
                    row[col_idx] = Self::format_number(new_val);
                }
            }
        }

        self.rebuild_csv_result(&headers, &all_values, column_name, processor_type, n)
    }

    /// Vymaže stĺpec z CSV dát
    #[wasm_bindgen(js_name = deleteColumn)]
    pub fn delete_column(&self, data: &str, column_name: &str) -> Result<JsValue, JsValue> {
        let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return Err(JsValue::from_str("Prázdne dáta"));
        }

        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let col_idx = headers.iter().position(|h| h == column_name)
            .ok_or_else(|| JsValue::from_str(&format!("Stĺpec '{}' nebol nájdený", column_name)))?;

        if headers.len() <= 1 {
            return Err(JsValue::from_str("Nemôžete vymazať posledný stĺpec"));
        }

        // Rebuild CSV without the column
        let new_headers: Vec<&str> = headers.iter()
            .enumerate()
            .filter(|(i, _)| *i != col_idx)
            .map(|(_, h)| h.as_str())
            .collect();

        let mut csv = new_headers.join(",");
        csv.push('\n');

        for line in &lines[1..] {
            let mut vals: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            // Pad to match header count
            while vals.len() < headers.len() {
                vals.push(String::new());
            }
            let new_vals: Vec<&str> = vals.iter()
                .enumerate()
                .filter(|(i, _)| *i != col_idx)
                .map(|(_, v)| v.as_str())
                .collect();
            csv.push_str(&new_vals.join(","));
            csv.push('\n');
        }

        let result = serde_json::json!({
            "csv": csv,
            "deleted_column": column_name,
            "remaining_columns": new_headers.len()
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Zmení hodnotu konkrétnej bunky v CSV
    #[wasm_bindgen(js_name = setCellValue)]
    pub fn set_cell_value(
        &self,
        data: &str,
        row_idx: usize,
        column_name: &str,
        new_value: &str,
    ) -> Result<JsValue, JsValue> {
        let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return Err(JsValue::from_str("Prázdne dáta"));
        }

        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let col_idx = headers.iter().position(|h| h == column_name)
            .ok_or_else(|| JsValue::from_str(&format!("Stĺpec '{}' nebol nájdený", column_name)))?;

        if row_idx + 1 >= lines.len() {
            return Err(JsValue::from_str("Riadok mimo rozsah"));
        }

        // Rebuild CSV with modified cell
        let mut csv = headers.join(",");
        csv.push('\n');

        for (i, line) in lines[1..].iter().enumerate() {
            let mut vals: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            // Pad to match header count
            while vals.len() < headers.len() {
                vals.push(String::new());
            }
            if i == row_idx && col_idx < vals.len() {
                vals[col_idx] = new_value.to_string();
            }
            // Only keep as many values as there are headers
            vals.truncate(headers.len());
            csv.push_str(&vals.join(","));
            csv.push('\n');
        }

        let result = serde_json::json!({
            "csv": csv,
            "row": row_idx,
            "column": column_name,
            "new_value": new_value
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

    /// Nahradí všetky výskyty hodnoty v stĺpci (Replace All)
    #[wasm_bindgen(js_name = replaceAllInColumn)]
    pub fn replace_all_in_column(
        &self,
        data: &str,
        column_name: &str,
        search_value: &str,
        replace_value: &str,
    ) -> Result<JsValue, JsValue> {
        let lines: Vec<&str> = data.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.is_empty() {
            return Err(JsValue::from_str("Prázdne dáta"));
        }

        let headers: Vec<String> = lines[0].split(',').map(|s| s.trim().to_string()).collect();
        let col_idx = headers.iter().position(|h| h == column_name)
            .ok_or_else(|| JsValue::from_str(&format!("Stĺpec '{}' nebol nájdený", column_name)))?;

        let mut replaced_count = 0;
        let mut csv = headers.join(",");
        csv.push('\n');

        for line in &lines[1..] {
            let mut vals: Vec<String> = line.split(',').map(|s| s.trim().to_string()).collect();
            // Pad to match header count
            while vals.len() < headers.len() {
                vals.push(String::new());
            }
            if col_idx < vals.len() && vals[col_idx] == search_value {
                vals[col_idx] = replace_value.to_string();
                replaced_count += 1;
            }
            // Only keep as many values as there are headers
            vals.truncate(headers.len());
            csv.push_str(&vals.join(","));
            csv.push('\n');
        }

        let result = serde_json::json!({
            "csv": csv,
            "column": column_name,
            "search_value": search_value,
            "replace_value": replace_value,
            "replaced_count": replaced_count
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }
}

// Helper methods (not exported to WASM)
impl WasmMLPipeline {
    /// Parsuje HH:MM:SS alebo HH:MM formát na sekundy
    fn parse_time_string(time_str: &str) -> Option<f64> {
        let parts: Vec<&str> = time_str.split(':').collect();
        match parts.len() {
            2 => {
                // HH:MM or MM:SS
                let a: f64 = parts[0].parse().ok()?;
                let b: f64 = parts[1].parse().ok()?;
                Some(a * 3600.0 + b * 60.0)
            },
            3 => {
                // HH:MM:SS or HH:MM:SS.s
                let hours: f64 = parts[0].parse().ok()?;
                let minutes: f64 = parts[1].parse().ok()?;
                let seconds: f64 = parts[2].parse().ok()?;
                Some(hours * 3600.0 + minutes * 60.0 + seconds)
            },
            _ => None,
        }
    }

    /// Formátuje číslo pre CSV výstup
    fn format_number(val: f64) -> String {
        if val.is_nan() || val.is_infinite() {
            return String::new();
        }
        // Ak je celé číslo, formátuj bez zbytočných desatinných miest
        if val.fract() == 0.0 && val.abs() < 1e15 {
            format!("{:.0}", val)
        } else {
            // Zaokrúhli na 6 desatinných miest, odstráň koncové nuly
            let formatted = format!("{:.6}", val);
            let trimmed = formatted.trim_end_matches('0').trim_end_matches('.');
            trimmed.to_string()
        }
    }

    /// Postaví CSV výsledok z headers a values
    fn rebuild_csv_result(
        &self,
        headers: &[String],
        all_values: &[Vec<String>],
        column_name: &str,
        processor_type: &str,
        rows_affected: usize,
    ) -> Result<JsValue, JsValue> {
        let mut csv = headers.join(",");
        csv.push('\n');
        for row in all_values {
            let truncated: Vec<&str> = row.iter().take(headers.len()).map(|s| s.as_str()).collect();
            csv.push_str(&truncated.join(","));
            csv.push('\n');
        }

        let result = serde_json::json!({
            "csv": csv,
            "column": column_name,
            "processor": processor_type,
            "rows_affected": rows_affected
        });

        Ok(serde_wasm_bindgen::to_value(&result).unwrap())
    }

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
        target_index: usize,
    ) -> String {
        let mut html = String::new();
        html.push_str("<div style='font-family:Segoe UI,sans-serif;'>");

        // Summary cards
        html.push_str("<div style='display:flex;gap:15px;flex-wrap:wrap;margin-bottom:20px;'>");
        for r in selector_results {
            let name = r["selector_name"].as_str().unwrap_or("");
            let err = r["error"].as_str();
            let count = r["selected_count"].as_u64().unwrap_or(0);
            let border_color = if err.is_some() { "#e74c3c" } else { "#cc0000" };
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
                    "<div style='font-size:1.5em;font-weight:bold;color:#cc0000;'>{}/{}</div>\
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
            // Interactive feature map with checkboxes and selector vote counts
            let total_valid = valid_selectors.len();
            let mut feature_votes: Vec<usize> = vec![0; total_features];
            for r in &valid_selectors {
                if let Some(arr) = r["selected_indices"].as_array() {
                    for v in arr {
                        if let Some(fi) = v.as_u64() {
                            if (fi as usize) < total_features {
                                feature_votes[fi as usize] += 1;
                            }
                        }
                    }
                }
            }

            html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Mapa features \u{2013} vyberte príznaky pre vlastný tréning</h4>");
            html.push_str("<p style='font-size:12px;color:#6c757d;margin-bottom:10px;'>Zaškrtnite features, s ktorými chcete natrénovať model. Farba pozadia a indikátor bodiek ukazujú, koľko selektorov danú feature vybralo.</p>");
            html.push_str("<div id='userFeatureMap' style='display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:6px;'>");

            for (fidx, fname) in feature_names.iter().enumerate() {
                // Skip target column
                if fidx == target_index {
                    continue;
                }
                
                // Safety check: ensure fidx is within bounds
                if fidx >= feature_votes.len() {
                    continue;
                }
                
                let votes = feature_votes[fidx];
                let bg = if total_valid > 0 && votes == total_valid { "#c8e6c9" }
                    else if total_valid > 0 && votes * 2 >= total_valid { "#fff9c4" }
                    else if votes > 0 { "#ffe0b2" }
                    else { "#ffcdd2" };
                let checked = if total_valid > 0 && votes * 2 >= total_valid { " checked" } else { "" };
                let dots: String = (0..total_valid).map(|i| if i < votes { "\u{25cf}" } else { "\u{25cb}" }).collect();

                html.push_str(&format!(
                    "<label style='display:flex;align-items:center;gap:6px;padding:6px 10px;\
                    background:{};border:1px solid #dee2e6;cursor:pointer;font-size:12px;\
                    user-select:none;'>\
                    <input type='checkbox' data-feature-idx='{}'{} \
                    style='cursor:pointer;width:16px;height:16px;flex-shrink:0;'>\
                    <span style='flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;\
                    white-space:nowrap;'><strong style='color:#cc0000;'>F{}</strong>: {}</span>\
                    <span style='flex-shrink:0;font-size:11px;color:#495057;\
                    white-space:nowrap;font-weight:bold;'>{}/{} {}</span>\
                    </label>",
                    bg, fidx, checked, fidx, fname, votes, total_valid, dots
                ));
            }
            html.push_str("</div>");

            // Select all / deselect all
            html.push_str("<div style='margin-top:8px;display:flex;gap:8px;font-size:12px;'>");
            html.push_str("<button id='featureMapSelectAll' style='padding:4px 12px;cursor:pointer;font-size:12px;background:#f0f0f0;border:1px solid #ccc;'>Vybrať všetky</button>");
            html.push_str("<button id='featureMapDeselectAll' style='padding:4px 12px;cursor:pointer;font-size:12px;background:#f0f0f0;border:1px solid #ccc;'>Odznačiť všetky</button>");
            html.push_str("</div>");

            // Train button + status
            html.push_str("<div style='margin-top:15px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;'>");
            html.push_str("<button id='trainFromComparisonBtn' style='background:#27ae60;color:white;border:none;padding:10px 20px;cursor:pointer;font-size:14px;font-weight:bold;'>Natrénovať a porovnať varianty</button>");
            html.push_str("<span id='comparisonTrainStatus' style='font-size:0.9em;color:#6c757d;'></span>");
            html.push_str("</div>");

            // Training results placeholder
            html.push_str("<div id='comparisonTrainingResults'></div>");

            html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Porovnávacia matica</h4>");
            html.push_str("<div style='overflow-x:auto;'>");
            html.push_str("<table style='width:100%;border-collapse:collapse;font-size:13px;'>");

            // Header
            html.push_str("<thead><tr>");
            html.push_str("<th style='padding:10px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:left;min-width:120px;position:sticky;left:0;z-index:1;'>Príznak</th>");
            for r in &valid_selectors {
                let name = r["selector_name"].as_str().unwrap_or("");
                let count = r["selected_count"].as_u64().unwrap_or(0);
                html.push_str(&format!(
                    "<th style='padding:10px;border:1px solid #dee2e6;background:#cc0000;color:white;text-align:center;min-width:100px;'>{}<br><small>({}/{})</small></th>",
                    name, count, total_features
                ));
            }
            html.push_str("<th style='padding:10px;border:1px solid #dee2e6;background:#8b0000;color:white;text-align:center;'>Zhoda</th>");
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

                    let (bg, icon) = if is_selected {
                        ("rgba(52,152,219,0.08)", "<span style='color:#28a745;font-weight:bold;'>✓</span>")
                    } else {
                        ("rgba(189,195,199,0.08)", "<span style='color:#6c757d;'>✗</span>")
                    };
                    cells.push_str(&format!(
                        "<td style='padding:8px;border:1px solid #dee2e6;text-align:center;background:{};'>{}</td>",
                        bg, icon
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
            html.push_str("<span style='background:rgba(52,152,219,0.08);padding:3px 10px;border:1px solid #dee2e6;'><span style='color:#28a745;font-weight:bold;'>✓</span> Vybraný</span>");
            html.push_str("<span style='background:rgba(189,195,199,0.08);padding:3px 10px;border:1px solid #dee2e6;'><span style='color:#6c757d;'>✗</span> Nevybraný</span>");
            html.push_str("<span style='background:#c8e6c9;padding:3px 10px;'>100% zhoda</span>");
            html.push_str("<span style='background:#fff9c4;padding:3px 10px;'>&ge;50% zhoda</span>");
            html.push_str("<span style='background:#ffcdd2;padding:3px 10px;'>0% zhoda</span>");
            html.push_str("<div style='color:#6c757d;padding:3px 10px;'><strong>Čísla v hranatých zátvorkách [0], [1], [2]...</strong> označujú poradie features v datasete</div>");
            html.push_str("</div>");

            // Per-selector details in collapsible sections
            html.push_str("<h4 style='color:#495057;margin:25px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Detaily jednotlivých selektorov</h4>");
            
            for r in &valid_selectors {
                let name = r["selector_name"].as_str().unwrap_or("");
                let details = r["details_html"].as_str().unwrap_or("");
                if !details.is_empty() {
                    html.push_str(&format!(
                        "<details style='margin-bottom:10px;border:1px solid #dee2e6;'>\
                        <summary style='padding:12px;background:#f8f9fa;cursor:pointer;font-weight:600;color:#cc0000;'>{} - Detailná analýza</summary>\
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
