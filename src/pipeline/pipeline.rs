use crate::models::IModel;
use crate::processing::DataProcessor;
use crate::feature_selection_strategies::FeatureSelector;
use crate::evaluation::{ModelEvaluator, EvaluationReport};
use super::builder::MLPipelineBuilder;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

/// Facade trieda pre celý ML Pipeline
/// Zapuzdruje loading, processing, feature selection, training a evaluation
pub struct MLPipeline {
    pub(crate) model: Box<dyn IModel>,
    pub(crate) processor: Option<Box<dyn DataProcessor>>,
    pub(crate) selector: Option<Box<dyn FeatureSelector>>,
    pub(crate) model_name: String,
    pub(crate) evaluation_mode: String,
    pub(crate) selected_indices: Option<Vec<usize>>,
    pub(crate) expected_features: Option<usize>, // Number of features model expects after selection
}

impl MLPipeline {
    /// Vytvorí builder pre konfiguráciu pipeline
    pub fn builder() -> MLPipelineBuilder {
        MLPipelineBuilder::new()
    }
    
    /// Nastaví cached feature indices (pre externú kontrolu)
    pub fn set_selected_indices(&mut self, indices: Vec<usize>) {
        self.selected_indices = Some(indices);
    }

    /// Resetuje tréningový stav (pre opakované trénovanie s rôznymi features)
    pub fn reset_training_state(&mut self) {
        self.selected_indices = None;
        self.expected_features = None;
    }

    /// Spracuje dáta cez processor (ak existuje)
    pub fn preprocess(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        if let Some(ref processor) = self.processor {
            processor.process(data)
        } else {
            data.clone()
        }
    }

    /// Vykoná feature selection (ak existuje)
    pub fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> {
        if let Some(ref selector) = self.selector {
            selector.select_features(x, y)
        } else {
            x.clone()
        }
    }
    
    /// Vykoná feature selection pomocou cached indices (pre predikciu)
    fn select_features_cached(&self, x: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        if let Some(ref indices) = self.selected_indices {
            web_sys::console::log_1(&format!("✅ select_features_cached: input {}x{}, cached_indices: {:?}", 
                x.shape().0, x.shape().1, indices).into());
            
            // CRITICAL: Check for empty indices
            if indices.is_empty() {
                web_sys::console::error_1(&"🔴 CRITICAL BUG: cached indices is empty! This will cause 0-feature matrix!".into());
                panic!("Feature selector returned empty indices - cannot proceed with 0 features!");
            }
            
            // Extrahujeme stĺpce podľa cached indices
            let shape = x.shape();
            let rows = shape.0;
            let cols = indices.len();
            let mut data = vec![vec![0.0; cols]; rows];
            
            for (new_col, &old_col) in indices.iter().enumerate() {
                for row in 0..rows {
                    data[row][new_col] = *x.get((row, old_col));
                }
            }
            
            let result = DenseMatrix::from_2d_vec(&data).unwrap_or_else(|e| {
                panic!("CRITICAL BUG: Failed to create matrix from selected features! Error: {:?}, data shape: {}x{}, indices: {:?}", 
                    e, rows, cols, indices);
            });
            web_sys::console::log_1(&format!("✅ select_features_cached: output {}x{}", 
                result.shape().0, result.shape().1).into());
            result
        } else {
            web_sys::console::log_1(&format!("⚠️ select_features_cached: NO SELECTION - indices: {}", 
                self.selected_indices.is_some()).into());
            x.clone()
        }
    }

    /// Vykoná preprocessing + feature selection
    pub fn prepare_data(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> {
        let processed = self.preprocess(x);
        self.select_features(&processed, y)
    }

    /// Trénuje model
    pub fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>) -> Result<(), String> {
        web_sys::console::log_1(&format!("🚀 train() START: input {}x{}, selector: {}, cached_indices: {:?}", 
            x.shape().0, x.shape().1, self.selector.is_some(), self.selected_indices).into());
        
        // First apply feature selection (if exists) to get the reduced feature set
        let selected_x = if let Some(ref selector) = self.selector {
            if self.selected_indices.is_none() {
                web_sys::console::log_1(&"📌 Getting selected indices from selector...".into());
                let indices = selector.get_selected_indices(&x, &y);
                web_sys::console::log_1(&format!("📌 Got indices: {:?}", indices).into());
                
                // CRITICAL: Validate indices
                if indices.is_empty() {
                    web_sys::console::error_1(&"🔴 CRITICAL: Selector returned EMPTY indices!".into());
                    return Err(format!(
                        "Feature selector '{}' returned no features! Input was {}x{}, selector may be too aggressive.",
                        selector.get_name(), x.shape().0, x.shape().1
                    ));
                }
                
                self.selected_indices = Some(indices);
            }
            let selected = self.select_features_cached(&x);
            web_sys::console::log_1(&format!("🎯 after feature selection: {}x{}", selected.shape().0, selected.shape().1).into());
            selected
        } else if self.selected_indices.is_some() {
            // External indices set (e.g. from comparison training) - use them without a selector
            web_sys::console::log_1(&"📌 Using externally set feature indices (no selector)".into());
            let selected = self.select_features_cached(&x);
            web_sys::console::log_1(&format!("🎯 after external feature selection: {}x{}", selected.shape().0, selected.shape().1).into());
            selected
        } else {
            web_sys::console::log_1(&"⚠️ NO selector and NO indices - using x as-is".into());
            x.clone()
        };
        
        // Now fit processor on the SELECTED features
        if let Some(ref mut processor) = self.processor {
            web_sys::console::log_1(&format!("📐 Fitting processor '{}' on SELECTED {}x{} data", 
                processor.get_name(), selected_x.shape().0, selected_x.shape().1).into());
            processor.fit(&selected_x);
            web_sys::console::log_1(&"✅ Processor fitted on selected features".into());
        }
        
        // Apply preprocessing to the selected features
        web_sys::console::log_1(&"🔧 Applying preprocessing to selected features...".into());
        let processed_x = self.preprocess(&selected_x);
        web_sys::console::log_1(&format!("🔧 after preprocess: {}x{}", processed_x.shape().0, processed_x.shape().1).into());
        
        // Store expected feature count for validation during predict
        self.expected_features = Some(processed_x.shape().1);
        web_sys::console::log_1(&format!("📊 prepared_x for training: {}x{}, expected_features: {:?}", 
            processed_x.shape().0, processed_x.shape().1, self.expected_features).into());
        
        // Tréning
        web_sys::console::log_1(&format!("🎓 Calling model.train() with data: {}x{}, y len: {}", 
            processed_x.shape().0, processed_x.shape().1, y.len()).into());
        self.model.train(processed_x, y);
        web_sys::console::log_1(&"✅ model.train() completed".into());
        Ok(())
    }

    /// Predikcia s preprocessing a feature selection
    pub fn predict(&self, input: Vec<f64>) -> Result<Vec<f64>, String> {
        web_sys::console::log_1(&format!("� predict() START: input len={}, selector={}, cached_indices={:?}, expected={:?}", 
            input.len(), self.selector.is_some(), self.selected_indices, self.expected_features).into());
        
        // Konvertujeme Vec<f64> na DenseMatrix (1 riadok)
        let input_matrix = DenseMatrix::from_2d_vec(&vec![input.clone()])
            .map_err(|e| format!("Chyba pri konverzii input vektora: {:?}", e))?;
        
        web_sys::console::log_1(&format!("🔮 input_matrix: {}x{}", input_matrix.shape().0, input_matrix.shape().1).into());
        
        // Apply feature selection FIRST (using cached indices)
        let selected = self.select_features_cached(&input_matrix);
        web_sys::console::log_1(&format!("🎯 after feature selection: {}x{}", selected.shape().0, selected.shape().1).into());
        
        // Then apply preprocessing to the SELECTED features
        let prepared = self.preprocess(&selected);
        web_sys::console::log_1(&format!("🔧 after preprocessing: {}x{}", prepared.shape().0, prepared.shape().1).into());
        
        // Validation: check feature count matches what model expects
        if let Some(expected_count) = self.expected_features {
            let actual_count = prepared.shape().1;
            if actual_count != expected_count {
                return Err(format!(
                    "CRITICAL MISMATCH: Model trained on {} features but received {}. Selector: {}, Indices: {:?}",
                    expected_count, actual_count, self.selector.is_some(), self.selected_indices
                ));
            }
        }
        web_sys::console::log_1(&"✅ Feature count validation passed".into());
        if let Some(expected) = self.expected_features {
            web_sys::console::log_1(&format!("🔵 Validating: expected {} features, got {}", expected, prepared.shape().1).into());
            if prepared.shape().1 != expected {
                let err_msg = format!(
                    "Feature count mismatch: model expects {} features, got {} after selection",
                    expected, prepared.shape().1
                );
                web_sys::console::log_1(&format!("❌ {}", err_msg).into());
                return Err(err_msg);
            }
        } else {
            web_sys::console::log_1(&"⚠️ No expected_features set - skipping validation".into());
        }
        
        // Extrahujeme prvý riadok ako Vec<f64>
        let prepared_vec: Vec<f64> = (0..prepared.shape().1)
            .map(|col_idx| *prepared.get((0, col_idx)))
            .collect();
        
        web_sys::console::log_1(&format!("🔵 prepared_vec len: {}, calling model.predict()", prepared_vec.len()).into());
        
        // Predikcia
        let result = self.model.predict(&prepared_vec);
        web_sys::console::log_1(&format!("✅ predict() succeeded, result: {:?}", result).into());
        Ok(result)
    }

    /// Vyhodnotí model na testovacích dátach
    pub fn evaluate(
        &self,
        y_true: &[f64],
        y_pred: &[f64]
    ) -> Result<EvaluationReport, String> {
        let report = ModelEvaluator::evaluate_auto(
            y_true,
            y_pred,
            &self.model_name,
            &self.evaluation_mode
        )?;

        Ok(report)
    }

    /// Kompletny workflow: train + evaluate
    pub fn train_and_evaluate(
        &mut self,
        x_train: DenseMatrix<f64>,
        y_train: Vec<f64>,
        x_test: DenseMatrix<f64>,
        y_test: Vec<f64>
    ) -> Result<EvaluationReport, String> {
        // Tréning
        self.train(x_train, y_train)?;

        // Predikcia na test set
        let mut predictions = Vec::new();
        let shape = x_test.shape();
        for row_idx in 0..shape.0 {
            let row: Vec<f64> = (0..shape.1)
                .map(|col_idx| *x_test.get((row_idx, col_idx)))
                .collect();
            
            let pred = self.predict(row)?;
            predictions.extend(pred);
        }

        // Evaluácia
        self.evaluate(&y_test, &predictions)
    }

    /// Získa informácie o pipeline
    pub fn info(&self) -> PipelineInfo {
        PipelineInfo {
            model_name: self.model.get_name().to_string(),
            model_type: self.model_name.clone(),
            processor: self.processor.as_ref().map(|p| p.get_name().to_string()),
            selector: self.selector.as_ref().map(|s| s.get_name().to_string()),
            evaluation_mode: self.evaluation_mode.clone(),
        }
    }
}

/// Informácie o nakonfigurovanom pipeline
#[derive(Debug, Clone)]
pub struct PipelineInfo {
    pub model_name: String,
    pub model_type: String,
    pub processor: Option<String>,
    pub selector: Option<String>,
    pub evaluation_mode: String,
}

impl PipelineInfo {
    pub fn print(&self) {
        println!("=== ML Pipeline Info ===");
        println!("Model: {} ({})", self.model_name, self.model_type);
        println!("Processor: {}", self.processor.as_ref().unwrap_or(&"None".to_string()));
        println!("Feature Selector: {}", self.selector.as_ref().unwrap_or(&"None".to_string()));
        println!("Evaluation Mode: {}", self.evaluation_mode);
        println!("=======================");
    }
}
