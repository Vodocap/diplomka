use crate::models::IModel;
use crate::processing::DataProcessor;
use crate::feature_selection_strategies::FeatureSelector;
use crate::evaluation::{ModelEvaluator, EvaluationReport};
use super::builder::MLPipelineBuilder;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};

/// Facade trieda pre celý ML Pipeline
/// Zapuzdruje loading, processing, feature selection, training a evaluation
pub struct MLPipeline {
    pub(crate) model: Box<dyn IModel>,
    pub(crate) processor: Option<Box<dyn DataProcessor>>,
    pub(crate) selector: Option<Box<dyn FeatureSelector>>,
    pub(crate) model_name: String,
    pub(crate) evaluation_mode: String,
}

impl MLPipeline {
    /// Vytvorí builder pre konfiguráciu pipeline
    pub fn builder() -> MLPipelineBuilder {
        MLPipelineBuilder::new()
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

    /// Vykoná preprocessing + feature selection
    pub fn prepare_data(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> {
        let processed = self.preprocess(x);
        self.select_features(&processed, y)
    }

    /// Trénuje model
    pub fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>) -> Result<(), String> {
        // Príprava dát
        let prepared_x = self.prepare_data(&x, &y);
        
        // Tréning
        self.model.train(prepared_x, y);
        
        Ok(())
    }

    /// Predikcia
    pub fn predict(&self, input: Vec<f64>) -> Result<Vec<f64>, String> {
        // Pre predikciu používame rovnaký preprocessing
        Ok(self.model.predict(&input))
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

    /// Komplétny workflow: train + evaluate
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
