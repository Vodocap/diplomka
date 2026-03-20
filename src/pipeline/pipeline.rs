use crate::models::IModel;
use crate::processing::DataProcessor;
use crate::feature_selection_strategies::FeatureSelector;
use crate::evaluation::{ModelEvaluator, EvaluationReport};
use super::builder::MLPipelineBuilder;
use super::pipeline_info::PipelineInfo;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

/// Facade trieda pre cely ML pipeline.
/// Zapuzdruje processing, feature selection, training a evaluation do jedneho rozhrania.
/// Vytvara sa cez MLPipelineBuilder. Klienti volaju train() a predict() bez starostlivosti
/// o vnutorne kroky (preprocessing, selekcia, cachovanie vybranych indexov).
pub struct MLPipeline
{
    pub(crate) model: Box<dyn IModel>,
    pub(crate) processors: Vec<Box<dyn DataProcessor>>,
    pub(crate) selector: Option<Box<dyn FeatureSelector>>,
    pub(crate) model_name: String,
    pub(crate) evaluation_mode: String,
    pub(crate) selected_indices: Option<Vec<usize>>,
    pub(crate) expected_features: Option<usize>, // Number of features model expects after selection
}

impl MLPipeline
{
    /// Vytvorí builder pre konfiguráciu pipeline
    pub fn builder() -> MLPipelineBuilder
    {
        MLPipelineBuilder::new()
    }

    /// Nastaví cached feature indices (pre externú kontrolu)
    pub fn set_selected_indices(&mut self, indices: Vec<usize>)
    {
        self.selected_indices = Some(indices);
    }

    /// Resetuje tréningový stav (pre opakované trénovanie s rôznymi features)
    pub fn reset_training_state(&mut self)
    {
        self.selected_indices = None;
        self.expected_features = None;
    }

    /// Spracuje dáta cez vsetky procesory v poradi
    pub fn preprocess(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        let mut result = data.clone();
        for processor in &self.processors
        {
            result = processor.process(&result);
        }
        result
    }

    /// Vykoná feature selection (ak existuje)
    pub fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64>
    {
        if let Some(ref selector) = self.selector
        {
            selector.select_features(x, y)
        }
        else
        {
            x.clone()
        }
    }

    /// Vykoná feature selection pomocou cached indices (pre predikciu)
    fn select_features_cached(&self, x: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        if let Some(ref indices) = self.selected_indices
        {
            if indices.is_empty()
            {
                web_sys::console::error_1(&"select_features_cached: cached indices is empty, returning original data".into());
                return x.clone();
            }

            let shape = x.shape();
            let rows = shape.0;
            let cols = indices.len();
            let mut data = vec![vec![0.0; cols]; rows];

            for (new_col, &old_col) in indices.iter().enumerate()
            {
                for row in 0..rows
                {
                    data[row][new_col] = *x.get((row, old_col));
                }
            }

            DenseMatrix::from_2d_vec(&data).unwrap_or_else(|e|
            {
                web_sys::console::error_1(&format!(
                    "select_features_cached: failed to create matrix: {:?}, returning original data", e
                ).into());
                x.clone()
            })
        }
        else
        {
            x.clone()
        }
    }

    /// Vykoná preprocessing + feature selection
    pub fn prepare_data(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64>
    {
        let processed = self.preprocess(x);
        self.select_features(&processed, y)
    }

    /// Trénuje model
    pub fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>) -> Result<(), String>
    {
        web_sys::console::log_1(&format!("🚀 train() START: {}x{}, selector: {}",
            x.shape().0, x.shape().1, self.selector.is_some()).into());

        // First apply feature selection (if exists) to get the reduced feature set
        let selected_x = if let Some(ref selector) = self.selector
        {
            if self.selected_indices.is_none()
            {
                let indices = selector.get_selected_indices(&x, &y);

                // CRITICAL: Validate indices
                if indices.is_empty()
                {
                    web_sys::console::error_1(&"🔴 CRITICAL: Selector returned EMPTY indices!".into());
                    return Err(format!(
                        "Feature selector '{}' returned no features! Input was {}x{}, selector may be too aggressive.",
                        selector.get_name(), x.shape().0, x.shape().1
                    ));
                }

                self.selected_indices = Some(indices);
            }
            self.select_features_cached(&x)
        }
        else if self.selected_indices.is_some()
        {
            // External indices set (e.g. from comparison training) - use them without a selector
            self.select_features_cached(&x)
        }
        else
        {
            x.clone()
        };

        // Fit kazdy procesor na transformovanom vystupe predchadzajuceho
        {
            let mut current = selected_x.clone();
            for processor in self.processors.iter_mut()
            {
                processor.fit(&current);
                current = processor.transform(&current);
            }
        }
        let processed_x = self.preprocess(&selected_x);

        // Store expected feature count for validation during predict
        self.expected_features = Some(processed_x.shape().1);

        self.model.train(processed_x, y);
        web_sys::console::log_1(&format!("✅ train() completed: model={}, features={}",
            self.model_name, self.expected_features.unwrap_or(0)).into());
        Ok(())
    }

    /// Predikcia s preprocessing a feature selection
    pub fn predict(&self, input: Vec<f64>) -> Result<Vec<f64>, String>
    {
        // Konvertujeme Vec<f64> na DenseMatrix (1 riadok)
        let input_matrix = DenseMatrix::from_2d_vec(&vec![input.clone()])
            .map_err(|e| format!("Chyba pri konverzii input vektora: {:?}", e))?;

        // Apply feature selection FIRST (using cached indices)
        let selected = self.select_features_cached(&input_matrix);

        // Then apply preprocessing to the SELECTED features
        let prepared = self.preprocess(&selected);

        // Validation: check feature count matches what model expects
        if let Some(expected_count) = self.expected_features
        {
            let actual_count = prepared.shape().1;
            if actual_count != expected_count
            {
                return Err(format!(
                    "CRITICAL MISMATCH: Model trained on {} features but received {}. Selector: {}, Indices: {:?}",
                    expected_count, actual_count, self.selector.is_some(), self.selected_indices
                ));
            }
        }

        // Extrahujeme prvy riadok ako Vec<f64>
        let prepared_vec: Vec<f64> = (0..prepared.shape().1)
            .map(|col_idx| *prepared.get((0, col_idx)))
            .collect();

        Ok(self.model.predict(&prepared_vec))
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
        for row_idx in 0..shape.0
        {
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
    pub fn info(&self) -> PipelineInfo
    {
        PipelineInfo {
            model_name: self.model.get_name().to_string(),
            model_type: self.model_name.clone(),
            processors: self.processors.iter().map(|p| p.get_name().to_string()).collect(),
            selector: self.selector.as_ref().map(|s| s.get_name().to_string()),
            evaluation_mode: self.evaluation_mode.clone(),
        }
    }
}
