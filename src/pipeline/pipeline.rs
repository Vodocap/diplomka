use crate::models::IModel;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::builder::MLPipelineBuilder;
use super::pipeline_info::PipelineInfo;

/// Kontajner pre natrénovaný ML model s eval mode a cache-ovanými feature indexmi.
/// Vytvára sa cez `MLPipeline::builder()` (Builder pattern).
/// Preprocessing a feature selection prebieha externe (data editor & compareSelectors);
/// pipeline drží len model, eval_mode a cached selected_indices pre konzistentnú predikciu.
pub struct MLPipeline
{
    pub(crate) model: Box<dyn IModel>,
    pub(crate) model_name: String,
    pub(crate) evaluation_mode: String,
    pub(crate) selected_indices: Option<Vec<usize>>,
    pub(crate) expected_features: Option<usize>,
}

impl MLPipeline
{
    /// Vytvorí builder pre konfiguráciu pipeline
    pub fn builder() -> MLPipelineBuilder
    {
        MLPipelineBuilder::new()
    }

    /// Nastaví cached feature indices (nastavuje sa externe z compareSelectors)
    pub fn set_selected_indices(&mut self, indices: Vec<usize>)
    {
        self.selected_indices = Some(indices);
    }

    /// Resetuje tréningový stav (pred opakovaným trénovaním s novými features)
    pub fn reset_training_state(&mut self)
    {
        self.selected_indices = None;
        self.expected_features = None;
    }

    /// Vyberie stĺpce podľa cached indices (pre tréning aj predikciu)
    fn apply_feature_selection(&self, x: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        let indices = match &self.selected_indices
        {
            Some(idx) if !idx.is_empty() => idx,
            _ => return x.clone(),
        };

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
                "apply_feature_selection: failed to create matrix: {:?}", e
            ).into());
            x.clone()
        })
    }

    /// Trénuje model (aplikuje feature selection podľa cached indices, ak sú nastavené)
    pub fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>) -> Result<(), String>
    {
        let selected_x = self.apply_feature_selection(&x);

        if selected_x.shape().1 == 0
        {
            return Err("Feature selection vrátila 0 stĺpcov".to_string());
        }

        self.expected_features = Some(selected_x.shape().1);
        self.model.train(selected_x, y);
        Ok(())
    }

    /// Predikcia pre jeden riadok (Vec<f64>); aplikuje feature selection pomocou cached indices
    pub fn predict(&self, input: Vec<f64>) -> Result<Vec<f64>, String>
    {
        let input_matrix = DenseMatrix::from_2d_vec(&vec![input])
            .map_err(|e| format!("Chyba pri konverzii vstupu: {:?}", e))?;

        let prepared = self.apply_feature_selection(&input_matrix);

        if let Some(expected) = self.expected_features
        {
            let actual = prepared.shape().1;
            if actual != expected
            {
                return Err(format!(
                    "Nesúlad features: model bol natrénovaný na {} features, vstup má {}",
                    expected, actual
                ));
            }
        }

        let row: Vec<f64> = (0..prepared.shape().1)
            .map(|col| *prepared.get((0, col)))
            .collect();

        Ok(self.model.predict(&row))
    }

    /// Vráti informácie o pipeline (pre logging a diagnostiku)
    pub fn info(&self) -> PipelineInfo
    {
        PipelineInfo {
            model_name: self.model.get_name().to_string(),
            model_type: self.model_name.clone(),
            evaluation_mode: self.evaluation_mode.clone(),
        }
    }
}
