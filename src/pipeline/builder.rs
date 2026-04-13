use crate::models::model_factory::ModelFactory;
use crate::pipeline::MLPipeline;
use std::collections::HashMap;

/// Builder pre konfiguráciu ML Pipeline
pub struct MLPipelineBuilder
{
    model_type: Option<String>,
    model_params: HashMap<String, String>,
    evaluation_mode: Option<String>, // "classification" alebo "regression"
}

impl MLPipelineBuilder
{
    pub fn new() -> Self
    {
        Self {
            model_type: None,
            model_params: HashMap::new(),
            evaluation_mode: None,
        }
    }

    /// Nastaví model
    pub fn model(mut self, model_type: &str) -> Self
    {
        self.model_type = Some(model_type.to_string());
        self
    }

    /// Nastaví parameter modelu
    pub fn model_param(mut self, key: &str, value: &str) -> Self
    {
        self.model_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Explicitne nastaví evaluation mode (classification/regression)
    /// Ak nie je nastavené, automaticky sa detekuje z typu modelu
    pub fn evaluation_mode(mut self, mode: &str) -> Self
    {
        self.evaluation_mode = Some(mode.to_string());
        self
    }

    /// Vytvorí MLPipeline s validáciou kompatibility
    pub fn build(self) -> Result<MLPipeline, String>
    {
        // Validácia že model je nastavený
        let model_type = self.model_type
            .ok_or("Model musí byť nastavený")?;

        // Vytvorenie modelu
        let mut model = ModelFactory::create(&model_type)?;

        // Nastavenie parametrov modelu
        for (key, value) in &self.model_params
        {
            model.set_param(key, value)?;
        }

        // Určenie evaluation mode
        let eval_mode = if let Some(mode) = self.evaluation_mode
        {
            mode
        }
        else
        {
            // Automatická detekcia z typu modelu
            let model_type_detected = ModelFactory::get_model_type(&model_type)
                .ok_or("Nepodarilo sa určiť typ modelu")?;

            if model_type_detected == "both"
            {
                return Err("Model podporuje obe typy (classification/regression). Prosím explicitne nastavte evaluation_mode()".to_string());
            }

            model_type_detected.to_string()
        };

        Ok(MLPipeline {
            model,
            model_name: model_type.clone(),
            evaluation_mode: eval_mode,
            selected_indices: None,
            expected_features: None,
        })
    }
}

impl Default for MLPipelineBuilder
{
    fn default() -> Self
    {
        Self::new()
    }
}
