use crate::models::model_factory::ModelFactory;
use super::pipeline::MLPipeline;

use std::collections::HashMap;

/// Builder pre postupnú konfiguráciu ML pipeline (Builder pattern).
/// Fluent API: `MLPipelineBuilder::new().model("knn").model_param("k","7").evaluation_mode("classification").build()`
pub struct MLPipelineBuilder
{
    model_type: Option<String>,
    model_params: HashMap<String, String>,
    evaluation_mode: Option<String>,
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

    /// Nastaví model (povinné)
    pub fn model(mut self, model_type: &str) -> Self
    {
        self.model_type = Some(model_type.to_string());
        self
    }

    /// Nastaví parameter modelu (napr. k, max_depth)
    pub fn model_param(mut self, key: &str, value: &str) -> Self
    {
        self.model_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Explicitne nastaví evaluation_mode ("classification" alebo "regression").
    /// Ak nie je nastavené, automaticky sa odvodí z typu modelu cez ModelFactory.
    pub fn evaluation_mode(mut self, mode: &str) -> Self
    {
        self.evaluation_mode = Some(mode.to_string());
        self
    }

    /// Zostaví MLPipeline. Zlyhá ak model nie je nastavený alebo evaluation_mode
    /// nie je možné odvodiť (modely s typom "both" ho vyžadujú explicitne).
    pub fn build(self) -> Result<MLPipeline, String>
    {
        let model_type = self.model_type
            .ok_or("Model musí byť nastavený")?;

        // Vytvorenie a konfigurácia modelu
        let mut model = ModelFactory::create(&model_type)?;
        for (key, value) in &self.model_params
        {
            model.set_param(key, value)?;
        }

        // Určenie evaluation_mode: explicitné > auto-detekcia z ModelFactory
        let eval_mode = match self.evaluation_mode
        {
            Some(mode) => mode,
            None =>
            {
                let mt = ModelFactory::get_model_type(&model_type)
                    .ok_or_else(|| format!("Neznámy model: {}", model_type))?;
                if mt == "both"
                {
                    return Err(format!(
                        "Model '{}' podporuje classification aj regression. Nastavte evaluation_mode() explicitne.",
                        model_type
                    ));
                }
                mt.to_string()
            }
        };

        Ok(MLPipeline {
            model,
            model_name: model_type,
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

