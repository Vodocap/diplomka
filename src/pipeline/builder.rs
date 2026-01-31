use crate::models::{IModel, factory::ModelFactory};
use crate::processing::{DataProcessor, factory::ProcessorFactory};
use crate::feature_selection_strategies::{FeatureSelector, factory::FeatureSelectorFactory};
use crate::evaluation::{ModelEvaluator, EvaluationReport};
use super::{compatibility::CompatibilityRegistry, pipeline::MLPipeline};
use smartcore::linalg::basic::matrix::DenseMatrix;
use std::collections::HashMap;

/// Builder pre konfiguráciu ML Pipeline
pub struct MLPipelineBuilder {
    model_type: Option<String>,
    model_params: HashMap<String, String>,
    processor_type: Option<String>,
    selector_type: Option<String>,
    selector_params: HashMap<String, String>,
    evaluation_mode: Option<String>, // "classification" alebo "regression"
}

impl MLPipelineBuilder {
    pub fn new() -> Self {
        Self {
            model_type: None,
            model_params: HashMap::new(),
            processor_type: None,
            selector_type: None,
            selector_params: HashMap::new(),
            evaluation_mode: None,
        }
    }

    /// Nastaví model
    pub fn model(mut self, model_type: &str) -> Self {
        self.model_type = Some(model_type.to_string());
        self
    }

    /// Nastaví parameter modelu
    pub fn model_param(mut self, key: &str, value: &str) -> Self {
        self.model_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Nastaví data processor
    pub fn processor(mut self, processor_type: &str) -> Self {
        self.processor_type = Some(processor_type.to_string());
        self
    }

    /// Nastaví feature selector
    pub fn feature_selector(mut self, selector_type: &str) -> Self {
        self.selector_type = Some(selector_type.to_string());
        self
    }

    /// Nastaví parameter feature selektora
    pub fn selector_param(mut self, key: &str, value: &str) -> Self {
        self.selector_params.insert(key.to_string(), value.to_string());
        self
    }

    /// Explicitne nastaví evaluation mode (classification/regression)
    /// Ak nie je nastavené, automaticky sa detekuje z typu modelu
    pub fn evaluation_mode(mut self, mode: &str) -> Self {
        self.evaluation_mode = Some(mode.to_string());
        self
    }

    /// Vytvorí MLPipeline s validáciou kompatibility
    pub fn build(self) -> Result<MLPipeline, String> {
        // Validácia že model je nastavený
        let model_type = self.model_type
            .ok_or("Model musí byť nastavený")?;

        // Kontrola kompatibility
        CompatibilityRegistry::check_compatibility(
            &model_type,
            self.processor_type.as_deref(),
            self.selector_type.as_deref()
        )?;

        // Vytvorenie modelu
        let mut model = ModelFactory::create(&model_type)?;

        // Nastavenie parametrov modelu
        for (key, value) in &self.model_params {
            model.set_param(key, value)?;
        }

        // Vytvorenie procesora (optional)
        let processor = if let Some(proc_type) = &self.processor_type {
            Some(ProcessorFactory::create(proc_type)?)
        } else {
            None
        };

        // Vytvorenie selektora (optional)
        let mut selector = if let Some(sel_type) = &self.selector_type {
            let mut sel = FeatureSelectorFactory::create(sel_type)?;
            
            // Nastavenie parametrov selektora
            for (key, value) in &self.selector_params {
                sel.set_param(key, value)?;
            }
            
            Some(sel)
        } else {
            None
        };

        // Určenie evaluation mode
        let eval_mode = if let Some(mode) = self.evaluation_mode {
            mode
        } else {
            // Automatická detekcia z typu modelu
            let registry = CompatibilityRegistry::instance().lock().unwrap();
            let model_type_detected = registry.get_model_type(&model_type)
                .ok_or("Nepodarilo sa určiť typ modelu")?;
            
            if model_type_detected == "both" {
                return Err("Model podporuje obe typy (classification/regression). Prosím explicitne nastavte evaluation_mode()".to_string());
            }
            
            model_type_detected
        };

        Ok(MLPipeline {
            model,
            processor,
            selector,
            model_name: model_type.clone(),
            evaluation_mode: eval_mode,
        })
    }
}

impl Default for MLPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
