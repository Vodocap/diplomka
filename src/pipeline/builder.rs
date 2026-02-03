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
    processor_types: Vec<String>, // Zmenené na Vec pre viacero procesorov
    selector_type: Option<String>,
    selector_params: HashMap<String, String>,
    evaluation_mode: Option<String>, // "classification" alebo "regression"
}

impl MLPipelineBuilder {
    pub fn new() -> Self {
        Self {
            model_type: None,
            model_params: HashMap::new(),
            processor_types: Vec::new(), // Zmenené
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

    /// Nastaví data processor (jeden)
    pub fn processor(mut self, processor_type: &str) -> Self {
        self.processor_types = vec![processor_type.to_string()];
        self
    }

    /// Nastaví viacero data procesorov (chain)
    pub fn processors(mut self, processor_types: Vec<String>) -> Self {
        self.processor_types = processor_types;
        self
    }

    /// Pridá data processor do chainu
    pub fn add_processor(mut self, processor_type: &str) -> Self {
        self.processor_types.push(processor_type.to_string());
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

        // Kontrola kompatibility (použijeme prvý procesor pre spätnú kompatibilitu)
        let first_processor = self.processor_types.first().map(|s| s.as_str());
        CompatibilityRegistry::check_compatibility(
            &model_type,
            first_processor,
            self.selector_type.as_deref()
        )?;

        // Vytvorenie modelu
        let mut model = ModelFactory::create(&model_type)?;

        // Nastavenie parametrov modelu
        for (key, value) in &self.model_params {
            model.set_param(key, value)?;
        }

        // Vytvorenie processora/procesorov (optional)
        let processor = if !self.processor_types.is_empty() {
            let proc_refs: Vec<&str> = self.processor_types.iter().map(|s| s.as_str()).collect();
            Some(ProcessorFactory::create_chain(proc_refs)?)
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
            selected_indices: None,
            expected_features: None,
        })
    }
}

impl Default for MLPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
