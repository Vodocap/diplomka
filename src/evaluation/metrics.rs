use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EvaluationReport {
    pub metrics: HashMap<String, f64>,
    pub model_name: String,
    pub evaluation_type: String,
}

impl EvaluationReport {
    pub fn new(model_name: String, evaluation_type: String) -> Self {
        Self {
            metrics: HashMap::new(),
            model_name,
            evaluation_type,
        }
    }

    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    pub fn get_metric(&self, name: &str) -> Option<f64> {
        self.metrics.get(name).copied()
    }

    pub fn get_all_metrics(&self) -> &HashMap<String, f64> {
        &self.metrics
    }
}
