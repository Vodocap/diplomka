use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;

/// Singleton registry pre kompatibilitu modelov, procesorov a selektorov
pub struct CompatibilityRegistry {
    model_types: HashMap<String, String>, // model_name -> classification/regression
    compatible_processors: HashMap<String, Vec<String>>, // model_name -> compatible processors
    compatible_selectors: HashMap<String, Vec<String>>, // model_type -> compatible selectors
}

// Globálna singleton instancia
static REGISTRY: Lazy<Mutex<CompatibilityRegistry>> = Lazy::new(|| {
    Mutex::new(CompatibilityRegistry::new())
});

impl CompatibilityRegistry {
    fn new() -> Self {
        let mut registry = Self {
            model_types: HashMap::new(),
            compatible_processors: HashMap::new(),
            compatible_selectors: HashMap::new(),
        };
        
        registry.initialize_defaults();
        registry
    }

    /// Získa singleton instanciu
    pub fn instance() -> &'static Lazy<Mutex<CompatibilityRegistry>> {
        &REGISTRY
    }

    /// Inicializuje default kompatibility
    fn initialize_defaults(&mut self) {
        // Typy modelov
        self.model_types.insert("linreg".to_string(), "regression".to_string());
        self.model_types.insert("logreg".to_string(), "classification".to_string());
        self.model_types.insert("knn".to_string(), "both".to_string());
        self.model_types.insert("tree".to_string(), "both".to_string());

        // Kompatibilné procesory (všetky modely môžu použiť všetky procesory)
        let all_processors = vec!["scaler".to_string(), "binner".to_string(), "onehot".to_string()];
        self.compatible_processors.insert("linreg".to_string(), all_processors.clone());
        self.compatible_processors.insert("logreg".to_string(), all_processors.clone());
        self.compatible_processors.insert("knn".to_string(), all_processors.clone());
        self.compatible_processors.insert("tree".to_string(), all_processors.clone());

        // Kompatibilné selektory pre regression
        self.compatible_selectors.insert(
            "regression".to_string(), 
            vec!["variance".to_string(), "correlation".to_string(), "mutual_information".to_string()]
        );

        // Kompatibilné selektory pre classification
        self.compatible_selectors.insert(
            "classification".to_string(),
            vec![
                "variance".to_string(), 
                "correlation".to_string(), 
                "chi_square".to_string(), 
                "information_gain".to_string(),
                "mutual_information".to_string()
            ]
        );
    }

    /// Získa typ modelu (regression/classification/both)
    pub fn get_model_type(&self, model_name: &str) -> Option<String> {
        self.model_types.get(model_name).cloned()
    }

    /// Získa evaluation type pre model
    pub fn get_evaluation_type(&self, model_name: &str) -> Option<String> {
        self.get_model_type(model_name)
    }

    /// Skontroluje, či je processor kompatibilný s modelom
    pub fn is_processor_compatible(&self, model_name: &str, processor_name: &str) -> bool {
        self.compatible_processors
            .get(model_name)
            .map(|processors| processors.iter().any(|p| p == processor_name))
            .unwrap_or(false)
    }

    /// Skontroluje, či je selector kompatibilný s modelom
    pub fn is_selector_compatible(&self, model_name: &str, selector_name: &str) -> bool {
        let model_type = match self.get_model_type(model_name) {
            Some(t) => t,
            None => return false,
        };

        // Ak model podporuje "both", skontrolujeme obe kategórie
        if model_type == "both" {
            return self.is_selector_compatible_with_type("regression", selector_name)
                || self.is_selector_compatible_with_type("classification", selector_name);
        }

        self.is_selector_compatible_with_type(&model_type, selector_name)
    }

    fn is_selector_compatible_with_type(&self, model_type: &str, selector_name: &str) -> bool {
        self.compatible_selectors
            .get(model_type)
            .map(|selectors| selectors.iter().any(|s| s == selector_name))
            .unwrap_or(false)
    }

    /// Získa všetky kompatibilné procesory pre model
    pub fn get_compatible_processors(&self, model_name: &str) -> Vec<String> {
        self.compatible_processors
            .get(model_name)
            .cloned()
            .unwrap_or_default()
    }

    /// Získa všetky kompatibilné selektory pre model
    pub fn get_compatible_selectors(&self, model_name: &str) -> Vec<String> {
        let model_type = match self.get_model_type(model_name) {
            Some(t) => t,
            None => return vec![],
        };

        if model_type == "both" {
            // Zjednotenie oboch typov
            let mut selectors = self.compatible_selectors
                .get("regression")
                .cloned()
                .unwrap_or_default();
            
            if let Some(class_selectors) = self.compatible_selectors.get("classification") {
                for s in class_selectors {
                    if !selectors.contains(s) {
                        selectors.push(s.clone());
                    }
                }
            }
            selectors
        } else {
            self.compatible_selectors
                .get(&model_type)
                .cloned()
                .unwrap_or_default()
        }
    }
}

// Helper funkcie pre jednoduchšie použitie
impl CompatibilityRegistry {
    /// Statická helper metóda
    pub fn check_compatibility(
        model: &str,
        processor: Option<&str>,
        selector: Option<&str>
    ) -> Result<(), String> {
        let registry = REGISTRY.lock().unwrap();
        
        if let Some(proc) = processor {
            if !registry.is_processor_compatible(model, proc) {
                return Err(format!(
                    "Procesor '{}' nie je kompatibilný s modelom '{}'", 
                    proc, model
                ));
            }
        }

        if let Some(sel) = selector {
            if !registry.is_selector_compatible(model, sel) {
                return Err(format!(
                    "Feature selector '{}' nie je kompatibilný s modelom '{}'", 
                    sel, model
                ));
            }
        }

        Ok(())
    }
}
