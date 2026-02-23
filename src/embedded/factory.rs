use crate::embedded::{EmbeddedFeatureSelector, RandomForestSelector, RidgeSelector};

pub struct EmbeddedSelectorFactory;

impl EmbeddedSelectorFactory {
    /// Vytvorí embedded selector podľa typu úlohy
    pub fn create_for_task(is_classification: bool) -> Box<dyn EmbeddedFeatureSelector> {
        if is_classification {
            Box::new(RandomForestSelector::new(true))
        } else {
            // Pre regresiu použijeme Ridge (L2 regularization)
            Box::new(RidgeSelector::new())
        }
    }
    
    /// Vytvorí konkrétny embedded selector podľa názvu
    pub fn create(name: &str, is_classification: bool) -> Result<Box<dyn EmbeddedFeatureSelector>, String> {
        match name.to_lowercase().as_str() {
            "tree" | "random_forest" | "tree_importance" => {
                Ok(Box::new(RandomForestSelector::new(is_classification)))
            },
            "ridge" | "l2" => {
                if is_classification {
                    Err("Ridge je len pre regresiu, pre klasifikáciu použite tree".to_string())
                } else {
                    Ok(Box::new(RidgeSelector::new()))
                }
            },
            _ => Err(format!("Neznámy embedded selector: {}", name))
        }
    }
    
    /// Zoznam dostupných embedded metód
    pub fn available() -> Vec<String> {
        vec![
            "tree".to_string(),
            "ridge".to_string(),
        ]
    }
}
