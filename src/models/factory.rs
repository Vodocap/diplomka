use super::{IModel, LinRegWrapper, LogRegWrapper, KnnWrapper, TreeWrapper};

/// Factory pre vytváranie modelov podľa názvu
pub struct ModelFactory;

impl ModelFactory {
    /// Vytvorí model na základe názvu
    pub fn create(model_type: &str) -> Result<Box<dyn IModel>, String> {
        match model_type {
            "linreg" | "linear_regression" => Ok(Box::new(LinRegWrapper::new())),
            "logreg" | "logistic_regression" => Ok(Box::new(LogRegWrapper::new())),
            "knn" => Ok(Box::new(KnnWrapper::new())),
            "tree" | "decision_tree" => Ok(Box::new(TreeWrapper::new())),
            _ => Err(format!("Neznámy model: {}", model_type)),
        }
    }

    /// Vráti zoznam všetkých dostupných modelov
    pub fn available_models() -> Vec<&'static str> {
        vec![
            "linreg",
            "logreg", 
            "knn",
            "tree",
        ]
    }

    /// Vráti popis modelu
    pub fn get_model_description(model_type: &str) -> Option<&'static str> {
        match model_type {
            "linreg" => Some("Lineárna Regresia - predikcia spojitých hodnôt"),
            "logreg" => Some("Logistická Regresia - binárna klasifikácia"),
            "knn" => Some("K-Nearest Neighbors - klasifikácia alebo regresia"),
            "tree" => Some("Rozhodovací strom - klasifikácia alebo regresia"),
            _ => None,
        }
    }

    /// Určí typ modelu (classification/regression)
    pub fn get_model_type(model_type: &str) -> Option<&'static str> {
        match model_type {
            "linreg" => Some("regression"),
            "logreg" => Some("classification"),
            "knn" => Some("both"), // KNN môže byť oboje
            "tree" => Some("both"),
            _ => None,
        }
    }
}
