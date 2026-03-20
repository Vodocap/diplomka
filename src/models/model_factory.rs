use super::{IModel, LinRegWrapper, LogRegWrapper, KnnWrapper, TreeWrapper, RandomForestWrapper, SvmWrapper, GradientBoostingWrapper, PolyRegWrapper};

/// Factory pre vytváranie modelov podľa názvu
pub struct ModelFactory;

impl ModelFactory
{
    /// Vytvorí model na základe názvu
    pub fn create(model_type: &str) -> Result<Box<dyn IModel>, String>
    {
        match model_type
        {
            "linreg" | "linear_regression" => Ok(Box::new(LinRegWrapper::new())),
            "logreg" | "logistic_regression" => Ok(Box::new(LogRegWrapper::new())),
            "knn" => Ok(Box::new(KnnWrapper::new())),
            "tree" | "decision_tree" => Ok(Box::new(TreeWrapper::new())),
            "rf" | "random_forest" => Ok(Box::new(RandomForestWrapper::new())),
            "svm" => Ok(Box::new(SvmWrapper::new())),
            "gbt" | "gradient_boosting" => Ok(Box::new(GradientBoostingWrapper::new())),
            "polynom" | "polynomial_regression" => Ok(Box::new(PolyRegWrapper::new())),
            _ => Err(format!("Neznámy model: {}", model_type)),
        }
    }

    /// Vráti zoznam všetkých dostupných modelov
    pub fn available_models() -> Vec<&'static str>
    {
        vec![
            "linreg",
            "logreg",
            "knn",
            "tree",
            "rf",
            "svm",
            "gbt",
            "polynom",
        ]
    }

    /// Vráti popis modelu
    pub fn get_model_description(model_type: &str) -> Option<&'static str>
    {
        match model_type
        {
            "linreg" => Some("Lineárna Regresia - predikcia spojitých hodnôt"),
            "logreg" => Some("Logistická Regresia - binárna klasifikácia"),
            "knn" => Some("K-Nearest Neighbors - klasifikácia alebo regresia"),
            "tree" => Some("Rozhodovací strom - klasifikácia alebo regresia"),
            "rf" => Some("Random Forest - ensemble stromov, regresia alebo klasifikácia"),
            "svm" => Some("Support Vector Machine - SVM s RBF/lineárnym kernelom"),
            "gbt" => Some("Gradient Boosting Trees - boosting stromov, regresia alebo klasifikácia"),
            "polynom" => Some("Polynomiálna Regresia - OLS s polynomiálnymi features (degree 1–8)"),
            _ => None,
        }
    }

    /// Určí typ modelu (classification/regression)
    pub fn get_model_type(model_type: &str) -> Option<&'static str>
    {
        match model_type
        {
            "linreg" => Some("regression"),
            "logreg" => Some("classification"),
            "knn" => Some("both"),
            "tree" => Some("both"),
            "rf" => Some("both"),
            "svm" => Some("both"),
            "gbt" => Some("both"),
            "polynom" => Some("regression"),
            _ => None,
        }
    }

    /// Vráti podporované parametre pre model
    pub fn get_supported_params(model_type: &str) -> Vec<&'static str>
    {
        match model_type
        {
            "knn" => vec!["k"],
            "tree" => vec!["max_depth", "min_samples_split"],
            "linreg" => vec!["solver"],
            "logreg" => vec!["alpha"],
            "rf" => vec!["n_estimators", "max_depth", "min_samples_leaf"],
            "svm" => vec!["c", "eps", "kernel", "gamma"],
            "gbt" => vec!["n_estimators", "max_depth", "learning_rate"],
            "polynom" => vec!["degree"],
            _ => vec![],
        }
    }
}
