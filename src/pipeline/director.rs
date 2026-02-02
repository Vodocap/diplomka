use super::builder::MLPipelineBuilder;
use super::pipeline::MLPipeline;
use serde::{Serialize, Deserialize};

/// Director pre Builder pattern - obsahuje hotové "recepty" na vytváranie pipeline
/// Zapuzdruje komplexnú logiku konštrukcie a ponúka predpripravené konfigurácie
pub struct MLPipelineDirector;

impl MLPipelineDirector {
    /// Vytvorí základný klasifikačný pipeline
    /// Model: Logistic Regression, Processor: Standard Scaler, Selector: Variance
    pub fn build_basic_classification(model: &str) -> Result<MLPipeline, String> {
        MLPipelineBuilder::new()
            .model(model)
            .processor("scaler")
            .feature_selector("variance")
            .selector_param("threshold", "0.01")
            .evaluation_mode("classification")
            .build()
    }

    /// Vytvorí základný regresný pipeline
    /// Model: Linear Regression, Processor: Standard Scaler, Selector: Correlation
    pub fn build_basic_regression(model: &str) -> Result<MLPipeline, String> {
        MLPipelineBuilder::new()
            .model(model)
            .processor("scaler")
            .feature_selector("correlation")
            .evaluation_mode("regression")
            .build()
    }

    /// Vytvorí pokročilý klasifikačný pipeline s Chi-Square selection
    pub fn build_advanced_classification(
        model: &str,
        k_features: usize
    ) -> Result<MLPipeline, String> {
        MLPipelineBuilder::new()
            .model(model)
            .processor("scaler")
            .feature_selector("chi_square")
            .selector_param("k", &k_features.to_string())
            .evaluation_mode("classification")
            .build()
    }

    /// Vytvorí pokročilý regresný pipeline s mutual information selection
    pub fn build_advanced_regression(
        model: &str,
        alpha: f64
    ) -> Result<MLPipeline, String> {
        MLPipelineBuilder::new()
            .model(model)
            .model_param("alpha", &alpha.to_string())
            .processor("scaler")
            .feature_selector("mutual_information")
            .evaluation_mode("regression")
            .build()
    }

    /// Vytvorí minimálny pipeline bez processingu a feature selection
    pub fn build_minimal(model: &str, eval_mode: &str) -> Result<MLPipeline, String> {
        MLPipelineBuilder::new()
            .model(model)
            .evaluation_mode(eval_mode)
            .build()
    }

    /// Vytvorí KNN klasifikátor s optimálnymi nastaveniami
    pub fn build_knn_classifier(k: usize) -> Result<MLPipeline, String> {
        MLPipelineBuilder::new()
            .model("knn")
            .model_param("k", &k.to_string())
            .processor("scaler")
            .feature_selector("variance")
            .selector_param("threshold", "0.05")
            .evaluation_mode("classification")
            .build()
    }

    /// Vytvorí KNN regressor s optimálnymi nastaveniami
    pub fn build_knn_regressor(k: usize) -> Result<MLPipeline, String> {
        MLPipelineBuilder::new()
            .model("knn")
            .model_param("k", &k.to_string())
            .processor("scaler")
            .feature_selector("correlation")
            .evaluation_mode("regression")
            .build()
    }

    /// Vytvorí Decision Tree s information gain selection
    pub fn build_decision_tree_classifier() -> Result<MLPipeline, String> {
        MLPipelineBuilder::new()
            .model("tree")
            .feature_selector("information_gain")
            .evaluation_mode("classification")
            .build()
    }

    /// Vytvorí custom pipeline pomocou builder pattern s validáciou
    pub fn build_custom() -> MLPipelineBuilder {
        MLPipelineBuilder::new()
    }

    /// Vytvorí pipeline na základe stringu konfigurácie (pre jednoduchšie volanie z frontendu)
    pub fn build_from_config(
        model: &str,
        processor: Option<&str>,
        selector: Option<&str>,
        eval_mode: &str
    ) -> Result<MLPipeline, String> {
        let mut builder = MLPipelineBuilder::new()
            .model(model)
            .evaluation_mode(eval_mode);

        if let Some(proc) = processor {
            builder = builder.processor(proc);
        }

        if let Some(sel) = selector {
            builder = builder.feature_selector(sel);
        }

        builder.build()
    }

    /// Vráti všetky dostupné predpripravené konfigurácie
    pub fn available_presets() -> Vec<PresetInfo> {
        vec![
            PresetInfo {
                name: "basic_classification",
                description: "Základný klasifikačný pipeline (LogReg + Scaler + Variance)",
                model_type: "classification",
                model: Some("logreg"),
                processor: Some("scaler"),
                selector: Some("variance"),
            },
            PresetInfo {
                name: "basic_regression",
                description: "Základný regresný pipeline (LinReg + Scaler + Correlation)",
                model_type: "regression",
                model: Some("linreg"),
                processor: Some("scaler"),
                selector: Some("correlation"),
            },
            PresetInfo {
                name: "advanced_classification",
                description: "Pokročilý klasifikačný pipeline (Model + Scaler + Chi-Square)",
                model_type: "classification",
                model: None,
                processor: Some("scaler"),
                selector: Some("chi_square"),
            },
            PresetInfo {
                name: "advanced_regression",
                description: "Pokročilý regresný pipeline (Model + Scaler + MI)",
                model_type: "regression",
                model: None,
                processor: Some("scaler"),
                selector: Some("mutual_information"),
            },
            PresetInfo {
                name: "knn_classifier",
                description: "KNN klasifikátor s optimálnymi nastaveniami",
                model_type: "classification",
                model: Some("knn"),
                processor: Some("scaler"),
                selector: Some("variance"),
            },
            PresetInfo {
                name: "knn_regressor",
                description: "KNN regressor s optimálnymi nastaveniami",
                model_type: "regression",
                model: Some("knn"),
                processor: Some("scaler"),
                selector: Some("correlation"),
            },
            PresetInfo {
                name: "decision_tree",
                description: "Decision Tree s Information Gain selection",
                model_type: "classification",
                model: Some("tree"),
                processor: None,
                selector: Some("information_gain"),
            },
            PresetInfo {
                name: "minimal",
                description: "Minimálny pipeline bez preprocessingu",
                model_type: "both",
                model: None,
                processor: None,
                selector: None,
            },
        ]
    }
}

/// Informácie o predpripravenej konfigurácii
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub model_type: &'static str,
    pub model: Option<&'static str>,
    pub processor: Option<&'static str>,
    pub selector: Option<&'static str>,
}
