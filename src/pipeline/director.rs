use super::builder::MLPipelineBuilder;
use super::pipeline::MLPipeline;

/// Director pre Builder pattern — obsahuje hotove "recepty" na zostavenie pipeline.
/// Zapuzdruje komplexnu logiku konfiguracie
/// pre typicke ulohý (klasifikacia, regresia, KNN, minimal pipeline).
pub struct MLPipelineDirector;

impl MLPipelineDirector
{
    /// Vytvorí základný klasifikačný pipeline
    /// Model: Logistic Regression, Processor: Standard Scaler, Selector: Variance
    pub fn build_basic_classification(model: &str) -> Result<MLPipeline, String>
    {
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
    pub fn build_basic_regression(model: &str) -> Result<MLPipeline, String>
    {
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
            .selector_param("num_features", &k_features.to_string())
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
    pub fn build_minimal(model: &str, eval_mode: &str) -> Result<MLPipeline, String>
    {
        MLPipelineBuilder::new()
            .model(model)
            .evaluation_mode(eval_mode)
            .build()
    }

    /// Vytvorí KNN klasifikátor s optimálnymi nastaveniami
    pub fn build_knn_classifier(k: usize) -> Result<MLPipeline, String>
    {
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
    pub fn build_knn_regressor(k: usize) -> Result<MLPipeline, String>
    {
        MLPipelineBuilder::new()
            .model("knn")
            .model_param("k", &k.to_string())
            .processor("scaler")
            .feature_selector("correlation")
            .evaluation_mode("regression")
            .build()
    }

    /// Vytvorí Decision Tree so SMC selection
    pub fn build_decision_tree_classifier() -> Result<MLPipeline, String>
    {
        MLPipelineBuilder::new()
            .model("tree")
            .feature_selector("smc")
            .evaluation_mode("classification")
            .build()
    }

    /// Vytvorí custom pipeline pomocou builder pattern s validáciou
    pub fn build_custom() -> MLPipelineBuilder
    {
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

        if let Some(proc) = processor
        {
            builder = builder.processor(proc);
        }

        if let Some(sel) = selector
        {
            builder = builder.feature_selector(sel);
        }

        builder.build()
    }

}
