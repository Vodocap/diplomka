pub trait IModel 
{
    fn get_name(&self) -> &str;

    fn train(&mut self, x_train: [&Vec<f64>], y_train: [&Vec<f64>]);

    fn predict(&self, input: &[f64]) -> Vec<f64>;

    fn get_supported_params(&self) -> Vec<&str>;

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;

}

pub mod linreg;
pub mod logreg;
pub mod tree;
pub mod knn;

pub use knn::KnnWrapper;
pub use linreg::LinRegWrapper;
pub use logreg::LogRegWrapper;
pub use tree::TreeWrapper;

use std::collections::HashMap;
use smartcore::metrics::{
    accuracy, f1, precision, recall,
    mean_squared_error, r2, mean_absolute_error,
};

pub struct EvaluationReport {
    pub metrics: HashMap<String, f64>,
}

pub struct ModelEvaluator;

impl ModelEvaluator {
    /// Vypočíta metriky pre klasifikačné modely (Logistic Regression, KNN Classifier, atď.)
    pub fn evaluate_classification(y_true: &[f64], y_pred: &[f64]) -> EvaluationReport {
        let mut metrics = HashMap::new();

        metrics.insert("accuracy".to_string(), accuracy(y_true, y_pred));
        metrics.insert("precision".to_string(), precision(y_true, y_pred));
        metrics.insert("recall".to_string(), recall(y_true, y_pred));
        metrics.insert("f1_score".to_string(), f1(y_true, y_pred));

        EvaluationReport { metrics }
    }

    /// Vypočíta metriky pre regresné modely (Linear Regression, KNN Regressor, atď.)
    pub fn evaluate_regression(y_true: &[f64], y_pred: &[f64]) -> EvaluationReport {
        let mut metrics = HashMap::new();

        metrics.insert("mse".to_string(), mean_squared_error(y_true, y_pred));
        metrics.insert("mae".to_string(), mean_absolute_error(y_true, y_pred));
        metrics.insert("r2_score".to_string(), r2(y_true, y_pred));

        EvaluationReport { metrics }
    }
}
