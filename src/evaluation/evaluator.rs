use super::metrics::EvaluationReport;
use smartcore::metrics::{
    accuracy, f1, precision, recall,
    mean_squared_error, r2, mean_absolute_error,
};

pub struct ModelEvaluator;

impl ModelEvaluator {
    /// Vypočíta metriky pre klasifikačné modely (Logistic Regression, KNN Classifier, atď.)
    pub fn evaluate_classification(y_true: &[f64], y_pred: &[f64], model_name: &str) -> EvaluationReport {
        let mut report = EvaluationReport::new(
            model_name.to_string(),
            "classification".to_string()
        );

        // Konverzia na Vec pre smartcore metrics
        let y_true_vec: Vec<f64> = y_true.to_vec();
        let y_pred_vec: Vec<f64> = y_pred.to_vec();

        // Accuracy pre klasifik\u00e1ciu potrebuje porovnanie cel\u00fdch \u010d\u00edsel
        let correct = y_true_vec.iter().zip(y_pred_vec.iter())
            .filter(|(t, p)| (t.round() - p.round()).abs() < 0.1)
            .count();
        let acc = correct as f64 / y_true_vec.len() as f64;
        
        report.add_metric("accuracy".to_string(), acc);
        report.add_metric("precision".to_string(), precision(&y_true_vec, &y_pred_vec));
        report.add_metric("recall".to_string(), recall(&y_true_vec, &y_pred_vec));
        report.add_metric("f1_score".to_string(), f1(&y_true_vec, &y_pred_vec, 1.0));

        report
    }

    /// Vypočíta metriky pre regresné modely (Linear Regression, KNN Regressor, atď.)
    pub fn evaluate_regression(y_true: &[f64], y_pred: &[f64], model_name: &str) -> EvaluationReport {
        let mut report = EvaluationReport::new(
            model_name.to_string(),
            "regression".to_string()
        );

        // Konverzia na Vec pre smartcore metrics
        let y_true_vec: Vec<f64> = y_true.to_vec();
        let y_pred_vec: Vec<f64> = y_pred.to_vec();

        report.add_metric("mse".to_string(), mean_squared_error(&y_true_vec, &y_pred_vec));
        report.add_metric("mae".to_string(), mean_absolute_error(&y_true_vec, &y_pred_vec));
        report.add_metric("r2_score".to_string(), r2(&y_true_vec, &y_pred_vec));

        report
    }

    /// Automaticky vyberie správny typ evaluácie na základe typu modelu
    pub fn evaluate_auto(
        y_true: &[f64], 
        y_pred: &[f64], 
        model_name: &str,
        model_type: &str
    ) -> Result<EvaluationReport, String> {
        match model_type {
            "classification" => Ok(Self::evaluate_classification(y_true, y_pred, model_name)),
            "regression" => Ok(Self::evaluate_regression(y_true, y_pred, model_name)),
            _ => Err(format!("Neznámy typ modelu: {}", model_type)),
        }
    }
}
