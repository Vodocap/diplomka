use super::metrics::EvaluationReport;
use smartcore::metrics::{
    f1, precision, recall,
    mean_squared_error, r2, mean_absolute_error,
};

pub struct ModelEvaluator;

impl ModelEvaluator {
    /// Vypočíta confusion matrix pre binárnu klasifikáciu
    /// Vracia (TP, TN, FP, FN)
    fn confusion_matrix(y_true: &[f64], y_pred: &[f64]) -> (f64, f64, f64, f64) {
        let mut tp = 0.0;  // True Positives
        let mut tn = 0.0;  // True Negatives
        let mut fp = 0.0;  // False Positives
        let mut fn_ = 0.0; // False Negatives
        
        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let t_val = t.round();
            let p_val = p.round();
            
            if p_val == 1.0 && t_val == 1.0 { tp += 1.0; }
            else if p_val == 0.0 && t_val == 0.0 { tn += 1.0; }
            else if p_val == 1.0 && t_val == 0.0 { fp += 1.0; }
            else if p_val == 0.0 && t_val == 1.0 { fn_ += 1.0; }
        }
        
        (tp, tn, fp, fn_)
    }

    /// Vypočíta metriky pre klasifikačné modely
    pub fn evaluate_classification(y_true: &[f64], y_pred: &[f64], model_name: &str) -> EvaluationReport {
        let mut report = EvaluationReport::new(
            model_name.to_string(),
            "classification".to_string()
        );

        let y_true_vec: Vec<f64> = y_true.to_vec();
        let y_pred_vec: Vec<f64> = y_pred.to_vec();

        // Accuracy
        let correct = y_true_vec.iter().zip(y_pred_vec.iter())
            .filter(|(t, p)| (t.round() - p.round()).abs() < 0.1)
            .count();
        let acc = correct as f64 / y_true_vec.len() as f64;
        
        // Štandardné metriky
        report.add_metric("accuracy".to_string(), acc);
        report.add_metric("precision".to_string(), precision(&y_true_vec, &y_pred_vec));
        report.add_metric("recall".to_string(), recall(&y_true_vec, &y_pred_vec));
        report.add_metric("f1_score".to_string(), f1(&y_true_vec, &y_pred_vec, 1.0));

        // Confusion Matrix
        let (tp, tn, fp, fn_) = Self::confusion_matrix(&y_true_vec, &y_pred_vec);
        report.add_metric("true_positives".to_string(), tp);
        report.add_metric("true_negatives".to_string(), tn);
        report.add_metric("false_positives".to_string(), fp);
        report.add_metric("false_negatives".to_string(), fn_);

        // Specificity (TNR - True Negative Rate)
        let specificity = if tn + fp > 0.0 { tn / (tn + fp) } else { 0.0 };
        report.add_metric("specificity".to_string(), specificity);

        // Sensitivity = Recall (pre úplnosť)
        let sensitivity = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        report.add_metric("sensitivity".to_string(), sensitivity);

        // False Positive Rate (FPR)
        let fpr = if fp + tn > 0.0 { fp / (fp + tn) } else { 0.0 };
        report.add_metric("false_positive_rate".to_string(), fpr);

        // False Negative Rate (FNR)
        let fnr = if fn_ + tp > 0.0 { fn_ / (fn_ + tp) } else { 0.0 };
        report.add_metric("false_negative_rate".to_string(), fnr);

        // Matthews Correlation Coefficient (MCC) - lepší ako accuracy pre imbalanced datasets
        let mcc_denom = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();
        let mcc = if mcc_denom > 0.0 {
            (tp * tn - fp * fn_) / mcc_denom
        } else {
            0.0
        };
        report.add_metric("mcc".to_string(), mcc);

        report
    }

    /// Vypočíta metriky pre regresné modely
    pub fn evaluate_regression(y_true: &[f64], y_pred: &[f64], model_name: &str) -> EvaluationReport {
        let mut report = EvaluationReport::new(
            model_name.to_string(),
            "regression".to_string()
        );

        let y_true_vec: Vec<f64> = y_true.to_vec();
        let y_pred_vec: Vec<f64> = y_pred.to_vec();

        // Štandardné metriky
        let mse = mean_squared_error(&y_true_vec, &y_pred_vec);
        let mae = mean_absolute_error(&y_true_vec, &y_pred_vec);
        let r2_val = r2(&y_true_vec, &y_pred_vec);

        report.add_metric("mse".to_string(), mse);
        report.add_metric("mae".to_string(), mae);
        report.add_metric("r2_score".to_string(), r2_val);

        // RMSE - Root Mean Squared Error (v rovnakých jednotkách ako y)
        let rmse = mse.sqrt();
        report.add_metric("rmse".to_string(), rmse);

        // MAPE - Mean Absolute Percentage Error (% error)
        let mape = Self::calculate_mape(&y_true_vec, &y_pred_vec);
        report.add_metric("mape".to_string(), mape);

        // Mean Absolute Scaled Error (MASE) - užitočné pre porovnávanie
        let naive_mae = Self::calculate_naive_mae(&y_true_vec);
        if naive_mae > 0.0 {
            report.add_metric("mase".to_string(), mae / naive_mae);
        }

        // Medián Absolute Error
        let median_ae = Self::calculate_median_absolute_error(&y_true_vec, &y_pred_vec);
        report.add_metric("median_absolute_error".to_string(), median_ae);

        // Explained Variance Score (podobný R² ale inak počítaný)
        let explained_var = Self::calculate_explained_variance(&y_true_vec, &y_pred_vec);
        report.add_metric("explained_variance".to_string(), explained_var);

        // Pearsonov korelačný koeficient medzi y_true a y_pred
        let corr = Self::calculate_pearson_correlation(&y_true_vec, &y_pred_vec);
        report.add_metric("pearson_correlation".to_string(), corr);

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

    // ============= Pomocné funkcie =============

    fn calculate_mape(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len();
        if n == 0 { return 0.0; }
        
        let sum: f64 = y_true.iter().zip(y_pred.iter())
            .map(|(t, p)| {
                if t.abs() > 1e-10 {
                    ((t - p).abs() / t.abs()) * 100.0
                } else {
                    0.0
                }
            })
            .sum();
        
        sum / n as f64
    }

    fn calculate_naive_mae(y_true: &[f64]) -> f64 {
        let n = y_true.len();
        if n < 2 { return 0.0; }
        
        // Naive forecast: predikcia je hodnota z predchádzajúceho riadku
        let sum: f64 = y_true.windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .sum();
        
        sum / (n - 1) as f64
    }

    fn calculate_median_absolute_error(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let mut errors: Vec<f64> = y_true.iter().zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .collect();
        
        if errors.is_empty() { return 0.0; }
        
        errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        
        let n = errors.len();
        if n % 2 == 0 {
            (errors[n / 2 - 1] + errors[n / 2]) / 2.0
        } else {
            errors[n / 2]
        }
    }

    fn calculate_explained_variance(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len() as f64;
        if n == 0.0 { return 0.0; }
        
        let mean_y = y_true.iter().sum::<f64>() / n;
        let ss_res: f64 = y_true.iter().zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum();
        let ss_tot: f64 = y_true.iter()
            .map(|t| (t - mean_y).powi(2))
            .sum();
        
        if ss_tot == 0.0 { return 0.0; }
        1.0 - (ss_res / ss_tot)
    }

    fn calculate_pearson_correlation(y_true: &[f64], y_pred: &[f64]) -> f64 {
        let n = y_true.len() as f64;
        if n == 0.0 { return 0.0; }
        
        let mean_true = y_true.iter().sum::<f64>() / n;
        let mean_pred = y_pred.iter().sum::<f64>() / n;
        
        let mut numerator = 0.0;
        let mut sum_sq_true = 0.0;
        let mut sum_sq_pred = 0.0;
        
        for (t, p) in y_true.iter().zip(y_pred.iter()) {
            let dt = t - mean_true;
            let dp = p - mean_pred;
            numerator += dt * dp;
            sum_sq_true += dt * dt;
            sum_sq_pred += dp * dp;
        }
        
        let denom = (sum_sq_true * sum_sq_pred).sqrt();
        if denom == 0.0 { return 0.0; }
        
        numerator / denom
    }
}
