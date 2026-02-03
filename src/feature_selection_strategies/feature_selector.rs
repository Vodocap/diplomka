use smartcore::linalg::basic::matrix::DenseMatrix;

pub trait FeatureSelector 
{
    fn get_name(&self) -> &str;
    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64>;
    fn get_selected_indices(&self, data: &DenseMatrix<f64>) -> Vec<usize>;
    fn get_supported_params(&self) -> Vec<&str>;
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
    
    /// Get feature scores/metrics used for selection (if available)
    /// Returns Vec of (feature_index, score) tuples
    fn get_feature_scores(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Option<Vec<(usize, f64)>> {
        None // Default implementation returns None
    }
    
    /// Get metric name (e.g., "Variance", "MI Score", "Chi-Square")
    fn get_metric_name(&self) -> &str {
        "Score"
    }
}

pub mod variance;
pub mod correlation;
