use smartcore::linalg::basic::matrix::DenseMatrix;

pub trait FeatureSelector 
{
    fn get_name(&self) -> &str;
    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64>;
    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize>;
    fn get_supported_params(&self) -> Vec<&str>;
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
    
    /// Get feature scores/metrics used for selection (if available)
    /// Returns Vec of (feature_index, score) tuples
    fn get_feature_scores(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Option<Vec<(usize, f64)>> {
        None // Default implementation returns None
    }
    
    /// Get metric name (e.g., "Variance", "Chi-Square", "Correlation")
    fn get_metric_name(&self) -> &str {
        "Score"
    }
    
    /// Get detailed selection information as HTML
    /// Returns HTML string with visualization of selection process (e.g., correlation matrix)
    /// Must return empty string if no visualization available
    fn get_selection_details(&self, x: &DenseMatrix<f64>, y: &[f64]) -> String {
        let _ = (x, y);
        String::new()
    }
    
    /// Helper method to extract columns by indices
    fn extract_columns(&self, x: &DenseMatrix<f64>, indices: &[usize]) -> DenseMatrix<f64> {
        let shape = x.shape();
        let rows = shape.0;
        let cols = indices.len();
        let mut data = vec![vec![0.0; cols]; rows];
        
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..rows {
                data[row][new_col] = *x.get((row, old_col));
            }
        }
        
        DenseMatrix::from_2d_vec(&data).unwrap()
    }
}

pub mod variance;
pub mod correlation;
