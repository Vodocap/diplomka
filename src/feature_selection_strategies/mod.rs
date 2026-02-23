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
    fn get_feature_scores(&self, _x: &DenseMatrix<f64>, _y: &[f64]) -> Option<Vec<(usize, f64)>> {
        None // Default implementation returns None
    }
    
    /// Get metric name (e.g., "Variance", "MI Score", "Chi-Square")
    fn get_metric_name(&self) -> &str {
        "Score"
    }
    
    /// Get detailed HTML about why each feature was selected/rejected.
    /// Selectors cache this info during get_selected_indices and return it here.
    fn get_selection_details(&self) -> String {
        String::new()
    }
    
    /// Helper method to extract columns by indices
    fn extract_columns(&self, x: &DenseMatrix<f64>, indices: &[usize]) -> DenseMatrix<f64> {
        use smartcore::linalg::basic::arrays::Array;
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

pub mod variance_selector;
pub mod correlation_selector;
pub mod chi_square_selector;
pub mod information_gain_selector;
pub mod mutual_information_selector;
pub mod smc_selector;
pub mod synergy_vns_selector;
pub mod factory;

pub use variance_selector::VarianceSelector;
pub use correlation_selector::CorrelationSelector;
pub use chi_square_selector::ChiSquareSelector;
pub use information_gain_selector::InformationGainSelector;
pub use mutual_information_selector::MutualInformationSelector;
pub use smc_selector::SmcSelector;
pub use synergy_vns_selector::SynergyVNSSelector;
// FeatureSelectorFactory removed - not used directly
