use smartcore::linalg::basic::matrix::DenseMatrix;

pub trait FeatureSelector 
{
    fn get_name(&self) -> &str;
    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64>;
    fn get_selected_indices(&self, data: &DenseMatrix<f64>) -> Vec<usize>;
    fn get_supported_params(&self) -> Vec<&str>;
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
}

pub mod variance;
pub mod correlation;
