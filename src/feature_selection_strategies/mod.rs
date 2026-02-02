use smartcore::linalg::basic::matrix::DenseMatrix;

pub trait FeatureSelector 
{
    fn get_name(&self) -> &str;
    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64>;
    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize>;
    fn get_supported_params(&self) -> Vec<&str>;
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
}

pub mod variance_selector;
pub mod correlation_selector;
pub mod chi_square_selector;
pub mod information_gain_selector;
pub mod mutual_information_selector;
pub mod factory;

pub use variance_selector::VarianceSelector;
pub use correlation_selector::CorrelationSelector;
pub use chi_square_selector::ChiSquareSelector;
pub use information_gain_selector::InformationGainSelector;
pub use mutual_information_selector::MutualInformationSelector;
pub use factory::FeatureSelectorFactory;
