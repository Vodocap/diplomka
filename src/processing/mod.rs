pub trait DataProcessor 
{
    fn get_name(&self) -> &str;
    fn process(&self, data: &smartcore::linalg::basic::matrix::DenseMatrix<f64>) -> smartcore::linalg::basic::matrix::DenseMatrix<f64>;
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
    fn get_supported_params(&self) -> Vec<&str>;
}

pub mod scaler;
pub mod binner;
pub mod ohencoder;
pub mod null_handler;
pub mod processor_decorator;
pub mod factory;

pub use scaler::StandardScaler;
pub use binner::Binner;
pub use ohencoder::OneHotEncoder;
pub use null_handler::NullValueHandler;
pub use processor_decorator::ProcessorChain;
pub use factory::ProcessorFactory;
