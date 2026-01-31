pub trait DataProcessor 
{
    fn get_name(&self) -> &str;
    fn process(&self, data: &smartcore::linalg::basic::matrix::DenseMatrix<f64>) -> smartcore::linalg::basic::matrix::DenseMatrix<f64>;
}

pub mod scaler;
pub mod binner;
pub mod ohencoder;
pub mod factory;

pub use scaler::StandardScaler;
pub use binner::Binner;
pub use ohencoder::OneHotEncoder;
pub use factory::ProcessorFactory;
