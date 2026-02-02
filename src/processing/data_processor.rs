use smartcore::linalg::basic::matrix::DenseMatrix;

pub trait DataProcessor 
{
    fn get_name(&self) -> &str;
    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>;
}

pub mod scaler;
pub mod binner;
