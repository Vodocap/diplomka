pub trait IModel 
{
    fn get_name(&self) -> &str;

    fn train(&mut self, x_train: [&Vec<f64>], y_train: [&Vec<f64>]);

    fn predict(&self, input: &[f64]) -> Vec<f64>;

    fn get_supported_params(&self) -> Vec<&str>;

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;

}

pub mod linreg;
pub mod logreg;
pub mod tree;
pub mod knn;

pub use knn::KnnWrapper;
pub use linreg::LinRegWrapper;
pub use logreg::LogRegWrapper;
pub use tree::TreeWrapper;