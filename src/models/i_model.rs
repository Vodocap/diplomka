use smartcore::linalg::basic::matrix::DenseMatrix;

/// Strategy pattern pre načítanie dát z rôznych zdrojov
pub trait IModel
{
    fn get_name(&self) -> &str;

    fn train(&mut self, x_train: DenseMatrix<f64>, y_train: Vec<f64>);

    fn predict(&self, input: &[f64]) -> Vec<f64>;

    fn get_supported_params(&self) -> Vec<&str>;

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
}
