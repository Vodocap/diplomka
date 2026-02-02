use smartcore::linear::linear_regression::{LinearRegression, LinearRegressionParameters, LinearRegressionSolverName};
use smartcore::linalg::basic::matrix::DenseMatrix;
use super::IModel; 

pub struct LinRegWrapper 
{
    pub(crate) model: Option<LinearRegression<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    solver: String,
}

impl LinRegWrapper 
{
    pub fn new() -> Self 
    {
        Self { model: None, solver: "qr".to_string() }
    }
}

impl IModel for LinRegWrapper 
{
    fn get_name(&self) -> &str { "Lineárna Regresia" }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["solver"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        match key {
            "solver" => {
                if value == "qr" || value == "svd" 
                {
                    self.solver = value.to_string();
                    Ok(())
                } else {
                    Err("Podporované solver-y sú: qr, svd".into())
                }
            }
            _ => Err("Parameter neexistuje".into())
        }
    }

    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>) 
    {
        let mut params = LinearRegressionParameters::default();
        params.solver = match self.solver.as_str() 
        {
            "svd" => LinearRegressionSolverName::SVD,
            _ => LinearRegressionSolverName::QR,
        };
        
        self.model = Some(LinearRegression::fit(&x, &y, params).unwrap());
    }

    fn predict(&self, input: &[f64]) -> Vec<f64> 
    {
        let x = DenseMatrix::from_2d_vec(&vec![input.to_vec()]).unwrap();
        self.model.as_ref()
            .map(|m| m.predict(&x).unwrap())
            .unwrap_or_default()
    }
}
