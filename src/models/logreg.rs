use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use super::IModel;

pub struct LogRegWrapper {
    model: Option<LogisticRegression<f64, DenseMatrix<f64>>>,
    alpha: f64,
}

impl LogRegWrapper {
    pub fn new() -> Self {
        Self { 
            model: None, 
            alpha: 0.0, 
        }
    }
}

impl IModel for LogRegWrapper {
    fn get_name(&self) -> &str { "Logistická Regresia (Klasifikácia)" }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["alpha"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "alpha" => {
                self.alpha = value.parse().map_err(|_| "Alpha musí byť desatinné číslo")?;
                Ok(())
            }
            _ => Err("Parameter neexistuje".into())
        }
    }

    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>) {
        let mut params = LogisticRegressionParameters::default();
        params.alpha = self.alpha;
        
        // Smartcore v logistic regression predpokladá, že y obsahuje label-y
        self.model = Some(LogisticRegression::fit(&x, &y, params).unwrap());
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> {
        let x = DenseMatrix::from_2d_vec(&vec![input]);
        self.model.as_ref()
            .map(|m| m.predict(&x).unwrap())
            .unwrap_or_default()
    }
}