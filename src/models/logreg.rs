use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use super::IModel;

pub struct LogRegWrapper {
    model: Option<LogisticRegression<f64, u32, DenseMatrix<f64>, Vec<u32>>>,
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
        
        // Konverzia f64 na u32 pre klasifikáciu
        let y_labels: Vec<u32> = y.iter().map(|&v| v.round() as u32).collect();
        self.model = Some(LogisticRegression::fit(&x, &y_labels, params).unwrap());
    }

    fn predict(&self, input: &[f64]) -> Vec<f64> {
        let x = DenseMatrix::from_2d_vec(&vec![input.to_vec()]).unwrap();
        self.model.as_ref()
            .map(|m| {
                let predictions: Vec<u32> = m.predict(&x).unwrap();
                predictions.iter().map(|&v| v as f64).collect()
            })
            .unwrap_or_default()
    }
}
