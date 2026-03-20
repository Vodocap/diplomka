use smartcore::linear::logistic_regression::{LogisticRegression, LogisticRegressionParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use super::IModel;

/// Wrapper okolo smartcore LogisticRegression pre binarnu klasifikaciu.
/// Cielu premennu konvertuje na Vec<u32> zaokruhlenim, co predpoklada binarny vstup (0/1).
/// Podporuje L2 regularizaciu cez parameter alpha.
pub struct LogRegWrapper
{
    model: Option<LogisticRegression<f64, u32, DenseMatrix<f64>, Vec<u32>>>,
    alpha: f64,
}

impl LogRegWrapper
{
    /// Vytvori novu instanciu s nulovym alpha (bez regularizacie).
    pub fn new() -> Self
    {
        Self {
            model: None,
            alpha: 0.0,
        }
    }
}

impl IModel for LogRegWrapper
{
    fn get_name(&self) -> &str { "Logistická Regresia (Klasifikácia)" }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["alpha"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>
    {
        match key
        {
            "alpha" =>
            {
                self.alpha = value.parse().map_err(|_| "Alpha musí byť desatinné číslo")?;
                Ok(())
            }
            _ => Err("Parameter neexistuje".into())
        }
    }

    /// Natrenuuje model. Hodnoty y sa zaokruhlia na u32 — predpoklada sa binarny label (0/1).
    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>)
    {
        let mut params = LogisticRegressionParameters::default();
        params.alpha = self.alpha;

        let y_labels: Vec<u32> = y.iter().map(|&v| v.round() as u32).collect();
        match LogisticRegression::fit(&x, &y_labels, params)
        {
            Ok(m) => self.model = Some(m),
            Err(e) => web_sys::console::error_1(&format!("Logistic Regression fit failed: {:?}", e).into()),
        }
    }

    /// Predikuje triedu (0.0 alebo 1.0) pre jeden vstupny vektor.
    fn predict(&self, input: &[f64]) -> Vec<f64>
    {
        let x = DenseMatrix::from_2d_vec(&vec![input.to_vec()]).unwrap();
        self.model.as_ref()
            .and_then(|m|
            {
                let predictions: Result<Vec<u32>, _> = m.predict(&x);
                predictions.ok().map(|p| p.iter().map(|&v| v as f64).collect())
            })
            .unwrap_or_default()
    }
}
