use smartcore::ensemble::random_forest_regressor::{RandomForestRegressor, RandomForestRegressorParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use super::IModel;

pub struct RandomForestWrapper
{
    model: Option<RandomForestRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    n_estimators: usize,
    max_depth: u16,
    min_samples_leaf: usize,
}

impl RandomForestWrapper
{
    pub fn new() -> Self
    {
        Self
        {
            model: None,
            n_estimators: 100,
            max_depth: 10,
            min_samples_leaf: 1,
        }
    }
}

impl IModel for RandomForestWrapper
{
    fn get_name(&self) -> &str { "Random Forest" }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["n_estimators", "max_depth", "min_samples_leaf"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>
    {
        match key
        {
            "n_estimators" =>
            {
                self.n_estimators = value.parse().map_err(|_| "n_estimators musí byť celé číslo")?;
                Ok(())
            }
            "max_depth" =>
            {
                self.max_depth = value.parse().map_err(|_| "max_depth musí byť celé číslo")?;
                Ok(())
            }
            "min_samples_leaf" =>
            {
                self.min_samples_leaf = value.parse().map_err(|_| "min_samples_leaf musí byť celé číslo")?;
                Ok(())
            }
            _ => Err(format!("Neznámy parameter '{}' pre Random Forest", key)),
        }
    }

    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>)
    {
        let mut params = RandomForestRegressorParameters::default();
        params.max_depth = Some(self.max_depth);
        params.n_trees = self.n_estimators;
        params.min_samples_leaf = self.min_samples_leaf;

        match RandomForestRegressor::fit(&x, &y, params)
        {
            Ok(m) => self.model = Some(m),
            Err(e) => web_sys::console::error_1(&format!("Random Forest fit failed: {:?}", e).into()),
        }
    }

    fn predict(&self, input: &[f64]) -> Vec<f64>
    {
        let x = DenseMatrix::from_2d_vec(&vec![input.to_vec()]).unwrap();
        self.model.as_ref()
            .and_then(|m| m.predict(&x).ok())
            .unwrap_or_default()
    }
}
