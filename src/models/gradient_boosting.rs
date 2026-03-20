use gbdt::config::Config;
use gbdt::decision_tree::{DataVec, Data};
use gbdt::gradient_boost::GBDT;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::IModel;

pub struct GradientBoostingWrapper
{
    model: Option<GBDT>,
    n_estimators: usize,
    max_depth: u32,
    learning_rate: f64,
}

impl GradientBoostingWrapper
{
    pub fn new() -> Self
    {
        Self
        {
            model: None,
            n_estimators: 50,
            max_depth: 3,
            learning_rate: 0.1,
        }
    }
}

impl IModel for GradientBoostingWrapper
{
    fn get_name(&self) -> &str { "Gradient Boosting" }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["n_estimators", "max_depth", "learning_rate"]
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
            "learning_rate" =>
            {
                self.learning_rate = value.parse().map_err(|_| "learning_rate musí byť desatinné číslo")?;
                Ok(())
            }
            _ => Err(format!("Neznámy parameter '{}' pre Gradient Boosting", key)),
        }
    }

    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>)
    {
        let (n_rows, n_cols) = x.shape();

        let mut cfg = Config::new();
        cfg.set_feature_size(n_cols);
        cfg.set_max_depth(self.max_depth);
        cfg.set_iterations(self.n_estimators);
        cfg.set_shrinkage(self.learning_rate as f32);
        cfg.set_loss("SquaredError");
        cfg.set_data_sample_ratio(1.0);
        cfg.set_feature_sample_ratio(1.0);
        cfg.set_training_optimization_level(2);
        cfg.set_debug(false);

        let mut train_dv: DataVec = (0..n_rows).map(|i|
        {
            let features: Vec<f32> = (0..n_cols).map(|j| *x.get((i, j)) as f32).collect();
            let label = y[i] as f32;
            Data::new_training_data(features, 1.0, label, None)
        }).collect();

        let mut gbt = GBDT::new(&cfg);
        gbt.fit(&mut train_dv);
        self.model = Some(gbt);
    }

    fn predict(&self, input: &[f64]) -> Vec<f64>
    {
        match &self.model
        {
            None => vec![],
            Some(gbt) =>
            {
                let features: Vec<f32> = input.iter().map(|&v| v as f32).collect();
                let test_dv: DataVec = vec![Data::new_test_data(features, None)];
                let predictions = gbt.predict(&test_dv);
                predictions.iter().map(|&v| v as f64).collect()
            }
        }
    }
}
