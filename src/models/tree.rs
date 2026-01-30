use smartcore::tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use super::IModel;


pub struct TreeWrapper 
{
    model: Option<DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    max_depth: u16,
    min_samples_split: u16,
}

impl TreeWrapper
{
    pub fn new() -> Self 
    {
        Self 
        {  //Default 
            model: None,
            max_depth: 10,       
            min_samples_split: 2, 
        }
    }
}

impl IModel for TreeWrapper 
{
    fn get_name(&self) -> &str 
    {
        "Decision Tree"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["max_depth", "min_samples_split"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        match key 
        {
            "max_depth" => 
            {
                self.max_depth = value.parse().map_err(|_| "Invalid depth")?;
                Ok(())
            }
            "min_samples_split" => 
            {
                self.min_samples_split = value.parse().map_err(|_| "Invalid split value")?;
                Ok(())
            }
            _ => Err("Param not found".into())
        }
    }

    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>) 
    {
        let mut params = DecisionTreeRegressorParameters::default();
        params.max_depth = Some(self.max_depth as u16);
        params.min_samples_split = self.min_samples_split as usize;

        self.model = Some(DecisionTreeRegressor::fit(&x, &y, params).unwrap());
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> 
    {
        if let Some(ref m) = self.model 
        {
            let x = DenseMatrix::from_2d_vec(&vec![input]);
            m.predict(&x).unwrap_or_default()
        } 
        else 
        {
            vec![]
        }
    }
}