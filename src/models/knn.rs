use smartcore::neighbors::knn_regressor::KNNRegressor;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use super::Model;

pub struct KnnWrapper 
{
    model: Option<KNNRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    k: usize,
}

impl KnnWrapper 
{
    pub fn new() -> Self 
    {
        Self { model: None }
    }
}

impl IModel for KnnWrapper 
{
    fn get_name(&self) -> &str { "K-Nearest Neighbors" }

    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>) 
    {
        let mut params = KNNRegressorParameters::default();
        params.k = self.k; 
        self.model = Some(KNNRegressor::fit(&x, &y, params).unwrap());
    }

    fn predict(&self, input: Vec<f64>) -> Vec<f64> 
    {
        let x = DenseMatrix::from_2d_vec(&vec![input]);
        self.model.as_ref()
            .map(|m| m.predict(&x).unwrap())
            .unwrap_or_default()
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["k"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        match key {
            "k" => {
                let new_k = value.parse::<usize>().map_err(|_| "K must be a number")?;
                self.k = new_k;
                Ok(())
            }
            _ => Err(format!("Unknown parameter {} for KNN", key)),
        }
    }

}