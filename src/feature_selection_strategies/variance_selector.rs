use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use ndarray::Array2;
use super::FeatureSelector;

pub struct VarianceSelector 
{
    threshold: f64,
}

impl VarianceSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            threshold: 0.0 
        }
    }
}

impl FeatureSelector for VarianceSelector 
{
    fn get_name(&self) -> &str 
    {
        "Variance Threshold Selector"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["threshold"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        match key 
        {
            "threshold" => 
            {
                self.threshold = value.parse().map_err(|_| "Invalid threshold".to_string())?;
                Ok(())
            }
            _ => Err("Unknown parameter".into()),
        }
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, _y: &[f64]) -> Vec<usize> 
    {
        let (rows, cols) = x.shape();
        let mut selected = Vec::new();
        let raw_data: Vec<f64> = x.all_elements().collect();
        let nd_array = Array2::from_shape_vec((rows, cols), raw_data).unwrap();

        for j in 0..cols 
        {
            // var(0.0) počíta populačný rozptyl
            if nd_array.column(j).var(0.0) > self.threshold 
            {
                selected.push(j);
            }
        }
        selected
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        x.get_columns(&indices)
    }
}