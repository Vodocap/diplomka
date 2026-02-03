use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};
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
        let shape = x.shape();
        let mut selected = Vec::new();

        for j in 0..shape.1 
        {
            // Extrakcia stĺpca do Vec
            let col: Vec<f64> = (0..shape.0).map(|i| *x.get((i, j))).collect();
            let mean: f64 = col.iter().sum::<f64>() / shape.0 as f64;
            let variance: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / shape.0 as f64;
            
            if variance > self.threshold 
            {
                selected.push(j);
            }
        }
        selected
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }
    
    fn get_feature_scores(&self, x: &DenseMatrix<f64>, _y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let shape = x.shape();
        let mut scores = Vec::new();

        for j in 0..shape.1 {
            let col: Vec<f64> = (0..shape.0).map(|i| *x.get((i, j))).collect();
            let mean: f64 = col.iter().sum::<f64>() / shape.0 as f64;
            let variance: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / shape.0 as f64;
            scores.push((j, variance));
        }
        Some(scores)
    }
    
    fn get_metric_name(&self) -> &str {
        "Variance"
    }
}

impl VarianceSelector {
    fn extract_columns(&self, x: &DenseMatrix<f64>, indices: &[usize]) -> DenseMatrix<f64> {
        let shape = x.shape();
        let rows = shape.0;
        let cols = indices.len();
        let mut data = vec![vec![0.0; cols]; rows];
        
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..rows {
                data[row][new_col] = *x.get((row, old_col));
            }
        }
        
        DenseMatrix::from_2d_vec(&data).unwrap()
    }
}
