use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use ndarray::{Array2, Array1};
use super::FeatureSelector;
use std::collections::HashSet;

pub struct CorrelationSelector 
{
    threshold: f64,
}

impl CorrelationSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            threshold: 0.95 
        }
    }

    fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 
    {
        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);
        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) 
        {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        let den = (den_x * den_y).sqrt();
        if den == 0.0 
        { 
            0.0 
        } 
        else 
        { 
            (num / den).abs() 
        }
    }
}

impl FeatureSelector for CorrelationSelector 
{
    fn get_name(&self) -> &str 
    {
        "Correlation Filter"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["threshold"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        if key == "threshold" 
        {
            self.threshold = value.parse().map_err(|_| "Invalid threshold")?;
            return Ok(());
        }
        Err("Param not found".into())
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, _y: &[f64]) -> Vec<usize> 
    {
        let (rows, cols) = x.shape();
        let raw_data: Vec<f64> = x.all_elements().collect();
        let nd_array = Array2::from_shape_vec((rows, cols), raw_data).unwrap();
        
        let mut dropped = HashSet::new();
        let mut selected = Vec::new();

        for i in 0..cols 
        {
            if dropped.contains(&i) 
            { 
                continue; 
            }
            
            selected.push(i);

            for j in (i + 1)..cols 
            {
                if dropped.contains(&j) 
                { 
                    continue; 
                }
                
                let corr = Self::pearson_correlation(
                    &nd_array.column(i).to_owned(), 
                    &nd_array.column(j).to_owned()
                );

                if corr > self.threshold 
                {
                    dropped.insert(j);
                }
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