use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};
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

    fn pearson_correlation_vec(x: &[f64], y: &[f64]) -> f64 
    {
        let n = x.len() as f64;
        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;
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
        let shape = x.shape();
        let cols = shape.1;
        
        let mut dropped = HashSet::new();
        let mut selected = Vec::new();

        for i in 0..cols 
        {
            if dropped.contains(&i) 
            { 
                continue; 
            }
            
            selected.push(i);

            // Extrakcia stĺpca i do Vec
            let col_i: Vec<f64> = (0..shape.0).map(|row| *x.get((row, i))).collect();
            for j in (i + 1)..cols 
            {
                if dropped.contains(&j) 
                { 
                    continue; 
                }
                
                // Extrakcia stĺpca j do Vec
                let col_j: Vec<f64> = (0..shape.0).map(|row| *x.get((row, j))).collect();
                let corr = Self::pearson_correlation_vec(&col_i, &col_j);

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
        self.extract_columns(x, &indices)
    }
    
    fn get_feature_scores(&self, x: &DenseMatrix<f64>, _y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let shape = x.shape();
        let cols = shape.1;
        let mut scores = Vec::new();

        for i in 0..cols {
            let col_i: Vec<f64> = (0..shape.0).map(|row| *x.get((row, i))).collect();
            let mut max_corr = 0.0;
            
            for j in 0..cols {
                if i == j { continue; }
                let col_j: Vec<f64> = (0..shape.0).map(|row| *x.get((row, j))).collect();
                let corr = Self::pearson_correlation_vec(&col_i, &col_j).abs();
                if corr > max_corr {
                    max_corr = corr;
                }
            }
            scores.push((i, max_corr));
        }
        Some(scores)
    }
    
    fn get_metric_name(&self) -> &str {
        "Max Correlation"
    }
}

impl CorrelationSelector {
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
