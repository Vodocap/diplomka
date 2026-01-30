use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use std::collections::HashMap;
use super::DataProcessor;

pub struct OneHotEncoder;

impl DataProcessor for OneHotEncoder 
{
    fn get_name(&self) -> &str 
    {
        "One-Hot Encoder"
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> 
    {
        let (rows, cols) = data.shape();
        let mut column_maps = Vec::new();
        let mut new_total_cols = 0;

        // 1. Zistíme unikátne hodnoty pre každý stĺpec
        for j in 0..cols 
        {
            let mut unique_vals = HashMap::new();
            let mut count = 0;
            
            for i in 0..rows 
            {
                let val_bits = data.get(i, j).to_bits();
                if !unique_vals.contains_key(&val_bits) 
                {
                    unique_vals.insert(val_bits, count);
                    count += 1;
                }
            }
            new_total_cols += count;
            column_maps.push(unique_vals);
        }

        // 2. Vytvoríme novú, širšiu maticu
        let mut new_matrix = DenseMatrix::zeros(rows, new_total_cols);
        let mut current_new_col = 0;

        for j in 0..cols 
        {
            let map = &column_maps[j];
            let num_unique = map.len();

            for i in 0..rows 
            {
                let val_bits = data.get(i, j).to_bits();
                let offset = map[&val_bits];
                new_matrix.set(i, current_new_col + offset, 1.0);
            }
            current_new_col += num_unique;
        }

        new_matrix
    }
}