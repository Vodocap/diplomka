use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};
use std::collections::HashMap;
use super::FeatureSelector;

pub struct ChiSquareSelector 
{
    top_k: usize,
}

impl ChiSquareSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            top_k: 10 
        }
    }

    fn calculate_chi_square(feature_col: Vec<f64>, target: &[f64]) -> f64 
    {
        let mut contingency_table: HashMap<u64, HashMap<u64, usize>> = HashMap::new();
        let mut row_totals: HashMap<u64, usize> = HashMap::new();
        let mut col_totals: HashMap<u64, usize> = HashMap::new();
        let n = feature_col.len() as f64;

        // 1. Vybudovanie kontingenčnej tabuľky (pozorované frekvencie)
        for (f_val, t_val) in feature_col.iter().zip(target.iter()) 
        {
            let f_bits = f_val.to_bits();
            let t_bits = t_val.to_bits();

            *contingency_table.entry(f_bits).or_default().entry(t_bits).or_insert(0) += 1;
            *row_totals.entry(f_bits).or_insert(0) += 1;
            *col_totals.entry(t_bits).or_insert(0) += 1;
        }

        let mut chi_sq = 0.0;

        // 2. Výpočet Chi-kvadrát štatistiky: Sum((O - E)^2 / E)
        for (&f_bits, rows) in &contingency_table 
        {
            for (&t_bits, &observed) in rows 
            {
                let expected = (row_totals[&f_bits] as f64 * col_totals[&t_bits] as f64) / n;
                
                if expected > 0.0 
                {
                    let diff = observed as f64 - expected;
                    chi_sq += (diff * diff) / expected;
                }
            }
        }

        chi_sq
    }
}

impl FeatureSelector for ChiSquareSelector 
{
    fn get_name(&self) -> &str 
    {
        "Chi-Square Selector"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["top_k"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        if key == "top_k" 
        {
            self.top_k = value.parse().map_err(|_| "Invalid top_k".to_string())?;
            return Ok(());
        }
        Err("Unknown parameter".into())
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> 
    {
        let (_, cols) = x.shape();
        let mut scores: Vec<(usize, f64)> = Vec::new();

        for j in 0..cols 
        {
            // Extrakcia stĺpca do Vec
            let col_data: Vec<f64> = (0..x.shape().0).map(|i| *x.get((i, j))).collect();
            let score = Self::calculate_chi_square(col_data, y);
            scores.push((j, score));
        }

        // Zoradíme podľa skóre zostupne (vyššie skóre = silnejšia závislosť)
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.into_iter()
            .take(self.top_k)
            .map(|(idx, _)| idx)
            .collect()
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }
}

impl ChiSquareSelector {
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
