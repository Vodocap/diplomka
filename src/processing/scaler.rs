use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ColumnType};

pub struct StandardScaler {
    means: Option<Vec<f64>>,
    stds: Option<Vec<f64>>,
}

impl StandardScaler {
    pub fn new() -> Self {
        Self {
            means: None,
            stds: None,
        }
    }
}

impl DataProcessor for StandardScaler 
{
    fn get_name(&self) -> &str 
    {
        "Standard Scaler"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let mut means = vec![0.0; cols];
        let mut stds = vec![0.0; cols];

        for j in 0..cols {
            let col: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).collect();
            means[j] = col.iter().sum::<f64>() / rows as f64;
            let var = col.iter().map(|x| (x - means[j]).powi(2)).sum::<f64>() / rows as f64;
            stds[j] = var.sqrt().max(1e-8); // Avoid division by zero
        }

        self.means = Some(means);
        self.stds = Some(stds);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let (Some(ref means), Some(ref stds)) = (&self.means, &self.stds) {
            for j in 0..cols.min(means.len()) {
                for i in 0..rows {
                    let val = (data.get((i, j)) - means[j]) / stds[j];
                    result.set((i, j), val);
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> 
    {
        let (rows, cols) = data.shape();
        
        // Use transform if fitted, otherwise compute on-the-fly (for backward compatibility)
        if self.means.is_some() && self.stds.is_some() {
            self.transform(data)
        } else {
            let mut result = data.clone();

            for j in 0..cols 
            {
                let col: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).collect();
                let mean = col.iter().sum::<f64>() / rows as f64;
                let var = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / rows as f64;
                let std = var.sqrt().max(1e-8);

                for i in 0..rows 
                {
                    let val = (data.get((i, j)) - mean) / std;
                    result.set((i, j), val);
                }
            }
            result
        }
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String> {
        Err("StandardScaler has no configurable parameters".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Numeric])
    }
}
