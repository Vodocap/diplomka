use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ColumnType};

/// Robust Scaler - škálovanie pomocou mediánu a IQR (Inter-Quartile Range)
/// Odolné voči outlierom
pub struct RobustScaler {
    medians: Option<Vec<f64>>,
    iqrs: Option<Vec<f64>>,
}

impl RobustScaler {
    pub fn new() -> Self {
        Self {
            medians: None,
            iqrs: None,
        }
    }

    fn calculate_median(mut values: Vec<f64>) -> f64 {
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let len = values.len();
        if len % 2 == 0 {
            (values[len / 2 - 1] + values[len / 2]) / 2.0
        } else {
            values[len / 2]
        }
    }

    fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

impl DataProcessor for RobustScaler {
    fn get_name(&self) -> &str {
        "Robust Scaler"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let mut medians = vec![0.0; cols];
        let mut iqrs = vec![1.0; cols];

        for j in 0..cols {
            let col: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).collect();
            
            medians[j] = Self::calculate_median(col.clone());
            let q1 = Self::calculate_percentile(&col, 25.0);
            let q3 = Self::calculate_percentile(&col, 75.0);
            iqrs[j] = (q3 - q1).max(1e-8); // Avoid division by zero
        }

        self.medians = Some(medians);
        self.iqrs = Some(iqrs);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let (Some(ref medians), Some(ref iqrs)) = (&self.medians, &self.iqrs) {
            for j in 0..cols.min(medians.len()) {
                for i in 0..rows {
                    let val = (*data.get((i, j)) - medians[j]) / iqrs[j];
                    result.set((i, j), val);
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String> {
        Err("RobustScaler has no configurable parameters".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Numeric])
    }
}
