use crate::processing::DataProcessor;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};

/// Procesor pre nahradenie null hodnôt
pub struct NullValueHandler {
    null_values: Vec<String>, // Možné reprezentácie null hodnôt ("", "NA", "null", "NaN")
    replacement_strategy: ReplacementStrategy,
}

#[derive(Clone)]
pub enum ReplacementStrategy {
    Mean,           // Nahradiť priemerom stĺpca
    Median,         // Nahradiť mediánom stĺpca
    Constant(f64),  // Nahradiť konštantou
    Zero,           // Nahradiť nulou
}

impl NullValueHandler {
    pub fn new(null_values: Vec<String>, replacement_strategy: ReplacementStrategy) -> Self {
        Self {
            null_values,
            replacement_strategy,
        }
    }

    pub fn with_params(null_repr: &str, strategy: &str, constant_value: Option<f64>) -> Self {
        let null_values = null_repr.split(',')
            .map(|s| s.trim().to_string())
            .collect();

        let replacement_strategy = match strategy {
            "mean" => ReplacementStrategy::Mean,
            "median" => ReplacementStrategy::Median,
            "zero" => ReplacementStrategy::Zero,
            "constant" => ReplacementStrategy::Constant(constant_value.unwrap_or(0.0)),
            _ => ReplacementStrategy::Zero,
        };

        Self::new(null_values, replacement_strategy)
    }

    fn calculate_column_mean(&self, data: &DenseMatrix<f64>, col: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for row in 0..data.shape().0 {
            let val = *data.get((row, col));
            if !val.is_nan() {
                sum += val;
                count += 1;
            }
        }
        if count > 0 { sum / count as f64 } else { 0.0 }
    }

    fn calculate_column_median(&self, data: &DenseMatrix<f64>, col: usize) -> f64 {
        let mut values: Vec<f64> = Vec::new();
        for row in 0..data.shape().0 {
            let val = *data.get((row, col));
            if !val.is_nan() {
                values.push(val);
            }
        }
        if values.is_empty() {
            return 0.0;
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mid = values.len() / 2;
        if values.len() % 2 == 0 {
            (values[mid - 1] + values[mid]) / 2.0
        } else {
            values[mid]
        }
    }

    fn get_replacement_value(&self, data: &DenseMatrix<f64>, col: usize) -> f64 {
        match &self.replacement_strategy {
            ReplacementStrategy::Mean => self.calculate_column_mean(data, col),
            ReplacementStrategy::Median => self.calculate_column_median(data, col),
            ReplacementStrategy::Constant(val) => *val,
            ReplacementStrategy::Zero => 0.0,
        }
    }
}

impl DataProcessor for NullValueHandler {
    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let shape = data.shape();
        let mut result_data = vec![vec![0.0; shape.1]; shape.0];

        // Pre každý stĺpec vypočítaj replacement hodnotu
        let mut replacements = Vec::new();
        for col in 0..shape.1 {
            replacements.push(self.get_replacement_value(data, col));
        }

        // Nahraď null hodnoty
        for row in 0..shape.0 {
            for col in 0..shape.1 {
                let val = *data.get((row, col));
                result_data[row][col] = if val.is_nan() {
                    replacements[col]
                } else {
                    val
                };
            }
        }

        DenseMatrix::from_2d_vec(&result_data).unwrap()
    }

    fn get_name(&self) -> &str {
        "Null Value Handler"
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "null_repr" => {
                self.null_values = value.split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                Ok(())
            }
            "strategy" => {
                self.replacement_strategy = match value {
                    "mean" => ReplacementStrategy::Mean,
                    "median" => ReplacementStrategy::Median,
                    "zero" => ReplacementStrategy::Zero,
                    val if val.starts_with("constant:") => {
                        let const_val = val.strip_prefix("constant:")
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0.0);
                        ReplacementStrategy::Constant(const_val)
                    }
                    _ => return Err(format!("Unknown strategy: {}", value)),
                };
                Ok(())
            }
            _ => Err(format!("Unknown parameter: {}", key))
        }
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["null_repr", "strategy"]
    }
}
