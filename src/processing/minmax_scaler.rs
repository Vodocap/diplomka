use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ProcessorParam, ColumnType};

/// MinMax Scaler - normalizuje dáta do rozsahu [min_range, max_range]
pub struct MinMaxScaler {
    min_vals: Option<Vec<f64>>,
    max_vals: Option<Vec<f64>>,
    min_range: f64,
    max_range: f64,
}

impl MinMaxScaler {
    pub fn new() -> Self {
        Self {
            min_vals: None,
            max_vals: None,
            min_range: 0.0,
            max_range: 1.0,
        }
    }

    #[allow(dead_code)]
    pub fn with_range(min_range: f64, max_range: f64) -> Self {
        Self {
            min_vals: None,
            max_vals: None,
            min_range,
            max_range,
        }
    }
}

impl DataProcessor for MinMaxScaler {
    fn get_name(&self) -> &str {
        "MinMax Scaler"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let mut min_vals = vec![f64::INFINITY; cols];
        let mut max_vals = vec![f64::NEG_INFINITY; cols];

        for j in 0..cols {
            for i in 0..rows {
                let val = *data.get((i, j));
                if val < min_vals[j] {
                    min_vals[j] = val;
                }
                if val > max_vals[j] {
                    max_vals[j] = val;
                }
            }
        }

        self.min_vals = Some(min_vals);
        self.max_vals = Some(max_vals);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let (Some(ref min_vals), Some(ref max_vals)) = (&self.min_vals, &self.max_vals) {
            for j in 0..cols.min(min_vals.len()) {
                let range = max_vals[j] - min_vals[j];
                let scale = self.max_range - self.min_range;
                
                for i in 0..rows {
                    let val = *data.get((i, j));
                    let normalized = if range > 1e-8 {
                        (val - min_vals[j]) / range * scale + self.min_range
                    } else {
                        self.min_range
                    };
                    result.set((i, j), normalized);
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "min" => {
                self.min_range = value.parse().map_err(|_| "Invalid min value".to_string())?;
                Ok(())
            }
            "max" => {
                self.max_range = value.parse().map_err(|_| "Invalid max value".to_string())?;
                Ok(())
            }
            _ => Err("Unknown parameter".to_string())
        }
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["min", "max"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam> {
        vec![
            ProcessorParam {
                name: "min".to_string(),
                param_type: "number".to_string(),
                default_value: "0".to_string(),
                description: "Minim\u{00e1}lna hodnota rozsahu".to_string(),
                min: None,
                max: None,
                options: None,
            },
            ProcessorParam {
                name: "max".to_string(),
                param_type: "number".to_string(),
                default_value: "1".to_string(),
                description: "Maxim\u{00e1}lna hodnota rozsahu".to_string(),
                min: None,
                max: None,
                options: None,
            },
        ]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Numeric])
    }
}
