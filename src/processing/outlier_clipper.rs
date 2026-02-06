use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ColumnType};

/// Outlier Clipper - orezáva outliere na základe IQR alebo percentile metódy
pub struct OutlierClipper {
    method: ClippingMethod,
    lower_bounds: Option<Vec<f64>>,
    upper_bounds: Option<Vec<f64>>,
}

#[derive(Clone)]
#[allow(dead_code)]
pub enum ClippingMethod {
    IQR(f64),          // IQR metóda s multiplierom (typicky 1.5)
    Percentile(f64, f64), // Percentile metóda (napr. 1, 99)
    ZScore(f64),       // Z-score metóda s prahom (typicky 3.0)
}

impl OutlierClipper {
    pub fn new(method: ClippingMethod) -> Self {
        Self {
            method,
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    pub fn with_iqr(multiplier: f64) -> Self {
        Self::new(ClippingMethod::IQR(multiplier))
    }

    #[allow(dead_code)]
    pub fn with_percentile(lower: f64, upper: f64) -> Self {
        Self::new(ClippingMethod::Percentile(lower, upper))
    }

    #[allow(dead_code)]
    pub fn with_zscore(threshold: f64) -> Self {
        Self::new(ClippingMethod::ZScore(threshold))
    }

    fn calculate_percentile(values: &[f64], percentile: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

impl DataProcessor for OutlierClipper {
    fn get_name(&self) -> &str {
        "Outlier Clipper"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let mut lower_bounds = vec![f64::NEG_INFINITY; cols];
        let mut upper_bounds = vec![f64::INFINITY; cols];

        for j in 0..cols {
            let col: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).collect();

            match self.method {
                ClippingMethod::IQR(multiplier) => {
                    let q1 = Self::calculate_percentile(&col, 25.0);
                    let q3 = Self::calculate_percentile(&col, 75.0);
                    let iqr = q3 - q1;
                    lower_bounds[j] = q1 - multiplier * iqr;
                    upper_bounds[j] = q3 + multiplier * iqr;
                }
                ClippingMethod::Percentile(lower, upper) => {
                    lower_bounds[j] = Self::calculate_percentile(&col, lower);
                    upper_bounds[j] = Self::calculate_percentile(&col, upper);
                }
                ClippingMethod::ZScore(threshold) => {
                    let mean: f64 = col.iter().sum::<f64>() / col.len() as f64;
                    let variance = col.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
                    let std = variance.sqrt();
                    lower_bounds[j] = mean - threshold * std;
                    upper_bounds[j] = mean + threshold * std;
                }
            }
        }

        self.lower_bounds = Some(lower_bounds);
        self.upper_bounds = Some(upper_bounds);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let (Some(ref lower), Some(ref upper)) = (&self.lower_bounds, &self.upper_bounds) {
            for j in 0..cols.min(lower.len()) {
                for i in 0..rows {
                    let val = *data.get((i, j));
                    let clipped = val.max(lower[j]).min(upper[j]);
                    result.set((i, j), clipped);
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String> {
        Err("OutlierClipper parameters cannot be changed after initialization".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Numeric])
    }
}
