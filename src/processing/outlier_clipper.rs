use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ColumnType, ProcessorParam};

/// Outlier Clipper - orezáva outliere na základe IQR metódy
pub struct OutlierClipper
{
    method: ClippingMethod,
    lower_bounds: Option<Vec<f64>>,
    upper_bounds: Option<Vec<f64>>,
}

#[derive(Clone)]
pub enum ClippingMethod
{
    IQR(f64),          // IQR metóda s multiplierom (typicky 1.5)
}

impl OutlierClipper
{
    pub fn new(method: ClippingMethod) -> Self
    {
        Self {
            method,
            lower_bounds: None,
            upper_bounds: None,
        }
    }

    pub fn with_iqr(multiplier: f64) -> Self
    {
        Self::new(ClippingMethod::IQR(multiplier))
    }

    fn calculate_percentile(values: &[f64], percentile: f64) -> f64
    {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (percentile / 100.0 * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

impl DataProcessor for OutlierClipper
{
    fn get_name(&self) -> &str
    {
        "Outlier Clipper"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>)
    {
        let (rows, cols) = data.shape();
        let mut lower_bounds = vec![f64::NEG_INFINITY; cols];
        let mut upper_bounds = vec![f64::INFINITY; cols];

        for j in 0..cols
        {
            let col: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).filter(|v| !v.is_nan()).collect();

            if col.is_empty()
            {
                continue; // Skip columns with no valid values
            }

            match self.method
            {
                ClippingMethod::IQR(multiplier) =>
                {
                    let q1 = Self::calculate_percentile(&col, 25.0);
                    let q3 = Self::calculate_percentile(&col, 75.0);
                    let iqr = q3 - q1;
                    lower_bounds[j] = q1 - multiplier * iqr;
                    upper_bounds[j] = q3 + multiplier * iqr;
                }
            }
        }

        self.lower_bounds = Some(lower_bounds);
        self.upper_bounds = Some(upper_bounds);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let (Some(ref lower), Some(ref upper)) = (&self.lower_bounds, &self.upper_bounds)
        {
            for j in 0..cols.min(lower.len())
            {
                for i in 0..rows
                {
                    let val = *data.get((i, j));
                    let clipped = val.max(lower[j]).min(upper[j]);
                    result.set((i, j), clipped);
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        self.transform(data)
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>
    {
        match key
        {
            "method" =>
            {
                // Backward-compatible key: currently only IQR is implemented.
                if value.eq_ignore_ascii_case("iqr")
                {
                    Ok(())
                }
                else
                {
                    Err("OutlierClipper currently supports only method=iqr".to_string())
                }
            }
            "threshold" =>
            {
                let multiplier = value.parse::<f64>()
                    .map_err(|_| format!("Invalid threshold '{}': expected number", value))?;
                if !multiplier.is_finite() || multiplier <= 0.0
                {
                    return Err("Threshold must be a positive finite number".to_string());
                }
                self.method = ClippingMethod::IQR(multiplier);
                Ok(())
            }
            _ => Err(format!("Unknown parameter: {}", key)),
        }
    }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["method", "threshold"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam>
    {
        vec![
            ProcessorParam {
                name: "method".to_string(),
                param_type: "select".to_string(),
                default_value: "iqr".to_string(),
                description: "Metóda orezania outlierov (aktuálne dostupná: IQR)".to_string(),
                min: None,
                max: None,
                options: Some(vec!["iqr".to_string()]),
            },
            ProcessorParam {
                name: "threshold".to_string(),
                param_type: "number".to_string(),
                default_value: "1.5".to_string(),
                description: "IQR multiplikátor (bežne 1.5, prísnejšie 1.0, voľnejšie 2.0+)".to_string(),
                min: Some(0.1),
                max: Some(10.0),
                options: None,
            },
        ]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>>
    {
        Some(vec![ColumnType::Numeric])
    }
}
