use std::collections::HashMap;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

/// Výsledok načítania dát
#[derive(Debug, Clone)]
pub struct LoadedData
{
    pub headers: Vec<String>,
    pub x_data: DenseMatrix<f64>,
    pub y_data: Vec<f64>,
    pub raw_records: Vec<HashMap<String, String>>,
}

impl LoadedData
{
    pub fn new(
        headers: Vec<String>,
        x_data: DenseMatrix<f64>,
        y_data: Vec<f64>,
        raw_records: Vec<HashMap<String, String>>,
    ) -> Self {
        Self {
            headers,
            x_data,
            y_data,
            raw_records,
        }
    }

    pub fn num_features(&self) -> usize
    {
        self.x_data.shape().1
    }

    pub fn num_samples(&self) -> usize
    {
        self.x_data.shape().0
    }
}
