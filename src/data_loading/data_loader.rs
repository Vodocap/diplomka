use std::collections::HashMap;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

/// Výsledok načítania dát
#[derive(Debug, Clone)]
pub struct LoadedData {
    pub headers: Vec<String>,
    pub x_data: DenseMatrix<f64>,
    pub y_data: Vec<f64>,
    pub raw_records: Vec<HashMap<String, String>>,
}

impl LoadedData {
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

    pub fn num_features(&self) -> usize {
        self.x_data.shape().1
    }

    pub fn num_samples(&self) -> usize {
        self.x_data.shape().0
    }
}

/// Strategy pattern pre načítanie dát z rôznych zdrojov
pub trait DataLoader {
    /// Názov loadera
    fn get_name(&self) -> &str;

    /// Načíta dáta zo stringu
    fn load_from_string(&mut self, data: &str, target_column: &str) -> Result<LoadedData, String>;

    /// Získa dostupné stĺpce (headers) z dát
    fn get_available_columns(&self, data: &str) -> Result<Vec<String>, String>;

    /// Validuje formát dát pred načítaním
    fn validate_format(&self, data: &str) -> Result<(), String>;
}
