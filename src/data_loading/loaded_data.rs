use std::collections::HashMap;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

/// Vysledok nacitania dat z DataLoader.
/// Obsahuje features (x_data), target (y_data), hlavicky stlpcov
/// a surove zaznamy pre spatne mapovanie na povodne riadky.
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
    /// Vytvori novu instanciu s hlavickami, maticou features, vektorom targetu a surovymi zaznamami.
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

    /// Pocet feature stlpcov (bez target stlpca).
    pub fn num_features(&self) -> usize
    {
        self.x_data.shape().1
    }

    /// Pocet riadkov (vzoriek) v datach.
    pub fn num_samples(&self) -> usize
    {
        self.x_data.shape().0
    }
}
