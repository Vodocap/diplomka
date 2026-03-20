use smartcore::linalg::basic::matrix::DenseMatrix;
use super::{DataProcessor, ColumnType};

/// Tento procesor pracuje na textovej úrovni v WASM API (nahradzuje čiarky ako
/// oddeľovače tisícov v textových dátach). Na DenseMatrix úrovni je no-op,
/// pretože data sú už parsované ako f64.
pub struct ThousandsSeparatorRemover;

impl ThousandsSeparatorRemover
{
    pub fn new() -> Self
    {
        Self
    }
}

impl DataProcessor for ThousandsSeparatorRemover
{
    fn get_name(&self) -> &str
    {
        "Thousands Separator Remover"
    }

    fn fit(&mut self, _data: &DenseMatrix<f64>) {}

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        data.clone()
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        data.clone()
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String>
    {
        Err("Žiadne konfigurovateľné parametre".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>>
    {
        None
    }
}
