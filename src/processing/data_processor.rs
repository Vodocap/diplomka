use super::processor_param::{ProcessorParam, ColumnType};

pub trait DataProcessor
{
    fn get_name(&self) -> &str;
    fn process(&self, data: &smartcore::linalg::basic::matrix::DenseMatrix<f64>) -> smartcore::linalg::basic::matrix::DenseMatrix<f64>;
    fn fit(&mut self, data: &smartcore::linalg::basic::matrix::DenseMatrix<f64>);
    fn transform(&self, data: &smartcore::linalg::basic::matrix::DenseMatrix<f64>) -> smartcore::linalg::basic::matrix::DenseMatrix<f64>;
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
    fn get_supported_params(&self) -> Vec<&str>;

    /// Získa detailné informácie o parametroch procesora pre UI
    fn get_param_definitions(&self) -> Vec<ProcessorParam>
    {
        vec![] // Default - žiadne parametre
    }

    /// Určuje, na aké typy stĺpcov sa má procesor aplikovať
    /// None = aplikuje sa na všetky stĺpce
    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>>
    {
        None // Default - aplikuje sa na všetky
    }
}
