use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessorParam {
    pub name: String,
    pub param_type: String,  // "number", "text", "select"
    pub default_value: String,
    pub description: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub options: Option<Vec<String>>,  // Pre select type
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    Categorical,  // Malý počet unikátnych hodnôt (< 10% riadkov)
    Numeric,      // Spojité numerické hodnoty
    Discrete,     // Diskrétne celé čísla
}

pub trait DataProcessor 
{
    fn get_name(&self) -> &str;
    fn process(&self, data: &smartcore::linalg::basic::matrix::DenseMatrix<f64>) -> smartcore::linalg::basic::matrix::DenseMatrix<f64>;
    fn fit(&mut self, data: &smartcore::linalg::basic::matrix::DenseMatrix<f64>);
    fn transform(&self, data: &smartcore::linalg::basic::matrix::DenseMatrix<f64>) -> smartcore::linalg::basic::matrix::DenseMatrix<f64>;
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
    fn get_supported_params(&self) -> Vec<&str>;
    
    /// Získa detailné informácie o parametroch procesora pre UI
    fn get_param_definitions(&self) -> Vec<ProcessorParam> {
        vec![] // Default - žiadne parametre
    }
    
    /// Určuje, na aké typy stĺpcov sa má procesor aplikovať
    /// None = aplikuje sa na všetky stĺpce
    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        None // Default - aplikuje sa na všetky
    }
}

/// Pomocná funkcia na detekciu typu stĺpca
pub fn detect_column_type(column: &[f64], total_rows: usize) -> ColumnType {
    use std::collections::HashSet;
    
    let unique_values: HashSet<u64> = column.iter()
        .map(|&v| v.to_bits())
        .collect();
    
    let unique_count = unique_values.len();
    let unique_ratio = unique_count as f64 / total_rows as f64;
    
    // Ak má menej než 10% unikátnych hodnôt, považuj za kategorický
    if unique_ratio < 0.1 {
        return ColumnType::Categorical;
    }
    
    // Skontroluj či sú všetky hodnoty celé čísla
    let all_integers = column.iter().all(|&v| v.fract() == 0.0);
    
    if all_integers && unique_count < 50 {
        ColumnType::Discrete
    } else {
        ColumnType::Numeric
    }
}

pub mod scaler;
pub mod binner;
pub mod ohencoder;
pub mod null_handler;
pub mod processor_decorator;
pub mod factory;
pub mod minmax_scaler;
pub mod robust_scaler;
pub mod label_encoder;
pub mod outlier_clipper;
pub mod log_transformer;
pub mod power_transformer;
pub mod selective_processor;
pub mod time_converter;
pub mod extra_processors;

pub use scaler::StandardScaler;
pub use binner::Binner;
pub use ohencoder::OneHotEncoder;
pub use null_handler::NullValueHandler;
pub use processor_decorator::ProcessorChain;
pub use factory::ProcessorFactory;
pub use minmax_scaler::MinMaxScaler;
pub use robust_scaler::RobustScaler;
pub use label_encoder::LabelEncoder;
pub use outlier_clipper::{OutlierClipper, ClippingMethod};
pub use log_transformer::{LogTransformer, LogBase};
pub use power_transformer::{PowerTransformer, TransformMethod};
pub use selective_processor::SelectiveProcessor;
pub use time_converter::TimeConverter;
pub use extra_processors::{CommaToDotProcessor, ThousandsSeparatorRemover, OrdinalEncoder, FrequencyEncoder, TargetEncoder};
