use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ProcessorParam, ColumnType};

/// Log Transformer - aplikuje logaritmickú transformáciu
/// Používa log(x + offset) pre vyhnutie sa log(0)
pub struct LogTransformer {
    offset: f64,
    base: LogBase,
}

#[derive(Clone)]
#[allow(dead_code)]
pub enum LogBase {
    Natural,    // ln(x)
    Base10,     // log10(x)
    Base2,      // log2(x)
}

impl LogTransformer {
    pub fn new() -> Self {
        Self {
            offset: 1.0,
            base: LogBase::Natural,
        }
    }

    #[allow(dead_code)]
    pub fn with_offset(offset: f64) -> Self {
        Self {
            offset,
            base: LogBase::Natural,
        }
    }

    #[allow(dead_code)]
    pub fn with_base(base: LogBase, offset: f64) -> Self {
        Self {
            offset,
            base,
        }
    }

    fn apply_log(&self, value: f64) -> f64 {
        let adjusted = value + self.offset;
        if adjusted <= 0.0 {
            return 0.0; // Bezpečnosť pre negatívne hodnoty
        }
        
        match self.base {
            LogBase::Natural => adjusted.ln(),
            LogBase::Base10 => adjusted.log10(),
            LogBase::Base2 => adjusted.log2(),
        }
    }
}

impl DataProcessor for LogTransformer {
    fn get_name(&self) -> &str {
        "Log Transformer"
    }

    fn fit(&mut self, _data: &DenseMatrix<f64>) {
        // Log transformer nepotrebuje fitting
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        for j in 0..cols {
            for i in 0..rows {
                let val = *data.get((i, j));
                let transformed = self.apply_log(val);
                result.set((i, j), transformed);
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "offset" => {
                self.offset = value.parse().map_err(|_| "Invalid offset value".to_string())?;
                Ok(())
            }
            _ => Err("Unknown parameter".to_string())
        }
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["offset"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam> {
        vec![ProcessorParam {
            name: "offset".to_string(),
            param_type: "number".to_string(),
            default_value: "1".to_string(),
            description: "Offset pre log(x + offset)".to_string(),
            min: Some(0.0),
            max: Some(100.0),
            options: None,
        }]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Numeric])
    }
}
