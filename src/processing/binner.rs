use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ProcessorParam, ColumnType};

pub struct Binner 
{
    bins: usize,
}

impl Binner 
{
    pub fn new(bins: usize) -> Self 
    {
        Self 
        { 
            bins 
        }
    }
}

impl DataProcessor for Binner 
{
    fn get_name(&self) -> &str 
    {
        "Equal-width Binner"
    }

    fn fit(&mut self, _data: &DenseMatrix<f64>) {
        // Binner doesn't need fitting
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.process(data)
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> 
    {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        for j in 0..cols 
        {
            // Extrakcia stĺpca do Vec
            let col: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).collect();
            let min = col.iter().fold(f64::INFINITY, |a: f64, &b| a.min(b));
            let max = col.iter().fold(f64::NEG_INFINITY, |a: f64, &b| a.max(b));
            let range = max - min;

            if range > 0.0 
            {
                for i in 0..rows 
                {
                    let val = data.get((i, j));
                    let mut bin = ((val - min) / range * self.bins as f64).floor();
                    
                    if bin >= self.bins as f64 
                    {
                        bin = (self.bins - 1) as f64;
                    }
                    
                    result.set((i, j), bin);
                }
            }
        }
        result
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "bins" => {
                self.bins = value.parse().map_err(|_| format!("Invalid bins value: {}", value))?;
                Ok(())
            }
            _ => Err(format!("Unknown parameter: {}", key))
        }
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["bins"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam> {
        vec![ProcessorParam {
            name: "bins".to_string(),
            param_type: "number".to_string(),
            default_value: "10".to_string(),
            description: "Po\u{010d}et binov pre diskretiz\u{00e1}ciu".to_string(),
            min: Some(2.0),
            max: Some(100.0),
            options: None,
        }]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Numeric])
    }
}
