use crate::processing::DataProcessor;
use smartcore::linalg::basic::matrix::DenseMatrix;

/// Decorator pattern pre chain-ovanie viacerých procesorov
pub struct ProcessorChain {
    processors: Vec<Box<dyn DataProcessor>>,
}

impl ProcessorChain {
    pub fn new() -> Self {
        Self {
            processors: Vec::new(),
        }
    }

    pub fn add_mut(&mut self, processor: Box<dyn DataProcessor>) {
        self.processors.push(processor);
    }
}

impl DataProcessor for ProcessorChain {
    fn fit(&mut self, data: &DenseMatrix<f64>) {
        // Fit each processor on the transformed output of previous processors
        let mut current_data = data.clone();
        for processor in self.processors.iter_mut() {
            processor.fit(&current_data);
            current_data = processor.transform(&current_data);
        }
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let mut result = data.clone();
        for processor in &self.processors {
            result = processor.transform(&result);
        }
        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let mut result = data.clone();
        
        // Aplikuj procesory v poradí
        for processor in &self.processors {
            result = processor.process(&result);
        }
        
        result
    }

    fn get_name(&self) -> &str {
        "Processor Chain"
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String> {
        Err("ProcessorChain doesn't support direct parameters".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }
}
