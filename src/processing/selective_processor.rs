use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ColumnType, detect_column_type};

/// Wrapper procesor, ktorý automaticky aplikuje procesor len na relevantné stĺpce
pub struct SelectiveProcessor {
    processor: Box<dyn DataProcessor>,
    applicable_columns: Vec<usize>, // Indexy stĺpcov, na ktoré sa procesor aplikuje
    column_mapping: Vec<usize>, // Mapovanie pôvodných indexov na nové
}

impl SelectiveProcessor {
    pub fn new(processor: Box<dyn DataProcessor>) -> Self {
        Self {
            processor,
            applicable_columns: Vec::new(),
            column_mapping: Vec::new(),
        }
    }

    /// Detekuje typy stĺpcov a určí, na ktoré sa má procesor aplikovať
    fn detect_applicable_columns(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let applicable_types = self.processor.get_applicable_column_types();

        if applicable_types.is_none() {
            // Procesor sa aplikuje na všetky stĺpce
            self.applicable_columns = (0..cols).collect();
            self.column_mapping = (0..cols).collect();
            return;
        }

        let target_types = applicable_types.unwrap();
        self.applicable_columns.clear();

        for j in 0..cols {
            let column: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).collect();
            let col_type = detect_column_type(&column, rows);

            if target_types.contains(&col_type) {
                self.applicable_columns.push(j);
            }
        }

        // Vytvor mapovanie
        self.column_mapping = vec![0; cols];
        let mut new_idx = 0;
        for &old_idx in &self.applicable_columns {
            self.column_mapping[old_idx] = new_idx;
            new_idx += 1;
        }
    }

    /// Extrahuje len aplikovateľné stĺpce
    fn extract_applicable_columns(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let rows = data.shape().0;
        let new_cols = self.applicable_columns.len();

        if new_cols == 0 || new_cols == data.shape().1 {
            return data.clone();
        }

        let mut result_data = vec![vec![0.0; new_cols]; rows];
        for i in 0..rows {
            for (new_j, &old_j) in self.applicable_columns.iter().enumerate() {
                result_data[i][new_j] = *data.get((i, old_j));
            }
        }

        DenseMatrix::from_2d_vec(&result_data).unwrap()
    }

    /// Zlúči spracované stĺpce späť s nespracovanými
    fn merge_columns(&self, original: &DenseMatrix<f64>, processed: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let rows = original.shape().0;
        let orig_cols = original.shape().1;

        if self.applicable_columns.len() == orig_cols {
            return processed.clone();
        }

        let mut result = original.clone();
        for (new_j, &old_j) in self.applicable_columns.iter().enumerate() {
            for i in 0..rows {
                result.set((i, old_j), *processed.get((i, new_j)));
            }
        }

        result
    }
}

impl DataProcessor for SelectiveProcessor {
    fn get_name(&self) -> &str {
        self.processor.get_name()
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        self.detect_applicable_columns(data);
        
        if !self.applicable_columns.is_empty() {
            let applicable_data = self.extract_applicable_columns(data);
            self.processor.fit(&applicable_data);
        }
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        if self.applicable_columns.is_empty() {
            return data.clone();
        }

        if self.applicable_columns.len() == data.shape().1 {
            return self.processor.transform(data);
        }

        let applicable_data = self.extract_applicable_columns(data);
        let processed_data = self.processor.transform(&applicable_data);
        self.merge_columns(data, &processed_data)
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        self.processor.set_param(key, value)
    }

    fn get_supported_params(&self) -> Vec<&str> {
        self.processor.get_supported_params()
    }

    fn get_param_definitions(&self) -> Vec<super::ProcessorParam> {
        self.processor.get_param_definitions()
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        self.processor.get_applicable_column_types()
    }
}
