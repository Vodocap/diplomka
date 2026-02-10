use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use std::collections::HashMap;
use super::{DataProcessor, ColumnType};

/// Label Encoder - enkóduje kategorické hodnoty na čísla (0, 1, 2, ...)
/// Užitočné pre ordinálne premenné a šetrí dimenzie oproti One-Hot
pub struct LabelEncoder {
    mappings: Option<Vec<HashMap<u64, usize>>>, // Pre každý stĺpec mapa hodnôt na indexy
}

impl LabelEncoder {
    pub fn new() -> Self {
        Self {
            mappings: None,
        }
    }
}

impl DataProcessor for LabelEncoder {
    fn get_name(&self) -> &str {
        "Label Encoder"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let mut mappings = Vec::new();

        for j in 0..cols {
            let mut unique_map = HashMap::new();
            let mut label_counter = 0;

            for i in 0..rows {
                let val_bits = data.get((i, j)).to_bits();
                if !unique_map.contains_key(&val_bits) {
                    unique_map.insert(val_bits, label_counter);
                    label_counter += 1;
                }
            }
            mappings.push(unique_map);
        }

        self.mappings = Some(mappings);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let Some(ref mappings) = self.mappings {
            for j in 0..cols.min(mappings.len()) {
                let map = &mappings[j];
                for i in 0..rows {
                    let val_bits = data.get((i, j)).to_bits();
                    if let Some(&label) = map.get(&val_bits) {
                        result.set((i, j), label as f64);
                    }
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String> {
        Err("LabelEncoder has no configurable parameters".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Categorical, ColumnType::Discrete])
    }
}
