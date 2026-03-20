use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use std::collections::HashMap;
use super::{DataProcessor, ProcessorParam, ColumnType};

/// Ordinal Encoder - enkoduje kategoricke hodnoty s uchovanim poradia.
/// Hodnoty su zoradene bud podla poradia vyskytu (appearance) alebo vzostupne (ascending)
/// a priradene celociselnemu indexu.
pub struct OrdinalEncoder
{
    mappings: Option<Vec<HashMap<u64, usize>>>,
    sort_mode: String,
}

impl OrdinalEncoder
{
    pub fn new() -> Self
    {
        Self {
            mappings: None,
            sort_mode: "appearance".to_string(),
        }
    }
}

impl DataProcessor for OrdinalEncoder
{
    fn get_name(&self) -> &str
    {
        "Ordinal Encoder"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>)
    {
        let (rows, cols) = data.shape();
        let mut mappings = Vec::new();

        for j in 0..cols
        {
            let mut unique_vals: Vec<(u64, f64)> = Vec::new();
            let mut seen = std::collections::HashSet::new();

            for i in 0..rows
            {
                let val = *data.get((i, j));
                let bits = val.to_bits();
                if !seen.contains(&bits)
                {
                    seen.insert(bits);
                    unique_vals.push((bits, val));
                }
            }

            if self.sort_mode == "ascending" || self.sort_mode == "alphabetical"
            {
                unique_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            let mut map = HashMap::new();
            for (idx, (bits, _)) in unique_vals.iter().enumerate()
            {
                map.insert(*bits, idx);
            }
            mappings.push(map);
        }

        self.mappings = Some(mappings);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let Some(ref mappings) = self.mappings
        {
            for j in 0..cols.min(mappings.len())
            {
                let map = &mappings[j];
                for i in 0..rows
                {
                    let val_bits = data.get((i, j)).to_bits();
                    if let Some(&label) = map.get(&val_bits)
                    {
                        result.set((i, j), label as f64);
                    }
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        self.transform(data)
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>
    {
        match key
        {
            "sort_mode" =>
            {
                self.sort_mode = value.to_string();
                Ok(())
            }
            _ => Err(format!("Neznámy parameter: {}", key)),
        }
    }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["sort_mode"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam>
    {
        vec![
            ProcessorParam {
                name: "sort_mode".to_string(),
                param_type: "select".to_string(),
                default_value: "appearance".to_string(),
                description: "Spôsob zoradenia kategórií".to_string(),
                min: None,
                max: None,
                options: Some(vec![
                    "appearance".to_string(),
                    "ascending".to_string(),
                ]),
            },
        ]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>>
    {
        Some(vec![ColumnType::Categorical, ColumnType::Discrete])
    }
}
