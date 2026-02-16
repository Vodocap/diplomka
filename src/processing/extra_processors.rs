use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use std::collections::HashMap;
use super::{DataProcessor, ProcessorParam, ColumnType};

// ============================================================================
// CommaToDotProcessor - nahradí desatinné čiarky bodkami
// ============================================================================

/// Tento procesor pracuje na textovej úrovni v WASM API.
/// Na DenseMatrix úrovni je no-op, pretože data sú už f64.
pub struct CommaToDotProcessor;

impl CommaToDotProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl DataProcessor for CommaToDotProcessor {
    fn get_name(&self) -> &str {
        "Comma to Dot"
    }

    fn fit(&mut self, _data: &DenseMatrix<f64>) {}

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        data.clone()
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        data.clone()
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String> {
        Err("Žiadne konfigurovateľné parametre".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        None
    }
}

// ============================================================================
// ThousandsSeparatorRemover - odstráni čiarky ako oddeľovače tisícov
// ============================================================================

pub struct ThousandsSeparatorRemover;

impl ThousandsSeparatorRemover {
    pub fn new() -> Self {
        Self
    }
}

impl DataProcessor for ThousandsSeparatorRemover {
    fn get_name(&self) -> &str {
        "Thousands Separator Remover"
    }

    fn fit(&mut self, _data: &DenseMatrix<f64>) {}

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        data.clone()
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        data.clone()
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String> {
        Err("Žiadne konfigurovateľné parametre".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        None
    }
}

// ============================================================================
// OrdinalEncoder - enkódovanie s uchovaním poradia
// ============================================================================

pub struct OrdinalEncoder {
    mappings: Option<Vec<HashMap<u64, usize>>>,
    sort_mode: String,
}

impl OrdinalEncoder {
    pub fn new() -> Self {
        Self {
            mappings: None,
            sort_mode: "appearance".to_string(),
        }
    }
}

impl DataProcessor for OrdinalEncoder {
    fn get_name(&self) -> &str {
        "Ordinal Encoder"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let mut mappings = Vec::new();

        for j in 0..cols {
            let mut unique_vals: Vec<(u64, f64)> = Vec::new();
            let mut seen = std::collections::HashSet::new();

            for i in 0..rows {
                let val = *data.get((i, j));
                let bits = val.to_bits();
                if !seen.contains(&bits) {
                    seen.insert(bits);
                    unique_vals.push((bits, val));
                }
            }

            if self.sort_mode == "alphabetical" {
                unique_vals.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            }

            let mut map = HashMap::new();
            for (idx, (bits, _)) in unique_vals.iter().enumerate() {
                map.insert(*bits, idx);
            }
            mappings.push(map);
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

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "sort_mode" => {
                self.sort_mode = value.to_string();
                Ok(())
            }
            _ => Err(format!("Neznámy parameter: {}", key)),
        }
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["sort_mode"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam> {
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
                    "alphabetical".to_string(),
                ]),
            },
        ]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Categorical, ColumnType::Discrete])
    }
}

// ============================================================================
// FrequencyEncoder - enkódovanie podľa frekvencie výskytu
// ============================================================================

pub struct FrequencyEncoder {
    freq_maps: Option<Vec<HashMap<u64, f64>>>,
}

impl FrequencyEncoder {
    pub fn new() -> Self {
        Self {
            freq_maps: None,
        }
    }
}

impl DataProcessor for FrequencyEncoder {
    fn get_name(&self) -> &str {
        "Frequency Encoder"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let mut maps = Vec::new();

        for j in 0..cols {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            for i in 0..rows {
                let bits = data.get((i, j)).to_bits();
                *counts.entry(bits).or_insert(0) += 1;
            }

            let mut freq_map = HashMap::new();
            for (bits, count) in counts {
                freq_map.insert(bits, count as f64 / rows as f64);
            }
            maps.push(freq_map);
        }

        self.freq_maps = Some(maps);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let Some(ref maps) = self.freq_maps {
            for j in 0..cols.min(maps.len()) {
                let map = &maps[j];
                for i in 0..rows {
                    let bits = data.get((i, j)).to_bits();
                    if let Some(&freq) = map.get(&bits) {
                        result.set((i, j), freq);
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
        Err("Frequency Encoder nemá konfigurovateľné parametre".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Categorical, ColumnType::Discrete])
    }
}

// ============================================================================
// TargetEncoder - enkódovanie podľa priemeru (smoothed mean encoding)
// ============================================================================

pub struct TargetEncoder {
    mean_maps: Option<Vec<HashMap<u64, f64>>>,
    global_mean: f64,
    smoothing: f64,
}

impl TargetEncoder {
    pub fn new() -> Self {
        Self {
            mean_maps: None,
            global_mean: 0.0,
            smoothing: 10.0,
        }
    }
}

impl DataProcessor for TargetEncoder {
    fn get_name(&self) -> &str {
        "Target Encoder"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        if cols == 0 || rows == 0 { return; }

        let mut maps = Vec::new();

        for j in 0..cols {
            let all_vals: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).collect();
            let global_mean = all_vals.iter().sum::<f64>() / rows as f64;
            self.global_mean = global_mean;

            let mut group_sums: HashMap<u64, f64> = HashMap::new();
            let mut group_counts: HashMap<u64, usize> = HashMap::new();

            for i in 0..rows {
                let val = *data.get((i, j));
                let bits = val.to_bits();
                *group_sums.entry(bits).or_insert(0.0) += val;
                *group_counts.entry(bits).or_insert(0) += 1;
            }

            let mut mean_map = HashMap::new();
            for (bits, sum) in &group_sums {
                let count = group_counts[bits] as f64;
                let group_mean = sum / count;
                let smoothed = (count * group_mean + self.smoothing * global_mean) / (count + self.smoothing);
                mean_map.insert(*bits, smoothed);
            }
            maps.push(mean_map);
        }

        self.mean_maps = Some(maps);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let Some(ref maps) = self.mean_maps {
            for j in 0..cols.min(maps.len()) {
                let map = &maps[j];
                for i in 0..rows {
                    let bits = data.get((i, j)).to_bits();
                    if let Some(&encoded) = map.get(&bits) {
                        result.set((i, j), encoded);
                    }
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "smoothing" => {
                self.smoothing = value.parse().map_err(|_| "Neplatná hodnota smoothing".to_string())?;
                Ok(())
            }
            _ => Err(format!("Neznámy parameter: {}", key)),
        }
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["smoothing"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam> {
        vec![
            ProcessorParam {
                name: "smoothing".to_string(),
                param_type: "number".to_string(),
                default_value: "10".to_string(),
                description: "Faktor vyhladzovania (vyššia = bližšie ku globálnemu priemeru)".to_string(),
                min: Some(0.0),
                max: Some(1000.0),
                options: None,
            },
        ]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Categorical, ColumnType::Discrete])
    }
}
