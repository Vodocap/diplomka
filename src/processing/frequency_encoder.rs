use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use std::collections::HashMap;
use super::{DataProcessor, ColumnType};

/// Frequency Encoder - nahradzuje kazdu kategoricku hodnotu jej relativnou frekvenciou vyskytu.
/// Hodnota = pocet_vyskytu / celkovy_pocet_riadkov. Caste kategorie maju vyssie cislo.
pub struct FrequencyEncoder
{
    freq_maps: Option<Vec<HashMap<u64, f64>>>,
}

impl FrequencyEncoder
{
    pub fn new() -> Self
    {
        Self {
            freq_maps: None,
        }
    }
}

impl DataProcessor for FrequencyEncoder
{
    fn get_name(&self) -> &str
    {
        "Frequency Encoder"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>)
    {
        let (rows, cols) = data.shape();
        let mut maps = Vec::new();

        for j in 0..cols
        {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            for i in 0..rows
            {
                let bits = data.get((i, j)).to_bits();
                *counts.entry(bits).or_insert(0) += 1;
            }

            let mut freq_map = HashMap::new();
            for (bits, count) in counts
            {
                freq_map.insert(bits, count as f64 / rows as f64);
            }
            maps.push(freq_map);
        }

        self.freq_maps = Some(maps);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let Some(ref maps) = self.freq_maps
        {
            for j in 0..cols.min(maps.len())
            {
                let map = &maps[j];
                for i in 0..rows
                {
                    let bits = data.get((i, j)).to_bits();
                    if let Some(&freq) = map.get(&bits)
                    {
                        result.set((i, j), freq);
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

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String>
    {
        Err("Frequency Encoder nemá konfigurovateľné parametre".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>>
    {
        Some(vec![ColumnType::Categorical, ColumnType::Discrete])
    }
}
