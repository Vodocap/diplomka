use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use std::collections::HashMap;
use super::{DataProcessor, ProcessorParam, ColumnType};

/// Target Encoder - nahradzuje kategoricku hodnotu priemerom cielovej premennej pre danu kategoriu.
/// Pouziva smoothing (Bayesov priemer) na zabranenie overfittingu pri malych skupinach.
pub struct TargetEncoder
{
    mean_maps: Option<Vec<HashMap<u64, f64>>>,
    global_mean: f64,
    smoothing: f64,
}

impl TargetEncoder
{
    pub fn new() -> Self
    {
        Self {
            mean_maps: None,
            global_mean: 0.0,
            smoothing: 10.0,
        }
    }
}

impl DataProcessor for TargetEncoder
{
    fn get_name(&self) -> &str
    {
        "Target Encoder"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>)
    {
        let (rows, cols) = data.shape();
        if cols == 0 || rows == 0
        {
            return;
        }

        // Compute global mean across ALL values (not per-column)
        let total_vals: f64 = (0..rows).flat_map(|i| (0..cols).map(move |j| *data.get((i, j)))).sum();
        self.global_mean = total_vals / (rows * cols) as f64;

        let mut maps = Vec::new();

        for j in 0..cols
        {
            let global_mean = self.global_mean;

            let mut group_sums: HashMap<u64, f64> = HashMap::new();
            let mut group_counts: HashMap<u64, usize> = HashMap::new();

            for i in 0..rows
            {
                let val = *data.get((i, j));
                let bits = val.to_bits();
                *group_sums.entry(bits).or_insert(0.0) += val;
                *group_counts.entry(bits).or_insert(0) += 1;
            }

            let mut mean_map = HashMap::new();
            for (bits, sum) in &group_sums
            {
                let count = group_counts[bits] as f64;
                let group_mean = sum / count;
                let smoothed = (count * group_mean + self.smoothing * global_mean) / (count + self.smoothing);
                mean_map.insert(*bits, smoothed);
            }
            maps.push(mean_map);
        }

        self.mean_maps = Some(maps);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let Some(ref maps) = self.mean_maps
        {
            for j in 0..cols.min(maps.len())
            {
                let map = &maps[j];
                for i in 0..rows
                {
                    let bits = data.get((i, j)).to_bits();
                    if let Some(&encoded) = map.get(&bits)
                    {
                        result.set((i, j), encoded);
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
            "smoothing" =>
            {
                self.smoothing = value.parse().map_err(|_| "Neplatná hodnota smoothing".to_string())?;
                Ok(())
            }
            _ => Err(format!("Neznámy parameter: {}", key)),
        }
    }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["smoothing"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam>
    {
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

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>>
    {
        Some(vec![ColumnType::Categorical, ColumnType::Discrete])
    }
}
