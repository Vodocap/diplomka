use serde::{Serialize, Deserialize};

/// Parameter definícia pre procesor (používa sa v UI)
#[derive(Serialize, Deserialize, Clone)]
pub struct ProcessorParam
{
    pub name: String,
    pub param_type: String,  // "number", "text", "select"
    pub default_value: String,
    pub description: String,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub options: Option<Vec<String>>,  // Pre select type
}

/// Typ stĺpca detekovaný automaticky
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType
{
    Categorical,  // Malý počet unikátnych hodnôt (< 10% riadkov)
    Numeric,      // Spojité numerické hodnoty
    Discrete,     // Diskrétne celé čísla
}

/// Pomocná funkcia na detekciu typu stĺpca
pub fn detect_column_type(column: &[f64], total_rows: usize) -> ColumnType
{
    use std::collections::HashSet;

    let unique_values: HashSet<u64> = column.iter()
        .map(|&v| v.to_bits())
        .collect();

    let unique_count = unique_values.len();
    let unique_ratio = unique_count as f64 / total_rows as f64;

    // Ak má menej než 10% unikátnych hodnôt, považuj za kategorický
    if unique_ratio < 0.1
    {
        return ColumnType::Categorical;
    }

    // Skontroluj či sú všetky hodnoty celé čísla
    let all_integers = column.iter().all(|&v| v.fract() == 0.0);

    if all_integers && unique_count < 50
    {
        ColumnType::Discrete
    }
    else
    {
        ColumnType::Numeric
    }
}
