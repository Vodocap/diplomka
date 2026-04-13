/// Modul pre analýzu a výber cieľovej premennej.
/// Rozšíriteľný cez trait TargetAnalyzer - nové metódy stačí implementovať
/// a zaregistrovať vo factory.

pub mod target_candidate;          // TargetCandidate struct
pub mod target_analyzer;           // TargetAnalyzer trait
pub mod correlation_analyzer;
pub mod mutual_information_analyzer;
pub mod smc_analyzer;
pub mod target_analyzer_factory;

pub use target_candidate::TargetCandidate;
pub use target_analyzer::TargetAnalyzer;
pub use correlation_analyzer::CorrelationAnalyzer;
pub use mutual_information_analyzer::MutualInformationAnalyzer;
pub use smc_analyzer::SmcAnalyzer;
pub use target_analyzer_factory::TargetAnalyzerFactory;

use std::collections::HashSet;

/// Klasifikuje stĺpec hodnôt ako classification/regression na základe počtu unikátnych hodnôt.
/// Vracia (unique_count, suggested_type).
pub fn classify_column(values: &[f64], n: usize) -> (usize, String)
{
    let mut uniq = HashSet::new();
    for &v in values
    {
        uniq.insert(v.to_bits());
    }
    let unique_count = uniq.len();
    let is_cat = unique_count <= 10 || (unique_count as f64 / n as f64) < 0.05;
    let stype = if is_cat { "classification" } else { "regression" };
    (unique_count, stype.to_string())
}

/// Zdieľaný helper pre analyzátory: vytvorí a zoradí kandidátov.
/// `score_and_metrics` vracia (hlavné score, extra metriky) pre daný stĺpec.
pub fn build_ranked_candidates<F>(
    columns: &[Vec<f64>],
    headers: &[String],
    mut score_and_metrics: F,
) -> Vec<TargetCandidate>
where
    F: FnMut(usize) -> (f64, Vec<(String, f64)>),
{
    let num_cols = columns.len();
    let n = if num_cols > 0 { columns[0].len() } else { return vec![]; };

    let mut candidates = Vec::new();
    for col_idx in 0..num_cols
    {
        let (unique_count, stype) = classify_column(&columns[col_idx], n);

        let mean = columns[col_idx].iter().sum::<f64>() / n as f64;
        let variance = columns[col_idx].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

        let (score, extra_metrics) = score_and_metrics(col_idx);

        candidates.push(TargetCandidate {
            column_index: col_idx,
            column_name: headers[col_idx].clone(),
            score: (score * 10000.0).round() / 10000.0,
            unique_values: unique_count,
            variance: (variance * 10000.0).round() / 10000.0,
            suggested_type: stype,
            extra_metrics,
        });
    }

    candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    candidates
}
