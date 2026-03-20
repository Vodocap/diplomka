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
