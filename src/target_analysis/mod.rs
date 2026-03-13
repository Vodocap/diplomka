/// Modul pre analýzu a výber cieľovej premennej.
/// Rozšíriteľný cez trait TargetAnalyzer - nové metódy stačí implementovať
/// a zaregistrovať vo factory.

pub mod correlation_analyzer;
pub mod mutual_information_analyzer;
pub mod entropy_analyzer;
pub mod smc_analyzer;
pub mod factory;

pub use correlation_analyzer::CorrelationAnalyzer;
pub use mutual_information_analyzer::MutualInformationAnalyzer;
pub use entropy_analyzer::EntropyAnalyzer;
pub use smc_analyzer::SmcAnalyzer;
pub use factory::TargetAnalyzerFactory;

use std::collections::HashSet;

/// Klasifikuje stĺpec hodnôt ako classification/regression na základe počtu unikátnych hodnôt.
/// Vracia (unique_count, suggested_type).
pub fn classify_column(values: &[f64], n: usize) -> (usize, String) {
    let mut uniq = HashSet::new();
    for &v in values { uniq.insert(v.to_bits()); }
    let unique_count = uniq.len();
    let is_cat = unique_count <= 10 || (unique_count as f64 / n as f64) < 0.05;
    let stype = if is_cat { "classification" } else { "regression" };
    (unique_count, stype.to_string())
}

/// Výsledok analýzy jedného stĺpca ako potenciálnej cieľovej premennej
#[derive(Debug, Clone)]
pub struct TargetCandidate {
    pub column_index: usize,
    pub column_name: String,
    pub score: f64,
    pub unique_values: usize,
    pub variance: f64,
    pub suggested_type: String,
    /// Ďalšie metriky špecifické pre analyzátor (key → value)
    pub extra_metrics: Vec<(String, f64)>,
}

/// Trait pre analyzátory cieľovej premennej.
/// Každý analyzátor implementuje inú metódu hodnotenia stĺpcov.
/// 
/// Nový analyzátor pridáte takto:
/// 1. Vytvorte struct implementujúci TargetAnalyzer
/// 2. Zaregistrujte ho v TargetAnalyzerFactory::create()
/// 3. Pridajte do TargetAnalyzerFactory::available()
pub trait TargetAnalyzer {
    /// Unikátny identifikátor analyzátora
    fn get_name(&self) -> &str;
    
    /// Popis pre používateľa
    fn get_description(&self) -> &str;
    
    /// Názov hlavnej metriky (napr. "Priem. |korelácia|", "MI Score")
    fn get_metric_name(&self) -> &str;
    
    /// Vysvetlenie čo metrika znamená a ako sa interpretuje
    fn get_metric_explanation(&self) -> &str;
    
    /// Analyzuje všetky stĺpce a vráti zoradených kandidátov (najlepší prvý)
    /// `columns` - vektor stĺpcov (columns[i] = vektor hodnôt i-teho stĺpca)
    /// `headers` - názvy stĺpcov
    fn analyze(&self, columns: &[Vec<f64>], headers: &[String]) -> Vec<TargetCandidate>;
    
    /// Vráti detailný HTML s vizualizáciou (voliteľné)
    /// Volá sa po analyze() a môže využiť cached výsledky
    fn get_details_html(&self, columns: &[Vec<f64>], headers: &[String], candidates: &[TargetCandidate]) -> String {
        let _ = (columns, headers, candidates);
        String::new()
    }
}
