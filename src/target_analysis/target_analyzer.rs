use super::target_candidate::TargetCandidate;

/// Trait pre analyzátory cieľovej premennej.
/// Každý analyzátor implementuje inú metódu hodnotenia stĺpcov.
///
/// Nový analyzátor pridáte takto:
/// 1. Vytvorte struct implementujúci TargetAnalyzer
/// 2. Zaregistrujte ho v TargetAnalyzerFactory::create()
/// 3. Pridajte do TargetAnalyzerFactory::available()
pub trait TargetAnalyzer
{
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
    fn get_details_html(&self, columns: &[Vec<f64>], headers: &[String], candidates: &[TargetCandidate]) -> String
    {
        let _ = (columns, headers, candidates);
        String::new()
    }
}
