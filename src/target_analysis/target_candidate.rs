/// Výsledok analýzy jedného stĺpca ako potenciálnej cieľovej premennej
#[derive(Debug, Clone)]
pub struct TargetCandidate
{
    pub column_index: usize,
    pub column_name: String,
    pub score: f64,
    pub unique_values: usize,
    pub variance: f64,
    pub suggested_type: String,
    /// Ďalšie metriky špecifické pre analyzátor (key → value)
    pub extra_metrics: Vec<(String, f64)>,
}
