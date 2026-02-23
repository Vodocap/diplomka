use smartcore::linalg::basic::matrix::DenseMatrix;

/// Trait pre embedded feature selection metódy
/// Embedded metódy vyberajú features priamo počas trénovania modelu
/// (napr. Lasso koeficienty, Random Forest importance, Ridge weights)
pub trait EmbeddedFeatureSelector {
    /// Natrénuje model a vráti feature importance scores (index, score)
    /// Features sú zoradené zostupne podľa importance
    fn fit_and_rank(&mut self, x: &DenseMatrix<f64>, y: &[f64]) -> Result<Vec<(usize, f64)>, String>;
    
    /// Názov metódy
    fn get_name(&self) -> String;
    
    /// Typ úlohy (classification / regression)
    fn supports_classification(&self) -> bool;
    fn supports_regression(&self) -> bool;
}
