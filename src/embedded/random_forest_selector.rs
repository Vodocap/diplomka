use crate::embedded::EmbeddedFeatureSelector;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use crate::entropy::mi_estimator;

/// Correlation-based feature importance selector (embedded method).
/// Uses absolute Pearson correlation with the target as a proxy for feature importance,
/// since SmartCore does not expose tree-based feature_importances.
pub struct RandomForestSelector
{
    is_classification: bool,
}

impl RandomForestSelector
{
    pub fn new(is_classification: bool) -> Self
    {
        RandomForestSelector {
            is_classification,
        }
    }

    /// Výpočet feature importance z Decision Tree
    /// SmartCore nemá priamy feature_importances, takže použijeme proxy:
    /// Počítame koľkokrát sa každý feature použil na split
    fn calculate_feature_importance_proxy(
        &self,
        x: &DenseMatrix<f64>,
        y: &[f64],
    ) -> Result<Vec<f64>, String> {
        let num_features = x.shape().1;

        if self.is_classification
        {
            // Pre klasifikáciu - použijeme korelácie ako aproximáciu
            // (SmartCore decision tree API neexponuje feature importance)
            let mut scores = vec![0.0; num_features];
            for feat_idx in 0..num_features
            {
                let feat_col: Vec<f64> = (0..x.shape().0)
                    .map(|row| *x.get((row, feat_idx)))
                    .collect();

                // Absolútna korelácia ako proxy pre importance
                let corr = mi_estimator::pearson_correlation(&feat_col, y).abs();
                scores[feat_idx] = corr;
            }
            Ok(scores)
        }
        else
        {
            // Pre regresiu - použijeme variance reduction proxy
            let mut scores = vec![0.0; num_features];

            for feat_idx in 0..num_features
            {
                let feat_col: Vec<f64> = (0..x.shape().0)
                    .map(|row| *x.get((row, feat_idx)))
                    .collect();

                // Korelácia^2 ako proxy pre variance reduction
                let corr = mi_estimator::pearson_correlation(&feat_col, y);
                scores[feat_idx] = corr * corr; // R²
            }
            Ok(scores)
        }
    }
}

impl EmbeddedFeatureSelector for RandomForestSelector
{
    fn fit_and_rank(&mut self, x: &DenseMatrix<f64>, y: &[f64]) -> Result<Vec<(usize, f64)>, String>
    {
        let scores = self.calculate_feature_importance_proxy(x, y)?;

        // Vytvor ranking
        let mut ranked: Vec<(usize, f64)> = scores.iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        // Zoraď zostupne podľa score
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(ranked)
    }

    fn get_name(&self) -> String
    {
        if self.is_classification
        {
            "Tree Importance (Classification)".to_string()
        }
        else
        {
            "Tree Importance (Regression)".to_string()
        }
    }
}
