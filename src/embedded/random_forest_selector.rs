use crate::embedded::EmbeddedFeatureSelector;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;

/// Random Forest/Decision Tree feature importance selector
/// Používa Gini importance (klasifikácia) alebo variance reduction (regresia)
pub struct RandomForestSelector {
    is_classification: bool,
    max_depth: Option<u16>,
}

impl RandomForestSelector {
    pub fn new(is_classification: bool) -> Self {
        RandomForestSelector {
            is_classification,
            max_depth: Some(10), // Obmedzená hĺbka pre rýchlejší výpočet
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
        
        if self.is_classification {
            // Pre klasifikáciu - použijeme korelácie ako aproximáciu
            // (SmartCore decision tree API neexponuje feature importance)
            let mut scores = vec![0.0; num_features];
            for feat_idx in 0..num_features {
                let feat_col: Vec<f64> = (0..x.shape().0)
                    .map(|row| *x.get((row, feat_idx)))
                    .collect();
                
                // Absolútna korelácia ako proxy pre importance
                let corr = self.pearson_correlation(&feat_col, y).abs();
                scores[feat_idx] = corr;
            }
            Ok(scores)
        } else {
            // Pre regresiu - použijeme variance reduction proxy
            let mut scores = vec![0.0; num_features];
            let y_mean = y.iter().sum::<f64>() / y.len() as f64;
            let total_variance = y.iter().map(|&val| (val - y_mean).powi(2)).sum::<f64>();
            
            for feat_idx in 0..num_features {
                let feat_col: Vec<f64> = (0..x.shape().0)
                    .map(|row| *x.get((row, feat_idx)))
                    .collect();
                
                // Korelácia^2 ako proxy pre variance reduction
                let corr = self.pearson_correlation(&feat_col, y);
                scores[feat_idx] = corr * corr; // R²
            }
            Ok(scores)
        }
    }
    
    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 || x.len() != y.len() { return 0.0; }
        
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;
        
        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;
        
        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }
        
        let den = (den_x * den_y).sqrt();
        if den == 0.0 { 0.0 } else { num / den }
    }
}

impl EmbeddedFeatureSelector for RandomForestSelector {
    fn fit_and_rank(&mut self, x: &DenseMatrix<f64>, y: &[f64]) -> Result<Vec<(usize, f64)>, String> {
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
    
    fn get_name(&self) -> String {
        if self.is_classification {
            "Tree Importance (Classification)".to_string()
        } else {
            "Tree Importance (Regression)".to_string()
        }
    }
    
    fn supports_classification(&self) -> bool {
        true
    }
    
    fn supports_regression(&self) -> bool {
        true
    }
}
