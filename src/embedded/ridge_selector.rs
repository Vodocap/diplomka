use crate::embedded::EmbeddedFeatureSelector;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::ridge_regression::{RidgeRegression, RidgeRegressionParameters};

/// Ridge Regression (L2 regularization) feature importance
/// Používa absolútne hodnoty koeficientov z Ridge regresie
pub struct RidgeSelector {
    alpha: f64, // L2 regularization parameter
}

impl RidgeSelector {
    pub fn new() -> Self {
        RidgeSelector {
            alpha: 1.0, // Default regularization
        }
    }
    
    pub fn with_alpha(alpha: f64) -> Self {
        RidgeSelector { alpha }
    }
}

impl EmbeddedFeatureSelector for RidgeSelector {
    fn fit_and_rank(&mut self, x: &DenseMatrix<f64>, y: &[f64]) -> Result<Vec<(usize, f64)>, String> {
        // Natrénuj Ridge regression
        let params = RidgeRegressionParameters {
            alpha: self.alpha,
            normalize: true,
            solver: smartcore::linear::ridge_regression::RidgeRegressionSolverName::SVD,
        };
        
        let y_vec = y.to_vec(); // RidgeRegression potrebuje Vec, nie slice
        let model = RidgeRegression::fit(x, &y_vec, params)
            .map_err(|e| format!("Ridge regression fit error: {:?}", e))?;
        
        // Získaj koeficienty
        let coefficients = model.coefficients();
        
        // Vytvor feature scores (absolútne hodnoty koeficientov)
        let scores: Vec<f64> = coefficients.iter().map(|&c| c.abs()).collect();
        
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
        format!("Ridge (L2, α={:.2})", self.alpha)
    }
    
    fn supports_classification(&self) -> bool {
        false // Ridge je len pre regresiu
    }
    
    fn supports_regression(&self) -> bool {
        true
    }
}

impl Default for RidgeSelector {
    fn default() -> Self {
        Self::new()
    }
}
