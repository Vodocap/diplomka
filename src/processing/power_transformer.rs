use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ColumnType};

/// Power Transformer - Box-Cox alebo Yeo-Johnson transformácia
/// Transformuje dáta na normálne rozdelenie
pub struct PowerTransformer {
    method: TransformMethod,
    lambdas: Option<Vec<f64>>,
}

#[derive(Clone)]
#[allow(dead_code)]
pub enum TransformMethod {
    YeoJohnson,  // Funguje aj pre negatívne hodnoty
    BoxCox,      // Len pre pozitívne hodnoty
}

impl PowerTransformer {
    pub fn new(method: TransformMethod) -> Self {
        Self {
            method,
            lambdas: None,
        }
    }

    pub fn yeo_johnson() -> Self {
        Self::new(TransformMethod::YeoJohnson)
    }

    #[allow(dead_code)]
    pub fn box_cox() -> Self {
        Self::new(TransformMethod::BoxCox)
    }

    /// Estimates the optimal lambda via a simple grid search over candidate values.
    /// Uses log-likelihood as the criterion (normal distribution of transformed data).
    fn estimate_lambda(&self, values: &[f64]) -> f64 {
        let candidates = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0];
        let n = values.len() as f64;
        if n < 2.0 { return 0.0; }

        let mut best_lambda = 0.0;
        let mut best_ll = f64::NEG_INFINITY;

        for &lam in &candidates {
            let transformed: Vec<f64> = values.iter().map(|&x| {
                match self.method {
                    TransformMethod::YeoJohnson => self.yeo_johnson_transform(x, lam),
                    TransformMethod::BoxCox => self.box_cox_transform(x, lam),
                }
            }).collect();

            // Skip if any NaN/Inf produced
            if transformed.iter().any(|v| !v.is_finite()) { continue; }

            let mean = transformed.iter().sum::<f64>() / n;
            let var = transformed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            if var < 1e-15 { continue; }

            // Approximate log-likelihood ∝ -n/2 * ln(var)
            let ll = -n / 2.0 * var.ln();
            if ll > best_ll {
                best_ll = ll;
                best_lambda = lam;
            }
        }

        best_lambda
    }

    fn yeo_johnson_transform(&self, x: f64, lambda: f64) -> f64 {
        let eps = 1e-8;
        
        if x >= 0.0 {
            if lambda.abs() < eps {
                x.ln()
            } else {
                ((x + 1.0).powf(lambda) - 1.0) / lambda
            }
        } else {
            if (lambda - 2.0).abs() < eps {
                (-x).ln()
            } else {
                -((-x + 1.0).powf(2.0 - lambda) - 1.0) / (2.0 - lambda)
            }
        }
    }

    fn box_cox_transform(&self, x: f64, lambda: f64) -> f64 {
        let eps = 1e-8;
        
        if x <= 0.0 {
            return 0.0; // Box-Cox vyžaduje pozitívne hodnoty
        }
        
        if lambda.abs() < eps {
            x.ln()
        } else {
            (x.powf(lambda) - 1.0) / lambda
        }
    }
}

impl DataProcessor for PowerTransformer {
    fn get_name(&self) -> &str {
        match self.method {
            TransformMethod::YeoJohnson => "Yeo-Johnson Transformer",
            TransformMethod::BoxCox => "Box-Cox Transformer",
        }
    }

    fn fit(&mut self, data: &DenseMatrix<f64>) {
        let (rows, cols) = data.shape();
        let mut lambdas = Vec::new();

        for j in 0..cols {
            let col: Vec<f64> = (0..rows).map(|i| *data.get((i, j))).collect();
            let lambda = self.estimate_lambda(&col);
            lambdas.push(lambda);
        }

        self.lambdas = Some(lambdas);
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        if let Some(ref lambdas) = self.lambdas {
            for j in 0..cols.min(lambdas.len()) {
                let lambda = lambdas[j];
                for i in 0..rows {
                    let val = *data.get((i, j));
                    let transformed = match self.method {
                        TransformMethod::YeoJohnson => self.yeo_johnson_transform(val, lambda),
                        TransformMethod::BoxCox => self.box_cox_transform(val, lambda),
                    };
                    result.set((i, j), transformed);
                }
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String> {
        Err("PowerTransformer parameters cannot be changed after initialization".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Numeric])
    }
}
