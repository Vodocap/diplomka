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

    // Zjednodušená verzia - používame fixnú lambdu (môže sa optimalizovať)
    fn estimate_lambda(&self, _values: &[f64]) -> f64 {
        // Pre zjednodušenie používame lambda = 0.0 (log transform)
        // V production verzii by sa lambda optimalizovala pomocou max likelihood
        0.0
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
