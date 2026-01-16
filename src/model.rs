use wasm_bindgen::prelude::*;
use ndarray::Array1;

#[wasm_bindgen]
pub struct LinearRegression {
    weights: Vec<f64>,
}

#[wasm_bindgen]
impl LinearRegression {
    #[wasm_bindgen(constructor)]
    pub fn new() -> LinearRegression {
        LinearRegression { weights: vec![] }
    }

    // very simple fit: weights = sum of inputs / len (demo)
    pub fn fit(&mut self, x: Vec<f64>, y: Vec<f64>) {
        let n = x.len().max(1) as f64;
        let w = x.iter().zip(&y).map(|(a,b)| a*b).sum::<f64>() / n;
        self.weights = vec![w];
    }

    pub fn predict(&self, x: Vec<f64>) -> f64 {
        self.weights[0] * x[0]  // demo
    }
}
