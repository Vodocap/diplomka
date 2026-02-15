use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::FeatureSelector;
use std::cell::RefCell;

pub struct VarianceSelector 
{
    threshold: f64,
    details_cache: RefCell<String>,
}

impl VarianceSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            threshold: 0.01,
            details_cache: RefCell::new(String::new()),
        }
    }
}

impl FeatureSelector for VarianceSelector 
{
    fn get_name(&self) -> &str 
    {
        "Variance Threshold Selector"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["threshold"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        match key 
        {
            "threshold" => 
            {
                self.threshold = value.parse().map_err(|_| "Invalid threshold".to_string())?;
                Ok(())
            }
            _ => Err("Unknown parameter".into()),
        }
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, _y: &[f64]) -> Vec<usize> 
    {
        let shape = x.shape();
        let mut selected = Vec::new();
        let mut all_scores: Vec<(usize, f64, bool)> = Vec::new();

        for j in 0..shape.1 
        {
            let col: Vec<f64> = (0..shape.0).map(|i| *x.get((i, j))).collect();
            let mean: f64 = col.iter().sum::<f64>() / shape.0 as f64;
            let variance: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / shape.0 as f64;
            let sel = variance > self.threshold;
            all_scores.push((j, variance, sel));
            if sel { selected.push(j); }
        }
        
        // Cache details
        let mut html = String::from("<div style='margin:10px 0;'>");
        html.push_str("<h4>Variance Threshold Selection</h4>");
        html.push_str(&format!("<p>Threshold: <b>{:.4}</b> | Vybraných: <b>{}/{}</b></p>", self.threshold, selected.len(), shape.1));
        html.push_str("<table style='border-collapse:collapse;font-size:12px;width:100%;'>");
        html.push_str("<tr><th style='padding:4px;border:1px solid #ddd;'>Feature</th><th style='padding:4px;border:1px solid #ddd;'>Variance</th><th style='padding:4px;border:1px solid #ddd;'>Status</th></tr>");
        for (idx, var, sel) in &all_scores {
            let bg = if *sel { "rgba(52,152,219,0.08)" } else { "rgba(189,195,199,0.08)" };
            let status = if *sel { "<span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "<span style='color:#6c757d;'>✗</span>" };
            html.push_str(&format!("<tr style='background:{}'><td style='padding:4px;border:1px solid #ddd;'>F{}</td><td style='padding:4px;border:1px solid #ddd;'>{:.4}</td><td style='padding:4px;border:1px solid #ddd;text-align:center;'>{}</td></tr>", bg, idx, var, status));
        }
        html.push_str("</table></div>");
        *self.details_cache.borrow_mut() = html;
        
        selected
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }
    
    fn get_feature_scores(&self, x: &DenseMatrix<f64>, _y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let shape = x.shape();
        let mut scores = Vec::new();

        for j in 0..shape.1 {
            let col: Vec<f64> = (0..shape.0).map(|i| *x.get((i, j))).collect();
            let mean: f64 = col.iter().sum::<f64>() / shape.0 as f64;
            let variance: f64 = col.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / shape.0 as f64;
            scores.push((j, variance));
        }
        Some(scores)
    }
    
    fn get_metric_name(&self) -> &str {
        "Variance"
    }
    
    fn get_selection_details(&self) -> String {
        self.details_cache.borrow().clone()
    }
}

impl VarianceSelector {
    fn extract_columns(&self, x: &DenseMatrix<f64>, indices: &[usize]) -> DenseMatrix<f64> {
        let shape = x.shape();
        let rows = shape.0;
        let cols = indices.len();
        let mut data = vec![vec![0.0; cols]; rows];
        
        for (new_col, &old_col) in indices.iter().enumerate() {
            for row in 0..rows {
                data[row][new_col] = *x.get((row, old_col));
            }
        }
        
        DenseMatrix::from_2d_vec(&data).unwrap()
    }
}
