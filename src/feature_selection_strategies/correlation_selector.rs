use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::FeatureSelector;
use std::collections::HashSet;
use std::cell::RefCell;

pub struct CorrelationSelector 
{
    threshold: f64,
    correlation_matrix: RefCell<Option<Vec<Vec<f64>>>>,
    target_correlations: RefCell<Option<Vec<f64>>>,
    details_cache: RefCell<String>,
}

impl CorrelationSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            threshold: 0.95,
            correlation_matrix: RefCell::new(None),
            target_correlations: RefCell::new(None),
            details_cache: RefCell::new(String::new()),
        }
    }

    fn pearson_correlation_vec(x: &[f64], y: &[f64]) -> f64 
    {
        let n = x.len() as f64;
        let mean_x: f64 = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().sum::<f64>() / n;
        let mut num = 0.0;
        let mut den_x = 0.0;
        let mut den_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) 
        {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            num += dx * dy;
            den_x += dx * dx;
            den_y += dy * dy;
        }

        let den = (den_x * den_y).sqrt();
        if den == 0.0 
        { 
            0.0 
        } 
        else 
        { 
            (num / den).abs() 
        }
    }
}

impl FeatureSelector for CorrelationSelector 
{
    fn get_name(&self) -> &str 
    {
        "Correlation Filter"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["threshold"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        if key == "threshold" 
        {
            self.threshold = value.parse().map_err(|_| "Invalid threshold")?;
            return Ok(());
        }
        Err("Param not found".into())
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> 
    {
        let shape = x.shape();
        let cols = shape.1;
        
        // Vypočítaj a ulož korelačnú maticu
        let mut corr_matrix = vec![vec![0.0; cols]; cols];
        let mut target_corr = Vec::new();
        
        // Krok 1: Vypočítaj koreláciu každého feature s cieľovou premennou
        let mut feature_target_corr = Vec::new();
        for i in 0..cols {
            let col_i: Vec<f64> = (0..shape.0).map(|row| *x.get((row, i))).collect();
            let corr = Self::pearson_correlation_vec(&col_i, y);
            feature_target_corr.push((i, corr));
            target_corr.push(corr);
        }
        
        // Vypočítaj korelácie medzi features
        for i in 0..cols {
            let col_i: Vec<f64> = (0..shape.0).map(|row| *x.get((row, i))).collect();
            for j in i..cols {
                if i == j {
                    corr_matrix[i][j] = 1.0;
                } else {
                    let col_j: Vec<f64> = (0..shape.0).map(|row| *x.get((row, j))).collect();
                    let corr = Self::pearson_correlation_vec(&col_i, &col_j);
                    corr_matrix[i][j] = corr;
                    corr_matrix[j][i] = corr; // Symetrická matica
                }
            }
        }
        
        // Ulož maticu pomocou RefCell
        *self.correlation_matrix.borrow_mut() = Some(corr_matrix.clone());
        *self.target_correlations.borrow_mut() = Some(target_corr);
        
        // Krok 2: Zoraď features podľa korelácie s cieľom (zostupne)
        feature_target_corr.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Krok 3: Greedy selection
        let mut selected = Vec::new();
        let mut dropped = HashSet::new();
        
        for (feature_idx, _target_corr) in feature_target_corr {
            if dropped.contains(&feature_idx) {
                continue;
            }
            
            selected.push(feature_idx);
            
            for j in 0..cols {
                if j == feature_idx || dropped.contains(&j) || selected.contains(&j) {
                    continue;
                }
                
                let corr = corr_matrix[feature_idx][j];
                
                if corr.abs() > self.threshold {
                    dropped.insert(j);
                }
            }
        }
        
        
        // Vráť features v pôvodnom poradí
        selected.sort();
        
        // Cache details - korelačná matica HTML
        let mut html = String::from("<div style='margin:10px 0;'>");
        html.push_str("<h4>Correlation Filter - Greedy Selection</h4>");
        html.push_str(&format!("<p>Threshold: <b>{:.2}</b> | Vybraných: <b>{}/{}</b></p>", self.threshold, selected.len(), cols));
        
        // Korelačná matica
        html.push_str("<div style='overflow-x:auto;'>");
        html.push_str("<table style='border-collapse:collapse;font-size:11px;'>");
        html.push_str("<tr><th style='padding:4px;border:1px solid #ddd;'></th>");
        for j in 0..cols {
            let sel_mark = if selected.contains(&j) { "[+]" } else { "" };
            html.push_str(&format!("<th style='padding:4px;border:1px solid #ddd;background:#f0f0f0;'>F{}{}</th>", j, sel_mark));
        }
        html.push_str("<th style='padding:4px;border:1px solid #ddd;background:#ffe6e6;'>Target</th></tr>");
        
        let target_corr = self.target_correlations.borrow();
        let tc = target_corr.as_ref().unwrap();
        for i in 0..cols {
            let sel_mark = if selected.contains(&i) { "[+]" } else { "" };
            html.push_str(&format!("<tr><th style='padding:4px;border:1px solid #ddd;background:#f0f0f0;'>F{}{}</th>", i, sel_mark));
            for j in 0..cols {
                let corr = corr_matrix[i][j];
                let abs_corr = corr.abs();
                let color = if i == j {
                    "#e0e0e0".to_string()
                } else if abs_corr > self.threshold {
                    format!("rgba(255,0,0,{})", 0.3 + abs_corr * 0.4)
                } else if abs_corr > 0.7 {
                    format!("rgba(255,165,0,{})", 0.2 + abs_corr * 0.3)
                } else {
                    format!("rgba(0,200,0,{})", 0.1 + (1.0 - abs_corr) * 0.2)
                };
                html.push_str(&format!("<td style='padding:4px;border:1px solid #ddd;text-align:center;background:{};'>{:.3}</td>", color, corr));
            }
            let abs_t = tc[i].abs();
            let tc_color = if abs_t > 0.7 { format!("rgba(0,128,255,{})", 0.3 + abs_t * 0.4) } else { "rgba(200,200,200,0.2)".to_string() };
            html.push_str(&format!("<td style='padding:4px;border:1px solid #ddd;text-align:center;background:{};font-weight:bold;'>{:.3}</td>", tc_color, tc[i]));
            html.push_str("</tr>");
        }
        html.push_str("</table></div>");
        
        // Legenda
        html.push_str("<div style='margin-top:8px;font-size:11px;'>");
        html.push_str("<span style='background:rgba(255,0,0,0.5);padding:2px 5px;margin-right:5px;'>Vysoká inter-feature korelácia (&gt; threshold)</span> ");
        html.push_str("<span style='background:rgba(0,128,255,0.5);padding:2px 5px;margin-right:5px;'>Vysoká korelácia s cieľom</span> ");
        html.push_str("<span style='background:rgba(0,200,0,0.3);padding:2px 5px;'>Nízka korelácia</span>");
        html.push_str("</div>");
        
        // Dropped features info
        if !dropped.is_empty() {
            html.push_str("<div style='margin-top:8px;font-size:12px;'><b>Odstránené features (multikolinearita):</b> ");
            let mut dropped_list: Vec<usize> = dropped.iter().cloned().collect();
            dropped_list.sort();
            for d in &dropped_list {
                html.push_str(&format!("F{} ", d));
            }
            html.push_str("</div>");
        }
        html.push_str("</div>");
        *self.details_cache.borrow_mut() = html;
        
        selected
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }
    
    fn get_feature_scores(&self, _x: &DenseMatrix<f64>, _y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let target_corr_opt = self.target_correlations.borrow().clone();
        if let Some(target) = target_corr_opt {
            let scores: Vec<(usize, f64)> = target.iter().enumerate().map(|(i, &corr)| (i, corr.abs())).collect();
            Some(scores)
        } else {
            None
        }
    }
    
    fn get_metric_name(&self) -> &str {
        "Target Correlation"
    }
    
    fn get_selection_details(&self) -> String {
        self.details_cache.borrow().clone()
    }
}

impl CorrelationSelector {
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
