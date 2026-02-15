use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::FeatureSelector;
use std::collections::HashSet;
use std::cell::RefCell;

/// Info o jednej feature: Pearson, Spearman, normalita, vybrany metric
#[derive(Clone, Debug)]
struct FeatureCorrelation {
    index: usize,
    pearson: f64,
    spearman: f64,
    is_normal: bool,
    chosen_corr: f64,          // |Pearson| ak normal, |Spearman| ak nie
    chosen_label: &'static str, // "Pearson" alebo "Spearman"
}

pub struct CorrelationSelector 
{
    threshold: f64,
    correlation_matrix: RefCell<Option<Vec<Vec<f64>>>>,
    target_correlations: RefCell<Option<Vec<f64>>>,
    feature_correlations: RefCell<Vec<FeatureCorrelation>>,
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
            feature_correlations: RefCell::new(Vec::new()),
            details_cache: RefCell::new(String::new()),
        }
    }

    /// Pearson correlation (signed)
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 { return 0.0; }
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

    /// Spearman rank correlation = Pearson on ranks
    fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
        let ranks_x = Self::compute_ranks(x);
        let ranks_y = Self::compute_ranks(y);
        Self::pearson_correlation(&ranks_x, &ranks_y)
    }

    /// Compute ranks (average ranks for ties)
    fn compute_ranks(values: &[f64]) -> Vec<f64> {
        let n = values.len();
        let mut indexed: Vec<(usize, f64)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let mut ranks = vec![0.0; n];
        let mut i = 0;
        while i < n {
            let mut j = i;
            while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
                j += 1;
            }
            let avg_rank = (i + j) as f64 / 2.0 + 0.5;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            i = j;
        }
        ranks
    }

    /// Test normality: skewness + kurtosis heuristic.
    /// Normal distribution has skewness ~0, excess kurtosis ~0.
    /// Returns true if approximately normal.
    fn is_approximately_normal(values: &[f64]) -> bool {
        let n = values.len() as f64;
        if n < 8.0 { return false; }
        let mean = values.iter().sum::<f64>() / n;
        let mut m2 = 0.0;
        let mut m3 = 0.0;
        let mut m4 = 0.0;
        for &v in values {
            let d = v - mean;
            let d2 = d * d;
            m2 += d2;
            m3 += d2 * d;
            m4 += d2 * d2;
        }
        m2 /= n;
        m3 /= n;
        m4 /= n;

        if m2 < 1e-15 { return true; } // constant -> treat as normal

        let std_dev = m2.sqrt();
        let skewness = m3 / (std_dev * std_dev * std_dev);
        let kurtosis = m4 / (std_dev * std_dev * std_dev * std_dev) - 3.0; // excess kurtosis

        // Jarque-Bera-inspired: normal if |skew| < 1 and |kurt| < 2
        skewness.abs() < 1.0 && kurtosis.abs() < 2.0
    }

    /// Color for correlation values: linear opacity based on |r| (bounded [0,1])
    /// Correlation is naturally bounded, so direct mapping is valid.
    fn metric_color(abs_val: f64) -> String {
        format!("rgba(52,152,219,{})", 0.05 + abs_val * 0.55)
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
        
        // Extract all columns
        let columns: Vec<Vec<f64>> = (0..cols)
            .map(|i| (0..shape.0).map(|row| *x.get((row, i))).collect())
            .collect();

        // ─── Compute Pearson + Spearman for each feature vs target ───
        // ─── Auto-select based on normality of feature distribution ───
        let mut correlations: Vec<FeatureCorrelation> = Vec::with_capacity(cols);
        for i in 0..cols {
            let pearson = Self::pearson_correlation(&columns[i], y);
            let spearman = Self::spearman_correlation(&columns[i], y);
            let is_normal = Self::is_approximately_normal(&columns[i]);
            
            let (chosen_corr, chosen_label) = if is_normal {
                (pearson.abs(), "Pearson")
            } else {
                (spearman.abs(), "Spearman")
            };

            correlations.push(FeatureCorrelation {
                index: i,
                pearson,
                spearman,
                is_normal,
                chosen_corr,
                chosen_label,
            });
        }
        *self.feature_correlations.borrow_mut() = correlations.clone();
        
        // ─── Inter-feature correlation matrix (for multicollinearity removal) ───
        let mut corr_matrix = vec![vec![0.0f64; cols]; cols];
        let mut target_corr = Vec::with_capacity(cols);
        
        for i in 0..cols {
            corr_matrix[i][i] = 1.0;
            target_corr.push(correlations[i].chosen_corr);
            for j in (i+1)..cols {
                let c = Self::pearson_correlation(&columns[i], &columns[j]).abs();
                corr_matrix[i][j] = c;
                corr_matrix[j][i] = c;
            }
        }
        
        *self.correlation_matrix.borrow_mut() = Some(corr_matrix.clone());
        *self.target_correlations.borrow_mut() = Some(target_corr);
        
        // ─── Sort by chosen correlation for greedy ordering ───
        let mut feature_order: Vec<(usize, f64)> = correlations.iter()
            .map(|fc| (fc.index, fc.chosen_corr))
            .collect();
        feature_order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // ─── Greedy selection: remove multicollinear features ───
        let mut selected = Vec::new();
        let mut dropped = HashSet::new();
        
        for (feature_idx, _score) in &feature_order {
            if dropped.contains(feature_idx) { continue; }
            selected.push(*feature_idx);
            
            for j in 0..cols {
                if j == *feature_idx || dropped.contains(&j) || selected.contains(&j) { continue; }
                if corr_matrix[*feature_idx][j] > self.threshold {
                    dropped.insert(j);
                }
            }
        }
        
        selected.sort();
        
        // ─── Build detailed HTML ───
        let mut html = String::from("<div style='margin:10px 0;'>");
        html.push_str("<h4>Correlation Filter Selection</h4>");
        html.push_str(&format!("<p>Threshold multikolinearity: <b>{:.2}</b> | Vybranych: <b>{}/{}</b></p>", 
            self.threshold, selected.len(), cols));

        // Explanation
        html.push_str("<p style='font-size:11px;color:#6c757d;margin-bottom:8px;'>\
            Pre kazdu feature sa vypocita <b>Pearson</b> (linearna) aj <b>Spearman</b> (monotonna) korelacia s cielom.<br>\
            Ak ma premenna priblizne <b>normalne rozdelenie</b> (|skewness| &lt; 1, |excess kurtosis| &lt; 2), <b>pouzije sa Pearson</b>. Inak <b>Spearman</b>.<br>\
            Vybrany metric je <b>zvyrazneny</b> (podciarknuty). Multikolinearita sa odstranuje greedy algoritmom.\
        </p>");

        // Feature-target table
        html.push_str("<div style='overflow-x:auto;'>");
        html.push_str("<table style='border-collapse:collapse;font-size:12px;width:100%;'>");
        html.push_str("<thead><tr>");
        for h in &["Premenná", "Pearson", "Spearman", "Použitý", "Stav"] {
            let bg = if *h == "Stav" { "#8b0000" } else { "#cc0000" };
            html.push_str(&format!(
                "<th style='padding:8px 6px;border:1px solid #dee2e6;background:{};color:white;\
                text-align:center;'>{}</th>", bg, h));
        }
        html.push_str("</tr></thead><tbody>");
        
        // Sort by chosen_corr for display
        let mut sorted_corrs = correlations.clone();
        sorted_corrs.sort_by(|a, b| b.chosen_corr.partial_cmp(&a.chosen_corr).unwrap());

        for fc in &sorted_corrs {
            let is_selected = selected.contains(&fc.index);
            let row_bg = if is_selected { "rgba(52,152,219,0.08)" } else { "rgba(189,195,199,0.08)" };
            let status = if is_selected { "<span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "<span style='color:#6c757d;'>✗</span>" };

            // Highlight the chosen metric with bold + underline
            let pearson_style = if fc.chosen_label == "Pearson" {
                "font-weight:bold;text-decoration:underline;"
            } else { "" };
            let spearman_style = if fc.chosen_label == "Spearman" {
                "font-weight:bold;text-decoration:underline;"
            } else { "" };

            let pearson_bg = Self::metric_color(fc.pearson.abs());
            let spearman_bg = Self::metric_color(fc.spearman.abs());

            html.push_str(&format!(
                "<tr style='background:{};'>\
                <td style='padding:6px;border:1px solid #dee2e6;font-weight:bold;text-align:center;'>F{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};{}'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};{}'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>\
                </tr>",
                row_bg, fc.index,
                pearson_bg, pearson_style, fc.pearson,
                spearman_bg, spearman_style, fc.spearman,
                fc.chosen_label,
                status
            ));
        }
        html.push_str("</tbody></table></div>");
        
        // Legend
        html.push_str("<div style='margin-top:8px;font-size:11px;display:flex;gap:8px;flex-wrap:wrap;'>");
        html.push_str("<span style='background:rgba(52,152,219,0.2);padding:2px 8px;'>|r| nižšia</span>");
        html.push_str("<span style='background:rgba(52,152,219,0.6);padding:2px 8px;color:white;'>|r| vyššia</span>");
        html.push_str("<span style='padding:2px 8px;text-decoration:underline;font-weight:bold;'>Podčiarknutý = použitý metric</span>");
        html.push_str("<span style='color:#28a745;font-weight:bold;padding:2px 8px;'>✓ Vybraná</span>");
        html.push_str("<span style='color:#6c757d;font-weight:bold;padding:2px 8px;'>✗ Nevyradená</span>");
        html.push_str("<p style='font-size:10px;color:#999;margin-top:4px;'>Intenzita farieb = |r|. Korelacia je ohranicena [-1,1], priame mapovanie je korektne.</p>");
        html.push_str("</div>");

        // Inter-feature correlation matrix
        if cols <= 15 {
            html.push_str("<h5 style='color:#495057;margin:20px 0 8px;border-bottom:1px solid #dee2e6;\
                padding-bottom:5px;'>Korelacna matica medzi features (Pearson)</h5>");
            html.push_str("<p style='font-size:11px;color:#6c757d;margin-bottom:6px;'>\
                Cervena = inter-feature korelacia nad threshold (multikolinearita, feature odstraneny).</p>");
            html.push_str("<div style='overflow-x:auto;'>");
            html.push_str("<table style='border-collapse:collapse;font-size:11px;'>");
            html.push_str("<tr><th style='padding:4px;border:1px solid #ddd;'></th>");
            for j in 0..cols {
                let sel = if selected.contains(&j) { " <span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "" };
                html.push_str(&format!("<th style='padding:4px;border:1px solid #ddd;background:#cc0000;color:white;'>F{}{}</th>", j, sel));
            }
            html.push_str("</tr>");
            
            for i in 0..cols {
                let sel = if selected.contains(&i) { " <span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "" };
                html.push_str(&format!("<tr><th style='padding:4px;border:1px solid #ddd;background:#cc0000;color:white;'>F{}{}</th>", i, sel));
                for j in 0..cols {
                    let corr = corr_matrix[i][j];
                    let color = if i == j {
                        "#e0e0e0".to_string()
                    } else if corr > self.threshold {
                        // Over threshold = red (multicollinearity detected)
                        format!("rgba(231,76,60,{})", 0.2 + corr * 0.5)
                    } else {
                        // Linear blue intensity based on |r|
                        format!("rgba(52,152,219,{})", 0.05 + corr * 0.45)
                    };
                    html.push_str(&format!("<td style='padding:4px;border:1px solid #ddd;text-align:center;\
                        background:{};'>{:.3}</td>", color, corr));
                }
                html.push_str("</tr>");
            }
            html.push_str("</table></div>");
        }
        
        // Dropped features info
        if !dropped.is_empty() {
            html.push_str("<div style='margin-top:10px;font-size:12px;'><b>Odstranene features (multikolinearita):</b> ");
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
        let correlations = self.feature_correlations.borrow();
        if correlations.is_empty() { return None; }
        let scores: Vec<(usize, f64)> = correlations.iter()
            .map(|fc| (fc.index, fc.chosen_corr))
            .collect();
        Some(scores)
    }
    
    fn get_metric_name(&self) -> &str {
        "|Corr| (auto Pearson/Spearman)"
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

