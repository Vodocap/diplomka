use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::FeatureSelector;
use statrs::function::gamma::digamma;
use std::cell::RefCell;

/// Kombinovana analyza: Pearson + Spearman + MI s interpretaciou a odporucenim modelu.
/// Vysvetluje vztah kazdej feature voci cielu a doporucuje typ modelu.
#[derive(Clone, Debug)]
struct FeatureAnalysis {
    index: usize,
    pearson: f64,
    spearman: f64,
    mi: f64,
    is_normal: bool,
    recommendation: &'static str,
    model_hint: &'static str,
}

pub struct CombinedAnalysisSelector 
{
    top_k: usize,
    feature_analyses: RefCell<Vec<FeatureAnalysis>>,
    details_cache: RefCell<String>,
}

impl CombinedAnalysisSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            top_k: 10,
            feature_analyses: RefCell::new(Vec::new()),
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

    /// KSG estimator for Mutual Information
    fn estimate_mi_ksg(x: &[f64], y: &[f64], k: usize) -> f64 {
        let n = x.len();
        if n <= k { return 0.0; }

        let mut nx_vec = vec![0usize; n];
        let mut ny_vec = vec![0usize; n];

        for i in 0..n {
            let mut distances: Vec<f64> = Vec::with_capacity(n - 1);
            for j in 0..n {
                if i == j { continue; }
                let dx = (x[i] - x[j]).abs();
                let dy = (y[i] - y[j]).abs();
                distances.push(dx.max(dy));
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let epsilon = distances[k - 1];

            let mut nx = 0usize;
            let mut ny = 0usize;
            for j in 0..n {
                if i == j { continue; }
                if (x[i] - x[j]).abs() < epsilon { nx += 1; }
                if (y[i] - y[j]).abs() < epsilon { ny += 1; }
            }
            nx_vec[i] = nx;
            ny_vec[i] = ny;
        }

        let psi_k = digamma(k as f64);
        let psi_n = digamma(n as f64);
        let mut mean_psi = 0.0;
        for i in 0..n {
            mean_psi += digamma((nx_vec[i] + 1) as f64) + digamma((ny_vec[i] + 1) as f64);
        }
        mean_psi /= n as f64;
        (psi_k - mean_psi + psi_n).max(0.0)
    }

    /// Test normality: skewness + kurtosis heuristic
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
        if m2 < 1e-15 { return true; }
        let std_dev = m2.sqrt();
        let skewness = m3 / (std_dev * std_dev * std_dev);
        let kurtosis = m4 / (std_dev * std_dev * std_dev * std_dev) - 3.0;
        skewness.abs() < 1.0 && kurtosis.abs() < 2.0
    }

    /// Interpretuje kombinaciu Pearson, Spearman a MI
    /// MI prahy su relativne k max_mi v datasete
    fn interpret(pearson: f64, spearman: f64, mi: f64, max_mi: f64) -> (&'static str, &'static str) {
        let ap = pearson.abs();
        let asp = spearman.abs();
        let mi_rel = if max_mi > 1e-12 { mi / max_mi } else { 0.0 };

        // MI detects non-linear dependency invisible to correlation
        // Relative: MI is in top half of range, but both correlations are weak (Cohen 1988: < 0.3)
        if mi_rel > 0.5 && ap < 0.3 && asp < 0.3 {
            return ("Nelinearna zavislost (MI)", "Nelinearny model (strom, NN)");
        }

        // High Spearman, low Pearson -> monotonic but nonlinear
        // Cohen (1988): r >= 0.5 = large effect, r < 0.3 = small effect
        if asp >= 0.5 && ap < 0.3 {
            return ("Monotonny nelinearny vztah", "Nelinearny model (strom, NN)");
        }

        // Both high -> strong linear (Evans 1996: >= 0.6 strong)
        if ap >= 0.6 && asp >= 0.6 {
            return ("Silny linearny vztah", "Linearna regresia");
        }
        // Both moderate (Evans 1996: 0.4-0.59 moderate)
        if ap >= 0.4 && asp >= 0.4 {
            return ("Stredny linearny vztah", "Linearna regresia");
        }

        // Weak but present (Cohen 1988: r >= 0.1 = small effect)
        if ap >= 0.2 || asp >= 0.2 || mi_rel > 0.3 {
            return ("Slaby vztah", "Akykolvek model");
        }

        ("Zanedbatelny vztah", "Neurcite")
    }

    /// Composite score combining all three metrics
    fn composite_score(pearson: f64, spearman: f64, mi: f64, max_mi: f64) -> f64 {
        let linear = pearson.abs().max(spearman.abs());
        let mi_norm = if max_mi > 1e-12 { mi / max_mi } else { 0.0 };
        linear * 0.5 + mi_norm * 0.5
    }

    /// Color for correlation metric cells (Evans 1996: <0.4 weak, 0.4-0.59 moderate, >=0.6 strong)
    fn metric_color(abs_val: f64) -> String {
        // Linear opacity based on |r| (bounded [0,1])
        format!("rgba(52,152,219,{})", 0.05 + abs_val * 0.55)
    }

    /// Relative color for MI cells: intensity proportional to value/max in dataset
    fn rel_mi_color(mi: f64, max_mi: f64) -> String {
        let t = if max_mi > 1e-12 { (mi / max_mi).min(1.0) } else { 0.0 };
        format!("rgba(52,152,219,{})", 0.05 + t * 0.55)
    }
}

impl FeatureSelector for CombinedAnalysisSelector 
{
    fn get_name(&self) -> &str 
    {
        "Combined Analysis (Pearson + Spearman + MI)"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["num_features"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        match key {
            "num_features" | "top_k" => {
                self.top_k = value.parse().map_err(|_| "Invalid num_features".to_string())?;
                Ok(())
            }
            _ => Err(format!("Param not found: {}. Supported: num_features", key)),
        }
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> 
    {
        let shape = x.shape();
        let cols = shape.1;
        let effective_k = self.top_k.min(cols);
        
        // Extract all columns
        let columns: Vec<Vec<f64>> = (0..cols)
            .map(|i| (0..shape.0).map(|row| *x.get((row, i))).collect())
            .collect();

        // ─── Compute all three metrics for each feature vs target ───
        let mut analyses: Vec<FeatureAnalysis> = Vec::with_capacity(cols);
        for i in 0..cols {
            let pearson = Self::pearson_correlation(&columns[i], y);
            let spearman = Self::spearman_correlation(&columns[i], y);
            let mi = Self::estimate_mi_ksg(&columns[i], y, 3);
            let is_normal = Self::is_approximately_normal(&columns[i]);
            let (recommendation, model_hint) = Self::interpret(pearson, spearman, mi, 0.0); // preliminary, will recalc

            analyses.push(FeatureAnalysis {
                index: i,
                pearson,
                spearman,
                mi,
                is_normal,
                recommendation,
                model_hint,
            });
        }
        // ─── Recalculate interpretations with known max_mi ───
        let max_mi = analyses.iter().map(|a| a.mi).fold(0.0f64, f64::max);
        for a in analyses.iter_mut() {
            let (rec, hint) = Self::interpret(a.pearson, a.spearman, a.mi, max_mi);
            a.recommendation = rec;
            a.model_hint = hint;
        }
        *self.feature_analyses.borrow_mut() = analyses.clone();
        
        // ─── Rank by composite score and select top K ───
        let mut scored: Vec<(usize, f64)> = analyses.iter()
            .map(|a| (a.index, Self::composite_score(a.pearson, a.spearman, a.mi, max_mi)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected: Vec<usize> = scored.iter().take(effective_k).map(|(idx, _)| *idx).collect();
        
        // ─── Build detailed HTML ───
        let mut html = String::from("<div style='margin:10px 0;'>");
        html.push_str("<h4>Combined Analysis (Pearson + Spearman + MI)</h4>");
        html.push_str(&format!("<p>Vybranych: <b>{}</b> z <b>{}</b> (top K podla kompozitneho skore)</p>", 
            effective_k, cols));

        // Explanation from thesis advisor
        html.push_str("<p style='font-size:11px;color:#6c757d;margin-bottom:8px;'>\
            <b>Pearson</b> = linearna korelacia (pre normalne rozdelenie) | \
            <b>Spearman</b> = monotonny trend (aj nelinearny, pre nenormalne rozdelenie) | \
            <b>MI</b> = akakolvek statisticka zavislost (KSG estimator, k=3)<br>\
            Ak je <b>Spearman vysoky a Pearson nizky</b> = premenna je cenna, ale model musi byt <b>nelinearny</b>.<br>\
            Ak je <b>MI vysoke a obe korelacie nizke</b> = nejaka nelinearna zavislost (napr. kvadraticka).\
        </p>");

        // Combined analysis table
        html.push_str("<div style='overflow-x:auto;'>");
        html.push_str("<table style='border-collapse:collapse;font-size:12px;width:100%;'>");
        html.push_str("<thead><tr>");
        for h in &["Premenná", "Pearson", "Spearman", "MI", "Stav"] {
            let bg = if *h == "Stav" { "#8b0000" } else { "#cc0000" };
            html.push_str(&format!(
                "<th style='padding:8px 6px;border:1px solid #dee2e6;background:{};color:white;\
                text-align:center;font-size:11px;'>{}</th>", bg, h));
        }
        html.push_str("</tr></thead><tbody>");
        
        // Sort by composite score for display
        let mut sorted_analyses = analyses.clone();
        let max_mi_display = max_mi;
        sorted_analyses.sort_by(|a, b| {
            let sa = Self::composite_score(a.pearson, a.spearman, a.mi, max_mi_display);
            let sb = Self::composite_score(b.pearson, b.spearman, b.mi, max_mi_display);
            sb.partial_cmp(&sa).unwrap()
        });

        for a in &sorted_analyses {
            let is_selected = selected.contains(&a.index);
            let row_bg = if is_selected { "rgba(52,152,219,0.08)" } else { "rgba(189,195,199,0.08)" };
            let status = if is_selected { "<span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "<span style='color:#6c757d;'>✗</span>" };

            let pearson_bg = Self::metric_color(a.pearson.abs());
            let spearman_bg = Self::metric_color(a.spearman.abs());
            let mi_bg = Self::rel_mi_color(a.mi, max_mi);

            // Color the recommendation text
            let rec_color = match a.recommendation {
                "Silny linearny vztah" => "#8b0000",
                "Stredny linearny vztah" => "#cc0000",
                "Monotonny nelinearny vztah" | "Ciastocne nelinearny" => "#dc143c",
                "Nelinearna zavislost (MI)" => "#b22222",
                "Slaby vztah" => "#cd5c5c",
                _ => "#f08080",
            };

            html.push_str(&format!(
                "<tr style='background:{};'>\
                <td style='padding:6px;border:1px solid #dee2e6;font-weight:bold;text-align:center;'>F{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>\
                </tr>",
                row_bg, a.index,
                pearson_bg, a.pearson,
                spearman_bg, a.spearman,
                mi_bg, a.mi,
                status
            ));
        }
        html.push_str("</tbody></table></div>");
        
        // Color legend
        html.push_str("<div style='margin-top:8px;font-size:11px;display:flex;gap:8px;flex-wrap:wrap;'>");
        html.push_str("<span style='background:rgba(52,152,219,0.4);padding:2px 8px;'>|r| intenzita = veľkosť korelácie (Evans, 1996)</span>");
        html.push_str("<span style='background:rgba(52,152,219,0.4);padding:2px 8px;'>MI intenzita = relatívna k max v datasete</span>");
        html.push_str("<span style='color:#28a745;font-weight:bold;padding:2px 8px;'>✓ Vybraná</span>");
        html.push_str("<span style='color:#6c757d;font-weight:bold;padding:2px 8px;'>✗ Nevyradaná</span>");
        html.push_str("<p style='font-size:10px;color:#999;margin-top:4px;'>Korelačné prahy: Cohen (1988), Evans (1996). MI farbenie je relatívne k datasetu.</p>");
        html.push_str("</div>");

        html.push_str("</div>");
        *self.details_cache.borrow_mut() = html;
        
        let mut result = selected;
        result.sort();
        result
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }
    
    fn get_feature_scores(&self, _x: &DenseMatrix<f64>, _y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let analyses = self.feature_analyses.borrow();
        if analyses.is_empty() { return None; }
        let max_mi = analyses.iter().map(|a| a.mi).fold(0.0f64, f64::max);
        let scores: Vec<(usize, f64)> = analyses.iter()
            .map(|a| (a.index, Self::composite_score(a.pearson, a.spearman, a.mi, max_mi)))
            .collect();
        Some(scores)
    }
    
    fn get_metric_name(&self) -> &str {
        "Composite (Corr+MI)"
    }
    
    fn get_selection_details(&self) -> String {
        self.details_cache.borrow().clone()
    }
}

impl CombinedAnalysisSelector {
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
