use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::FeatureSelector;
use statrs::function::gamma::digamma;
use std::cell::RefCell;
use std::collections::HashSet;

pub struct MutualInformationSelector 
{
    top_k: usize,
    k_neighbors: usize,
    details_cache: RefCell<String>,
}

impl MutualInformationSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            top_k: 10,
            k_neighbors: 3,
            details_cache: RefCell::new(String::new()),
        }
    }

    /// KSG odhad vzajomnej informacie medzi dvoma spojitymi premennymi
    fn estimate_mi_ksg(x_col: &[f64], y: &[f64], k: usize) -> f64 
    {
        let n = x_col.len();
        if n <= k 
        { 
            return 0.0; 
        }

        let mut nx_vec = vec![0usize; n];
        let mut ny_vec = vec![0usize; n];

        for i in 0..n 
        {
            let mut distances = Vec::with_capacity(n - 1);
            for j in 0..n 
            {
                if i == j { continue; }
                let dx = (x_col[i] - x_col[j]).abs();
                let dy = (y[i] - y[j]).abs();
                distances.push(dx.max(dy));
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let epsilon = distances[k - 1];

            let mut nx = 0usize;
            let mut ny = 0usize;
            for j in 0..n 
            {
                if i == j { continue; }
                if (x_col[i] - x_col[j]).abs() < epsilon { nx += 1; }
                if (y[i] - y[j]).abs() < epsilon { ny += 1; }
            }
            nx_vec[i] = nx;
            ny_vec[i] = ny;
        }

        let psi_k = digamma(k as f64);
        let psi_n = digamma(n as f64);
        let mut mean_psi = 0.0;
        for i in 0..n 
        {
            mean_psi += digamma((nx_vec[i] + 1) as f64) + digamma((ny_vec[i] + 1) as f64);
        }
        mean_psi /= n as f64;
        (psi_k - mean_psi + psi_n).max(0.0)
    }

    /// Relative color for MI relevance: intensity proportional to value/max
    fn rel_mi_color(mi: f64, max_mi: f64) -> String {
        let t = if max_mi > 1e-12 { (mi / max_mi).min(1.0) } else { 0.0 };
        format!("rgba(52,152,219,{})", 0.05 + t * 0.55)
    }

    /// Relative color for inter-feature MI (redundancy): red intensity proportional to value/max
    fn rel_redundancy_color(mi: f64, max_mi: f64) -> String {
        let t = if max_mi > 1e-12 { (mi / max_mi).min(1.0) } else { 0.0 };
        format!("rgba(231,76,60,{})", 0.05 + t * 0.55)
    }
}

impl FeatureSelector for MutualInformationSelector 
{
    fn get_name(&self) -> &str 
    {
        "Mutual Information (mRMR)"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["num_features", "k_neighbors"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        match key 
        {
            "num_features" | "top_k" => self.top_k = value.parse().map_err(|_| "Invalid num_features".to_string())?,
            "k_neighbors" => self.k_neighbors = value.parse().map_err(|_| "Invalid k_neighbors".to_string())?,
            _ => return Err(format!("Param not found: {}. Supported: num_features, k_neighbors", key)),
        }
        Ok(())
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> 
    {
        let (rows, cols) = x.shape();
        let effective_k = self.top_k.min(cols);
        let k_nn = self.k_neighbors;

        // ─── Extract all columns ───
        let columns: Vec<Vec<f64>> = (0..cols)
            .map(|j| (0..rows).map(|i| *x.get((i, j))).collect())
            .collect();

        // ─── 1. Relevance: MI(feature_i, target) for each feature ───
        let relevance: Vec<f64> = (0..cols)
            .map(|j| Self::estimate_mi_ksg(&columns[j], y, k_nn))
            .collect();

        // ─── 2. Inter-feature MI matrix (symmetric) ───
        let mut mi_matrix = vec![vec![0.0f64; cols]; cols];
        for i in 0..cols {
            for j in (i+1)..cols {
                let mi_ij = Self::estimate_mi_ksg(&columns[i], &columns[j], k_nn);
                mi_matrix[i][j] = mi_ij;
                mi_matrix[j][i] = mi_ij;
            }
        }

        // ─── 3. mRMR greedy selection ───
        // mRMR score = relevance(f) - (1/|S|) * sum_{s in S} MI(f, s)
        // First feature: highest relevance
        let mut selected: Vec<usize> = Vec::with_capacity(effective_k);
        let mut remaining: HashSet<usize> = (0..cols).collect();
        let mut mrmr_scores: Vec<(usize, f64, f64, f64)> = Vec::with_capacity(cols); // (idx, mrmr, relevance, redundancy)

        // Select first feature (max relevance)
        let first = (0..cols).max_by(|&a, &b| relevance[a].partial_cmp(&relevance[b]).unwrap()).unwrap_or(0);
        selected.push(first);
        remaining.remove(&first);
        mrmr_scores.push((first, relevance[first], relevance[first], 0.0));

        // Greedy: select remaining features one at a time
        while selected.len() < effective_k && !remaining.is_empty() {
            let mut best_idx = 0;
            let mut best_mrmr = f64::NEG_INFINITY;
            let mut best_rel = 0.0;
            let mut best_red = 0.0;

            for &candidate in &remaining {
                let rel = relevance[candidate];
                // Average MI with already selected features (redundancy)
                let redundancy: f64 = selected.iter()
                    .map(|&s| mi_matrix[candidate][s])
                    .sum::<f64>() / selected.len() as f64;
                let mrmr = rel - redundancy;

                if mrmr > best_mrmr {
                    best_mrmr = mrmr;
                    best_idx = candidate;
                    best_rel = rel;
                    best_red = redundancy;
                }
            }

            selected.push(best_idx);
            remaining.remove(&best_idx);
            mrmr_scores.push((best_idx, best_mrmr, best_rel, best_red));
        }
        
        // ─── Build detailed HTML ───
        let mut html = String::from("<div style='margin:10px 0;'>");
        html.push_str("<h4>Mutual Information - mRMR (minimum Redundancy Maximum Relevance)</h4>");
        html.push_str(&format!("<p>K neighbors (KSG): <b>{}</b> | Vybranych: <b>{}</b> z <b>{}</b></p>", 
            k_nn, effective_k, cols));

        // Explanation
        html.push_str("<p style='font-size:11px;color:#6c757d;margin-bottom:8px;'>\
            <b>mRMR</b> vyberá features s vysokou <b>relevanciou</b> voci cielu (MI s targetom) \
            a zaroven nízkou <b>redundanciou</b> medzi sebou (MI medzi features).<br>\
            <b>mRMR skóre</b> = Relevancia - Priemerná redundancia s uz vybranymi features.<br>\
            Greedy: najprv sa vyberie feature s najvyssim MI voci targetu, potom iterativne dalsi s najvyssim mRMR.\
        </p>");

        // mRMR selection order table
        html.push_str("<h5 style='color:#495057;margin:15px 0 8px;border-bottom:1px solid #dee2e6;\
            padding-bottom:5px;'>Poradie vyberu (mRMR greedy)</h5>");
        html.push_str("<div style='overflow-x:auto;'>");
        html.push_str("<table style='border-collapse:collapse;font-size:12px;width:100%;'>");
        html.push_str("<thead><tr>");
        for h in &["Poradie", "Feature", "Relevancia MI(F,Y)", "Redundancia avg MI(F,S)", "mRMR Score"] {
            html.push_str(&format!(
                "<th style='padding:8px 6px;border:1px solid #dee2e6;background:#cc0000;color:white;\
                text-align:center;'>{}</th>", h));
        }
        html.push_str("</tr></thead><tbody>");

        // Compute max values for relative coloring
        let max_relevance = relevance.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
        let max_redundancy = mi_matrix.iter()
            .flat_map(|row| row.iter())
            .cloned().fold(0.0f64, f64::max).max(1e-12);
        let max_mrmr = mrmr_scores.iter().map(|&(_, m, _, _)| m).fold(f64::NEG_INFINITY, f64::max).max(1e-12);

        for (rank, &(idx, mrmr, rel, red)) in mrmr_scores.iter().enumerate() {
            let row_bg = "rgba(52,152,219,0.08)";
            let rel_bg = Self::rel_mi_color(rel, max_relevance);
            let red_bg = Self::rel_redundancy_color(red, max_redundancy);
            let mrmr_t = if max_mrmr > 1e-12 { (mrmr / max_mrmr).min(1.0).max(0.0) } else { 0.0 };
            let mrmr_color = format!("color:rgba(39,174,96,{});font-weight:bold;", 0.3 + mrmr_t * 0.7);

            html.push_str(&format!(
                "<tr style='background:{};'>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;'>#{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;font-weight:bold;text-align:center;'>F{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;{}'>{:.4}</td>\
                </tr>",
                row_bg, rank + 1, idx,
                rel_bg, rel,
                red_bg, red,
                mrmr_color, mrmr
            ));
        }
        html.push_str("</tbody></table></div>");

        // All features relevance ranking (including not selected)
        let mut all_scores: Vec<(usize, f64)> = (0..cols).map(|j| (j, relevance[j])).collect();
        all_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        html.push_str("<h5 style='color:#495057;margin:15px 0 8px;border-bottom:1px solid #dee2e6;\
            padding-bottom:5px;'>Relevancia vsetkych features MI(F, Target)</h5>");
        html.push_str("<div style='overflow-x:auto;'>");
        html.push_str("<table style='border-collapse:collapse;font-size:12px;width:100%;'>");
        html.push_str("<thead><tr>");
        for h in &["Poradie", "Premenná", "MI(F, Target)", "Stav"] {
            let bg = if *h == "Stav" { "#8b0000" } else { "#cc0000" };
            html.push_str(&format!(
                "<th style='padding:8px 6px;border:1px solid #dee2e6;background:{};color:white;\
                text-align:center;'>{}</th>", bg, h));
        }
        html.push_str("</tr></thead><tbody>");

        for (rank, &(idx, score)) in all_scores.iter().enumerate() {
            let is_selected = selected.contains(&idx);
            let row_bg = if is_selected { "rgba(52,152,219,0.08)" } else { "rgba(189,195,199,0.08)" };
            let status = if is_selected { "<span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "<span style='color:#6c757d;'>✗</span>" };
            let mi_bg = Self::rel_mi_color(score, max_relevance);

            html.push_str(&format!(
                "<tr style='background:{};'>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;'>#{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;font-weight:bold;text-align:center;'>F{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>\
                </tr>",
                row_bg, rank + 1, idx,
                mi_bg, score,
                status
            ));
        }
        html.push_str("</tbody></table></div>");

        // Inter-feature MI matrix
        if cols <= 15 {
            html.push_str("<h5 style='color:#495057;margin:20px 0 8px;border-bottom:1px solid #dee2e6;\
                padding-bottom:5px;'>Matica MI medzi features (redundancia)</h5>");
            html.push_str("<p style='font-size:11px;color:#6c757d;margin-bottom:6px;'>\
                Vysoke MI medzi features = redundancia (nesú podobnú informáciu). \
                mRMR penalizuje výber redundantných features.</p>");
            html.push_str("<div style='overflow-x:auto;'>");
            html.push_str("<table style='border-collapse:collapse;font-size:11px;'>");
            
            // Header row
            html.push_str("<tr><th style='padding:4px;border:1px solid #ddd;'></th>");
            for j in 0..cols {
                html.push_str(&format!(
                    "<th style='padding:4px;border:1px solid #ddd;background:#cc0000;color:white;'>F{}{}</th>", j, if selected.contains(&j) { " <span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "" }));
            }
            html.push_str("</tr>");
            
            // Matrix rows
            for i in 0..cols {
                html.push_str(&format!(
                    "<tr><th style='padding:4px;border:1px solid #ddd;background:#cc0000;color:white;'>F{}{}</th>", i, if selected.contains(&i) { " <span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "" }));
                for j in 0..cols {
                    if i == j {
                        html.push_str("<td style='padding:4px;border:1px solid #ddd;text-align:center;\
                            background:#e0e0e0;'>-</td>");
                    } else {
                        let mi_val = mi_matrix[i][j];
                        let color = Self::rel_redundancy_color(mi_val, max_redundancy);
                        html.push_str(&format!(
                            "<td style='padding:4px;border:1px solid #ddd;text-align:center;\
                            background:{};'>{:.3}</td>", color, mi_val));
                    }
                }
                html.push_str("</tr>");
            }
            html.push_str("</table></div>");
        }

        // Legend
        html.push_str("<div style='margin-top:8px;font-size:11px;display:flex;gap:8px;flex-wrap:wrap;'>");
        html.push_str("<span style='background:rgba(52,152,219,0.15);padding:2px 8px;'>Relevancia: </span>");
        html.push_str("<span style='background:rgba(52,152,219,0.2);padding:2px 8px;'>nižšia</span>");
        html.push_str("<span style='background:rgba(52,152,219,0.6);padding:2px 8px;color:white;'>vyššia</span>");
        html.push_str("<span style='padding:2px 4px;'>|</span>");
        html.push_str("<span style='background:rgba(231,76,60,0.15);padding:2px 8px;'>Redundancia: </span>");
        html.push_str("<span style='background:rgba(231,76,60,0.2);padding:2px 8px;'>nižšia</span>");
        html.push_str("<span style='background:rgba(231,76,60,0.6);padding:2px 8px;color:white;'>vyššia</span>");
        html.push_str("<span style='padding:2px 4px;'>|</span>");
        html.push_str("<span style='color:#28a745;font-weight:bold;padding:2px 8px;'>✓ Vybraná</span>");
        html.push_str("<span style='color:#6c757d;font-weight:bold;padding:2px 8px;'>✗ Nevyradená</span>");
        html.push_str("<p style='font-size:10px;color:#999;margin-top:4px;'>Intenzita farieb je relativna k hodnotam v aktualnom datasete (hodnota/maximum).</p>");
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
    
    fn get_feature_scores(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let (rows, cols) = x.shape();
        let columns: Vec<Vec<f64>> = (0..cols)
            .map(|j| (0..rows).map(|i| *x.get((i, j))).collect())
            .collect();
        let scores: Vec<(usize, f64)> = (0..cols)
            .map(|j| (j, Self::estimate_mi_ksg(&columns[j], y, self.k_neighbors)))
            .collect();
        Some(scores)
    }
    
    fn get_metric_name(&self) -> &str {
        "MI (mRMR)"
    }
    
    fn get_selection_details(&self) -> String {
        self.details_cache.borrow().clone()
    }
}

impl MutualInformationSelector {
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
