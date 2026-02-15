use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::FeatureSelector;
use std::cell::RefCell;

/// SMC (Squared Multiple Correlation) Feature Selector.
///
/// Pre každý feature vypočíta jeho príspevok k SMC targetu:
/// 1. Vypočíta korelačnú maticu všetkých features + target
/// 2. SMC targetu = 1 - 1/(R⁻¹)_{target,target} (koľko variability targetu vysvetľujú features)
/// 3. Pre každý feature vypočíta "drop" v SMC po jeho odstránení → importance
/// 4. Vyberie top-K features podľa SMC importance
pub struct SmcSelector {
    top_k: usize,
    details_cache: RefCell<String>,
}

impl SmcSelector {
    pub fn new() -> Self {
        Self {
            top_k: 10,
            details_cache: RefCell::new(String::new()),
        }
    }

    /// Pearson korelácia (signed)
    fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
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

    /// Invertuje maticu Gauss-Jordan elimináciou. Vracia None ak singulárna.
    fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = mat.len();
        let mut aug: Vec<Vec<f64>> = mat.iter().enumerate().map(|(i, row)| {
            let mut r = row.clone();
            for j in 0..n {
                r.push(if i == j { 1.0 } else { 0.0 });
            }
            r
        }).collect();

        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-12 { return None; }
            aug.swap(col, max_row);

            let pivot = aug[col][col];
            for j in 0..(2 * n) {
                aug[col][j] /= pivot;
            }
            for row in 0..n {
                if row == col { continue; }
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
        Some(aug.iter().map(|row| row[n..].to_vec()).collect())
    }

    /// Regularizuje korelačnú maticu (pridá lambda na diagonálu)
    fn regularize(corr: &mut [Vec<f64>], lambda: f64) {
        for i in 0..corr.len() {
            corr[i][i] += lambda;
        }
    }

    /// Vypočíta SMC targetu (posledný stĺpec) z korelačnej matice
    fn compute_target_smc(corr: &[Vec<f64>]) -> f64 {
        let n = corr.len();
        let inv = Self::invert_matrix(corr).unwrap_or_else(|| {
            let mut reg = corr.to_vec();
            Self::regularize(&mut reg, 0.01);
            Self::invert_matrix(&reg).unwrap_or_else(|| vec![vec![1.0; n]; n])
        });
        let target_idx = n - 1;
        let vif = inv[target_idx][target_idx];
        if vif > 1e-12 { (1.0 - 1.0 / vif).max(0.0).min(1.0) } else { 0.0 }
    }

    /// Vypočíta SMC targetu BEZ feature `exclude_idx`
    fn compute_target_smc_without(columns: &[Vec<f64>], y: &[f64], exclude_idx: usize) -> f64 {
        let num_features = columns.len();
        // Zostávajúce features (bez excluded) + target na konci
        let mut kept: Vec<usize> = (0..num_features).filter(|&i| i != exclude_idx).collect();
        let total = kept.len() + 1; // features + target

        let mut corr = vec![vec![0.0f64; total]; total];
        // Feature-feature korelácia
        for (ni, &i) in kept.iter().enumerate() {
            corr[ni][ni] = 1.0;
            for (nj, &j) in kept.iter().enumerate().skip(ni + 1) {
                let c = Self::pearson_corr(&columns[i], &columns[j]);
                corr[ni][nj] = c;
                corr[nj][ni] = c;
            }
        }
        // Feature-target korelácia
        let target_idx = total - 1;
        corr[target_idx][target_idx] = 1.0;
        for (ni, &i) in kept.iter().enumerate() {
            let c = Self::pearson_corr(&columns[i], y);
            corr[ni][target_idx] = c;
            corr[target_idx][ni] = c;
        }

        Self::compute_target_smc(&corr)
    }

    /// Farba pre SMC importance: intenzita úmerná hodnote/max
    fn importance_color(val: f64, max_val: f64) -> String {
        let t = if max_val > 1e-12 { (val / max_val).min(1.0).max(0.0) } else { 0.0 };
        format!("rgba(52,152,219,{})", 0.05 + t * 0.55)
    }
}

impl FeatureSelector for SmcSelector {
    fn get_name(&self) -> &str {
        "SMC Filter"
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["num_features"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "num_features" | "top_k" => {
                self.top_k = value.parse().map_err(|_| "Invalid num_features".to_string())?;
                Ok(())
            }
            _ => Err(format!("Param not found: {}. Supported: num_features", key)),
        }
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> {
        let (rows, cols) = x.shape();
        let effective_k = self.top_k.min(cols);

        // Extrahovať stĺpce
        let columns: Vec<Vec<f64>> = (0..cols)
            .map(|j| (0..rows).map(|i| *x.get((i, j))).collect())
            .collect();

        // 1. Korelačná matica všetkých features + target (target = posledný)
        let total = cols + 1;
        let mut corr_full = vec![vec![0.0f64; total]; total];
        for i in 0..cols {
            corr_full[i][i] = 1.0;
            for j in (i + 1)..cols {
                let c = Self::pearson_corr(&columns[i], &columns[j]);
                corr_full[i][j] = c;
                corr_full[j][i] = c;
            }
        }
        // Target korelácie
        let target_idx = cols;
        corr_full[target_idx][target_idx] = 1.0;
        let mut target_corrs = Vec::with_capacity(cols);
        for i in 0..cols {
            let c = Self::pearson_corr(&columns[i], y);
            corr_full[i][target_idx] = c;
            corr_full[target_idx][i] = c;
            target_corrs.push(c);
        }

        // 2. SMC targetu so všetkými features
        let baseline_smc = Self::compute_target_smc(&corr_full);

        // 3. Pre každý feature: SMC drop = baseline - SMC_without_feature
        let mut importance: Vec<(usize, f64, f64)> = Vec::with_capacity(cols);
        for i in 0..cols {
            let smc_without = Self::compute_target_smc_without(&columns, y, i);
            let drop = (baseline_smc - smc_without).max(0.0);
            importance.push((i, drop, smc_without));
        }

        // Sort by importance (drop) descending
        importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select top-K
        let selected: Vec<usize> = importance.iter().take(effective_k).map(|&(idx, _, _)| idx).collect();

        // 4. Compute SMC for each feature individually (how well other features predict it)
        let mut feature_smc: Vec<f64> = vec![0.0; cols];
        {
            let inv = Self::invert_matrix(&corr_full).unwrap_or_else(|| {
                let mut reg = corr_full.clone();
                Self::regularize(&mut reg, 0.01);
                Self::invert_matrix(&reg).unwrap_or_else(|| vec![vec![1.0; total]; total])
            });
            for i in 0..cols {
                let vif = inv[i][i];
                feature_smc[i] = if vif > 1e-12 { (1.0 - 1.0 / vif).max(0.0).min(1.0) } else { 0.0 };
            }
        }

        // ─── Build HTML ───
        let max_importance = importance.iter().map(|&(_, d, _)| d).fold(0.0f64, f64::max).max(1e-12);

        let mut html = String::from("<div style='margin:10px 0;'>");
        html.push_str("<h4>SMC (Squared Multiple Correlation) Feature Selection</h4>");
        html.push_str(&format!(
            "<p>Baseline SMC targetu: <b>{:.4}</b> | Vybraných: <b>{}</b> z <b>{}</b></p>",
            baseline_smc, effective_k, cols
        ));

        // Explanation
        html.push_str("<p style='font-size:11px;color:#6c757d;margin-bottom:8px;'>\
            <b>SMC</b> (Squared Multiple Correlation) meria, koľko variability premennej je vysvetlené \
            lineárnou kombináciou ostatných premenných.<br>\
            <b>SMC Importance</b> = pokles SMC targetu po odstránení daného feature. \
            Čím väčší pokles, tým dôležitejší feature pre predikciu targetu.<br>\
            SMC targetu = 1 − 1/VIF<sub>target</sub>, kde VIF pochádza z inverznej korelačnej matice.\
        </p>");

        // SMC baseline bar
        html.push_str(&format!(
            "<div style='margin-bottom:15px;'>\
            <div style='font-size:12px;margin-bottom:2px;'><strong>SMC targetu (všetky features)</strong>: {:.4}</div>\
            <div style='background:#eee;height:22px;position:relative;max-width:400px;'>\
            <div style='background:#cc0000;height:100%;width:{:.1}%;'></div>\
            </div></div>",
            baseline_smc, baseline_smc * 100.0
        ));

        // Feature importance table
        html.push_str("<h5 style='color:#495057;margin:15px 0 8px;border-bottom:1px solid #dee2e6;\
            padding-bottom:5px;'>Dôležitosť features (SMC Drop)</h5>");
        html.push_str("<div style='overflow-x:auto;'>");
        html.push_str("<table style='border-collapse:collapse;font-size:12px;width:100%;'>");
        html.push_str("<thead><tr>");
        for h in &["Poradie", "Feature", "|Corr s targetom|", "SMC feature", "SMC Importance (Drop)", "Stav"] {
            let bg = if *h == "Stav" { "#8b0000" } else { "#cc0000" };
            html.push_str(&format!(
                "<th style='padding:8px 6px;border:1px solid #dee2e6;background:{};color:white;\
                text-align:center;'>{}</th>", bg, h
            ));
        }
        html.push_str("</tr></thead><tbody>");

        for (rank, &(idx, drop, _smc_without)) in importance.iter().enumerate() {
            let is_selected = selected.contains(&idx);
            let row_bg = if is_selected { "rgba(52,152,219,0.08)" } else { "rgba(189,195,199,0.08)" };
            let status = if is_selected {
                "<span style='color:#28a745;font-weight:bold;'>✓</span>"
            } else {
                "<span style='color:#6c757d;'>✗</span>"
            };

            let corr_abs = target_corrs[idx].abs();
            let corr_bg = Self::importance_color(corr_abs, 1.0);
            let drop_bg = Self::importance_color(drop, max_importance);
            let smc_f = feature_smc[idx];

            html.push_str(&format!(
                "<tr style='background:{};'>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;'>#{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;font-weight:bold;text-align:center;'>F{}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;'>{:.4}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;background:{};font-weight:bold;'>{:.6}</td>\
                <td style='padding:6px;border:1px solid #dee2e6;text-align:center;font-weight:bold;'>{}</td>\
                </tr>",
                row_bg, rank + 1, idx,
                corr_bg, corr_abs,
                smc_f,
                drop_bg, drop,
                status
            ));
        }
        html.push_str("</tbody></table></div>");

        // SMC bar chart for all features
        html.push_str("<h5 style='color:#495057;margin:15px 0 8px;border-bottom:1px solid #dee2e6;\
            padding-bottom:5px;'>SMC jednotlivých features</h5>");
        html.push_str("<p style='font-size:11px;color:#6c757d;margin-bottom:6px;'>\
            SMC feature = koľko variability daného feature je vysvetlené ostatnými premennými + targetom. \
            Vysoké SMC → feature je lineárne závislý od ostatných (potenciálna multikolinearita).</p>");
        html.push_str("<div style='max-width:500px;'>");

        let mut smc_sorted: Vec<(usize, f64)> = (0..cols).map(|i| (i, feature_smc[i])).collect();
        smc_sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for &(idx, smc) in &smc_sorted {
            let is_sel = selected.contains(&idx);
            let pct = (smc * 100.0).min(100.0);
            let sel_marker = if is_sel { " ✓" } else { "" };
            html.push_str(&format!(
                "<div style='margin-bottom:6px;'>\
                <div style='display:flex;justify-content:space-between;font-size:11px;margin-bottom:1px;'>\
                <span style='font-weight:{};'>F{}{}</span>\
                <span style='color:#6c757d;'>SMC={:.4}</span>\
                </div>\
                <div style='background:#eee;height:16px;position:relative;'>\
                <div style='background:#cc0000;height:100%;width:{:.1}%;opacity:{};'></div>\
                </div></div>",
                if is_sel { "bold" } else { "normal" },
                idx, sel_marker,
                smc,
                pct, 0.4 + smc * 0.6
            ));
        }
        html.push_str("</div>");

        // Interpretácia
        html.push_str("<div style='margin-top:12px;font-size:11px;padding:10px;background:#f8f9fa;border:1px solid #dee2e6;'>");
        html.push_str("<strong>Interpretácia SMC Importance:</strong><br>");
        html.push_str("Veľký drop = feature je dôležitý pre predikciu targetu (bez neho SMC targetu výrazne klesne).<br>");
        html.push_str("Malý drop = feature je redundantný alebo neprispieva k predikcii targetu.<br>");
        html.push_str(&format!("<br><b>Baseline SMC targetu</b>: {:.4} — celkový R² lineárnej regresie targetu na všetky features.", baseline_smc));
        html.push_str("</div>");

        // Legend
        html.push_str("<div style='margin-top:8px;font-size:11px;display:flex;gap:8px;flex-wrap:wrap;'>");
        html.push_str("<span style='background:rgba(52,152,219,0.2);padding:2px 8px;'>Nižšia hodnota</span>");
        html.push_str("<span style='background:rgba(52,152,219,0.6);padding:2px 8px;color:white;'>Vyššia hodnota</span>");
        html.push_str("<span style='color:#28a745;font-weight:bold;padding:2px 8px;'>✓ Vybraná</span>");
        html.push_str("<span style='color:#6c757d;font-weight:bold;padding:2px 8px;'>✗ Nevybraná</span>");
        html.push_str("<p style='font-size:10px;color:#999;margin-top:4px;'>Intenzita farieb je relatívna k hodnotám v aktuálnom datasete.</p>");
        html.push_str("</div>");
        html.push_str("</div>");

        *self.details_cache.borrow_mut() = html;

        let mut result = selected;
        result.sort();
        result
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }

    fn get_feature_scores(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let (rows, cols) = x.shape();
        let columns: Vec<Vec<f64>> = (0..cols)
            .map(|j| (0..rows).map(|i| *x.get((i, j))).collect())
            .collect();

        // Build full correlation matrix
        let total = cols + 1;
        let mut corr = vec![vec![0.0f64; total]; total];
        for i in 0..cols {
            corr[i][i] = 1.0;
            for j in (i + 1)..cols {
                let c = Self::pearson_corr(&columns[i], &columns[j]);
                corr[i][j] = c;
                corr[j][i] = c;
            }
        }
        let target_idx = cols;
        corr[target_idx][target_idx] = 1.0;
        for i in 0..cols {
            let c = Self::pearson_corr(&columns[i], y);
            corr[i][target_idx] = c;
            corr[target_idx][i] = c;
        }

        let baseline_smc = Self::compute_target_smc(&corr);

        let scores: Vec<(usize, f64)> = (0..cols).map(|i| {
            let smc_without = Self::compute_target_smc_without(&columns, y, i);
            let drop = (baseline_smc - smc_without).max(0.0);
            (i, drop)
        }).collect();

        Some(scores)
    }

    fn get_metric_name(&self) -> &str {
        "SMC Importance (Drop)"
    }

    fn get_selection_details(&self) -> String {
        self.details_cache.borrow().clone()
    }
}

impl SmcSelector {
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
