use super::{TargetAnalyzer, TargetCandidate};
use statrs::function::gamma::digamma;
use std::collections::HashSet;

/// Analyzátor cieľovej premennej na základe Mutual Information (KSG estimátor).
/// Zachytáva aj nelineárne vzťahy medzi premennými, na rozdiel od korelácie.
pub struct MutualInformationAnalyzer {
    k_neighbors: usize,
}

impl MutualInformationAnalyzer {
    pub fn new() -> Self {
        Self { k_neighbors: 3 }
    }

    /// KSG estimátor vzájomnej informácie medzi dvoma spojitými premennými
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

    fn classify_column(values: &[f64], n: usize) -> (usize, String) {
        let mut uniq = HashSet::new();
        for &v in values { uniq.insert(v.to_bits()); }
        let unique_count = uniq.len();
        let is_cat = unique_count <= 10 || (unique_count as f64 / n as f64) < 0.05;
        let stype = if is_cat { "classification" } else { "regression" };
        (unique_count, stype.to_string())
    }
}

impl TargetAnalyzer for MutualInformationAnalyzer {
    fn get_name(&self) -> &str {
        "mutual_information"
    }

    fn get_description(&self) -> &str {
        "Mutual Information (KSG) - zachytáva aj nelineárne vzťahy medzi premennými"
    }

    fn get_metric_name(&self) -> &str {
        "Priem. MI"
    }

    fn get_metric_explanation(&self) -> &str {
        "Priemerná vzájomná informácia (Mutual Information) tohto stĺpca so všetkými ostatnými. \
        MI meria množstvo informácie, ktorú jedna premenná poskytuje o druhej. \
        Na rozdiel od korelácie zachytáva aj nelineárne závislosti. \
        Vyššia hodnota = stĺpec je lepšie predvídateľný (lineárne aj nelineárne). \
        Používa KSG estimátor (Kraskov-Stögbauer-Grassberger) pre spojité dáta."
    }

    fn analyze(&self, columns: &[Vec<f64>], headers: &[String]) -> Vec<TargetCandidate> {
        let num_cols = columns.len();
        let n = if num_cols > 0 { columns[0].len() } else { return vec![]; };

        // Pre-compute MI for all pairs
        let mut mi_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        for i in 0..num_cols {
            for j in (i+1)..num_cols {
                let mi = Self::estimate_mi_ksg(&columns[i], &columns[j], self.k_neighbors);
                mi_matrix[i][j] = mi;
                mi_matrix[j][i] = mi;
            }
        }

        let mut candidates = Vec::new();
        for col_idx in 0..num_cols {
            let (unique_count, stype) = Self::classify_column(&columns[col_idx], n);

            let mean = columns[col_idx].iter().sum::<f64>() / n as f64;
            let variance = columns[col_idx].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

            let mut total_mi = 0.0f64;
            let mut max_mi = 0.0f64;
            for j in 0..num_cols {
                if j == col_idx { continue; }
                total_mi += mi_matrix[col_idx][j];
                if mi_matrix[col_idx][j] > max_mi { max_mi = mi_matrix[col_idx][j]; }
            }
            let avg_mi = if num_cols > 1 { total_mi / (num_cols - 1) as f64 } else { 0.0 };

            // Normalize score to 0-100 range
            // MI doesn't have a fixed upper bound, so we use avg_mi directly
            // and later normalize relative to the best one
            candidates.push(TargetCandidate {
                column_index: col_idx,
                column_name: headers[col_idx].clone(),
                score: (avg_mi * 10000.0).round() / 10000.0,
                unique_values: unique_count,
                variance: (variance * 10000.0).round() / 10000.0,
                suggested_type: stype,
                extra_metrics: vec![
                    ("avg_mi".to_string(), (avg_mi * 10000.0).round() / 10000.0),
                    ("max_mi".to_string(), (max_mi * 10000.0).round() / 10000.0),
                ],
            });
        }
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Normalize scores: best = 100
        if let Some(max_score) = candidates.first().map(|c| c.score) {
            if max_score > 0.0 {
                for c in &mut candidates {
                    c.score = (c.score / max_score * 100.0 * 10.0).round() / 10.0;
                }
            }
        }
        candidates
    }

    fn get_details_html(&self, columns: &[Vec<f64>], headers: &[String], _candidates: &[TargetCandidate]) -> String {
        let num_cols = columns.len();
        if num_cols > 15 { return String::new(); }

        let mut mi_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        for i in 0..num_cols {
            for j in (i+1)..num_cols {
                let mi = Self::estimate_mi_ksg(&columns[i], &columns[j], self.k_neighbors);
                mi_matrix[i][j] = mi;
                mi_matrix[j][i] = mi;
            }
        }

        // Find max MI for color scaling
        let max_mi = mi_matrix.iter()
            .flat_map(|row| row.iter())
            .cloned()
            .fold(0.0f64, f64::max);
        let scale = if max_mi > 0.0 { max_mi } else { 1.0 };

        let mut html = String::new();
        html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Matica vzájomnej informácie (MI)</h4>");
        html.push_str("<div style='overflow-x:auto;'><table style='border-collapse:collapse;font-size:11px;'>");
        html.push_str("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;'></th>");
        for h in headers {
            html.push_str(&format!("<th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{}</th>", h));
        }
        html.push_str("</tr>");
        for i in 0..num_cols {
            html.push_str(&format!("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;white-space:nowrap;'>{}</th>", &headers[i]));
            for j in 0..num_cols {
                let mi = mi_matrix[i][j];
                let norm = mi / scale;
                let bg_color = if i == j {
                    "#e0e0e0".to_string()
                } else if norm > 0.7 {
                    format!("rgba(52,152,219,{})", 0.3 + norm * 0.5)
                } else if norm > 0.3 {
                    format!("rgba(46,204,113,{})", 0.2 + norm * 0.4)
                } else {
                    format!("rgba(200,200,200,{})", 0.1 + norm * 0.3)
                };
                html.push_str(&format!("<td style='padding:6px;border:1px solid #ddd;text-align:center;background:{};font-size:10px;'>{:.3}</td>", bg_color, mi));
            }
            html.push_str("</tr>");
        }
        html.push_str("</table></div>");
        html.push_str("<div style='margin-top:8px;font-size:11px;display:flex;gap:10px;flex-wrap:wrap;'>");
        html.push_str("<span style='background:rgba(52,152,219,0.7);padding:2px 8px;color:white;'>Silná MI</span>");
        html.push_str("<span style='background:rgba(46,204,113,0.4);padding:2px 8px;'>Stredná MI</span>");
        html.push_str("<span style='background:rgba(200,200,200,0.3);padding:2px 8px;'>Slabá MI</span>");
        html.push_str("</div>");
        html
    }
}
