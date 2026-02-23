use super::{TargetAnalyzer, TargetCandidate};
use statrs::function::gamma::digamma;
use std::collections::HashSet;
use std::cell::RefCell;

/// Analyzátor cieľovej premennej na základe Mutual Information (KSG estimátor).
/// Zachytáva aj nelineárne vzťahy medzi premennými, na rozdiel od korelácie.
pub struct MutualInformationAnalyzer {
    k_neighbors: usize,
    /// Cache pre MI maticu - aby sme ju nemuseli prerátavať v get_details_html()
    mi_cache: RefCell<Option<Vec<Vec<f64>>>>,
}

impl MutualInformationAnalyzer {
    pub fn new() -> Self {
        Self { 
            k_neighbors: 3,
            mi_cache: RefCell::new(None),
        }
    }

    /// Optimalizovaný KSG estimátor - použije prealokované buffery
    fn estimate_mi_ksg(x: &[f64], y: &[f64], k: usize) -> f64 {
        let n = x.len();
        if n <= k { return 0.0; }

        // Prealokujeme buffery pre zrýchlenie
        let mut distances = Vec::with_capacity(n - 1);
        let mut nx_vec = Vec::with_capacity(n);
        let mut ny_vec = Vec::with_capacity(n);

        for i in 0..n {
            distances.clear();
            
            // Optimalizácia: jednoduché absolútne vzdialenosti
            for j in 0..n {
                if i == j { continue; }
                let dx = (x[i] - x[j]).abs();
                let dy = (y[i] - y[j]).abs();
                distances.push(dx.max(dy));
            }
            
            // Partial sort - iba k-tý element (rýchlejšie ako full sort)
            distances.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let epsilon = distances[k - 1];

            let mut nx = 0usize;
            let mut ny = 0usize;
            
            // Počítame susedov v epsilon-okolí
            for j in 0..n {
                if i == j { continue; }
                if (x[i] - x[j]).abs() < epsilon { nx += 1; }
                if (y[i] - y[j]).abs() < epsilon { ny += 1; }
            }
            
            nx_vec.push(nx);
            ny_vec.push(ny);
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

    /// Vypočíta MI maticu pre všetky páry stĺpcov s cachovaním
    fn compute_mi_matrix(&self, columns: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let num_cols = columns.len();
        
        // Adaptive k_neighbors: pre veľké datasety použijeme menšie k
        let n = columns[0].len();
        let k = if n > 1000 { 2 } else if n > 500 { 3 } else { self.k_neighbors };
        
        let mut mi_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        
        // Počítame len hornú trojuholníkovú maticu (symetrická)
        for i in 0..num_cols {
            for j in (i+1)..num_cols {
                let mi = Self::estimate_mi_ksg(&columns[i], &columns[j], k);
                mi_matrix[i][j] = mi;
                mi_matrix[j][i] = mi; // Symetria
            }
        }
        
        mi_matrix
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
        "ΣMI"
    }

    fn get_metric_explanation(&self) -> &str {
        "Suma vzájomnej informácie (MI) s ostatnými premennými: Score_j = Σ MI(X_j, X_k). \
        MI meria množstvo informácie, ktorú jedna premenná poskytuje o druhej. \
        Na rozdiel od korelácie zachytáva aj nelineárne závislosti. \
        Vyššia hodnota = premenná zdieľa viac informácie s ostatnými. \
        Používa KSG estimátor (Kraskov-Stögbauer-Grassberger) pre spojité dáta."
    }

    fn analyze(&self, columns: &[Vec<f64>], headers: &[String]) -> Vec<TargetCandidate> {
        let num_cols = columns.len();
        let n = if num_cols > 0 { columns[0].len() } else { return vec![]; };

        // Vypočítame MI maticu a uložíme do cache
        let mi_matrix = self.compute_mi_matrix(columns);
        
        // Uložíme do cache pre get_details_html()
        *self.mi_cache.borrow_mut() = Some(mi_matrix.clone());

        let mut candidates = Vec::new();
        for col_idx in 0..num_cols {
            let (unique_count, stype) = Self::classify_column(&columns[col_idx], n);

            let mean = columns[col_idx].iter().sum::<f64>() / n as f64;
            let variance = columns[col_idx].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

            // Score_j = Σ MI(X_j, X_k)  (suma MI so všetkými ostatnými)
            let mut total_mi = 0.0f64;
            let mut max_mi = 0.0f64;
            for j in 0..num_cols {
                if j == col_idx { continue; }
                total_mi += mi_matrix[col_idx][j];
                if mi_matrix[col_idx][j] > max_mi { max_mi = mi_matrix[col_idx][j]; }
            }

            candidates.push(TargetCandidate {
                column_index: col_idx,
                column_name: headers[col_idx].clone(),
                score: (total_mi * 10000.0).round() / 10000.0,
                unique_values: unique_count,
                variance: (variance * 10000.0).round() / 10000.0,
                suggested_type: stype,
                extra_metrics: vec![
                    ("sum_mi".to_string(), (total_mi * 10000.0).round() / 10000.0),
                    ("max_mi".to_string(), (max_mi * 10000.0).round() / 10000.0),
                ],
            });
        }
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        candidates
    }

    fn get_details_html(&self, columns: &[Vec<f64>], headers: &[String], _candidates: &[TargetCandidate]) -> String {
        let num_cols = columns.len();
        if num_cols > 50 { return String::new(); }

        // Použijeme cache ak existuje, inak vypočítame znova
        let mi_matrix = if let Some(cached) = self.mi_cache.borrow().as_ref() {
            cached.clone()
        } else {
            self.compute_mi_matrix(columns)
        };

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
        html.push_str("<span style='background:rgba(52,152,219,0.7);padding:2px 8px;color:white;'>Vyššia</span>");
        html.push_str("<span style='background:rgba(46,204,113,0.4);padding:2px 8px;'>Stredná</span>");
        html.push_str("<span style='background:rgba(200,200,200,0.3);padding:2px 8px;'>Nižšia</span>");
        html.push_str("</div>");
        html
    }
}
