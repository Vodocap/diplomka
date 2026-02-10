use super::{TargetAnalyzer, TargetCandidate};
use std::collections::HashSet;

/// Analyzátor cieľovej premennej na základe Pearsonovej korelácie.
/// Pre každý stĺpec vypočíta priemernú absolútnu koreláciu so všetkými
/// ostatnými stĺpcami. Vyššie skóre = stĺpec je lepšie predvídateľný
/// z ostatných stĺpcov (lineárne vzťahy).
pub struct CorrelationAnalyzer;

impl CorrelationAnalyzer {
    pub fn new() -> Self {
        Self
    }

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

    fn classify_column(values: &[f64], n: usize) -> (usize, String) {
        let mut uniq = HashSet::new();
        for &v in values { uniq.insert(v.to_bits()); }
        let unique_count = uniq.len();
        let is_cat = unique_count <= 10 || (unique_count as f64 / n as f64) < 0.05;
        let stype = if is_cat { "classification" } else { "regression" };
        (unique_count, stype.to_string())
    }
}

impl TargetAnalyzer for CorrelationAnalyzer {
    fn get_name(&self) -> &str {
        "correlation"
    }

    fn get_description(&self) -> &str {
        "Pearsonova korelácia - meria lineárne vzťahy medzi stĺpcami"
    }

    fn get_metric_name(&self) -> &str {
        "Priem. |korelácia|"
    }

    fn get_metric_explanation(&self) -> &str {
        "Priemerná absolútna Pearsonova korelácia tohto stĺpca so všetkými ostatnými stĺpcami. \
        Vyššia hodnota znamená, že stĺpec má silnejšie lineárne vzťahy s ostatnými premennými, \
        a teda je lepšie predvídateľný. Rozsah: 0 (žiadna korelácia) - 1 (perfektná korelácia). \
        Obmedzenie: zachytáva len lineárne vzťahy."
    }

    fn analyze(&self, columns: &[Vec<f64>], headers: &[String]) -> Vec<TargetCandidate> {
        let num_cols = columns.len();
        let n = if num_cols > 0 { columns[0].len() } else { return vec![]; };

        // Full correlation matrix
        let mut corr_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        for i in 0..num_cols {
            corr_matrix[i][i] = 1.0;
            for j in (i+1)..num_cols {
                let c = Self::pearson_corr(&columns[i], &columns[j]);
                corr_matrix[i][j] = c;
                corr_matrix[j][i] = c;
            }
        }

        let mut candidates = Vec::new();
        for col_idx in 0..num_cols {
            let (unique_count, stype) = Self::classify_column(&columns[col_idx], n);

            let mean = columns[col_idx].iter().sum::<f64>() / n as f64;
            let variance = columns[col_idx].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

            let mut total = 0.0f64;
            let mut max_c = 0.0f64;
            for j in 0..num_cols {
                if j == col_idx { continue; }
                let ac = corr_matrix[col_idx][j].abs();
                total += ac;
                if ac > max_c { max_c = ac; }
            }
            let avg = if num_cols > 1 { total / (num_cols - 1) as f64 } else { 0.0 };
            let score = avg * 100.0;

            candidates.push(TargetCandidate {
                column_index: col_idx,
                column_name: headers[col_idx].clone(),
                score: (score * 10.0).round() / 10.0,
                unique_values: unique_count,
                variance: (variance * 10000.0).round() / 10000.0,
                suggested_type: stype,
                extra_metrics: vec![
                    ("avg_correlation".to_string(), (avg * 10000.0).round() / 10000.0),
                    ("max_correlation".to_string(), (max_c * 10000.0).round() / 10000.0),
                ],
            });
        }
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        candidates
    }

    fn get_details_html(&self, columns: &[Vec<f64>], headers: &[String], _candidates: &[TargetCandidate]) -> String {
        let num_cols = columns.len();
        if num_cols > 15 { return String::new(); }

        // Correlation matrix
        let mut corr_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        for i in 0..num_cols {
            corr_matrix[i][i] = 1.0;
            for j in (i+1)..num_cols {
                let c = Self::pearson_corr(&columns[i], &columns[j]);
                corr_matrix[i][j] = c;
                corr_matrix[j][i] = c;
            }
        }

        let mut html = String::new();
        html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Korelačná matica všetkých stĺpcov</h4>");
        html.push_str("<div style='overflow-x:auto;'><table style='border-collapse:collapse;font-size:11px;'>");
        html.push_str("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;'></th>");
        for h in headers {
            html.push_str(&format!("<th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{}</th>", h));
        }
        html.push_str("</tr>");
        for i in 0..num_cols {
            html.push_str(&format!("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;white-space:nowrap;'>{}</th>", &headers[i]));
            for j in 0..num_cols {
                let c = corr_matrix[i][j];
                let ac = c.abs();
                let bg_color = if i == j {
                    "#e0e0e0".to_string()
                } else if ac > 0.7 {
                    format!("rgba(52,152,219,{})", 0.3 + ac * 0.5)
                } else if ac > 0.4 {
                    format!("rgba(46,204,113,{})", 0.2 + ac * 0.3)
                } else {
                    format!("rgba(200,200,200,{})", 0.1 + ac * 0.2)
                };
                html.push_str(&format!("<td style='padding:6px;border:1px solid #ddd;text-align:center;background:{};font-size:10px;'>{:.3}</td>", bg_color, c));
            }
            html.push_str("</tr>");
        }
        html.push_str("</table></div>");
        html.push_str("<div style='margin-top:8px;font-size:11px;display:flex;gap:10px;flex-wrap:wrap;'>");
        html.push_str("<span style='background:rgba(52,152,219,0.7);padding:2px 8px;color:white;'>Silná korelácia (|r| &gt; 0.7)</span>");
        html.push_str("<span style='background:rgba(46,204,113,0.4);padding:2px 8px;'>Stredná korelácia (|r| &gt; 0.4)</span>");
        html.push_str("<span style='background:rgba(200,200,200,0.3);padding:2px 8px;'>Slabá korelácia</span>");
        html.push_str("</div>");
        html
    }
}
