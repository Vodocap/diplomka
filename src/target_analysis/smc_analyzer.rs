use super::{TargetAnalyzer, TargetCandidate};
use std::collections::HashSet;

/// Analyzátor cieľovej premennej na základe Squared Multiple Correlation (SMC).
/// SMC_j = 1 - 1/(R⁻¹)_jj, kde R je korelačná matica.
/// Udáva, koľko variability premennej Xj je vysvetlené lineárnou kombináciou
/// všetkých ostatných premenných. Presne aproximuje R² z lineárnej regresie
/// bez nutnosti trénovania modelu.
pub struct SmcAnalyzer;

impl SmcAnalyzer {
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

    /// Invertuje maticu pomocou Gauss-Jordan eliminácie.
    /// Vracia None ak je matica singulárna.
    fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
        let n = mat.len();
        // Augmented matrix [mat | I]
        let mut aug: Vec<Vec<f64>> = mat.iter().enumerate().map(|(i, row)| {
            let mut r = row.clone();
            for j in 0..n {
                r.push(if i == j { 1.0 } else { 0.0 });
            }
            r
        }).collect();

        for col in 0..n {
            // Partial pivoting
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            if max_val < 1e-12 {
                return None; // Singular
            }
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

        // Extract inverse from right half
        let inv: Vec<Vec<f64>> = aug.iter().map(|row| {
            row[n..].to_vec()
        }).collect();
        Some(inv)
    }

    /// Regularizuje korelačnú maticu pridaním malej hodnoty na diagonálu
    /// aby bola invertovateľná aj pri multikolinearite.
    fn regularize_corr_matrix(corr: &mut Vec<Vec<f64>>, lambda: f64) {
        let n = corr.len();
        for i in 0..n {
            corr[i][i] += lambda;
        }
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

impl TargetAnalyzer for SmcAnalyzer {
    fn get_name(&self) -> &str {
        "smc"
    }

    fn get_description(&self) -> &str {
        "SMC (Squared Multiple Correlation) - koľko variability je vysvetlené ostatnými premennými"
    }

    fn get_metric_name(&self) -> &str {
        "SMC"
    }

    fn get_metric_explanation(&self) -> &str {
        "SMC (Squared Multiple Correlation) = 1 - 1/VIF, kde VIF = diagonálny prvok inverznej korelačnej matice. \
        Udáva, aký podiel variability premennej je vysvetlený lineárnou kombináciou všetkých ostatných premenných. \
        Rozsah: 0 (nezávislá premenná) - 1 (úplne predikovateľná). \
        SMC > 0.70: vynikajúci kandidát na cieľovú premennú. \
        0.40 - 0.70: dobrý kandidát. \
        SMC < 0.20: slabý kandidát - bude potrebný nelineárny model alebo iné dáta."
    }

    fn analyze(&self, columns: &[Vec<f64>], headers: &[String]) -> Vec<TargetCandidate> {
        let num_cols = columns.len();
        let n = if num_cols > 0 { columns[0].len() } else { return vec![]; };

        // Compute correlation matrix
        let mut corr_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        for i in 0..num_cols {
            corr_matrix[i][i] = 1.0;
            for j in (i+1)..num_cols {
                let c = Self::pearson_corr(&columns[i], &columns[j]);
                corr_matrix[i][j] = c;
                corr_matrix[j][i] = c;
            }
        }

        // Try to invert; if singular, regularize
        let inv = Self::invert_matrix(&corr_matrix).unwrap_or_else(|| {
            let mut reg = corr_matrix.clone();
            Self::regularize_corr_matrix(&mut reg, 0.01);
            Self::invert_matrix(&reg).unwrap_or_else(|| {
                // Fallback: identity-like
                vec![vec![1.0; num_cols]; num_cols]
            })
        });

        let mut candidates = Vec::new();
        for col_idx in 0..num_cols {
            let (unique_count, stype) = Self::classify_column(&columns[col_idx], n);

            let mean = columns[col_idx].iter().sum::<f64>() / n as f64;
            let variance = columns[col_idx].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

            // SMC_j = 1 - 1/(R^{-1})_{jj}
            let vif = inv[col_idx][col_idx];
            let smc = if vif > 1e-12 { 1.0 - (1.0 / vif) } else { 0.0 };
            let smc_clamped = smc.max(0.0).min(1.0);

            // Interpretácia
            let quality = if smc_clamped > 0.70 {
                "vynikajuci"
            } else if smc_clamped > 0.40 {
                "dobry"
            } else if smc_clamped > 0.20 {
                "slaby"
            } else {
                "nevhodny"
            };

            candidates.push(TargetCandidate {
                column_index: col_idx,
                column_name: headers[col_idx].clone(),
                score: (smc_clamped * 10000.0).round() / 10000.0,
                unique_values: unique_count,
                variance: (variance * 10000.0).round() / 10000.0,
                suggested_type: stype,
                extra_metrics: vec![
                    ("smc".to_string(), (smc_clamped * 10000.0).round() / 10000.0),
                    ("vif".to_string(), (vif * 100.0).round() / 100.0),
                    ("quality".to_string(), match quality {
                        "vynikajuci" => 3.0,
                        "dobry" => 2.0,
                        "slaby" => 1.0,
                        _ => 0.0,
                    }),
                ],
            });
        }
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        candidates
    }

    fn get_details_html(&self, columns: &[Vec<f64>], headers: &[String], candidates: &[TargetCandidate]) -> String {
        let num_cols = columns.len();
        let mut html = String::new();

        // SMC bar chart
        html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>SMC hodnoty premenných</h4>");
        html.push_str("<div style='max-width:600px;'>");
        for cand in candidates {
            let smc = cand.score;
            let vif = cand.extra_metrics.iter().find(|(k, _)| k == "vif").map(|(_, v)| *v).unwrap_or(1.0);
            let pct = (smc * 100.0).min(100.0);
            let color = if smc > 0.70 {
                "rgba(204,0,0,1.0)"
            } else if smc > 0.40 {
                "rgba(204,0,0,0.75)"
            } else if smc > 0.20 {
                "rgba(204,0,0,0.50)"
            } else {
                "rgba(204,0,0,0.30)"
            };
            let label = if smc > 0.70 {
                "vynikajuci"
            } else if smc > 0.40 {
                "dobry"
            } else if smc > 0.20 {
                "slaby"
            } else {
                "nevhodny"
            };
            html.push_str(&format!(
                "<div style='margin-bottom:8px;'>\
                <div style='display:flex;justify-content:space-between;font-size:12px;margin-bottom:2px;'>\
                <span style='font-weight:bold;'>{}</span>\
                <span style='color:#6c757d;'>SMC={:.4} VIF={:.2} ({})</span>\
                </div>\
                <div style='background:#eee;height:20px;position:relative;'>\
                <div style='background:{};height:100%;width:{:.1}%;'></div>\
                </div></div>",
                cand.column_name, smc, vif, label, color, pct
            ));
        }
        html.push_str("</div>");

        // Interpretácia
        html.push_str("<div style='margin-top:15px;font-size:12px;padding:12px;background:#f8f9fa;border:1px solid #dee2e6;'>");
        html.push_str("<strong>Interpretácia SMC pre návrh cieľovej premennej:</strong><br>");
        html.push_str("<span style='color:rgba(204,0,0,1.0);'>SMC &gt; 0.70</span>: Vynikajúci kandidát - silná lineárna väzba s ostatnými.<br>");
        html.push_str("<span style='color:rgba(204,0,0,0.75);'>0.40 - 0.70</span>: Dobrý kandidát - predikcia bude mať istú chybovosť.<br>");
        html.push_str("<span style='color:rgba(204,0,0,0.50);'>0.20 - 0.40</span>: Slabý kandidát - potrebný komplexnejší model.<br>");
        html.push_str("<span style='color:rgba(204,0,0,0.30);'>SMC &lt; 0.20</span>: Nevhodný - premenná je takmer nezávislá od ostatných.");
        html.push_str("</div>");

        // Correlation matrix if not too many columns
        if num_cols <= 15 {
            let mut corr_matrix = vec![vec![0.0f64; num_cols]; num_cols];
            for i in 0..num_cols {
                corr_matrix[i][i] = 1.0;
                for j in (i+1)..num_cols {
                    let c = Self::pearson_corr(&columns[i], &columns[j]);
                    corr_matrix[i][j] = c;
                    corr_matrix[j][i] = c;
                }
            }

            html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Korelačná matica</h4>");
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
        }

        html
    }
}
