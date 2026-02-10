use super::{TargetAnalyzer, TargetCandidate};
use std::collections::{HashMap, HashSet};

/// Analyzátor cieľovej premennej na základe entropie a podmienenej entropie.
/// Meria, koľko informácie o cieľovom stĺpci sa dá získať z ostatných stĺpcov.
/// Score = priemerná redukcia entropie (Information Gain) naprieč všetkými
/// features: IG(target | feature) = H(target) - H(target | feature).
pub struct EntropyAnalyzer {
    num_bins: usize,
}

impl EntropyAnalyzer {
    pub fn new() -> Self {
        Self { num_bins: 10 }
    }

    /// Shannon entropy H(X) pre diskrétne hodnoty
    fn entropy_discrete(values: &[u64]) -> f64 {
        if values.is_empty() { return 0.0; }
        let mut counts: HashMap<u64, usize> = HashMap::new();
        for &v in values {
            *counts.entry(v).or_insert(0) += 1;
        }
        let n = values.len() as f64;
        counts.values().map(|&c| {
            let p = c as f64 / n;
            if p > 0.0 { -p * p.log2() } else { 0.0 }
        }).sum()
    }

    /// Bin spojité hodnoty do num_bins rovnomerných intervalov
    fn bin_values(values: &[f64], num_bins: usize) -> Vec<u64> {
        if values.is_empty() { return vec![]; }
        let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max - min;
        if range == 0.0 {
            return vec![0u64; values.len()];
        }
        values.iter().map(|&v| {
            let bin = ((v - min) / range * (num_bins as f64 - 1.0)).round() as u64;
            bin.min(num_bins as u64 - 1)
        }).collect()
    }

    /// Podmienená entropia H(Y|X) kde X a Y sú diskrétne (binned)
    fn conditional_entropy(target_bins: &[u64], feature_bins: &[u64]) -> f64 {
        let n = target_bins.len();
        if n == 0 { return 0.0; }

        // Group target values by feature bin
        let mut groups: HashMap<u64, Vec<u64>> = HashMap::new();
        for i in 0..n {
            groups.entry(feature_bins[i]).or_default().push(target_bins[i]);
        }

        let n_f64 = n as f64;
        groups.values().map(|subset| {
            let weight = subset.len() as f64 / n_f64;
            weight * Self::entropy_discrete(subset)
        }).sum()
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

impl TargetAnalyzer for EntropyAnalyzer {
    fn get_name(&self) -> &str {
        "entropy"
    }

    fn get_description(&self) -> &str {
        "Entropia a Information Gain - meria redukciu neistoty pri predikcii"
    }

    fn get_metric_name(&self) -> &str {
        "Priem. norm. IG"
    }

    fn get_metric_explanation(&self) -> &str {
        "Priemerný normalizovaný Information Gain: IG(target | feature) = H(target) - H(target | feature), \
        normalizovaný entropiou targetu. Udáva, aký podiel neistoty o cieľovej premennej sa dá \
        v priemere odstrániť pomocou jednej feature. Hodnota 0 = features neposkytujú žiadnu informáciu, \
        1 = features úplne predikujú target. Dáta sú automaticky binované do intervalov. \
        Výhoda: funguje aj pre nelineárne vzťahy. Nevýhoda: citlivý na počet binov."
    }

    fn analyze(&self, columns: &[Vec<f64>], headers: &[String]) -> Vec<TargetCandidate> {
        let num_cols = columns.len();
        let n = if num_cols > 0 { columns[0].len() } else { return vec![]; };

        // Pre-bin all columns
        let binned: Vec<Vec<u64>> = columns.iter()
            .map(|col| Self::bin_values(col, self.num_bins))
            .collect();

        let mut candidates = Vec::new();
        for col_idx in 0..num_cols {
            let (unique_count, stype) = Self::classify_column(&columns[col_idx], n);

            let mean = columns[col_idx].iter().sum::<f64>() / n as f64;
            let variance = columns[col_idx].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;

            let h_target = Self::entropy_discrete(&binned[col_idx]);

            // Compute average IG across all features
            let mut total_norm_ig = 0.0f64;
            let mut per_feature_ig: Vec<(String, f64)> = Vec::new();
            for j in 0..num_cols {
                if j == col_idx { continue; }
                let h_cond = Self::conditional_entropy(&binned[col_idx], &binned[j]);
                let ig = h_target - h_cond;
                let norm_ig = if h_target > 0.0 { ig / h_target } else { 0.0 };
                total_norm_ig += norm_ig;
                per_feature_ig.push((headers[j].clone(), (norm_ig * 10000.0).round() / 10000.0));
            }
            let avg_norm_ig = if num_cols > 1 { total_norm_ig / (num_cols - 1) as f64 } else { 0.0 };

            // Best feature IG for this target
            let max_ig = per_feature_ig.iter().map(|(_, v)| *v).fold(0.0f64, f64::max);

            let score = avg_norm_ig * 100.0;

            let mut extra = vec![
                ("entropy".to_string(), (h_target * 10000.0).round() / 10000.0),
                ("avg_norm_ig".to_string(), (avg_norm_ig * 10000.0).round() / 10000.0),
                ("max_norm_ig".to_string(), max_ig),
            ];
            // Add top-3 features
            per_feature_ig.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            for (i, (fname, ig)) in per_feature_ig.iter().take(3).enumerate() {
                extra.push((format!("top{}_feature", i + 1), *ig));
                // Store name separately — we can't put String in f64, but it's in the metric name
                let _ = fname; // name is implicit from sorted order
            }

            candidates.push(TargetCandidate {
                column_index: col_idx,
                column_name: headers[col_idx].clone(),
                score: (score * 10.0).round() / 10.0,
                unique_values: unique_count,
                variance: (variance * 10000.0).round() / 10000.0,
                suggested_type: stype,
                extra_metrics: extra,
            });
        }
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        candidates
    }

    fn get_details_html(&self, columns: &[Vec<f64>], headers: &[String], _candidates: &[TargetCandidate]) -> String {
        let num_cols = columns.len();
        if num_cols > 15 { return String::new(); }

        let binned: Vec<Vec<u64>> = columns.iter()
            .map(|col| Self::bin_values(col, self.num_bins))
            .collect();

        // IG matrix: ig_matrix[target][feature] = normalized IG
        let mut ig_matrix = vec![vec![0.0f64; num_cols]; num_cols];
        for t in 0..num_cols {
            let h_t = Self::entropy_discrete(&binned[t]);
            if h_t == 0.0 { continue; }
            for f in 0..num_cols {
                if t == f { continue; }
                let h_cond = Self::conditional_entropy(&binned[t], &binned[f]);
                ig_matrix[t][f] = ((h_t - h_cond) / h_t * 10000.0).round() / 10000.0;
            }
        }

        let mut html = String::new();
        html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Matica normalizovaného Information Gain</h4>");
        html.push_str("<p style='font-size:0.85em;color:#6c757d;margin-bottom:10px;'>Riadok = cieľová premenná, stĺpec = feature. Hodnota = IG(target|feature) / H(target)</p>");
        html.push_str("<div style='overflow-x:auto;'><table style='border-collapse:collapse;font-size:11px;'>");
        html.push_str("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;'>Target \\ Feature</th>");
        for h in headers {
            html.push_str(&format!("<th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{}</th>", h));
        }
        html.push_str("</tr>");
        for t in 0..num_cols {
            html.push_str(&format!("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;white-space:nowrap;'>{}</th>", &headers[t]));
            for f in 0..num_cols {
                let v = ig_matrix[t][f];
                let bg = if t == f {
                    "#e0e0e0".to_string()
                } else if v > 0.5 {
                    format!("rgba(52,152,219,{})", 0.3 + v * 0.5)
                } else if v > 0.2 {
                    format!("rgba(46,204,113,{})", 0.2 + v * 0.4)
                } else {
                    format!("rgba(200,200,200,{})", 0.1 + v * 0.3)
                };
                html.push_str(&format!("<td style='padding:6px;border:1px solid #ddd;text-align:center;background:{};font-size:10px;'>{:.3}</td>", bg, v));
            }
            html.push_str("</tr>");
        }
        html.push_str("</table></div>");

        // Entropy per column
        html.push_str("<h4 style='color:#495057;margin:20px 0 10px;border-bottom:2px solid #dee2e6;padding-bottom:8px;'>Entropia stĺpcov</h4>");
        html.push_str("<div style='display:flex;gap:10px;flex-wrap:wrap;'>");
        for (i, h) in headers.iter().enumerate() {
            let ent = Self::entropy_discrete(&binned[i]);
            let intensity = (ent / 4.0).min(1.0); // rough normalization
            html.push_str(&format!(
                "<div style='padding:8px 12px;background:rgba(52,152,219,{});border:1px solid #dee2e6;font-size:12px;'>\
                <b>{}</b><br>H = {:.3}\
                </div>",
                0.1 + intensity * 0.4, h, ent
            ));
        }
        html.push_str("</div>");
        html
    }
}
