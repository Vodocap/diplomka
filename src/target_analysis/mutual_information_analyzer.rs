use super::{TargetAnalyzer, TargetCandidate, build_ranked_candidates};
use crate::entropy::mi_estimator;

/// Analyzátor cieľovej premennej na základe Mutual Information (KSG estimátor).
/// Zachytáva aj nelineárne vzťahy medzi premennými, na rozdiel od korelácie.
pub struct MutualInformationAnalyzer
{
    k_neighbors: usize,
}

impl MutualInformationAnalyzer
{
    pub fn new() -> Self
    {
        Self { k_neighbors: 3 }
    }

}

impl TargetAnalyzer for MutualInformationAnalyzer
{
    fn get_name(&self) -> &str
    {
        "mutual_information"
    }

    fn get_description(&self) -> &str
    {
        "Mutual Information (KSG) - zachytáva aj nelineárne vzťahy medzi premennými"
    }

    fn get_metric_name(&self) -> &str
    {
        "ΣMI"
    }

    fn get_metric_explanation(&self) -> &str
    {
        "Suma vzájomnej informácie (MI) s ostatnými premennými: Score_j = Σ MI(X_j, X_k). \
        MI meria množstvo informácie, ktorú jedna premenná poskytuje o druhej. \
        Na rozdiel od korelácie zachytáva aj nelineárne závislosti. \
        Vyššia hodnota = premenná zdieľa viac informácie s ostatnými. \
        Používa KSG estimátor (Kraskov-Stögbauer-Grassberger) pre spojité dáta."
    }

    fn analyze(&self, columns: &[Vec<f64>], headers: &[String]) -> Vec<TargetCandidate>
    {
        // Vypočítame MI maticu cez zdieľanú cache.
        let mi_matrix = mi_estimator::compute_mi_matrix_cached(columns, self.k_neighbors);

        build_ranked_candidates(columns, headers, |col_idx|
        {
            let num_cols = mi_matrix.len();
            let mut total_mi = 0.0f64;
            let mut max_mi = 0.0f64;
            for j in 0..num_cols
            {
                if j == col_idx
                {
                    continue;
                }
                total_mi += mi_matrix[col_idx][j];
                if mi_matrix[col_idx][j] > max_mi
                {
                    max_mi = mi_matrix[col_idx][j];
                }
            }

            (
                total_mi,
                vec![
                    ("sum_mi".to_string(), (total_mi * 10000.0).round() / 10000.0),
                    ("max_mi".to_string(), (max_mi * 10000.0).round() / 10000.0),
                ],
            )
        })
    }

    fn get_details_html(&self, columns: &[Vec<f64>], headers: &[String], _candidates: &[TargetCandidate]) -> String
    {
        let num_cols = columns.len();
        if num_cols > 50
        {
            return String::new();
        }

        let mi_matrix = mi_estimator::compute_mi_matrix_cached(columns, self.k_neighbors);

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
        for h in headers
        {
            html.push_str(&format!("<th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;max-width:80px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;'>{}</th>", h));
        }
        html.push_str("</tr>");
        for i in 0..num_cols
        {
            html.push_str(&format!("<tr><th style='padding:6px;border:1px solid #ddd;background:#f0f0f0;font-size:10px;white-space:nowrap;'>{}</th>", &headers[i]));
            for j in 0..num_cols
            {
                let mi = mi_matrix[i][j];
                let norm = mi / scale;
                let bg_color = if i == j
                {
                    "#e0e0e0".to_string()
                }
                else if norm > 0.7
                {
                    format!("rgba(52,152,219,{})", 0.3 + norm * 0.5)
                }
                else if norm > 0.3
                {
                    format!("rgba(46,204,113,{})", 0.2 + norm * 0.4)
                }
                else
                {
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
