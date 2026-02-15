use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use std::collections::HashMap;
use super::FeatureSelector;
use std::cell::RefCell;

pub struct InformationGainSelector 
{
    /// top_k: Určuje počet najlepších príznakov, ktoré majú byť vybrané.
    top_k: usize,
    details_cache: RefCell<String>,
}

impl InformationGainSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            top_k: 10,
            details_cache: RefCell::new(String::new()),
        }
    }

    /// Pomocná metóda na výpočet Shannonovej entropie H(Y).
    /// Meria mieru neistoty v cieľovej premennej.
    fn entropy(labels: &[f64]) -> f64 
    {
        if labels.is_empty() 
        {
            return 0.0;
        }

        let mut counts = HashMap::new();
        for &l in labels 
        {
            *counts.entry(l.to_bits()).or_insert(0) += 1;
        }

        let n = labels.len() as f64;
        counts.values().map(|&c| 
        {
            let p = c as f64 / n;
            -p * p.log2()
        }).sum()
    }
    
    /// Kontroluje či sú dáta pravdepodobne diskrétne
    /// Loguje varovanie ak nie sú
    fn check_discrete_warning(col: &[f64]) -> bool {
        let unique_values: std::collections::HashSet<_> = col.iter().map(|v| v.to_bits()).collect();
        // Ak je viac ako 50 unikátnych hodnôt, pravdepodobne spojité dáta
        if unique_values.len() > 50 {
            web_sys::console::warn_1(&format!(
                "Information Gain: Stĺpec má {} unikátnych hodnôt. Pre správne výsledky použite Binner processor!",
                unique_values.len()
            ).into());
            return false;
        }
        true
    }
}

impl FeatureSelector for InformationGainSelector 
{
    fn get_name(&self) -> &str 
    {
        "Information Gain (vyžaduje diskrétne/binned dáta)"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["num_features"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        if key == "num_features" || key == "top_k"
        {
            self.top_k = value.parse().map_err(|_| "num_features musí byť celé číslo".to_string())?;
            return Ok(());
        }
        Err(format!("Parameter nenájdený: {}. Podporované: num_features", key))
    }

    /// Hlavná logika výberu príznakov.
    /// DÔLEŽITÉ: Information Gain vyžaduje diskrétne dáta!
    /// Pridajte Binner processor pred tento selektor v pipeline.
    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> 
    {
        let (_, cols) = x.shape();
        let effective_k = self.top_k.min(cols);
        let mut ig_scores = Vec::new();
        
        Self::check_discrete_warning(y);
        let base_entropy = Self::entropy(y);

        for j in 0..cols 
        {
            let col: Vec<f64> = (0..x.shape().0).map(|i| *x.get((i, j))).collect();
            Self::check_discrete_warning(&col);
            
            let mut conditional_entropy = 0.0;
            let mut map: HashMap<u64, Vec<f64>> = HashMap::new();
            for (val, target) in col.iter().zip(y.iter()) {
                map.entry((*val).to_bits()).or_default().push(*target);
            }
            for subset in map.values() {
                let weight = subset.len() as f64 / y.len() as f64;
                conditional_entropy += weight * Self::entropy(subset);
            }
            ig_scores.push((j, base_entropy - conditional_entropy));
        }

        ig_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<usize> = ig_scores.iter().take(effective_k).map(|(i, _)| *i).collect();
        
        // Cache details
        let mut html = String::from("<div style='margin:10px 0;'>");
        html.push_str("<h4>Information Gain Feature Selection</h4>");
        html.push_str(&format!("<p>Base Entropy H(Y): <b>{:.4}</b> | Top K: <b>{}</b> z <b>{}</b></p>", base_entropy, effective_k, cols));
        html.push_str("<table style='border-collapse:collapse;font-size:12px;width:100%;'>");
        html.push_str("<tr><th style='padding:4px;border:1px solid #ddd;'>Poradie</th><th style='padding:4px;border:1px solid #ddd;'>Feature</th><th style='padding:4px;border:1px solid #ddd;'>IG Score</th><th style='padding:4px;border:1px solid #ddd;'>Status</th></tr>");
        for (rank, (idx, score)) in ig_scores.iter().enumerate() {
            let sel = rank < effective_k;
            let bg = if sel { "rgba(52,152,219,0.08)" } else { "rgba(189,195,199,0.08)" };
            let status = if sel { "<span style='color:#28a745;font-weight:bold;'>✓</span>" } else { "<span style='color:#6c757d;'>✗</span>" };
            html.push_str(&format!("<tr style='background:{}'><td style='padding:4px;border:1px solid #ddd;'>#{}</td><td style='padding:4px;border:1px solid #ddd;'>F{}</td><td style='padding:4px;border:1px solid #ddd;'>{:.4}</td><td style='padding:4px;border:1px solid #ddd;text-align:center;'>{}</td></tr>", bg, rank+1, idx, score, status));
        }
        html.push_str("</table></div>");
        *self.details_cache.borrow_mut() = html;
        
        selected
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }
    
    fn get_feature_scores(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let (_, cols) = x.shape();
        let mut ig_scores = Vec::new();
        let base_entropy = Self::entropy(y);

        for j in 0..cols {
            let col: Vec<f64> = (0..x.shape().0).map(|i| *x.get((i, j))).collect();
            let mut conditional_entropy = 0.0;
            let mut map: HashMap<u64, Vec<f64>> = HashMap::new();

            for (val, target) in col.iter().zip(y.iter()) {
                map.entry((*val).to_bits()).or_default().push(*target);
            }

            for subset in map.values() {
                let weight = subset.len() as f64 / y.len() as f64;
                conditional_entropy += weight * Self::entropy(subset);
            }

            ig_scores.push((j, base_entropy - conditional_entropy));
        }
        Some(ig_scores)
    }
    
    fn get_metric_name(&self) -> &str {
        "Information Gain"
    }
    
    fn get_selection_details(&self) -> String {
        self.details_cache.borrow().clone()
    }
}

impl InformationGainSelector {
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
