use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use std::collections::HashMap;
use super::FeatureSelector;

pub struct InformationGainSelector 
{
    /// top_k: Určuje počet najlepších príznakov, ktoré majú byť vybrané.
    top_k: usize,
}

impl InformationGainSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            top_k: 10 
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
        
        // Limituj top_k na maximum dostupných features
        let effective_k = self.top_k.min(cols);
        
        let mut ig_scores = Vec::new();
        
        // Kontrola či target má rozumnú diskrétnosť
        Self::check_discrete_warning(y);
        
        let base_entropy = Self::entropy(y);

        for j in 0..cols 
        {
            // Extrakcia stĺpca
            let col: Vec<f64> = (0..x.shape().0).map(|i| *x.get((i, j))).collect();
            
            // Kontrola či stĺpec má rozumnú diskrétnosť
            Self::check_discrete_warning(&col);
            
            let mut conditional_entropy = 0.0;
            let mut map: HashMap<u64, Vec<f64>> = HashMap::new();

            // Rozdelenie targetov (y) podľa hodnôt v príznaku (Xi)
            for (val, target) in col.iter().zip(y.iter()) 
            {
                map.entry((*val).to_bits()).or_default().push(*target);
            }

            // Výpočet podmienenej entropie H(Y|Xi)
            for subset in map.values() 
            {
                let weight = subset.len() as f64 / y.len() as f64;
                conditional_entropy += weight * Self::entropy(subset);
            }

            // IG = H(Y) - H(Y|Xi)
            ig_scores.push((j, base_entropy - conditional_entropy));
        }

        // Zoradenie podľa IG (zostupne)
        ig_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Vrátime top_k indexov
        ig_scores.into_iter().take(effective_k).map(|(i, _)| i).collect()
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
