use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};
use std::collections::HashMap;
use super::FeatureSelector;

pub struct InformationGainSelector 
{
    /// top_k: Určuje počet najlepších príznakov, ktoré majú byť vybrané.
    /// Napríklad, ak je top_k = 5, selektor vráti len 5 stĺpcov s najvyšším informačným ziskom.
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
            // f64 neimplementuje Eq, preto používame to_bits() na hashovanie unikátnych kategórií
            *counts.entry(l.to_bits()).or_insert(0) += 1;
        }

        let n = labels.len() as f64;
        counts.values().map(|&c| 
        {
            let p = c as f64 / n;
            -p * p.log2() // Vzorec pre entropiu: -sum(p * log2(p))
        }).sum()
    }
}

impl FeatureSelector for InformationGainSelector 
{
    fn get_name(&self) -> &str 
    {
        "Information Gain (Diskrétny odhad)"
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
        Err(format!("Parameter nenájdený: {}. Podporované parametre: num_features", key))
    }

    /// Hlavná logika výberu príznakov.
    /// Vypočíta IG(Y, Xi) = H(Y) - H(Y|Xi) pre každý stĺpec.
    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> 
    {
        let (_, cols) = x.shape();
        let mut ig_scores = Vec::new();
        let base_entropy = Self::entropy(y); // H(Y)

        for j in 0..cols 
        {
            // Extrakcia stĺpca do Vec
            let col: Vec<f64> = (0..x.shape().0).map(|i| *x.get((i, j))).collect();
            let mut conditional_entropy = 0.0;
            let mut map: HashMap<u64, Vec<f64>> = HashMap::new();

            // Rozdelenie targetov (y) podľa unikátnych hodnôt v príznaku (Xi)
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

        // Zoradenie: Príznaky, ktoré najviac znižujú neistotu (najvyššie IG), idú na začiatok
        ig_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        // Vrátime len top_k indexov
        ig_scores.into_iter().take(self.top_k).map(|(i, _)| i).collect()
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
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
