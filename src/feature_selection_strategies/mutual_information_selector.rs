use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, Array2};
use super::FeatureSelector;
use statrs::function::gamma::digamma;

pub struct MutualInformationSelector 
{
    top_k: usize,
    k_neighbors: usize,
}

impl MutualInformationSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            top_k: 10,
            k_neighbors: 3 
        }
    }

    /// Implementácia KSG odhadu vzájomnej informácie
    fn estimate_mi_ksg(x_col: &[f64], y: &[f64], k: usize) -> f64 
    {
        let n = x_col.len();
        if n <= k 
        { 
            return 0.0; 
        }

        let mut mi = 0.0;
        let mut nx_vec = vec![0; n];
        let mut ny_vec = vec![0; n];

        for i in 0..n 
        {
            // 1. Nájdeme vzdialenosť k k-tému susedovi v spoločnom priestore (X, Y)
            // Používame Chebyshevovu metriku (L-infinity norma) ako v originálnom KSG papieri
            let mut distances = Vec::new();
            for j in 0..n 
            {
                if i == j { continue; }
                let dx = (x_col[i] - x_col[j]).abs();
                let dy = (y[i] - y[j]).abs();
                distances.push(dx.max(dy)); // L-infinity norma
            }
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            // epsilon je vzdialenosť k k-tému najbližšiemu susedovi
            let epsilon = distances[k - 1];

            // 2. Spočítať body v X a Y priestore, ktoré sú bližšie ako epsilon
            // n_x a n_y sú počty bodov, kde |xi - xj| < epsilon (resp. y)
            let mut nx = 0;
            let mut ny = 0;
            for j in 0..n 
            {
                if i == j { continue; }
                if (x_col[i] - x_col[j]).abs() < epsilon 
                {
                    nx += 1;
                }
                if (y[i] - y[j]).abs() < epsilon 
                {
                    ny += 1;
                }
            }
            nx_vec[i] = nx;
            ny_vec[i] = ny;
        }

        // 3. Finálny KSG vzorec: MI = psi(k) - <psi(nx+1) + psi(ny+1)> + psi(N)
        // psi je digamma funkcia (derivácia logaritmu Gamma funkcie)
        let psi_k = digamma(k as f64);
        let psi_n = digamma(n as f64);
        
        let mut mean_psi_nx_ny = 0.0;
        for i in 0..n 
        {
            mean_psi_nx_ny += digamma((nx_vec[i] + 1) as f64) + digamma((ny_vec[i] + 1) as f64);
        }
        mean_psi_nx_ny /= n as f64;

        mi = psi_k - mean_psi_nx_ny + psi_n;
        
        // MI nemôže byť záporná (v dôsledku šumu pri odhade môže vyjsť mierne pod 0)
        mi.max(0.0)
    }
}

impl FeatureSelector for MutualInformationSelector 
{
    fn get_name(&self) -> &str 
    {
        "Mutual Information (KSG Estimator)"
    }

    fn get_supported_params(&self) -> Vec<&str> 
    {
        vec!["num_features", "k_neighbors"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> 
    {
        match key 
        {
            "num_features" | "top_k" => self.top_k = value.parse().map_err(|_| "Invalid num_features".to_string())?,
            "k_neighbors" => self.k_neighbors = value.parse().map_err(|_| "Invalid k_neighbors".to_string())?,
            _ => return Err(format!("Param not found: {}. Supported: num_features, k_neighbors", key)),
        }
        Ok(())
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> 
    {
        let (_, cols) = x.shape();
        
        // Limituj top_k na maximum dostupných features
        let effective_k = self.top_k.min(cols);
        
        let mut scores = Vec::new();

        for j in 0..cols 
        {
            // Extrakcia stĺpca do Vec
            let col_vec: Vec<f64> = (0..x.shape().0).map(|i| *x.get((i, j))).collect();
            let score = Self::estimate_mi_ksg(&col_vec, y, self.k_neighbors);
            scores.push((j, score));
        }

        // Zoradíme podľa vypočítaného MI skóre
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        scores.into_iter()
            .take(effective_k)
            .map(|(idx, _)| idx)
            .collect()
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> 
    {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }
    
    fn get_feature_scores(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let (_, cols) = x.shape();
        let mut scores = Vec::new();

        for j in 0..cols {
            let col_vec: Vec<f64> = (0..x.shape().0).map(|i| *x.get((i, j))).collect();
            let score = Self::estimate_mi_ksg(&col_vec, y, self.k_neighbors);
            scores.push((j, score));
        }
        Some(scores)
    }
    
    fn get_metric_name(&self) -> &str {
        "MI Score"
    }
}

impl MutualInformationSelector {
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
