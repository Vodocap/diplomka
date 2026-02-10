use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::FeatureSelector;
use statrs::function::gamma::digamma;
use std::cell::RefCell;

pub struct MutualInformationSelector 
{
    top_k: usize,
    k_neighbors: usize,
    details_cache: RefCell<String>,
}

impl MutualInformationSelector 
{
    pub fn new() -> Self 
    {
        Self 
        { 
            top_k: 10,
            k_neighbors: 3,
            details_cache: RefCell::new(String::new()),
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

        let mi = psi_k - mean_psi_nx_ny + psi_n;
        
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
        let effective_k = self.top_k.min(cols);
        let mut scores = Vec::new();

        for j in 0..cols {
            let col_vec: Vec<f64> = (0..x.shape().0).map(|i| *x.get((i, j))).collect();
            let score = Self::estimate_mi_ksg(&col_vec, y, self.k_neighbors);
            scores.push((j, score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let selected: Vec<usize> = scores.iter().take(effective_k).map(|(idx, _)| *idx).collect();
        
        // Cache details
        let mut html = String::from("<div style='margin:10px 0;'>");
        html.push_str("<h4>Mutual Information (KSG) Feature Selection</h4>");
        html.push_str(&format!("<p>K neighbors: <b>{}</b> | Top K: <b>{}</b> z <b>{}</b></p>", self.k_neighbors, effective_k, cols));
        html.push_str("<table style='border-collapse:collapse;font-size:12px;width:100%;'>");
        html.push_str("<tr><th style='padding:4px;border:1px solid #ddd;'>Poradie</th><th style='padding:4px;border:1px solid #ddd;'>Feature</th><th style='padding:4px;border:1px solid #ddd;'>MI Score</th><th style='padding:4px;border:1px solid #ddd;'>Status</th></tr>");
        for (rank, (idx, score)) in scores.iter().enumerate() {
            let sel = rank < effective_k;
            let bg = if sel { "rgba(0,200,0,0.15)" } else { "rgba(255,0,0,0.15)" };
            let status = if sel { "✅ Vybraný" } else { "❌ Odstránený" };
            html.push_str(&format!("<tr style='background:{}'><td style='padding:4px;border:1px solid #ddd;'>#{}</td><td style='padding:4px;border:1px solid #ddd;'>F{}</td><td style='padding:4px;border:1px solid #ddd;'>{:.4}</td><td style='padding:4px;border:1px solid #ddd;'>{}</td></tr>", bg, rank+1, idx, score, status));
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
    
    fn get_selection_details(&self) -> String {
        self.details_cache.borrow().clone()
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
