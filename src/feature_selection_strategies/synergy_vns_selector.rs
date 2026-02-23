use super::FeatureSelector;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use std::collections::HashSet;

// Simple PRNG for WASM compatibility
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        // Use timestamp-like seed from WASM
        SimpleRng { state: 0x123456789ABCDEF }
    }

    fn next(&mut self) -> u64 {
        // Linear congruential generator
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn gen_range(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        let range = (max - min) as u64;
        (self.next() % range) as usize + min
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(0, i + 1);
            slice.swap(i, j);
        }
    }
}

/// Variable Neighborhood Search (VNS) selector založený na synergii features
/// Optimalizuje výber features na základe:
/// 1. Mutual Information s target
/// 2. Sinergia medzi features (joint MI s target pre páry)
/// 3. Redundancia (MI a korelácia medzi features)
pub struct SynergyVNSSelector {
    target_feature_count: usize,
    max_iterations: usize,
    k_max: usize,  // Počet rôznych neighborhoods
    alpha: f64,    // Váha pre jednotlivú MI s target (0.0 - 1.0)
    beta: f64,     // Váha pre synergiu (joint info párov s target)
    gamma: f64,    // Penalizácia za redundanciu (MI medzi features)
    initial_solution: String, // "greedy" alebo "random"
}

impl SynergyVNSSelector {
    pub fn new() -> Self {
        SynergyVNSSelector {
            target_feature_count: 10,
            max_iterations: 100,
            k_max: 4,
            alpha: 0.7,      // 70% váha na individuálnu MI s target
            beta: 0.2,       // 20% váha na synergiu (joint MI párov)
            gamma: 0.3,      // 30% penalizácia redundancie (MI medzi features)
            initial_solution: "greedy".to_string(),
        }
    }

    /// Konvertuje DenseMatrix na Vec<Vec<f64>> (stĺpce)
    fn matrix_to_columns(x: &DenseMatrix<f64>) -> Vec<Vec<f64>> {
        let shape = x.shape();
        let (rows, cols) = (shape.0, shape.1);
        let mut columns = vec![Vec::with_capacity(rows); cols];
        
        for col in 0..cols {
            for row in 0..rows {
                columns[col].push(*x.get((row, col)));
            }
        }
        columns
    }

    /// Fitness funkcia: kombinuje MI s target, synergiu (joint MI) a redundanciu
    fn fitness(
        &self,
        selected: &HashSet<usize>,
        mi_scores: &[f64],
        pairwise_mi_matrix: &[Vec<f64>],  // MI medzi features (redundancia!)
        _correlation_matrix: &[Vec<f64>],
    ) -> f64 {
        if selected.is_empty() {
            return 0.0;
        }

        let n = selected.len() as f64;

        // 1. Priemerna individuálna MI s target
        let avg_mi: f64 = selected.iter().map(|&idx| mi_scores[idx]).sum::<f64>() / n;

        // 2. Synergy proxy: pre páry features, predpokladáme že nízka redundancia = potenciál pre synergiu
        //    Skutočná synergy by bola: I(X1,X2; Y) - I(X1;Y) - I(X2;Y)
        //    Aproximujeme: čím nižšia redundancia, tým lepšia šanca na synergiu
        let mut diversity_score = 0.0;
        let mut pair_count = 0;
        for &i in selected.iter() {
            for &j in selected.iter() {
                if i < j && i < pairwise_mi_matrix.len() && j < pairwise_mi_matrix[i].len() {
                    // Invertujeme: nízka MI medzi features = vysoká diverzita = dobrá synergy
                    let mi_between = pairwise_mi_matrix[i][j];
                    diversity_score += 1.0 / (1.0 + mi_between);
                    pair_count += 1;
                }
            }
        }
        let avg_diversity = if pair_count > 0 {
            diversity_score / pair_count as f64
        } else {
            0.0
        };

        // 3. Redundancia: vysoká MI medzi features = zlé (features sú podobné)
        let mut total_redundancy = 0.0;
        let mut redundancy_count = 0;
        for &i in selected.iter() {
            for &j in selected.iter() {
                if i < j && i < pairwise_mi_matrix.len() && j < pairwise_mi_matrix[i].len() {
                    let mi_between = pairwise_mi_matrix[i][j];
                    total_redundancy += mi_between;
                    redundancy_count += 1;
                }
            }
        }
        let avg_redundancy = if redundancy_count > 0 {
            total_redundancy / redundancy_count as f64
        } else {
            0.0
        };

        // Kombinovaný score:
        // - Vysoká MI s target = dobré (alpha)
        // - Vysoká diverzita = dobrá synergy (beta)
        // - Vysoká redundancia = zlé (gamma)
        self.alpha * avg_mi + self.beta * avg_diversity - self.gamma * avg_redundancy
    }

    /// Vytvorí počiatočné riešenie (greedy alebo random)
    fn initial_solution(
        &self,
        num_features: usize,
        mi_scores: &[f64],
    ) -> HashSet<usize> {
        let mut selected = HashSet::new();

        if self.initial_solution == "greedy" {
            // Greedy: vyber top-k podľa MI
            let mut ranked: Vec<(usize, f64)> = mi_scores
                .iter()
                .enumerate()
                .map(|(idx, &score)| (idx, score))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            
            for (idx, _) in ranked.iter().take(self.target_feature_count.min(num_features)) {
                selected.insert(*idx);
            }
        } else {
            // Random
            let mut rng = SimpleRng::new();
            let mut indices: Vec<usize> = (0..num_features).collect();
            rng.shuffle(&mut indices);
            for idx in indices.iter().take(self.target_feature_count.min(num_features)) {
                selected.insert(*idx);
            }
        }

        selected
    }

    /// Neighborhood 1: Swap one feature (remove + add)
    fn neighborhood_swap(
        &self,
        current: &HashSet<usize>,
        num_features: usize,
        rng: &mut SimpleRng,
    ) -> HashSet<usize> {
        let mut new_solution = current.clone();
        
        if new_solution.is_empty() {
            return new_solution;
        }

        // Remove random feature
        let remove_idx = *new_solution.iter().nth(rng.gen_range(0, new_solution.len())).unwrap();
        new_solution.remove(&remove_idx);

        // Add random feature not in set
        let mut candidates: Vec<usize> = (0..num_features)
            .filter(|idx| !new_solution.contains(idx))
            .collect();
        if !candidates.is_empty() {
            rng.shuffle(&mut candidates);
            new_solution.insert(candidates[0]);
        }

        new_solution
    }

    /// Neighborhood 2: Flip k features (remove k, add k)
    fn neighborhood_flip_k(
        &self,
        current: &HashSet<usize>,
        num_features: usize,
        k: usize,
        rng: &mut SimpleRng,
    ) -> HashSet<usize> {
        let mut new_solution = current.clone();
        let flip_count = k.min(current.len());

        // Remove k random features
        let current_vec: Vec<usize> = current.iter().copied().collect();
        let mut remove_indices = current_vec.clone();
        rng.shuffle(&mut remove_indices);
        for idx in remove_indices.iter().take(flip_count) {
            new_solution.remove(idx);
        }

        // Add k random features
        let mut candidates: Vec<usize> = (0..num_features)
            .filter(|idx| !new_solution.contains(idx))
            .collect();
        rng.shuffle(&mut candidates);
        for idx in candidates.iter().take(flip_count) {
            new_solution.insert(*idx);
        }

        new_solution
    }

    /// Neighborhood 3: Add one feature (expand)
    fn neighborhood_add(
        &self,
        current: &HashSet<usize>,
        num_features: usize,
        rng: &mut SimpleRng,
    ) -> HashSet<usize> {
        let mut new_solution = current.clone();
        
        let mut candidates: Vec<usize> = (0..num_features)
            .filter(|idx| !new_solution.contains(idx))
            .collect();
        
        if !candidates.is_empty() {
            rng.shuffle(&mut candidates);
            new_solution.insert(candidates[0]);
        }

        new_solution
    }

    /// Neighborhood 4: Remove one feature (shrink)
    fn neighborhood_remove(
        &self,
        current: &HashSet<usize>,
        rng: &mut SimpleRng,
    ) -> HashSet<usize> {
        let mut new_solution = current.clone();
        
        if new_solution.len() > 1 {
            let remove_idx = *new_solution.iter().nth(rng.gen_range(0, new_solution.len())).unwrap();
            new_solution.remove(&remove_idx);
        }

        new_solution
    }

    /// Lokálne prehľadávanie: vyskúšaj všetky susedné riešenia a vyber najlepšie
    fn local_search(
        &self,
        current: &HashSet<usize>,
        num_features: usize,
        mi_scores: &[f64],
        pairwise_mi_matrix: &[Vec<f64>],
        correlation_matrix: &[Vec<f64>],
        rng: &mut SimpleRng,
    ) -> HashSet<usize> {
        let mut best_solution = current.clone();
        let mut best_fitness = self.fitness(current, mi_scores, pairwise_mi_matrix, correlation_matrix);

        // Vyskúšaj 10 náhodných swapov
        for _ in 0..10 {
            let neighbor = self.neighborhood_swap(current, num_features, rng);
            let fitness = self.fitness(&neighbor, mi_scores, pairwise_mi_matrix, correlation_matrix);
            if fitness > best_fitness {
                best_fitness = fitness;
                best_solution = neighbor;
            }
        }

        best_solution
    }
}

impl FeatureSelector for SynergyVNSSelector {
    fn get_name(&self) -> &str {
        "Synergy VNS"
    }

    fn select_features(&self, x: &DenseMatrix<f64>, y: &[f64]) -> DenseMatrix<f64> {
        let indices = self.get_selected_indices(x, y);
        self.extract_columns(x, &indices)
    }

    fn get_selected_indices(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Vec<usize> {
        let shape = x.shape();
        let num_features = shape.1;
        
        if num_features == 0 {
            return vec![];
        }

        // Konvertuj DenseMatrix na stĺpce
        let columns = Self::matrix_to_columns(x);

        // 1. Vypočítaj MI scores pre každý feature vs target
        let mi_scores: Vec<f64> = columns.iter().map(|col| {
            estimate_mi_ksg(col, y, 3)
        }).collect();

        // 2. Vypočítaj pairwise MI medzi features (REDUNDANCIA, nie synergy!)
        //    Vysoká MI medzi features = features sú podobné = redundantné
        let mut pairwise_mi_matrix = vec![vec![0.0; num_features]; num_features];
        for i in 0..num_features {
            for j in (i + 1)..num_features {
                let mi = estimate_mi_ksg(&columns[i], &columns[j], 3);
                pairwise_mi_matrix[i][j] = mi;
                pairwise_mi_matrix[j][i] = mi;
            }
        }

        // 3. Vypočítaj correlation matrix (pre dodatočnú info o redundancii)
        let mut correlation_matrix = vec![vec![0.0; num_features]; num_features];
        for i in 0..num_features {
            for j in (i + 1)..num_features {
                let corr = pearson_correlation(&columns[i], &columns[j]);
                correlation_matrix[i][j] = corr;
                correlation_matrix[j][i] = corr;
            }
        }

        // 4. VNS algoritmus
        let mut rng = SimpleRng::new();
        let mut current_solution = self.initial_solution(num_features, &mi_scores);
        let mut best_solution = current_solution.clone();
        let mut best_fitness = self.fitness(&best_solution, &mi_scores, &pairwise_mi_matrix, &correlation_matrix);

        for iteration in 0..self.max_iterations {
            let mut k = 1;

            while k <= self.k_max {
                // Vyber neighborhood podľa k
                let candidate = match k {
                    1 => self.neighborhood_swap(&current_solution, num_features, &mut rng),
                    2 => self.neighborhood_flip_k(&current_solution, num_features, 2, &mut rng),
                    3 => self.neighborhood_add(&current_solution, num_features, &mut rng),
                    4 => self.neighborhood_remove(&current_solution, &mut rng),
                    _ => self.neighborhood_swap(&current_solution, num_features, &mut rng),
                };

                // Enforce target count constraint (približne)
                let mut adjusted_candidate = candidate.clone();
                while adjusted_candidate.len() < self.target_feature_count && adjusted_candidate.len() < num_features {
                    let mut available: Vec<usize> = (0..num_features)
                        .filter(|idx| !adjusted_candidate.contains(idx))
                        .collect();
                    if available.is_empty() {
                        break;
                    }
                    rng.shuffle(&mut available);
                    adjusted_candidate.insert(available[0]);
                }
                while adjusted_candidate.len() > self.target_feature_count {
                    if let Some(&remove_idx) = adjusted_candidate.iter().nth(rng.gen_range(0, adjusted_candidate.len())) {
                        adjusted_candidate.remove(&remove_idx);
                    }
                }

                // Lokálne prehľadávanie
                let local_best = self.local_search(
                    &adjusted_candidate,
                    num_features,
                    &mi_scores,
                    &pairwise_mi_matrix,
                    &correlation_matrix,
                    &mut rng,
                );

                let fitness = self.fitness(&local_best, &mi_scores, &pairwise_mi_matrix, &correlation_matrix);

                // Acceptance criterion
                if fitness > best_fitness {
                    best_solution = local_best.clone();
                    best_fitness = fitness;
                    current_solution = local_best;
                    k = 1; // Reset neighborhood
                } else {
                    k += 1;
                }
            }

            // Diversifikácia: občas reštartuj z náhodného riešenia
            if iteration % 30 == 0 && iteration > 0 {
                current_solution = self.initial_solution(num_features, &mi_scores);
            }
        }

        // Vráť sorted výsledok
        let mut result: Vec<usize> = best_solution.into_iter().collect();
        result.sort();
        result
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["num_features", "max_iterations", "k_max", "alpha", "beta", "gamma", "initial_solution"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "num_features" => {
                let k = value.parse::<usize>()
                    .map_err(|_| format!("Neplatná hodnota pre num_features: {}", value))?;
                self.target_feature_count = k.max(1);
                Ok(())
            },
            "max_iterations" => {
                let iter = value.parse::<usize>()
                    .map_err(|_| format!("Neplatná hodnota pre max_iterations: {}", value))?;
                self.max_iterations = iter.max(1);
                Ok(())
            },
            "k_max" => {
                let k_max = value.parse::<usize>()
                    .map_err(|_| format!("Neplatná hodnota pre k_max: {}", value))?;
                self.k_max = k_max.max(1).min(4);
                Ok(())
            },
            "alpha" => {
                let alpha = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre alpha: {}", value))?;
                self.alpha = alpha.clamp(0.0, 1.0);
                Ok(())
            },
            "beta" => {
                let beta = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre beta: {}", value))?;
                self.beta = beta.clamp(0.0, 1.0);
                Ok(())
            },
            "gamma" => {
                let gamma = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre gamma: {}", value))?;
                self.gamma = gamma.clamp(0.0, 1.0);
                Ok(())
            },
            "initial_solution" => {
                if value == "greedy" || value == "random" {
                    self.initial_solution = value.to_string();
                    Ok(())
                } else {
                    Err(format!("Neplatná hodnota pre initial_solution: {} (použite 'greedy' alebo 'random')", value))
                }
            },
            _ => Err(format!("Neznámy parameter: {}", key))
        }
    }

    fn get_feature_scores(&self, x: &DenseMatrix<f64>, y: &[f64]) -> Option<Vec<(usize, f64)>> {
        let columns = Self::matrix_to_columns(x);
        let mi_scores: Vec<f64> = columns.iter().map(|col| estimate_mi_ksg(col, y, 3)).collect();
        
        Some(mi_scores.iter().enumerate().map(|(idx, &score)| (idx, score)).collect())
    }

    fn get_metric_name(&self) -> &str {
        "VNS Fitness"
    }
}

/// KSG estimator pre mutual information (simplified)
fn estimate_mi_ksg(x: &[f64], y: &[f64], _k: usize) -> f64 {
    if x.is_empty() || x.len() != y.len() {
        return 0.0;
    }

    // Simplified: použijeme normalized MI proxy
    let corr = pearson_correlation(x, y);
    let mi_proxy = -0.5 * (1.0 - corr * corr).max(0.0).ln();
    mi_proxy.max(0.0)
}

/// Pearson correlation
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    if n == 0.0 || x.len() != y.len() {
        return 0.0;
    }

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
