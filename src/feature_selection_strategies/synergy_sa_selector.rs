use super::FeatureSelector;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use std::collections::HashSet;
use crate::mi_estimator;

// Simple PRNG for WASM compatibility
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new() -> Self {
        SimpleRng { state: 0xDEADBEEF42 }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn gen_range(&mut self, min: usize, max: usize) -> usize {
        if min >= max {
            return min;
        }
        let range = (max - min) as u64;
        (self.next() % range) as usize + min
    }

    /// Generuje f64 v [0.0, 1.0)
    fn gen_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = self.gen_range(0, i + 1);
            slice.swap(i, j);
        }
    }
}

/// Simulated Annealing selector optimalizujúci synergiu medzi features.
///
/// Na rozdiel od VNS, SA používa pravdepodobnostné akceptovanie horších riešení
/// podľa Boltzmannovho kritéria: P(accept) = exp(-ΔE / T), čo umožňuje
/// efektívnejšie unikať z lokálnych optím.
///
/// Fitness funkcia kombinuje:
/// - **Alpha (α)**: Individuálna MI s target – relevancia features
/// - **Beta (β)**: Synergia (diverzita) – features nesúce rôznu informáciu
/// - **Gamma (γ)**: Penalizácia redundancie – vylučuje duplicitné features
pub struct SynergySASelector {
    target_feature_count: usize,
    max_iterations: usize,
    initial_temp: f64,      // Počiatočná teplota
    cooling_rate: f64,       // Rýchlosť chladnutia (0.9 - 0.999)
    min_temp: f64,           // Minimálna teplota (stopping criterion)
    reheat_interval: usize,  // Po koľkých iteráciách sa reheating
    alpha: f64,              // Váha pre individuálnu MI s target
    beta: f64,               // Váha pre synergiu (diverzita features)
    gamma: f64,              // Penalizácia za redundanciu
    initial_solution: String, // "greedy" alebo "random"
}

impl SynergySASelector {
    pub fn new() -> Self {
        SynergySASelector {
            target_feature_count: 10,
            max_iterations: 200,
            initial_temp: 1.0,
            cooling_rate: 0.95,
            min_temp: 0.001,
            reheat_interval: 50,
            alpha: 0.7,
            beta: 0.2,
            gamma: 0.3,
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

    /// Fitness funkcia: alpha * MI_relevancia + beta * synergia - gamma * redundancia
    fn fitness(
        &self,
        selected: &HashSet<usize>,
        mi_scores: &[f64],
        pairwise_mi_matrix: &[Vec<f64>],
    ) -> f64 {
        if selected.is_empty() {
            return 0.0;
        }

        let n = selected.len() as f64;

        // 1. Priemerná individuálna MI s target (relevancia)
        let avg_mi: f64 = selected.iter().map(|&idx| mi_scores[idx]).sum::<f64>() / n;

        // 2. Synergia: invertovaná MI medzi features = diverzita
        let mut diversity_score = 0.0;
        let mut pair_count = 0;
        for &i in selected.iter() {
            for &j in selected.iter() {
                if i < j && i < pairwise_mi_matrix.len() && j < pairwise_mi_matrix[i].len() {
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

        // 3. Redundancia: vysoká MI medzi features = zlé
        let mut total_redundancy = 0.0;
        let mut redundancy_count = 0;
        for &i in selected.iter() {
            for &j in selected.iter() {
                if i < j && i < pairwise_mi_matrix.len() && j < pairwise_mi_matrix[i].len() {
                    total_redundancy += pairwise_mi_matrix[i][j];
                    redundancy_count += 1;
                }
            }
        }
        let avg_redundancy = if redundancy_count > 0 {
            total_redundancy / redundancy_count as f64
        } else {
            0.0
        };

        self.alpha * avg_mi + self.beta * avg_diversity - self.gamma * avg_redundancy
    }

    /// Vytvorí počiatočné riešenie
    fn create_initial_solution(
        &self,
        num_features: usize,
        mi_scores: &[f64],
        rng: &mut SimpleRng,
    ) -> HashSet<usize> {
        let count = self.target_feature_count.min(num_features);
        let mut selected = HashSet::new();

        if self.initial_solution == "greedy" {
            let mut ranked: Vec<(usize, f64)> = mi_scores
                .iter()
                .enumerate()
                .map(|(idx, &score)| (idx, score))
                .collect();
            ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (idx, _) in ranked.iter().take(count) {
                selected.insert(*idx);
            }
        } else {
            let mut indices: Vec<usize> = (0..num_features).collect();
            rng.shuffle(&mut indices);
            for idx in indices.iter().take(count) {
                selected.insert(*idx);
            }
        }

        selected
    }

    /// Generuje susedné riešenie – náhodne zamení 1-2 features
    fn generate_neighbor(
        &self,
        current: &HashSet<usize>,
        num_features: usize,
        rng: &mut SimpleRng,
    ) -> HashSet<usize> {
        let mut new_solution = current.clone();
        
        if new_solution.is_empty() || num_features <= new_solution.len() {
            return new_solution;
        }

        // Rozhodne sa medzi swap(1) a swap(2)
        let num_swaps = if rng.gen_f64() < 0.7 { 1 } else { 2.min(current.len()) };
        
        for _ in 0..num_swaps {
            // Remove random feature
            let current_vec: Vec<usize> = new_solution.iter().copied().collect();
            if current_vec.is_empty() { break; }
            let remove_idx = current_vec[rng.gen_range(0, current_vec.len())];
            new_solution.remove(&remove_idx);

            // Add random feature not in current set
            let mut candidates: Vec<usize> = (0..num_features)
                .filter(|idx| !new_solution.contains(idx))
                .collect();
            if !candidates.is_empty() {
                rng.shuffle(&mut candidates);
                new_solution.insert(candidates[0]);
            }
        }

        new_solution
    }

    /// Boltzmann acceptance: vráti true ak sa má akceptovať nové riešenie
    fn accept(&self, delta_fitness: f64, temperature: f64, rng: &mut SimpleRng) -> bool {
        if delta_fitness >= 0.0 {
            // Lepšie riešenie – vždy akceptuj
            return true;
        }
        // Horšie riešenie – akceptuj s pravdepodobnosťou exp(delta / T)
        let prob = (delta_fitness / temperature).exp();
        rng.gen_f64() < prob
    }
}

impl FeatureSelector for SynergySASelector {
    fn get_name(&self) -> &str {
        "Synergy SA"
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

        let columns = Self::matrix_to_columns(x);

        // 1. MI scores pre každý feature vs target
        let mi_scores: Vec<f64> = columns.iter().map(|col| {
            mi_estimator::estimate_mi_proxy(col, y)
        }).collect();

        // 2. Pairwise MI medzi features (pre redundanciu/synergiu)
        let mut pairwise_mi_matrix = vec![vec![0.0; num_features]; num_features];
        for i in 0..num_features {
            for j in (i + 1)..num_features {
                let mi = mi_estimator::estimate_mi_proxy(&columns[i], &columns[j]);
                pairwise_mi_matrix[i][j] = mi;
                pairwise_mi_matrix[j][i] = mi;
            }
        }

        // 3. Simulated Annealing
        let mut rng = SimpleRng::new();
        let mut current = self.create_initial_solution(num_features, &mi_scores, &mut rng);
        let mut current_fitness = self.fitness(&current, &mi_scores, &pairwise_mi_matrix);
        
        let mut best = current.clone();
        let mut best_fitness = current_fitness;

        let mut temperature = self.initial_temp;
        let mut iterations_since_improvement = 0;

        for _iteration in 0..self.max_iterations {
            if temperature < self.min_temp {
                break;
            }

            // Generuj susedné riešenie
            let neighbor = self.generate_neighbor(&current, num_features, &mut rng);
            let neighbor_fitness = self.fitness(&neighbor, &mi_scores, &pairwise_mi_matrix);
            
            let delta = neighbor_fitness - current_fitness;

            // Boltzmann acceptance criterion
            if self.accept(delta, temperature, &mut rng) {
                current = neighbor;
                current_fitness = neighbor_fitness;

                if current_fitness > best_fitness {
                    best = current.clone();
                    best_fitness = current_fitness;
                    iterations_since_improvement = 0;
                } else {
                    iterations_since_improvement += 1;
                }
            } else {
                iterations_since_improvement += 1;
            }

            // Chladnutie
            temperature *= self.cooling_rate;

            // Reheating: ak dlho nebolo zlepšenie, čiastočne zvýš teplotu
            if self.reheat_interval > 0 && iterations_since_improvement > 0
                && iterations_since_improvement % self.reheat_interval == 0
            {
                temperature = (self.initial_temp * 0.5).max(temperature * 3.0);
                // Reštartuj z najlepšieho riešenia
                current = best.clone();
                current_fitness = best_fitness;
            }
        }

        let mut result: Vec<usize> = best.into_iter().collect();
        result.sort();
        result
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec![
            "num_features", "max_iterations", "initial_temp", "cooling_rate",
            "min_temp", "reheat_interval", "alpha", "beta", "gamma", "initial_solution"
        ]
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
            "initial_temp" => {
                let t = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre initial_temp: {}", value))?;
                self.initial_temp = t.max(0.001);
                Ok(())
            },
            "cooling_rate" => {
                let cr = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre cooling_rate: {}", value))?;
                self.cooling_rate = cr.clamp(0.5, 0.9999);
                Ok(())
            },
            "min_temp" => {
                let mt = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre min_temp: {}", value))?;
                self.min_temp = mt.max(0.0);
                Ok(())
            },
            "reheat_interval" => {
                let ri = value.parse::<usize>()
                    .map_err(|_| format!("Neplatná hodnota pre reheat_interval: {}", value))?;
                self.reheat_interval = ri;
                Ok(())
            },
            "alpha" => {
                let a = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre alpha: {}", value))?;
                self.alpha = a.clamp(0.0, 1.0);
                Ok(())
            },
            "beta" => {
                let b = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre beta: {}", value))?;
                self.beta = b.clamp(0.0, 1.0);
                Ok(())
            },
            "gamma" => {
                let g = value.parse::<f64>()
                    .map_err(|_| format!("Neplatná hodnota pre gamma: {}", value))?;
                self.gamma = g.clamp(0.0, 1.0);
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
        let mi_scores: Vec<f64> = columns.iter().map(|col| mi_estimator::estimate_mi_proxy(col, y)).collect();
        Some(mi_scores.iter().enumerate().map(|(idx, &score)| (idx, score)).collect())
    }

    fn get_metric_name(&self) -> &str {
        "SA Fitness"
    }
}
