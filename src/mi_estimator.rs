//! Zdieľaný modul pre výpočet Mutual Information (KSG estimátor).
//! Používa sa v `feature_selection_strategies::mutual_information_selector`
//! aj v `target_analysis::mutual_information_analyzer`.
//!
//! Optimalizácie:
//! - KD-tree (crate `kdtree`) s Chebyshevovou vzdialenosťou pre k-NN query
//!   → O(k log n) na query namiesto O(n)
//! - Zoradené polia + binárne vyhľadávanie pre počítanie marginálnych susedov
//!   → O(log n) namiesto O(n)
//! - Celková zložitosť: ~O(n k log n) namiesto O(n²)

use statrs::function::gamma::digamma;
use kdtree::KdTree;

// ═══════════════════════════════════════════════════════════════════════
//  Zdieľané štatistické utility
// ═══════════════════════════════════════════════════════════════════════

/// Pearsonov korelačný koeficient medzi dvoma vektormi.
/// Vracia hodnotu v rozsahu [-1, 1].
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
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

/// Zjednodušený odhad Mutual Information na báze Pearsonovej korelácie.
/// Používa sa v SA/VNS optimalizačných selektoroch kde je dôležitá
/// rýchlosť (volaná opakovane v každej iterácii).
/// Pre presnejší odhad použite `estimate_mi_ksg`.
pub fn estimate_mi_proxy(x: &[f64], y: &[f64]) -> f64 {
    if x.is_empty() || x.len() != y.len() {
        return 0.0;
    }
    let corr = pearson_correlation(x, y);
    let mi_proxy = -0.5 * (1.0 - corr * corr).max(0.0).ln();
    mi_proxy.max(0.0)
}

/// Chebyshevova (L∞) vzdialenosť medzi dvoma bodmi.
#[inline]
fn chebyshev(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

// ═══════════════════════════════════════════════════════════════════════
//  Zoradený index pre rýchle 1D range-counting
// ═══════════════════════════════════════════════════════════════════════

/// Zoradené pole hodnôt pre efektívne počítanie bodov v 1D intervale
/// pomocou binárneho vyhľadávania (O(log n) namiesto O(n)).
struct SortedIndex {
    sorted: Vec<(f64, usize)>, // (hodnota, pôvodný index)
}

impl SortedIndex {
    /// Zostaví zoradený index. Zložitosť: O(n log n).
    fn build(values: &[f64]) -> Self {
        let mut sorted: Vec<(f64, usize)> = values.iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        sorted.sort_unstable_by(|a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        SortedIndex { sorted }
    }

    /// Spočíta body kde |value - center| < epsilon (striktne), okrem skip_index.
    /// Používa binárne vyhľadávanie na nájdenie rozsahu.
    fn count_within(&self, center: f64, epsilon: f64, skip_index: usize) -> usize {
        let lo = center - epsilon;
        let hi = center + epsilon;

        // Binárne vyhľadávanie: prvý index kde value > lo
        let start = self.sorted.partition_point(|&(v, _)| v <= lo);
        // Binárne vyhľadávanie: prvý index kde value >= hi
        let end = self.sorted.partition_point(|&(v, _)| v < hi);

        // Počítame body v rozsahu [start, end), preskočíme skip_index
        let mut count = 0usize;
        for i in start..end {
            if self.sorted[i].1 != skip_index {
                count += 1;
            }
        }
        count
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Verejné API
// ═══════════════════════════════════════════════════════════════════════

/// KSG (Kraskov-Stögbauer-Grassberger) estimátor vzájomnej informácie
/// medzi dvoma spojitými premennými.
///
/// Používa:
/// - KD-tree (`kdtree` crate) s Chebyshevovou vzdialenosťou na k-NN hľadanie
/// - Zoradené polia na počítanie marginálnych susedov (1D range queries)
///
/// # Argumenty
/// * `x` - prvá premenná (vektor hodnôt)
/// * `y` - druhá premenná (vektor hodnôt)
/// * `k` - počet najbližších susedov pre KSG odhad
pub fn estimate_mi_ksg(x: &[f64], y: &[f64], k: usize) -> f64 {
    let n = x.len();
    if n <= k {
        return 0.0;
    }

    let k_eff = k.min(n - 1);
    if n <= k_eff {
        return 0.0;
    }

    // Postavíme KD-tree s 2D bodmi: O(n log n)
    let points: Vec<[f64; 2]> = (0..n).map(|i| [x[i], y[i]]).collect();
    let mut tree = KdTree::with_capacity(2, n);
    for i in 0..n {
        tree.add(&points[i], i).unwrap();
    }

    // Zoradené indexy pre marginálne 1D range-counting: O(n log n)
    let x_sorted = SortedIndex::build(x);
    let y_sorted = SortedIndex::build(y);

    // Pre každý bod nájdeme epsilon a spočítame marginálnych susedov
    let mut nx_vec = Vec::with_capacity(n);
    let mut ny_vec = Vec::with_capacity(n);

    for i in 0..n {
        // KD-tree query: k+1 najbližších (vrátane seba), Chebyshevova vzdialenosť
        let neighbors = tree.nearest(&[x[i], y[i]], k_eff + 1, &chebyshev).unwrap();

        // Nájdi k-tu vzdialenosť (preskočíme seba — vzdialenosť 0)
        let epsilon = neighbors.iter()
            .filter(|(_, &idx)| idx != i)
            .nth(k_eff - 1)
            .map(|(d, _)| *d)
            .unwrap_or(f64::INFINITY);

        // Zoradené polia: spočítaj marginálnych susedov — O(log n + m)
        let nx = x_sorted.count_within(x[i], epsilon, i);
        let ny = y_sorted.count_within(y[i], epsilon, i);

        nx_vec.push(nx);
        ny_vec.push(ny);
    }

    // Výpočet MI podľa KSG vzorca
    let psi_k = digamma(k_eff as f64);
    let psi_n = digamma(n as f64);
    let mean_psi: f64 = (0..n)
        .map(|i| digamma((nx_vec[i] + 1) as f64) + digamma((ny_vec[i] + 1) as f64))
        .sum::<f64>() / n as f64;

    (psi_k - mean_psi + psi_n).max(0.0)
}

/// Adaptívne k: pre veľké datasety sa automaticky zníži k,
/// pretože väčší počet bodov kompenzuje menší k_neighbors.
fn adaptive_k(n: usize, k: usize) -> usize {
    if n > 1000 { k.min(2) }
    else if n > 500 { k.min(3) }
    else { k }
}

/// Vypočíta symetrickú MI maticu pre všetky páry stĺpcov.
/// Počíta len vrchný trojuholník a zrkadlí (MI je symetrická).
///
/// # Argumenty
/// * `columns` - slice vektorov, kde každý vektor je jeden stĺpec
/// * `k` - počet najbližších susedov pre KSG odhad (automaticky sa adaptuje podľa veľkosti dát)
pub fn compute_mi_matrix(columns: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let num_cols = columns.len();
    let n = if num_cols > 0 { columns[0].len() } else { 0 };
    let k_eff = adaptive_k(n, k);

    let mut mi_matrix = vec![vec![0.0f64; num_cols]; num_cols];

    for i in 0..num_cols {
        for j in (i + 1)..num_cols {
            let mi = estimate_mi_ksg(&columns[i], &columns[j], k_eff);
            mi_matrix[i][j] = mi;
            mi_matrix[j][i] = mi;
        }
    }

    mi_matrix
}
