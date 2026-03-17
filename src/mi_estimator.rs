//! Zdieľaný modul pre výpočet Mutual Information.
//! Používa sa v `feature_selection_strategies::mutual_information_selector`
//! aj v `target_analysis::mutual_information_analyzer`.
//!
//! Estimátory:
//! - **Diskrétny (histogram)**: presný pre ordinálne/diskrétne premenné (málo unikátnych hodnôt).
//!   O(n) – veľmi rýchly. Auto-detekcia: ak oba stĺpce majú ≤ 100 unikátnych hodnôt.
//! - **KSG (k-NN)**: pre skutočne spojité premenné. O(n·k·log n).
//!
//! Optimalizácie KSG:
//! - KD-tree (crate `kdtree`) s Chebyshevovou vzdialenosťou pre k-NN query
//! - Zoradené polia + binárne vyhľadávanie pre marginálne 1D range queries
//! - Joint MI: 2D KD-tree pre (x1, x2) marginál – O(n log n)

use statrs::function::gamma::digamma;
use kdtree::KdTree;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use once_cell::sync::Lazy;
use ndarray::{Array2, Axis};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

#[derive(Default)]
struct MatrixCache {
    corr_matrices: HashMap<u64, Vec<Vec<f64>>>,
    mi_matrices: HashMap<u64, Vec<Vec<f64>>>,
}

impl MatrixCache {
    const MAX_CACHE_ITEMS: usize = 8;

    fn get_corr(&self, key: u64) -> Option<Vec<Vec<f64>>> {
        self.corr_matrices.get(&key).cloned()
    }

    fn insert_corr(&mut self, key: u64, value: Vec<Vec<f64>>) {
        self.corr_matrices.insert(key, value);
        self.trim_if_needed();
    }

    fn get_mi(&self, key: u64) -> Option<Vec<Vec<f64>>> {
        self.mi_matrices.get(&key).cloned()
    }

    fn insert_mi(&mut self, key: u64, value: Vec<Vec<f64>>) {
        self.mi_matrices.insert(key, value);
        self.trim_if_needed();
    }

    fn trim_if_needed(&mut self) {
        if self.corr_matrices.len() > Self::MAX_CACHE_ITEMS {
            self.corr_matrices.clear();
        }
        if self.mi_matrices.len() > Self::MAX_CACHE_ITEMS {
            self.mi_matrices.clear();
        }
    }
}

static MATRIX_CACHE_SINGLETON: Lazy<Mutex<MatrixCache>> =
    Lazy::new(|| Mutex::new(MatrixCache::default()));

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

// ═══════════════════════════════════════════════════════════════════════
//  Diskrétny (histogram) MI estimátor
// ═══════════════════════════════════════════════════════════════════════

/// Vráti počet unikátnych hodnôt v stĺpci.
fn unique_count(x: &[f64]) -> usize {
    let mut seen: HashMap<u64, ()> = HashMap::new();
    for &v in x { seen.insert(v.to_bits(), ()); }
    seen.len()
}

/// Diskrétny MI estimátor pre ordinálne/kategoriálne premenné.
/// Buduje spoločnú frekvenčnú tabuľku a počíta MI = Σ P(x,y) log(P(x,y)/(P(x)P(y))).
/// Presný, O(n), vhodný pre dáta s malým počtom unikátnych hodnôt.
fn estimate_mi_discrete(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 || x.len() != y.len() { return 0.0; }

    // Priradíme index každej unikátnej hodnote
    let mut x_map: HashMap<u64, usize> = HashMap::new();
    let mut y_map: HashMap<u64, usize> = HashMap::new();
    for &v in x { let l = x_map.len(); x_map.entry(v.to_bits()).or_insert(l); }
    for &v in y { let l = y_map.len(); y_map.entry(v.to_bits()).or_insert(l); }

    let nx = x_map.len();
    let ny = y_map.len();

    // Spoločná frekvenčná tabuľka
    let mut joint = vec![vec![0usize; ny]; nx];
    for i in 0..n {
        let xi = x_map[&x[i].to_bits()];
        let yi = y_map[&y[i].to_bits()];
        joint[xi][yi] += 1;
    }

    // Marginálne početnosti
    let x_count: Vec<usize> = joint.iter().map(|row| row.iter().sum()).collect();
    let y_count: Vec<usize> = (0..ny)
        .map(|j| joint.iter().map(|row| row[j]).sum())
        .collect();

    let n_f = n as f64;
    let mut mi = 0.0f64;
    for xi in 0..nx {
        for yi in 0..ny {
            let c = joint[xi][yi];
            if c == 0 { continue; }
            let pxy = c as f64 / n_f;
            let px  = x_count[xi] as f64 / n_f;
            let py  = y_count[yi] as f64 / n_f;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    mi.max(0.0)
}

/// Diskrétny Joint MI estimátor: MI((X1,X2); Y).
/// Používa 3-premennú frekvenčnú tabuľku.
fn estimate_joint_mi_discrete(x1: &[f64], x2: &[f64], y: &[f64]) -> f64 {
    let n = x1.len();
    if n == 0 || x2.len() != n || y.len() != n { return 0.0; }

    let mut x1_map: HashMap<u64, usize> = HashMap::new();
    let mut x2_map: HashMap<u64, usize> = HashMap::new();
    let mut y_map:  HashMap<u64, usize> = HashMap::new();
    for &v in x1 { let l = x1_map.len(); x1_map.entry(v.to_bits()).or_insert(l); }
    for &v in x2 { let l = x2_map.len(); x2_map.entry(v.to_bits()).or_insert(l); }
    for &v in y  { let l = y_map.len();  y_map.entry(v.to_bits()).or_insert(l); }

    let n1 = x1_map.len();
    let n2 = x2_map.len();
    let ny = y_map.len();

    // 3D joint count: joint[x1][x2][y]
    let mut joint = vec![vec![vec![0usize; ny]; n2]; n1];
    for i in 0..n {
        let i1 = x1_map[&x1[i].to_bits()];
        let i2 = x2_map[&x2[i].to_bits()];
        let iy = y_map[&y[i].to_bits()];
        joint[i1][i2][iy] += 1;
    }

    // Marginálne: P(x1,x2) a P(y)
    let mut xy_count = vec![vec![0usize; n2]; n1];
    let mut y_count  = vec![0usize; ny];
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            for iy in 0..ny {
                let c = joint[i1][i2][iy];
                xy_count[i1][i2] += c;
                y_count[iy] += c;
            }
        }
    }

    let n_f = n as f64;
    let mut mi = 0.0f64;
    for i1 in 0..n1 {
        for i2 in 0..n2 {
            for iy in 0..ny {
                let c = joint[i1][i2][iy];
                if c == 0 { continue; }
                let pxxy = c as f64 / n_f;
                let pxy  = xy_count[i1][i2] as f64 / n_f;
                let py   = y_count[iy] as f64 / n_f;
                mi += pxxy * (pxxy / (pxy * py)).ln();
            }
        }
    }
    mi.max(0.0)
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
/// * `k` - počet najbližších susedov pre KSG odhad (ignoruje sa pri diskrétnych dátach)
pub fn estimate_mi_ksg(x: &[f64], y: &[f64], k: usize) -> f64 {
    // Auto-detekcia: diskrétne dáta (málo unikátnych hodnôt) → histogram estimátor
    // KSG predpokladá spojité dáta a zlyhá pri väzoch (epsilon=0 → artefakty MI≈10)
    if unique_count(x) <= 100 && unique_count(y) <= 100 {
        return estimate_mi_discrete(x, y);
    }
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

/// KSG estimátor Joint Mutual Information MI((X1, X2); Y)
/// Meria koľko informácie dvojica premenných (X1, X2) spoločne nesie o cieľovej Y.
/// Používa 3D KD-tree s Chebyshevovou vzdialenosťou.
///
/// # Argumenty
/// * `x1` - prvá premenná dvojice
/// * `x2` - druhá premenná dvojice
/// * `y`  - cieľová premenná
/// * `k`  - počet najbližších susedov pre KSG odhad (ignoruje sa pri diskrétnych dátach)
pub fn estimate_joint_mi_ksg(x1: &[f64], x2: &[f64], y: &[f64], k: usize) -> f64 {
    // Auto-detekcia: diskrétne dáta → 3-premenný histogram
    if unique_count(x1) <= 100 && unique_count(x2) <= 100 && unique_count(y) <= 100 {
        return estimate_joint_mi_discrete(x1, x2, y);
    }
    let n = x1.len();
    if n <= k || x1.len() != x2.len() || x1.len() != y.len() {
        return 0.0;
    }

    let k_eff = k.min(n - 1);
    if n <= k_eff {
        return 0.0;
    }

    // 3D KD-tree: body (x1, x2, y)
    let points: Vec<[f64; 3]> = (0..n).map(|i| [x1[i], x2[i], y[i]]).collect();
    let mut tree = KdTree::with_capacity(3, n);
    for i in 0..n {
        tree.add(&points[i], i).unwrap();
    }

    // 2D KD-tree pre (x1, x2) marginál — O(log n) range query namiesto O(n) brute-force
    let points_xy: Vec<[f64; 2]> = (0..n).map(|i| [x1[i], x2[i]]).collect();
    let mut tree_xy = KdTree::with_capacity(2, n);
    for i in 0..n {
        tree_xy.add(&points_xy[i], i).unwrap();
    }

    // Zoradený index pre y marginal (1D range-counting)
    let y_sorted = SortedIndex::build(y);

    let mut nxy_vec = Vec::with_capacity(n);
    let mut ny_vec = Vec::with_capacity(n);

    for i in 0..n {
        // k-NN v 3D priestore (Chebyshevova vzdialenosť)
        let neighbors = tree.nearest(&points[i], k_eff + 1, &chebyshev).unwrap();

        let epsilon = neighbors.iter()
            .filter(|(_, &idx)| idx != i)
            .nth(k_eff - 1)
            .map(|(d, _)| *d)
            .unwrap_or(f64::INFINITY);

        // O(log n) range query v (x1, x2) marginále pomocou 2D KD-tree
        let nxy = if epsilon == f64::INFINITY {
            0
        } else {
            tree_xy
                .within(&points_xy[i], epsilon, &chebyshev)
                .unwrap()
                .into_iter()
                .filter(|(_, idx)| **idx != i)
                .count()
        };

        // Počítanie v y marginále (zoradený index pre O(log n))
        let ny = y_sorted.count_within(y[i], epsilon, i);

        nxy_vec.push(nxy);
        ny_vec.push(ny);
    }

    // KSG vzorec: MI = ψ(k) - <ψ(n_xy+1) + ψ(n_y+1)> + ψ(N)
    let psi_k = digamma(k_eff as f64);
    let psi_n = digamma(n as f64);
    let mean_psi: f64 = (0..n)
        .map(|i| digamma((nxy_vec[i] + 1) as f64) + digamma((ny_vec[i] + 1) as f64))
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

fn columns_fingerprint(columns: &[Vec<f64>], tag: u64, k: usize) -> u64 {
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    columns.len().hash(&mut hasher);
    k.hash(&mut hasher);

    if let Some(first) = columns.first() {
        first.len().hash(&mut hasher);
    }

    for col in columns {
        col.len().hash(&mut hasher);
        for v in col {
            v.to_bits().hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Prevedie `DenseMatrix<f64>` na stĺpcovú reprezentáciu cez `ndarray`.
///
/// Výhoda: údaje sa do dočasnej 2D štruktúry načítajú raz a následne sa
/// stĺpce vyťahujú cez `Axis(1)` namiesto opakovaného `x.get((row,col))`
/// v každom selektore osobitne.
pub fn dense_matrix_to_columns_ndarray(x: &DenseMatrix<f64>) -> Vec<Vec<f64>> {
    let (rows, cols) = x.shape();
    let mut flat = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            flat.push(*x.get((i, j)));
        }
    }

    let arr = match Array2::from_shape_vec((rows, cols), flat) {
        Ok(v) => v,
        Err(_) => return vec![vec![]; cols],
    };

    arr.axis_iter(Axis(1)).map(|col| col.to_vec()).collect()
}

/// Vypočíta (alebo načíta z cache) Pearsonovu korelačnú maticu.
pub fn compute_corr_matrix_cached(columns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let key = columns_fingerprint(columns, 0xC0AA_u64, 0);

    if let Ok(cache) = MATRIX_CACHE_SINGLETON.lock() {
        if let Some(hit) = cache.get_corr(key) {
            return hit;
        }
    }

    let num_cols = columns.len();
    let mut corr_matrix = vec![vec![0.0f64; num_cols]; num_cols];

    for i in 0..num_cols {
        corr_matrix[i][i] = 1.0;
        for j in (i + 1)..num_cols {
            let c = pearson_correlation(&columns[i], &columns[j]);
            corr_matrix[i][j] = c;
            corr_matrix[j][i] = c;
        }
    }

    if let Ok(mut cache) = MATRIX_CACHE_SINGLETON.lock() {
        cache.insert_corr(key, corr_matrix.clone());
    }

    corr_matrix
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

/// Vypočíta (alebo načíta z cache) MI maticu pre dané stĺpce a `k`.
pub fn compute_mi_matrix_cached(columns: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let key = columns_fingerprint(columns, 0xBEEF_u64, k);

    if let Ok(cache) = MATRIX_CACHE_SINGLETON.lock() {
        if let Some(hit) = cache.get_mi(key) {
            return hit;
        }
    }

    let mi = compute_mi_matrix(columns, k);

    if let Ok(mut cache) = MATRIX_CACHE_SINGLETON.lock() {
        cache.insert_mi(key, mi.clone());
    }

    mi
}

/// Invertuje maticu pomocou Gauss-Jordan eliminácie s parciálnym pivotovaním.
/// Vracia None ak je matica singulárna (max pivot < 1e-12).
pub fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 { return None; }

    // Augmented matrix [mat | I]
    let mut aug: Vec<Vec<f64>> = mat.iter().enumerate().map(|(i, row)| {
        let mut r = row.clone();
        for j in 0..n {
            r.push(if i == j { 1.0 } else { 0.0 });
        }
        r
    }).collect();

    for col in 0..n {
        // Partial pivoting
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n {
            if aug[row][col].abs() > max_val {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return None; // Singulárna
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n) {
            aug[col][j] /= pivot;
        }

        for row in 0..n {
            if row == col { continue; }
            let factor = aug[row][col];
            for j in 0..(2 * n) {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    Some(aug.iter().map(|row| row[n..].to_vec()).collect())
}
