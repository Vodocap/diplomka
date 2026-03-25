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

mod matrix_cache;   // MatrixCache struct + MATRIX_CACHE_SINGLETON + cache helpers
mod sorted_index;   // SortedIndex struct + chebyshev distance fn

use matrix_cache::{MATRIX_CACHE_SINGLETON, columns_fingerprint, adaptive_k};
use sorted_index::{SortedIndex, chebyshev};

use statrs::function::gamma::digamma;
use kdtree::KdTree;
use std::collections::HashMap;
use ndarray::{Array2, Axis};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

// Manuálny SIMD-like súčet pre zrýchlenie
fn fast_sum(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    let len = values.len();
    let mut i = 0;
    while i + 4 <= len {
        sum += values[i] + values[i+1] + values[i+2] + values[i+3];
        i += 4;
    }
    for j in i..len {
        sum += values[j];
    }
    sum
}

// ═══════════════════════════════════════════════════════════════════════
//  Zdieľané štatistické utility
// ═══════════════════════════════════════════════════════════════════════

/// Pearsonov korelačný koeficient medzi dvoma vektormi.
/// Vracia hodnotu v rozsahu [-1, 1].
/// Optimalizované pre rýchlosť s manuálnym SIMD-like spracovaním.
pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64
{
    let n = x.len() as f64;
    if n == 0.0 || x.len() != y.len()
    {
        return 0.0;
    }

    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    // Manuálne SIMD-like spracovanie po 4 prvkoch
    let len = x.len();
    let mut i = 0;
    while i + 4 <= len {
        let dx0 = x[i] - mean_x;
        let dx1 = x[i+1] - mean_x;
        let dx2 = x[i+2] - mean_x;
        let dx3 = x[i+3] - mean_x;

        let dy0 = y[i] - mean_y;
        let dy1 = y[i+1] - mean_y;
        let dy2 = y[i+2] - mean_y;
        let dy3 = y[i+3] - mean_y;

        num += dx0 * dy0 + dx1 * dy1 + dx2 * dy2 + dx3 * dy3;
        den_x += dx0 * dx0 + dx1 * dx1 + dx2 * dx2 + dx3 * dx3;
        den_y += dy0 * dy0 + dy1 * dy1 + dy2 * dy2 + dy3 * dy3;

        i += 4;
    }

    // Zvyšné prvky
    for j in i..len {
        let dx = x[j] - mean_x;
        let dy = y[j] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }

    let den = (den_x * den_y).sqrt();
    if den == 0.0
    {
        0.0
    }
    else
    {
        num / den
    }
}

// ═══════════════════════════════════════════════════════════════════════
//  Diskrétny (histogram) MI estimátor
// ═══════════════════════════════════════════════════════════════════════

/// Vráti počet unikátnych hodnôt v stĺpci.
fn unique_count(x: &[f64]) -> usize
{
    let mut seen: HashMap<u64, ()> = HashMap::new();
    for &v in x
    {
        seen.insert(v.to_bits(), ());
    }
    seen.len()
}

/// Diskrétny MI estimátor pre ordinálne/kategoriálne premenné.
/// Buduje spoločnú frekvenčnú tabuľku a počíta MI = Σ P(x,y) log(P(x,y)/(P(x)P(y))).
/// Presný, O(n), vhodný pre dáta s malým počtom unikátnych hodnôt.
fn estimate_mi_discrete(x: &[f64], y: &[f64]) -> f64
{
    let n = x.len();
    if n == 0 || x.len() != y.len()
    {
        return 0.0;
    }

    let mut x_map: HashMap<u64, usize> = HashMap::new();
    let mut y_map: HashMap<u64, usize> = HashMap::new();
    for &v in x
    {
        let l = x_map.len();
        x_map.entry(v.to_bits()).or_insert(l);
    }
    for &v in y
    {
        let l = y_map.len();
        y_map.entry(v.to_bits()).or_insert(l);
    }

    let nx = x_map.len();
    let ny = y_map.len();

    let mut joint = vec![vec![0usize; ny]; nx];
    for i in 0..n
    {
        let xi = x_map[&x[i].to_bits()];
        let yi = y_map[&y[i].to_bits()];
        joint[xi][yi] += 1;
    }

    let mut x_count: Vec<usize> = Vec::with_capacity(nx);
    for row in &joint
    {
        let mut sum = 0usize;
        for &c in row
        {
            sum += c;
        }
        x_count.push(sum);
    }
    let mut y_count: Vec<usize> = vec![0usize; ny];
    for row in &joint
    {
        for (j, &c) in row.iter().enumerate()
        {
            y_count[j] += c;
        }
    }

    let n_f = n as f64;
    let mut mi = 0.0f64;
    for xi in 0..nx
    {
        for yi in 0..ny
        {
            let c = joint[xi][yi];
            if c == 0
            {
                continue;
            }
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
fn estimate_joint_mi_discrete(x1: &[f64], x2: &[f64], y: &[f64]) -> f64
{
    let n = x1.len();
    if n == 0 || x2.len() != n || y.len() != n
    {
        return 0.0;
    }

    let mut x1_map: HashMap<u64, usize> = HashMap::new();
    let mut x2_map: HashMap<u64, usize> = HashMap::new();
    let mut y_map:  HashMap<u64, usize> = HashMap::new();
    for &v in x1
    {
        let l = x1_map.len();
        x1_map.entry(v.to_bits()).or_insert(l);
    }
    for &v in x2
    {
        let l = x2_map.len();
        x2_map.entry(v.to_bits()).or_insert(l);
    }
    for &v in y
    {
        let l = y_map.len();
        y_map.entry(v.to_bits()).or_insert(l);
    }

    let n1 = x1_map.len();
    let n2 = x2_map.len();
    let ny = y_map.len();

    let mut joint = vec![vec![vec![0usize; ny]; n2]; n1];
    for i in 0..n
    {
        let i1 = x1_map[&x1[i].to_bits()];
        let i2 = x2_map[&x2[i].to_bits()];
        let iy = y_map[&y[i].to_bits()];
        joint[i1][i2][iy] += 1;
    }

    let mut xy_count = vec![vec![0usize; n2]; n1];
    let mut y_count  = vec![0usize; ny];
    for i1 in 0..n1
    {
        for i2 in 0..n2
        {
            for iy in 0..ny
            {
                let c = joint[i1][i2][iy];
                xy_count[i1][i2] += c;
                y_count[iy] += c;
            }
        }
    }

    let n_f = n as f64;
    let mut mi = 0.0f64;
    for i1 in 0..n1
    {
        for i2 in 0..n2
        {
            for iy in 0..ny
            {
                let c = joint[i1][i2][iy];
                if c == 0
                {
                    continue;
                }
                let pxxy = c as f64 / n_f;
                let pxy  = xy_count[i1][i2] as f64 / n_f;
                let py   = y_count[iy] as f64 / n_f;
                mi += pxxy * (pxxy / (pxy * py)).ln();
            }
        }
    }
    mi.max(0.0)
}

// ═══════════════════════════════════════════════════════════════════════
//  KSG estimátor
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
/// * `k` - počet najbližších susedov (ignoruje sa pri diskrétnych dátach)
pub fn estimate_mi_ksg(x: &[f64], y: &[f64], k: usize) -> f64
{
    if unique_count(x) <= 100 && unique_count(y) <= 100
    {
        return estimate_mi_discrete(x, y);
    }
    let n = x.len();
    if n <= k
    {
        return 0.0;
    }

    let k_eff = k.min(n - 1);
    if n <= k_eff
    {
        return 0.0;
    }

    let mut points: Vec<[f64; 2]> = Vec::with_capacity(n);
    for i in 0..n
    {
        points.push([x[i], y[i]]);
    }
    let mut tree = KdTree::with_capacity(2, n);
    for i in 0..n
    {
        tree.add(&points[i], i).unwrap();
    }

    let x_sorted = SortedIndex::build(x);
    let y_sorted = SortedIndex::build(y);

    let mut nx_vec = Vec::with_capacity(n);
    let mut ny_vec = Vec::with_capacity(n);

    for i in 0..n
    {
        let neighbors = tree.nearest(&[x[i], y[i]], k_eff + 1, &chebyshev).unwrap();

        let mut count = 0usize;
        let mut epsilon = f64::INFINITY;
        for (d, &idx) in neighbors.iter()
        {
            if idx == i
            {
                continue;
            }
            if count == k_eff - 1
            {
                epsilon = *d;
                break;
            }
            count += 1;
        }

        let nx = x_sorted.count_within(x[i], epsilon, i);
        let ny = y_sorted.count_within(y[i], epsilon, i);

        nx_vec.push(nx);
        ny_vec.push(ny);
    }

    let psi_k = digamma(k_eff as f64);
    let psi_n = digamma(n as f64);
    let mut psi_values = Vec::with_capacity(n);
    for i in 0..n
    {
        psi_values.push(digamma((nx_vec[i] + 1) as f64) + digamma((ny_vec[i] + 1) as f64));
    }
    let mean_psi = fast_sum(&psi_values) / n as f64;

    (psi_k - mean_psi + psi_n).max(0.0)
}

/// KSG estimátor Joint Mutual Information MI((X1, X2); Y).
/// Meria koľko informácie dvojica premenných (X1, X2) spoločne nesie o cieľovej Y.
/// Používa 3D KD-tree s Chebyshevovou vzdialenosťou.
pub fn estimate_joint_mi_ksg(x1: &[f64], x2: &[f64], y: &[f64], k: usize) -> f64
{
    if unique_count(x1) <= 100 && unique_count(x2) <= 100 && unique_count(y) <= 100
    {
        return estimate_joint_mi_discrete(x1, x2, y);
    }
    let n = x1.len();
    if n <= k || x1.len() != x2.len() || x1.len() != y.len()
    {
        return 0.0;
    }

    let k_eff = k.min(n - 1);
    if n <= k_eff
    {
        return 0.0;
    }

    let mut points: Vec<[f64; 3]> = Vec::with_capacity(n);
    for i in 0..n
    {
        points.push([x1[i], x2[i], y[i]]);
    }
    let mut tree = KdTree::with_capacity(3, n);
    for i in 0..n
    {
        tree.add(&points[i], i).unwrap();
    }

    let mut points_xy: Vec<[f64; 2]> = Vec::with_capacity(n);
    for i in 0..n
    {
        points_xy.push([x1[i], x2[i]]);
    }
    let mut tree_xy = KdTree::with_capacity(2, n);
    for i in 0..n
    {
        tree_xy.add(&points_xy[i], i).unwrap();
    }

    let y_sorted = SortedIndex::build(y);

    let mut nxy_vec = Vec::with_capacity(n);
    let mut ny_vec = Vec::with_capacity(n);

    for i in 0..n
    {
        let neighbors = tree.nearest(&points[i], k_eff + 1, &chebyshev).unwrap();

        let mut count = 0usize;
        let mut epsilon = f64::INFINITY;
        for (d, &idx) in neighbors.iter()
        {
            if idx == i
            {
                continue;
            }
            if count == k_eff - 1
            {
                epsilon = *d;
                break;
            }
            count += 1;
        }

        let nxy;
        if epsilon == f64::INFINITY
        {
            nxy = 0;
        }
        else
        {
            let within = tree_xy.within(&points_xy[i], epsilon, &chebyshev).unwrap();
            let mut within_count = 0usize;
            for (_, idx) in within
            {
                if *idx != i
                {
                    within_count += 1;
                }
            }
            nxy = within_count;
        }

        let ny = y_sorted.count_within(y[i], epsilon, i);

        nxy_vec.push(nxy);
        ny_vec.push(ny);
    }

    let psi_k = digamma(k_eff as f64);
    let psi_n = digamma(n as f64);
    let mut psi_values = Vec::with_capacity(n);
    for i in 0..n
    {
        psi_values.push(digamma((nxy_vec[i] + 1) as f64) + digamma((ny_vec[i] + 1) as f64));
    }
    let mean_psi = fast_sum(&psi_values) / n as f64;

    (psi_k - mean_psi + psi_n).max(0.0)
}

// ═══════════════════════════════════════════════════════════════════════
//  Maticové operácie a cache
// ═══════════════════════════════════════════════════════════════════════

/// Prevedie `DenseMatrix<f64>` na stĺpcovú reprezentáciu cez `ndarray`.
pub fn dense_matrix_to_columns_ndarray(x: &DenseMatrix<f64>) -> Vec<Vec<f64>>
{
    let (rows, cols) = x.shape();
    let mut flat = Vec::with_capacity(rows * cols);
    for i in 0..rows
    {
        for j in 0..cols
        {
            flat.push(*x.get((i, j)));
        }
    }

    let arr = match Array2::from_shape_vec((rows, cols), flat)
    {
        Ok(v) => v,
        Err(_) => return vec![vec![]; cols],
    };

    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(cols);
    for col in arr.axis_iter(Axis(1))
    {
        columns.push(col.to_vec());
    }
    columns
}

/// Vypočíta (alebo načíta z cache) Pearsonovu korelačnú maticu.
pub fn compute_corr_matrix_cached(columns: &[Vec<f64>]) -> Vec<Vec<f64>>
{
    let key = columns_fingerprint(columns, 0xC0AA_u64, 0);

    if let Ok(cache) = MATRIX_CACHE_SINGLETON.lock()
    {
        if let Some(hit) = cache.get_corr(key)
        {
            return hit;
        }
    }

    let num_cols = columns.len();
    let mut corr_matrix = vec![vec![0.0f64; num_cols]; num_cols];

    for i in 0..num_cols
    {
        corr_matrix[i][i] = 1.0;
        for j in (i + 1)..num_cols
        {
            let c = pearson_correlation(&columns[i], &columns[j]);
            corr_matrix[i][j] = c;
            corr_matrix[j][i] = c;
        }
    }

    if let Ok(mut cache) = MATRIX_CACHE_SINGLETON.lock()
    {
        cache.insert_corr(key, corr_matrix.clone());
    }

    corr_matrix
}

/// Vypočíta symetrickú MI maticu pre všetky páry stĺpcov.
/// Počíta len vrchný trojuholník a zrkadlí (MI je symetrická).
pub fn compute_mi_matrix(columns: &[Vec<f64>], k: usize) -> Vec<Vec<f64>>
{
    let num_cols = columns.len();
    let n;
    if num_cols > 0
    {
        n = columns[0].len();
    }
    else
    {
        n = 0;
    }
    let k_eff = adaptive_k(n, k);

    let mut mi_matrix = vec![vec![0.0f64; num_cols]; num_cols];

    for i in 0..num_cols
    {
        for j in (i + 1)..num_cols
        {
            let mi = estimate_mi_ksg(&columns[i], &columns[j], k_eff);
            mi_matrix[i][j] = mi;
            mi_matrix[j][i] = mi;
        }
    }

    mi_matrix
}

/// Vypočíta (alebo načíta z cache) MI maticu pre dané stĺpce a `k`.
pub fn compute_mi_matrix_cached(columns: &[Vec<f64>], k: usize) -> Vec<Vec<f64>>
{
    let key = columns_fingerprint(columns, 0xBEEF_u64, k);

    if let Ok(cache) = MATRIX_CACHE_SINGLETON.lock()
    {
        if let Some(hit) = cache.get_mi(key)
        {
            return hit;
        }
    }

    let mi = compute_mi_matrix(columns, k);

    if let Ok(mut cache) = MATRIX_CACHE_SINGLETON.lock()
    {
        cache.insert_mi(key, mi.clone());
    }

    mi
}

/// Invertuje maticu pomocou Gauss-Jordan eliminácie s parciálnym pivotovaním.
/// Vracia None ak je matica singulárna (max pivot < 1e-12).
pub fn invert_matrix(mat: &[Vec<f64>]) -> Option<Vec<Vec<f64>>>
{
    let n = mat.len();
    if n == 0
    {
        return None;
    }

    let mut aug: Vec<Vec<f64>> = Vec::with_capacity(n);
    for (i, row) in mat.iter().enumerate()
    {
        let mut r = row.clone();
        for j in 0..n
        {
            if i == j
            {
                r.push(1.0);
            }
            else
            {
                r.push(0.0);
            }
        }
        aug.push(r);
    }

    for col in 0..n
    {
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..n
        {
            if aug[row][col].abs() > max_val
            {
                max_val = aug[row][col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-12
        {
            return None; // Singulárna
        }
        aug.swap(col, max_row);

        let pivot = aug[col][col];
        for j in 0..(2 * n)
        {
            aug[col][j] /= pivot;
        }

        for row in 0..n
        {
            if row == col
            {
                continue;
            }
            let factor = aug[row][col];
            for j in 0..(2 * n)
            {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    let mut result: Vec<Vec<f64>> = Vec::with_capacity(n);
    for row in &aug
    {
        result.push(row[n..].to_vec());
    }
    Some(result)
}
