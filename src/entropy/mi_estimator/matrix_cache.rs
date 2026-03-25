use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Mutex;
use once_cell::sync::Lazy;

/// Interná cache pre korelačné a MI matice.
/// Zabraňuje opakovaným výpočtom pri rovnakých vstupných dátach.
#[derive(Default)]
pub(super) struct MatrixCache
{
    pub corr_matrices: HashMap<u64, Vec<Vec<f64>>>,
    pub mi_matrices: HashMap<u64, Vec<Vec<f64>>>,
}

impl MatrixCache
{
    const MAX_CACHE_ITEMS: usize = 8;

    pub fn get_corr(&self, key: u64) -> Option<Vec<Vec<f64>>>
    {
        self.corr_matrices.get(&key).cloned()
    }

    pub fn insert_corr(&mut self, key: u64, value: Vec<Vec<f64>>)
    {
        self.corr_matrices.insert(key, value);
        self.trim_if_needed();
    }

    pub fn get_mi(&self, key: u64) -> Option<Vec<Vec<f64>>>
    {
        self.mi_matrices.get(&key).cloned()
    }

    pub fn insert_mi(&mut self, key: u64, value: Vec<Vec<f64>>)
    {
        self.mi_matrices.insert(key, value);
        self.trim_if_needed();
    }

    fn trim_if_needed(&mut self)
    {
        if self.corr_matrices.len() > Self::MAX_CACHE_ITEMS
        {
            self.corr_matrices.clear();
        }
        if self.mi_matrices.len() > Self::MAX_CACHE_ITEMS
        {
            self.mi_matrices.clear();
        }
    }
}

pub(super) static MATRIX_CACHE_SINGLETON: Lazy<Mutex<MatrixCache>> =
    Lazy::new(|| Mutex::new(MatrixCache::default()));

/// Vypočíta hash vstupných stĺpcov spolu s tagom a k pre cache kľúč.
pub(super) fn columns_fingerprint(columns: &[Vec<f64>], tag: u64, k: usize) -> u64
{
    let mut hasher = DefaultHasher::new();
    tag.hash(&mut hasher);
    columns.len().hash(&mut hasher);
    k.hash(&mut hasher);

    if let Some(first) = columns.first()
    {
        first.len().hash(&mut hasher);
    }

    for col in columns
    {
        col.len().hash(&mut hasher);
        for v in col
        {
            v.to_bits().hash(&mut hasher);
        }
    }

    hasher.finish()
}

/// Adaptívne k: pre veľké datasety sa automaticky zníži k,
/// pretože väčší počet bodov kompenzuje menší k_neighbors.
pub(super) fn adaptive_k(n: usize, k: usize) -> usize
{
    if n > 1000
    {
        k.min(4)
    }
    else if n > 500
    {
        k.min(8)
    }
    else
    {
        k
    }
}
