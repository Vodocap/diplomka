/// Chebyshevova (L∞) vzdialenosť medzi dvoma bodmi.
/// Používa sa ako metrika v KD-tree pre KSG estimátor.
#[inline]
pub(super) fn chebyshev(a: &[f64], b: &[f64]) -> f64
{
    let mut max_dist = 0.0f64;
    for (x, y) in a.iter().zip(b.iter())
    {
        let dist = (x - y).abs();
        if dist > max_dist
        {
            max_dist = dist;
        }
    }
    max_dist
}

/// Zoradené pole hodnôt pre efektívne počítanie bodov v 1D intervale
/// pomocou binárneho vyhľadávania — O(log n) namiesto O(n).
pub(super) struct SortedIndex
{
    sorted: Vec<(f64, usize)>, // (hodnota, pôvodný index)
}

impl SortedIndex
{
    /// Zostaví zoradený index. Zložitosť: O(n log n).
    pub fn build(values: &[f64]) -> Self
    {
        let mut sorted: Vec<(f64, usize)> = Vec::with_capacity(values.len());
        for (i, &v) in values.iter().enumerate()
        {
            sorted.push((v, i));
        }
        sorted.sort_unstable_by(|a, b|
        {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        SortedIndex { sorted }
    }

    /// Spočíta body kde |value - center| < epsilon (striktne), okrem skip_index.
    /// Používa binárne vyhľadávanie na nájdenie rozsahu.
    pub fn count_within(&self, center: f64, epsilon: f64, skip_index: usize) -> usize
    {
        let lo = center - epsilon;
        let hi = center + epsilon;

        // Binárne vyhľadávanie: prvý index kde value > lo
        let start = self.sorted.partition_point(|&(v, _)| v <= lo);
        // Binárne vyhľadávanie: prvý index kde value >= hi
        let end = self.sorted.partition_point(|&(v, _)| v < hi);

        // Počítame body v rozsahu [start, end), preskočíme skip_index
        let mut count = 0usize;
        for i in start..end
        {
            if self.sorted[i].1 != skip_index
            {
                count += 1;
            }
        }
        count
    }
}
