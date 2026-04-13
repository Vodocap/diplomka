use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ColumnType};

/// One-Hot Encoder - kazdu unikatnu hodnotu v stlpci premeni na samostatny binarny stlpec.
/// Vystupna matica ma viac stlpcov ako vstupna (pocet stlpcov = sum(unikatnych hodnot)).
pub struct OneHotEncoder
{
    /// Kategórie pre každý pôvodný stĺpec v poradí výskytu (uložené ako f64 bits).
    categories_per_column: Option<Vec<Vec<u64>>>,
}

impl OneHotEncoder
{
    pub fn new() -> Self
    {
        Self {
            categories_per_column: None,
        }
    }

    fn infer_categories(data: &DenseMatrix<f64>) -> Vec<Vec<u64>>
    {
        let (rows, cols) = data.shape();
        let mut all_categories = Vec::with_capacity(cols);

        for j in 0..cols
        {
            let mut categories: Vec<u64> = Vec::new();
            for i in 0..rows
            {
                let bits = data.get((i, j)).to_bits();
                if !categories.contains(&bits)
                {
                    categories.push(bits);
                }
            }
            all_categories.push(categories);
        }

        all_categories
    }

    fn encode_with_categories(data: &DenseMatrix<f64>, categories_per_column: &[Vec<u64>]) -> DenseMatrix<f64>
    {
        let (rows, cols) = data.shape();
        let used_cols = cols.min(categories_per_column.len());
        let total_new_cols: usize = categories_per_column
            .iter()
            .take(used_cols)
            .map(|c| c.len())
            .sum();

        if total_new_cols == 0
        {
            return data.clone();
        }

        let mut encoded = DenseMatrix::from_2d_vec(&vec![vec![0.0; total_new_cols]; rows]).unwrap();
        let mut col_offset = 0;

        for j in 0..used_cols
        {
            let categories = &categories_per_column[j];

            for i in 0..rows
            {
                let bits = data.get((i, j)).to_bits();
                if let Some(local_idx) = categories.iter().position(|&cat| cat == bits)
                {
                    encoded.set((i, col_offset + local_idx), 1.0);
                }
            }

            col_offset += categories.len();
        }

        encoded
    }

    fn category_bits_to_label(bits: u64) -> String
    {
        let value = f64::from_bits(bits);

        if value.is_nan()
        {
            return "nan".to_string();
        }

        if value.is_infinite()
        {
            return if value.is_sign_positive() { "inf" } else { "-inf" }.to_string();
        }

        if value.fract().abs() < 1e-12
        {
            return format!("{:.0}", value);
        }

        let mut s = format!("{:.12}", value);
        while s.contains('.') && s.ends_with('0')
        {
            s.pop();
        }
        if s.ends_with('.')
        {
            s.pop();
        }
        s
    }

    fn sanitize_label(label: &str) -> String
    {
        let mut out = String::with_capacity(label.len());
        let mut prev_underscore = false;

        for ch in label.chars()
        {
            let keep = ch.is_ascii_alphanumeric() || ch == '-' || ch == '.';
            if keep
            {
                out.push(ch);
                prev_underscore = false;
            }
            else if !prev_underscore
            {
                out.push('_');
                prev_underscore = true;
            }
        }

        out.trim_matches('_').to_string()
    }
}

impl DataProcessor for OneHotEncoder
{
    fn get_name(&self) -> &str
    {
        "One-Hot Encoder"
    }

    fn fit(&mut self, data: &DenseMatrix<f64>)
    {
        self.categories_per_column = Some(Self::infer_categories(data));
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        match &self.categories_per_column
        {
            Some(categories) => Self::encode_with_categories(data, categories),
            None =>
            {
                // Fallback: keď nebolo volané fit(), použijeme kategórie z aktuálnych dát.
                let inferred = Self::infer_categories(data);
                Self::encode_with_categories(data, &inferred)
            }
        }
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64>
    {
        self.transform(data)
    }

    fn set_param(&mut self, _key: &str, _value: &str) -> Result<(), String>
    {
        Err("OneHotEncoder has no configurable parameters".to_string())
    }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec![]
    }

    fn get_output_feature_names(&self, input_feature_names: &[String]) -> Vec<String>
    {
        let categories = match &self.categories_per_column
        {
            Some(c) => c,
            None => return input_feature_names.to_vec(),
        };

        let used_cols = input_feature_names.len().min(categories.len());
        let mut output_names = Vec::new();

        for col_idx in 0..used_cols
        {
            let base = &input_feature_names[col_idx];
            let col_categories = &categories[col_idx];

            if col_categories.is_empty()
            {
                output_names.push(base.clone());
                continue;
            }

            for &bits in col_categories
            {
                let raw = Self::category_bits_to_label(bits);
                let suffix = Self::sanitize_label(&raw);
                let suffix = if suffix.is_empty() { "value" } else { &suffix };
                output_names.push(format!("{}_{}", base, suffix));
            }
        }

        output_names
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>>
    {
        Some(vec![ColumnType::Categorical, ColumnType::Discrete])
    }
}
