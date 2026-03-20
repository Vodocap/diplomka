use linreg_core::core::ols_regression;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::Array;
use super::IModel;

/// Polynomial Regression — expanduje každý feature na stĺpce x, x², ..., x^d
/// potom fituje OLS na rozšírenej matici.
/// Interakčné termy NIE SÚ zahrnuté (model neexploduje pre veľa features).
pub struct PolyRegWrapper
{
    /// Koeficienty: [intercept, x1, x2, ..., xp, x1², x2², ..., xp², ...]
    coefficients: Option<Vec<f64>>,
    /// Fallback predikcia pri numerickom zlyhaní modelu.
    y_mean: f64,
    degree: usize,
    n_features: usize,
}

impl PolyRegWrapper
{
    /// Vytvori novu instanciu s degree=2.
    pub fn new() -> Self
    {
        Self
        {
            coefficients: None,
            y_mean: 0.0,
            degree: 2,
            n_features: 0,
        }
    }

    /// Expanduje jeden riadok features na polynomiálne features.
    /// [x1, x2, ..., xp] → [x1, ..., xp, x1², ..., xp², ..., x1^d, ..., xp^d]
    fn expand_row(row: &[f64], degree: usize) -> Vec<f64>
    {
        let mut expanded = row.to_vec();
        for d in 2..=degree
        {
            for &x in row
            {
                expanded.push(x.powi(d as i32));
            }
        }
        expanded
    }
}

impl IModel for PolyRegWrapper
{
    fn get_name(&self) -> &str { "Polynomiálna Regresia" }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["degree"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>
    {
        match key
        {
            "degree" =>
            {
                let d = value.parse::<usize>().map_err(|_| "degree musí byť celé číslo (1-8)")?;
                if d < 1 || d > 8
                {
                    return Err("degree musí byť medzi 1 a 8".into());
                }
                self.degree = d;
                Ok(())
            }
            _ => Err(format!("Neznámy parameter '{}' pre Polynomial Regression", key)),
        }
    }

    /// Natrenuuje OLS na polynomialne expandovanej matici. Pri neplatnych koeficientoch
    /// (napr. silna multikolinearita) ulozi None a predict() pouzije fallback y_mean.
    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>)
    {
        let (n_rows, n_cols) = x.shape();
        self.n_features = n_cols;
        self.y_mean = if y.is_empty()
        {
            0.0
        }
        else
        {
            y.iter().sum::<f64>() / y.len() as f64
        };

        // Zostav stĺpcové vektory pre ols_regression.
        // Poradie: x1..xp, x1²..xp², ..., x1^d..xp^d
        let total_poly_cols = n_cols * self.degree;
        let mut columns: Vec<Vec<f64>> = vec![Vec::with_capacity(n_rows); total_poly_cols];

        for i in 0..n_rows
        {
            let row: Vec<f64> = (0..n_cols).map(|j| *x.get((i, j))).collect();
            let expanded = Self::expand_row(&row, self.degree);
            for (col_idx, &val) in expanded.iter().enumerate()
            {
                columns[col_idx].push(val);
            }
        }

        // Zostav názvy pre koeficienty
        let mut names: Vec<String> = vec!["Intercept".to_string()];
        for j in 0..n_cols
        {
            names.push(format!("x{}", j + 1));
        }
        for d in 2..=self.degree
        {
            for j in 0..n_cols
            {
                names.push(format!("x{}^{}", j + 1, d));
            }
        }

        match ols_regression(&y, &columns, &names)
        {
            Ok(result) =>
            {
                // OLS môže vrátiť neplatné koeficienty pri silnej multikolinearite.
                // V takom prípade nepoužijeme model a predict() použije fallback y_mean.
                if result.coefficients.iter().all(|c| c.is_finite())
                {
                    self.coefficients = Some(result.coefficients);
                }
                else
                {
                    self.coefficients = None;
                    web_sys::console::warn_1(
                        &"Polynomial Regression produced non-finite coefficients; using fallback mean prediction".into()
                    );
                }
            }
            Err(e) =>
            {
                self.coefficients = None;
                web_sys::console::error_1(
                    &format!("Polynomial Regression fit failed: {:?}", e).into()
                );
            }
        }
    }

    /// Expanduje vstup, aplikuje OLS koeficienty a vrati predikciu.
    /// Ak model zlyhal pocas trenovania, vracia fallback y_mean.
    fn predict(&self, input: &[f64]) -> Vec<f64>
    {
        match &self.coefficients
        {
            None => vec![self.y_mean],
            Some(coefs) =>
            {
                let expanded = Self::expand_row(input, self.degree);

                // coefs[0] = intercept, coefs[1..] = feature koeficienty
                let mut pred = coefs[0];
                for (i, &x_val) in expanded.iter().enumerate()
                {
                    if i + 1 < coefs.len()
                    {
                        pred += coefs[i + 1] * x_val;
                    }
                }
                if pred.is_finite() { vec![pred] } else { vec![self.y_mean] }
            }
        }
    }
}
