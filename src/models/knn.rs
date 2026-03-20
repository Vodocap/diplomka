use smartcore::neighbors::knn_regressor::{KNNRegressor, KNNRegressorParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::distance::euclidian::Euclidian;
use super::IModel;

/// Wrapper okolo smartcore KNNRegressor pouzivajuci Euklidovsku vzdialenost.
/// KNN je lazy learner — trenovacie data sa iba ulozia; skutocna praca prebieha az pri predikcii.
/// Predikcia je priemer cielovych hodnot k najblizich susedov.
pub struct KnnWrapper
{
    model: Option<KNNRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>, Euclidian<f64>>>,
    k: usize,
}

impl KnnWrapper
{
    /// Vytvori novu instanciu s k=5.
    pub fn new() -> Self
    {
        Self { model: None, k: 5 }
    }
}

impl IModel for KnnWrapper
{
    fn get_name(&self) -> &str { "K-Nearest Neighbors" }

    /// Ulozi trenovacie data do internej struktury (lazy fitting).
    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>)
    {
        let mut params = KNNRegressorParameters::default();
        params.k = self.k;
        match KNNRegressor::fit(&x, &y, params)
        {
            Ok(m) => self.model = Some(m),
            Err(e) => web_sys::console::error_1(&format!("KNN fit failed: {:?}", e).into()),
        }
    }

    /// Najde k najblizsich susedov pomocou Euklidovskej vzdialenosti a vrati priemer ich hodnot.
    fn predict(&self, input: &[f64]) -> Vec<f64>
    {
        let x = DenseMatrix::from_2d_vec(&vec![input.to_vec()]).unwrap();
        self.model.as_ref()
            .and_then(|m: &KNNRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>, Euclidian<f64>>| m.predict(&x).ok())
            .unwrap_or_default()
    }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["k"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>
    {
        match key
        {
            "k" =>
            {
                let new_k = value.parse::<usize>().map_err(|_| "K must be a number")?;
                self.k = new_k;
                Ok(())
            }
            _ => Err(format!("Unknown parameter {} for KNN", key)),
        }
    }

    fn get_param_definitions(&self) -> Vec<crate::processing::processor_param::ProcessorParam>
    {
        vec![
            crate::processing::processor_param::ProcessorParam {
                name: "k".to_string(),
                param_type: "number".to_string(),
                default_value: "5".to_string(),
                description: "Pocet najblizsich susedov".to_string(),
                min: Some(1.0),
                max: Some(100.0),
                options: None,
            },
        ]
    }

}
