use smartcore::tree::decision_tree_regressor::{DecisionTreeRegressor, DecisionTreeRegressorParameters};
use smartcore::linalg::basic::matrix::DenseMatrix;
use super::IModel;

/// Wrapper okolo smartcore DecisionTreeRegressor.
/// Strom rekurzivne deli priestor features podla minimalizacie MSE.
/// max_depth ohranicuje hĺbku stromu, min_samples_split minimálny pocet vzoriek pre delenie uzla.
pub struct TreeWrapper
{
    model: Option<DecisionTreeRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    max_depth: u16,
    min_samples_split: u16,
}

impl TreeWrapper
{
    /// Vytvori novu instanciu s max_depth=10 a min_samples_split=2.
    pub fn new() -> Self
    {
        Self
        {
            model: None,
            max_depth: 10,
            min_samples_split: 2,
        }
    }
}

impl IModel for TreeWrapper
{
    fn get_name(&self) -> &str
    {
        "Decision Tree"
    }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["max_depth", "min_samples_split"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>
    {
        match key
        {
            "max_depth" =>
            {
                self.max_depth = value.parse().map_err(|_| "Invalid depth")?;
                Ok(())
            }
            "min_samples_split" =>
            {
                self.min_samples_split = value.parse().map_err(|_| "Invalid split value")?;
                Ok(())
            }
            _ => Err("Param not found".into())
        }
    }

    /// Postavi rozhodovaci strom na trenovacich datach podla nastavenych parametrov.
    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>)
    {
        let mut params = DecisionTreeRegressorParameters::default();
        params.max_depth = Some(self.max_depth as u16);
        params.min_samples_split = self.min_samples_split as usize;

        match DecisionTreeRegressor::fit(&x, &y, params)
        {
            Ok(m) => self.model = Some(m),
            Err(e) => web_sys::console::error_1(&format!("Decision Tree fit failed: {:?}", e).into()),
        }
    }

    /// Prechádza natrenuovanym stromom od korena a vracia predikciu listu, v ktorom vstup skonci.
    fn predict(&self, input: &[f64]) -> Vec<f64>
    {
        if let Some(ref m) = self.model
        {
            let x = DenseMatrix::from_2d_vec(&vec![input.to_vec()]).unwrap();
            m.predict(&x).unwrap_or_default()
        }
        else
        {
            vec![]
        }
    }

    fn get_param_definitions(&self) -> Vec<crate::processing::processor_param::ProcessorParam>
    {
        vec![
            crate::processing::processor_param::ProcessorParam {
                name: "max_depth".to_string(),
                param_type: "number".to_string(),
                default_value: "10".to_string(),
                description: "Maximalna hlbka stromu".to_string(),
                min: Some(1.0),
                max: Some(50.0),
                options: None,
            },
        ]
    }
}
