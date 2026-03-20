use smartcore::svm::svr::{SVR, SVRParameters};
use smartcore::svm::Kernels;
use smartcore::linalg::basic::matrix::DenseMatrix;
use super::IModel;

/// Wrapper okolo smartcore SVR (Support Vector Regression).
/// Podporuje RBF a linearny kernel. Obsahuje lifetime workaround cez Box::leak,
/// pretoze smartcore SVR viaze lifetimes vstupnych dat a parametrov — pozri komentare v train().
pub struct SvmWrapper
{
    // Safety: SVR<'static> is valid because we use Box::leak for params.
    // predict() is called with a pointer cast (see predict impl).
    model: Option<SVR<'static, f64, DenseMatrix<f64>, Vec<f64>>>,
    c: f64,
    eps: f64,
    kernel: String,  // "rbf" alebo "linear"
    gamma: f64,      // pre RBF kernel
}

impl SvmWrapper
{
    /// Vytvori novu instanciu s C=1.0, eps=0.1, RBF kernelom a gamma=0.1.
    pub fn new() -> Self
    {
        Self
        {
            model: None,
            c: 1.0,
            eps: 0.1,
            kernel: "rbf".to_string(),
            gamma: 0.1,
        }
    }
}

impl IModel for SvmWrapper
{
    fn get_name(&self) -> &str { "Support Vector Machine" }

    fn get_supported_params(&self) -> Vec<&str>
    {
        vec!["c", "eps", "kernel", "gamma"]
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>
    {
        match key
        {
            "c" =>
            {
                self.c = value.parse().map_err(|_| "C musí byť desatinné číslo")?;
                Ok(())
            }
            "eps" =>
            {
                self.eps = value.parse().map_err(|_| "eps musí byť desatinné číslo")?;
                Ok(())
            }
            "kernel" =>
            {
                if value == "rbf" || value == "linear"
                {
                    self.kernel = value.to_string();
                    Ok(())
                }
                else
                {
                    Err("Podporované kernely sú: rbf, linear".into())
                }
            }
            "gamma" =>
            {
                self.gamma = value.parse().map_err(|_| "gamma musí byť desatinné číslo")?;
                Ok(())
            }
            _ => Err(format!("Neznámy parameter '{}' pre SVM", key)),
        }
    }

    /// Natrenuuje SVR. Pouziva Box::leak na ziskanie 'static referencie pre params a pointer casty pre x/y.
    /// Toto je nutne kvoli obmedzeniu smartcore API, nie kvoli skutocnemu zdielaniu dat.
    fn train(&mut self, x: DenseMatrix<f64>, y: Vec<f64>)
    {
        let mut params = SVRParameters::default();
        params.c = self.c;
        params.eps = self.eps;

        if self.kernel == "linear"
        {
            params.kernel = Some(Kernels::linear());
        }
        else
        {
            params.kernel = Some(Kernels::RBF { gamma: Some(self.gamma) });
        }

        // Box::leak gives a 'static reference for params (leaks ~40 bytes, acceptable).
        let params_static: &'static SVRParameters<f64> = Box::leak(Box::new(params));

        // Safety: x and y are alive during the entire SVR::fit() call.
        // SVR stores only owned copies of support vectors, not references to x/y.
        // Casting to 'static is needed because fit ties all three lifetimes together.
        let x_ptr: *const DenseMatrix<f64> = &x;
        let y_ptr: *const Vec<f64> = &y;
        let x_static: &'static DenseMatrix<f64> = unsafe { &*x_ptr };
        let y_static: &'static Vec<f64> = unsafe { &*y_ptr };

        match SVR::fit(x_static, y_static, params_static)
        {
            Ok(m) => self.model = Some(m),
            Err(e) => web_sys::console::error_1(&format!("SVM fit failed: {:?}", e).into()),
        }
    }

    /// Predikuje hodnotu pre jeden vstupny vektor. Pouziva pointer cast pre kompatibilitu s 'static.
    fn predict(&self, input: &[f64]) -> Vec<f64>
    {
        let x = DenseMatrix::from_2d_vec(&vec![input.to_vec()]).unwrap();
        match &self.model
        {
            None => vec![],
            Some(m) =>
            {
                // Safety: x is alive for the entire duration of predict().
                // The result is owned Vec<f64>, so no reference escapes.
                let x_ptr: *const DenseMatrix<f64> = &x;
                let x_static: &'static DenseMatrix<f64> = unsafe { &*x_ptr };
                m.predict(x_static).unwrap_or_default()
            }
        }
    }

    fn get_param_definitions(&self) -> Vec<crate::processing::processor_param::ProcessorParam>
    {
        vec![
            crate::processing::processor_param::ProcessorParam {
                name: "c".to_string(),
                param_type: "number".to_string(),
                default_value: "1.0".to_string(),
                description: "Regularizacny parameter C".to_string(),
                min: Some(0.001),
                max: Some(1000.0),
                options: None,
            },
            crate::processing::processor_param::ProcessorParam {
                name: "eps".to_string(),
                param_type: "number".to_string(),
                default_value: "0.1".to_string(),
                description: "Epsilon-tube sirka (tolerancia SVR)".to_string(),
                min: Some(0.001),
                max: Some(10.0),
                options: None,
            },
            crate::processing::processor_param::ProcessorParam {
                name: "kernel".to_string(),
                param_type: "select".to_string(),
                default_value: "rbf".to_string(),
                description: "Typ kernelu".to_string(),
                min: None,
                max: None,
                options: Some(vec!["rbf".to_string(), "linear".to_string()]),
            },
            crate::processing::processor_param::ProcessorParam {
                name: "gamma".to_string(),
                param_type: "number".to_string(),
                default_value: "0.1".to_string(),
                description: "RBF kernel gamma koeficient".to_string(),
                min: Some(0.0001),
                max: Some(100.0),
                options: None,
            },
        ]
    }
}
