use smartcore::linalg::basic::matrix::DenseMatrix;
use crate::processing::processor_param::ProcessorParam;

/// Spolocne rozhranie pre vsetky ML modely (Strategy pattern).
/// Kazdy model implementuje trato, co umoznuje pipeline pracovat s lubovolnym modelom
/// bez zavislosti na konkretnom type.
pub trait IModel
{
    /// Vracia zobrazitelny nazov modelu.
    fn get_name(&self) -> &str;

    /// Natrenuuje model na poskytnutych trenovacich datach.
    /// x_train je matica vzoriek x atributov, y_train je cielovy vektor.
    fn train(&mut self, x_train: DenseMatrix<f64>, y_train: Vec<f64>);

    /// Vrati predikciu pre jeden vstupny riadok ako vektor (zvycajne jeden prvok).
    fn predict(&self, input: &[f64]) -> Vec<f64>;

    /// Vracia zoznam nazvov hyperparametrov, ktore model podporuje.
    fn get_supported_params(&self) -> Vec<&str>;

    /// Nastavi hyperparameter podla nazvu a hodnoty zadanej ako retazec.
    /// Vracia chybu, ak parameter neexistuje alebo sa neda parsovat.
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;

    /// Vracia detailne definicie parametrov pre dynamicke generovanie UI.
    fn get_param_definitions(&self) -> Vec<ProcessorParam>
    {
        vec![]
    }
}
