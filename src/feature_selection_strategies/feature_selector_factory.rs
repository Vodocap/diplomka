use super::{
    FeatureSelector,
    VarianceSelector,
    ChiSquareSelector,
    MutualInformationSelector,
    SmcSelector,
};

/// Factory pre vytváranie feature selektorov podla nazvu.
/// Centralizuje registraciu vsetkych selektorov a poskytuje metadáta
/// (popis, podporovane typy uloh, parametrove schema) pre UI.
pub struct FeatureSelectorFactory;

impl FeatureSelectorFactory
{
    /// Vytvorí feature selektor na základe názvu
    pub fn create(selector_type: &str) -> Result<Box<dyn FeatureSelector>, String>
    {
        match selector_type
        {
            "variance" => Ok(Box::new(VarianceSelector::new())),
            "chi_square" | "chi2" => Ok(Box::new(ChiSquareSelector::new())),
            "mutual_information" | "mi" => Ok(Box::new(MutualInformationSelector::new())),
            "smc" => Ok(Box::new(SmcSelector::new())),
            _ => Err(format!("Neznámy feature selektor: {}", selector_type)),
        }
    }

    /// Vráti zoznam všetkých dostupných feature selektorov
    pub fn available() -> Vec<&'static str>
    {
        vec![
            "variance",
            "chi_square",
            "mutual_information",
            "smc",
        ]
    }

    /// Vráti popis feature selektora
    pub fn get_description(selector_type: &str) -> Option<&'static str>
    {
        match selector_type
        {
            "variance" => Some("Variance Threshold - odstraňuje features s nízkou varianciou (konštantné hodnoty)"),
            "chi_square" => Some("Chi-Square Test - testuje nezávislosť medzi features a targetom (len klasifikácia)"),
            "mutual_information" => Some("Mutual Information (KSG) - meria vzájomnú závislosť (funguje na spojitých dátach)"),
            "smc" => Some("SMC (Squared Multiple Correlation) - meria príspevok features k predikcii targetu cez drop v R²"),
            _ => None,
        }
    }

    /// Vráti či selector vyžaduje preprocessing (napr. binning)
    pub fn requires_binning(_selector_type: &str) -> bool
    {
        false
    }

    /// Vráti podporované typy problémov pre selector
    pub fn get_supported_types(selector_type: &str) -> Vec<&'static str>
    {
        match selector_type
        {
            "variance" => vec!["regression", "classification"],
            "chi_square" => vec!["classification"],
            "mutual_information" => vec!["regression", "classification"],
            "smc" => vec!["regression", "classification"],
            _ => vec![],
        }
    }

    /// Vráti podporované parametre pre selector
    pub fn get_supported_params(selector_type: &str) -> Vec<&'static str>
    {
        match selector_type
        {
            "variance" => vec!["threshold"],
            "chi_square" | "mutual_information" | "smc" => vec!["num_features"],
            _ => vec![],
        }
    }
}
