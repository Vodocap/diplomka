use super::{
    FeatureSelector, 
    VarianceSelector, 
    CorrelationSelector,
    ChiSquareSelector,
    InformationGainSelector,
    MutualInformationSelector
};

/// Factory pre vytváranie feature selektorov podľa názvu
pub struct FeatureSelectorFactory;

impl FeatureSelectorFactory {
    /// Vytvorí feature selektor na základe názvu
    pub fn create(selector_type: &str) -> Result<Box<dyn FeatureSelector>, String> {
        match selector_type {
            "variance" => Ok(Box::new(VarianceSelector::new())),
            "correlation" => Ok(Box::new(CorrelationSelector::new())),
            "chi_square" | "chi2" => Ok(Box::new(ChiSquareSelector::new())),
            "information_gain" | "infogain" => Ok(Box::new(InformationGainSelector::new())),
            "mutual_information" | "mi" => Ok(Box::new(MutualInformationSelector::new())),
            _ => Err(format!("Neznámy feature selektor: {}", selector_type)),
        }
    }

    /// Vráti zoznam všetkých dostupných feature selektorov
    pub fn available() -> Vec<&'static str> {
        vec![
            "variance",
            "correlation",
            "chi_square",
            "information_gain",
            "mutual_information",
        ]
    }

    /// Vráti popis feature selektora
    pub fn get_description(selector_type: &str) -> Option<&'static str> {
        match selector_type {
            "variance" => Some("Variance Threshold - odstráni features s nízkou varianciou"),
            "correlation" => Some("Correlation - odstráni vysoko korelované features"),
            "chi_square" => Some("Chi-Square Test - pre kategorické features"),
            "information_gain" => Some("Information Gain - meria redukciu entropie"),
            "mutual_information" => Some("Mutual Information - meria závislosti"),
            _ => None,
        }
    }

    /// Vráti podporované typy problémov pre selector
    pub fn get_supported_types(selector_type: &str) -> Vec<&'static str> {
        match selector_type {
            "variance" => vec!["regression", "classification"],
            "correlation" => vec!["regression", "classification"],
            "chi_square" => vec!["classification"],
            "information_gain" => vec!["classification"],
            "mutual_information" => vec!["regression", "classification"],
            _ => vec![],
        }
    }

    /// Vráti podporované parametre pre selector
    pub fn get_supported_params(selector_type: &str) -> Vec<&'static str> {
        match selector_type {
            "variance" => vec!["threshold"],
            "correlation" => vec!["threshold"],
            "chi_square" | "information_gain" | "mutual_information" => vec!["num_features"],
            _ => vec![],
        }
    }
}
