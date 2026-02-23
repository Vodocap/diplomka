use super::{
    FeatureSelector, 
    VarianceSelector, 
    CorrelationSelector,
    ChiSquareSelector,
    InformationGainSelector,
    MutualInformationSelector,
    SmcSelector,
    SynergyVNSSelector
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
            "smc" => Ok(Box::new(SmcSelector::new())),
            "synergy_vns" | "vns" => Ok(Box::new(SynergyVNSSelector::new())),
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
            "smc",
            "synergy_vns",
        ]
    }

    /// Vráti popis feature selektora
    pub fn get_description(selector_type: &str) -> Option<&'static str> {
        match selector_type {
            "variance" => Some("Variance Threshold - odstraňuje features s nízkou varianciou (konštantné hodnoty)"),
            "correlation" => Some("Correlation - vyberie features s najvyššou koreláciou k targetu (pre regression)"),
            "chi_square" => Some("Chi-Square Test - testuje nezávislosť medzi features a targetom (len klasifikácia)"),
            "information_gain" => Some("Information Gain - meria redukciu entropie (Pozor: vyžaduje Binner processor!)"),
            "mutual_information" => Some("Mutual Information (KSG) - meria vzájomnú závislosť (funguje na spojitých dátach)"),
            "smc" => Some("SMC (Squared Multiple Correlation) - meria príspevok features k predikcii targetu cez drop v R²"),
            "synergy_vns" => Some("Synergy VNS - Variable Neighborhood Search optimalizujúci synergiu medzi features (MI + synergy - redundancy)"),
            _ => None,
        }
    }
    
    /// Vráti či selector vyžaduje preprocessing (napr. binning)
    pub fn requires_binning(selector_type: &str) -> bool {
        matches!(selector_type, "information_gain")
    }

    /// Vráti podporované typy problémov pre selector
    pub fn get_supported_types(selector_type: &str) -> Vec<&'static str> {
        match selector_type {
            "variance" => vec!["regression", "classification"],
            "correlation" => vec!["regression", "classification"],
            "chi_square" => vec!["classification"],
            "information_gain" => vec!["classification"],
            "mutual_information" => vec!["regression", "classification"],
            "smc" => vec!["regression", "classification"],
            "synergy_vns" => vec!["regression", "classification"],
            _ => vec![],
        }
    }

    /// Vráti podporované parametre pre selector
    pub fn get_supported_params(selector_type: &str) -> Vec<&'static str> {
        match selector_type {
            "variance" => vec!["threshold"],
            "correlation" => vec!["threshold"],
            "chi_square" | "information_gain" | "mutual_information" | "smc" => vec!["num_features"],
            "synergy_vns" => vec!["num_features", "max_iterations", "k_max", "alpha", "beta", "gamma", "initial_solution"],
            _ => vec![],
        }
    }
}
