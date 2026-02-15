use super::{
    TargetAnalyzer,
    CorrelationAnalyzer,
    MutualInformationAnalyzer,
    EntropyAnalyzer,
    SmcAnalyzer,
};

/// Factory pre vytváranie analyzátorov cieľovej premennej podľa názvu.
/// Rovnaký vzor ako FeatureSelectorFactory.
pub struct TargetAnalyzerFactory;

impl TargetAnalyzerFactory {
    /// Vytvorí analyzátor na základe názvu
    pub fn create(analyzer_type: &str) -> Result<Box<dyn TargetAnalyzer>, String> {
        match analyzer_type {
            "correlation" => Ok(Box::new(CorrelationAnalyzer::new())),
            "mutual_information" | "mi" => Ok(Box::new(MutualInformationAnalyzer::new())),
            "entropy" | "information_gain" => Ok(Box::new(EntropyAnalyzer::new())),
            "smc" => Ok(Box::new(SmcAnalyzer::new())),
            _ => Err(format!("Neznámy analyzátor cieľovej premennej: {}", analyzer_type)),
        }
    }

    /// Vráti zoznam všetkých dostupných analyzátorov
    pub fn available() -> Vec<(&'static str, &'static str)> {
        vec![
            ("correlation", "Pearsonova korelácia - Σr² meria lineárne väzby medzi premennými"),
            ("mutual_information", "Mutual Information (KSG) - ΣMI zachytáva aj nelineárne vzťahy"),
            ("smc", "SMC (Squared Multiple Correlation) - koľko variability vysvetľujú ostatné premenné"),
            ("entropy", "Entropia a Information Gain - meria redukciu neistoty pri predikcii"),
        ]
    }

    /// Vráti popis analyzátora
    pub fn get_description(analyzer_type: &str) -> Option<&'static str> {
        match analyzer_type {
            "correlation" => Some("Pearsonova korelácia - Σr² meria lineárne väzby medzi premennými"),
            "mutual_information" | "mi" => Some("Mutual Information (KSG) - ΣMI zachytáva aj nelineárne vzťahy"),
            "smc" => Some("SMC (Squared Multiple Correlation) - koľko variability vysvetľujú ostatné premenné"),
            "entropy" | "information_gain" => Some("Entropia a Information Gain - meria redukciu neistoty pri predikcii"),
            _ => None,
        }
    }
}
