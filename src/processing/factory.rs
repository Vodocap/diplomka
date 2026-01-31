use super::{DataProcessor, StandardScaler, Binner, OneHotEncoder};

/// Factory pre vytváranie data procesorov podľa názvu
pub struct ProcessorFactory;

impl ProcessorFactory {
    /// Vytvorí procesor na základe názvu
    pub fn create(processor_type: &str) -> Result<Box<dyn DataProcessor>, String> {
        match processor_type {
            "scaler" | "standard_scaler" => Ok(Box::new(StandardScaler)),
            "binner" => Ok(Box::new(Binner::new(10))), // default 10 bins
            "onehot" | "one_hot_encoder" => Ok(Box::new(OneHotEncoder)),
            _ => Err(format!("Neznámy procesor: {}", processor_type)),
        }
    }

    /// Vráti zoznam všetkých dostupných procesorov
    pub fn available() -> Vec<&'static str> {
        vec![
            "scaler",
            "binner",
            "onehot",
        ]
    }

    /// Vráti popis procesora
    pub fn get_description(processor_type: &str) -> Option<&'static str> {
        match processor_type {
            "scaler" => Some("Standard Scaler - normalizácia dát (mean=0, std=1)"),
            "binner" => Some("Binner - diskretizácia spojitých hodnôt do binov"),
            "onehot" => Some("One-Hot Encoder - kódovanie kategorických premenných"),
            _ => None,
        }
    }
}
