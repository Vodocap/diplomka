use super::{DataProcessor, StandardScaler, Binner, OneHotEncoder, NullValueHandler, ProcessorChain};

/// Factory pre vytváranie data procesorov podľa názvu
pub struct ProcessorFactory;

impl ProcessorFactory {
    /// Vytvorí procesor na základe názvu
    pub fn create(processor_type: &str) -> Result<Box<dyn DataProcessor>, String> {
        match processor_type {
            "scaler" | "standard_scaler" => Ok(Box::new(StandardScaler::new())),
            "binner" => Ok(Box::new(Binner::new(10))), // default 10 bins
            "onehot" | "one_hot_encoder" => Ok(Box::new(OneHotEncoder)),
            "null_handler" => Ok(Box::new(NullValueHandler::with_params("NA,null,NaN", "mean", None))),
            _ => Err(format!("Neznámy procesor: {}", processor_type)),
        }
    }

    /// Vytvorí chain procesorov z viacerých typov
    pub fn create_chain(processor_types: Vec<&str>) -> Result<Box<dyn DataProcessor>, String> {
        if processor_types.is_empty() {
            return Err("No processors specified".to_string());
        }

        if processor_types.len() == 1 {
            return Self::create(processor_types[0]);
        }

        let mut chain = ProcessorChain::new();
        for proc_type in processor_types {
            let processor = Self::create(proc_type)?;
            chain.add_mut(processor);
        }

        Ok(Box::new(chain))
    }

    /// Vráti zoznam všetkých dostupných procesorov
    pub fn available() -> Vec<&'static str> {
        vec![
            "scaler",
            "binner",
            "onehot",
            "null_handler",
        ]
    }

    /// Vráti popis procesora
    pub fn get_description(processor_type: &str) -> Option<&'static str> {
        match processor_type {
            "scaler" => Some("Standard Scaler - normalizácia dát (mean=0, std=1)"),
            "binner" => Some("Binner - diskretizácia spojitých hodnôt do binov"),
            "onehot" => Some("One-Hot Encoder - kódovanie kategorických premenných"),
            "null_handler" => Some("Null Handler - nahradenie null hodnôt"),
            _ => None,
        }
    }

    /// Vráti podporované parametre pre procesor
    pub fn get_processor_params(processor_type: &str) -> Vec<&'static str> {
        match processor_type {
            "binner" => vec!["bins"],
            "null_handler" => vec!["null_repr", "strategy"],
            _ => vec![],
        }
    }
}
