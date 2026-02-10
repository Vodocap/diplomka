use super::{
    DataProcessor, StandardScaler, Binner, OneHotEncoder, NullValueHandler, ProcessorChain,
    MinMaxScaler, RobustScaler, LabelEncoder, OutlierClipper, LogTransformer, PowerTransformer,
    ProcessorParam, SelectiveProcessor,
};

/// Factory pre vytváranie data procesorov podľa názvu
pub struct ProcessorFactory;

impl ProcessorFactory {
    /// Vytvorí procesor na základe názvu
    pub fn create(processor_type: &str) -> Result<Box<dyn DataProcessor>, String> {
        let base_processor: Box<dyn DataProcessor> = match processor_type {
            "scaler" | "standard_scaler" => Box::new(StandardScaler::new()),
            "minmax_scaler" => Box::new(MinMaxScaler::new()),
            "robust_scaler" => Box::new(RobustScaler::new()),
            "binner" => Box::new(Binner::new(10)), // default 10 bins
            "onehot" | "one_hot_encoder" => Box::new(OneHotEncoder),
            "label_encoder" => Box::new(LabelEncoder::new()),
            "null_handler" => Box::new(NullValueHandler::with_params("NA,null,NaN", "mean", None)),
            "outlier_clipper" => Box::new(OutlierClipper::with_iqr(1.5)), // default IQR method
            "log_transformer" => Box::new(LogTransformer::new()),
            "power_transformer" => Box::new(PowerTransformer::yeo_johnson()), // default Yeo-Johnson
            _ => return Err(format!("Neznámy procesor: {}", processor_type)),
        };
        
        // Wrap procesor v SelectiveProcessor pre automatickú detekciu stĺpcov
        Ok(Box::new(SelectiveProcessor::new(base_processor)))
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
            "minmax_scaler",
            "robust_scaler",
            "binner",
            "onehot",
            "label_encoder",
            "null_handler",
            "outlier_clipper",
            "log_transformer",
            "power_transformer",
        ]
    }

    /// Vráti popis procesora
    pub fn get_description(processor_type: &str) -> Option<&'static str> {
        match processor_type {
            "scaler" => Some("Standard Scaler - normalizácia dát (mean=0, std=1)"),
            "minmax_scaler" => Some("MinMax Scaler - normalizácia do rozsahu [0, 1]"),
            "robust_scaler" => Some("Robust Scaler - škálovanie pomocou mediánu a IQR (odolné voči outlierom)"),
            "binner" => Some("Binner - diskretizácia spojitých hodnôt do binov"),
            "onehot" => Some("One-Hot Encoder - kódovanie kategorických premenných"),
            "label_encoder" => Some("Label Encoder - enkódovanie kategorických hodnôt na čísla"),
            "null_handler" => Some("Null Handler - nahradenie null hodnôt"),
            "outlier_clipper" => Some("Outlier Clipper - orezávanie outlierov (IQR/percentile/z-score)"),
            "log_transformer" => Some("Log Transformer - logaritmická transformácia"),
            "power_transformer" => Some("Power Transformer - Box-Cox/Yeo-Johnson transformácia"),
            _ => None,
        }
    }

    /// Vráti podporované parametre pre procesor
    pub fn get_processor_params(processor_type: &str) -> Vec<&'static str> {
        match processor_type {
            "minmax_scaler" => vec!["min", "max"],
            "binner" => vec!["bins"],
            "null_handler" => vec!["null_repr", "strategy"],
            "log_transformer" => vec!["offset"],
            _ => vec![],
        }
    }

    /// Vráti detailné definície parametrov pre procesor
    pub fn get_param_definitions(processor_type: &str) -> Vec<ProcessorParam> {
        if let Ok(processor) = Self::create(processor_type) {
            processor.get_param_definitions()
        } else {
            vec![]
        }
    }
}
