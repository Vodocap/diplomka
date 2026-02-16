use super::{
    DataProcessor, StandardScaler, Binner, OneHotEncoder, NullValueHandler, ProcessorChain,
    MinMaxScaler, RobustScaler, LabelEncoder, OutlierClipper, LogTransformer, PowerTransformer,
    ProcessorParam, SelectiveProcessor, TimeConverter,
    CommaToDotProcessor, ThousandsSeparatorRemover, OrdinalEncoder, FrequencyEncoder, TargetEncoder,
};

/// Factory pre vytváranie data procesorov podľa názvu
pub struct ProcessorFactory;

impl ProcessorFactory {
    /// Vytvorí procesor na základe názvu (zabalený v SelectiveProcessor)
    pub fn create(processor_type: &str) -> Result<Box<dyn DataProcessor>, String> {
        let base_processor = Self::create_raw(processor_type)?;
        
        // Wrap procesor v SelectiveProcessor pre automatickú detekciu stĺpcov
        Ok(Box::new(SelectiveProcessor::new(base_processor)))
    }

    /// Vytvorí surový (raw) procesor BEZ SelectiveProcessor wrappera.
    /// Použitie: keď chceme aplikovať procesor priamo na konkrétny stĺpec (editor)
    pub fn create_raw(processor_type: &str) -> Result<Box<dyn DataProcessor>, String> {
        match processor_type {
            "scaler" | "standard_scaler" => Ok(Box::new(StandardScaler::new())),
            "minmax_scaler" => Ok(Box::new(MinMaxScaler::new())),
            "robust_scaler" => Ok(Box::new(RobustScaler::new())),
            "binner" => Ok(Box::new(Binner::new(10))),
            "onehot" | "one_hot_encoder" => Ok(Box::new(OneHotEncoder)),
            "label_encoder" => Ok(Box::new(LabelEncoder::new())),
            "null_handler" => Ok(Box::new(NullValueHandler::with_params("NA,null,NaN", "mean", None))),
            "outlier_clipper" => Ok(Box::new(OutlierClipper::with_iqr(1.5))),
            "log_transformer" => Ok(Box::new(LogTransformer::new())),
            "power_transformer" => Ok(Box::new(PowerTransformer::yeo_johnson())),
            "time_converter" => Ok(Box::new(TimeConverter::new())),
            "comma_to_dot" => Ok(Box::new(CommaToDotProcessor::new())),
            "thousands_separator_remover" => Ok(Box::new(ThousandsSeparatorRemover::new())),
            "ordinal_encoder" => Ok(Box::new(OrdinalEncoder::new())),
            "frequency_encoder" => Ok(Box::new(FrequencyEncoder::new())),
            "target_encoder" => Ok(Box::new(TargetEncoder::new())),
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
            "minmax_scaler",
            "robust_scaler",
            "binner",
            "onehot",
            "label_encoder",
            "ordinal_encoder",
            "frequency_encoder",
            "target_encoder",
            "null_handler",
            "outlier_clipper",
            "log_transformer",
            "power_transformer",
            "time_converter",
            "comma_to_dot",
            "thousands_separator_remover",
        ]
    }

    /// Vráti popis procesora
    pub fn get_description(processor_type: &str) -> Option<&'static str> {
        match processor_type {
            "scaler" => Some("Standard Scaler - normalizácia dát (mean=0, std=1)"),
            "minmax_scaler" => Some("MinMax Scaler - normalizácia do rozsahu [0, 1]"),
            "robust_scaler" => Some("Robust Scaler - škálovanie pomocou mediánu a IQR (odolné voči outlierom)"),
            "binner" => Some("Binner - diskretizácia spojitých hodnôt do binov"),
            "onehot" => Some("One-Hot Encoder - kódovanie kategorických premenných (vytvorí binárne stĺpce)"),
            "label_encoder" => Some("Label Encoder - enkódovanie kategórií na čísla (0, 1, 2, ...)"),
            "ordinal_encoder" => Some("Ordinal Encoder - enkódovanie s uchovaním poradia hodnôt"),
            "frequency_encoder" => Some("Frequency Encoder - enkódovanie podľa frekvencie výskytu"),
            "target_encoder" => Some("Target Encoder - enkódovanie podľa priemeru cieľovej premennej"),
            "null_handler" => Some("Null Handler - nahradenie chýbajúcich hodnôt (mean/median/mode)"),
            "outlier_clipper" => Some("Outlier Clipper - orezávanie outlierov (IQR/percentile/z-score)"),
            "log_transformer" => Some("Log Transformer - logaritmická transformácia"),
            "power_transformer" => Some("Power Transformer - Box-Cox/Yeo-Johnson transformácia"),
            "time_converter" => Some("Time Converter - konverzia časových hodnôt (sekundy/minúty/hodiny)"),
            "comma_to_dot" => Some("Čiarka → Bodka - nahradí desatinné čiarky bodkami pre float interpretáciu"),
            "thousands_separator_remover" => Some("Odstrániť oddeľovač tisícov - odstráni čiarky z čísiel (napr. 1,000 → 1000)"),
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
