use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linalg::basic::arrays::{Array, MutArray};
use super::{DataProcessor, ProcessorParam, ColumnType};

/// Podporované časové formáty pre vstup
#[derive(Debug, Clone, PartialEq)]
pub enum TimeInputFormat {
    /// HH:MM:SS (napr. "01:30:00")
    HHMMSS,
    /// HH:MM (napr. "01:30")
    HHMM,
    /// MM:SS (napr. "90:00")
    MMSS,
    /// Sekundy (napr. "5400")
    Seconds,
    /// Minúty (napr. "90")
    Minutes,
    /// Hodiny (napr. "1.5")
    Hours,
}

/// Cieľová jednotka pre výstup
#[derive(Debug, Clone, PartialEq)]
pub enum TimeOutputUnit {
    Seconds,
    Minutes,
    Hours,
}

/// Time Converter - konvertuje časové hodnoty z rôznych formátov na sekundy/minúty/hodiny
pub struct TimeConverter {
    input_format: TimeInputFormat,
    output_unit: TimeOutputUnit,
}

impl TimeConverter {
    pub fn new() -> Self {
        Self {
            input_format: TimeInputFormat::Seconds,
            output_unit: TimeOutputUnit::Seconds,
        }
    }

    /// Parsuje reťazec input formátu
    fn parse_input_format(s: &str) -> TimeInputFormat {
        match s.to_lowercase().as_str() {
            "hh:mm:ss" | "hhmmss" => TimeInputFormat::HHMMSS,
            "hh:mm" | "hhmm" => TimeInputFormat::HHMM,
            "mm:ss" | "mmss" => TimeInputFormat::MMSS,
            "seconds" | "s" => TimeInputFormat::Seconds,
            "minutes" | "m" => TimeInputFormat::Minutes,
            "hours" | "h" => TimeInputFormat::Hours,
            _ => TimeInputFormat::Seconds,
        }
    }

    /// Parsuje reťazec output jednotky
    fn parse_output_unit(s: &str) -> TimeOutputUnit {
        match s.to_lowercase().as_str() {
            "seconds" | "s" => TimeOutputUnit::Seconds,
            "minutes" | "m" => TimeOutputUnit::Minutes,
            "hours" | "h" => TimeOutputUnit::Hours,
            _ => TimeOutputUnit::Seconds,
        }
    }

    /// Konvertuje hodnotu na sekundy podľa vstupného formátu.
    /// Keďže dáta sú už numerické (f64), formáty ako HH:MM:SS sa interpretujú cez zakódovanie.
    /// Pre HH:MM:SS: hodnota = H * 10000 + M * 100 + S (napr. 13000 = 01:30:00)
    /// Pre HH:MM: hodnota = H * 100 + M (napr. 130 = 01:30)
    /// Pre MM:SS: hodnota = M * 100 + S (napr. 9000 = 90:00)
    fn to_seconds(&self, value: f64) -> f64 {
        match self.input_format {
            TimeInputFormat::HHMMSS => {
                // Zakódované ako HHMMSS: napr. 13000.0 = 1h 30m 0s
                let total = value as i64;
                let hours = total / 10000;
                let minutes = (total % 10000) / 100;
                let seconds = total % 100;
                (hours * 3600 + minutes * 60 + seconds) as f64
            },
            TimeInputFormat::HHMM => {
                // Zakódované ako HHMM: napr. 130.0 = 1h 30m
                let total = value as i64;
                let hours = total / 100;
                let minutes = total % 100;
                (hours * 3600 + minutes * 60) as f64
            },
            TimeInputFormat::MMSS => {
                // Zakódované ako MMSS: napr. 9000.0 = 90m 0s
                let total = value as i64;
                let minutes = total / 100;
                let seconds = total % 100;
                (minutes * 60 + seconds) as f64
            },
            TimeInputFormat::Seconds => value,
            TimeInputFormat::Minutes => value * 60.0,
            TimeInputFormat::Hours => value * 3600.0,
        }
    }

    /// Konvertuje sekundy na výstupnú jednotku
    fn from_seconds(&self, seconds: f64) -> f64 {
        match self.output_unit {
            TimeOutputUnit::Seconds => seconds,
            TimeOutputUnit::Minutes => seconds / 60.0,
            TimeOutputUnit::Hours => seconds / 3600.0,
        }
    }

    /// Konvertuje jednu hodnotu
    fn convert(&self, value: f64) -> f64 {
        let seconds = self.to_seconds(value);
        self.from_seconds(seconds)
    }
}

impl DataProcessor for TimeConverter {
    fn get_name(&self) -> &str {
        "Time Converter"
    }

    fn fit(&mut self, _data: &DenseMatrix<f64>) {
        // Time converter nepotrebuje fit - je stateless
    }

    fn transform(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        let (rows, cols) = data.shape();
        let mut result = data.clone();

        for j in 0..cols {
            for i in 0..rows {
                let val = *data.get((i, j));
                let converted = if val.is_nan() {
                    f64::NAN
                } else {
                    self.convert(val)
                };
                result.set((i, j), converted);
            }
        }

        result
    }

    fn process(&self, data: &DenseMatrix<f64>) -> DenseMatrix<f64> {
        self.transform(data)
    }

    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String> {
        match key {
            "input_format" => {
                self.input_format = Self::parse_input_format(value);
                Ok(())
            },
            "output_unit" => {
                self.output_unit = Self::parse_output_unit(value);
                Ok(())
            },
            _ => Err(format!("Neznámy parameter: {}", key)),
        }
    }

    fn get_supported_params(&self) -> Vec<&str> {
        vec!["input_format", "output_unit"]
    }

    fn get_param_definitions(&self) -> Vec<ProcessorParam> {
        vec![
            ProcessorParam {
                name: "input_format".to_string(),
                param_type: "select".to_string(),
                default_value: "seconds".to_string(),
                description: "Vstupný formát času".to_string(),
                min: None,
                max: None,
                options: Some(vec![
                    "seconds".to_string(),
                    "minutes".to_string(),
                    "hours".to_string(),
                    "hh:mm:ss".to_string(),
                    "hh:mm".to_string(),
                    "mm:ss".to_string(),
                ]),
            },
            ProcessorParam {
                name: "output_unit".to_string(),
                param_type: "select".to_string(),
                default_value: "seconds".to_string(),
                description: "Výstupná jednotka".to_string(),
                min: None,
                max: None,
                options: Some(vec![
                    "seconds".to_string(),
                    "minutes".to_string(),
                    "hours".to_string(),
                ]),
            },
        ]
    }

    fn get_applicable_column_types(&self) -> Option<Vec<ColumnType>> {
        Some(vec![ColumnType::Numeric, ColumnType::Discrete])
    }
}
