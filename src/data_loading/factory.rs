use super::data_loader::DataLoader;
use super::csv_data_loader::CsvDataLoader;
use super::json_data_loader::JsonDataLoader;

/// Factory pre vytváranie data loaderov podľa typu
pub struct DataLoaderFactory;

impl DataLoaderFactory {
    /// Vytvorí loader na základe typu
    pub fn create(loader_type: &str) -> Result<Box<dyn DataLoader>, String> {
        match loader_type.to_lowercase().as_str() {
            "csv" => Ok(Box::new(CsvDataLoader::new())),
            "json" => Ok(Box::new(JsonDataLoader::new())),
            _ => Err(format!("Neznámy typ loadera: {}", loader_type)),
        }
    }

    /// Automaticky detekuje formát na základe obsahu
    pub fn create_auto(data: &str) -> Result<Box<dyn DataLoader>, String> {
        let trimmed = data.trim();
        
        if trimmed.starts_with('[') && trimmed.contains('{') {
            Ok(Box::new(JsonDataLoader::new()))
        } else if trimmed.contains(',') || trimmed.contains('\n') {
            Ok(Box::new(CsvDataLoader::new()))
        } else {
            Err("Nepodarilo sa automaticky detekovať formát dát".to_string())
        }
    }

    /// Vráti zoznam všetkých podporovaných formátov
    pub fn available_formats() -> Vec<&'static str> {
        vec!["csv", "json"]
    }

    /// Vráti popis formátu
    pub fn get_format_description(format: &str) -> Option<&'static str> {
        match format.to_lowercase().as_str() {
            "csv" => Some("CSV (Comma-Separated Values) - štandardný formát pre tabuľkové dáta"),
            "json" => Some("JSON (JavaScript Object Notation) - formát array of objects"),
            _ => None,
        }
    }
}
