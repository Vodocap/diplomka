use super::loaded_data::LoadedData;

/// Strategy pattern pre načítanie dát z rôznych zdrojov
pub trait DataLoader
{
    /// Názov loadera
    fn get_name(&self) -> &str;

    /// Načíta dáta zo stringu
    fn load_from_string(&mut self, data: &str, target_column: &str) -> Result<LoadedData, String>;

    /// Získa dostupné stĺpce (headers) z dát
    fn get_available_columns(&self, data: &str) -> Result<Vec<String>, String>;

    /// Validuje formát dát pred načítaním
    fn validate_format(&self, data: &str) -> Result<(), String>;
}
