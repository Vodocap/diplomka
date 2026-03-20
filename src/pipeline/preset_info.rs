use serde::{Serialize, Deserialize};

/// Informácie o predpripravenej konfigurácii pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PresetInfo
{
    pub name: &'static str,
    pub description: &'static str,
    pub model_type: &'static str,
    pub model: Option<&'static str>,
    pub processor: Option<&'static str>,
    pub selector: Option<&'static str>,
}
