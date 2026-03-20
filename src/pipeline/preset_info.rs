use serde::{Serialize, Deserialize};

/// Metadata o predpripravenej konfigurácii pipeline (preset).
/// Serializuje sa do JS pre zobrazenie v UI selectboxe.
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
