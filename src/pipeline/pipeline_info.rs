/// Informacie o nakonfigurovanom pipeline — model, eval mode.
/// Pouziva sa pre logovanie a diagnostiku.
#[derive(Debug, Clone)]
pub struct PipelineInfo
{
    pub model_name: String,
    pub model_type: String,
    pub evaluation_mode: String,
}
