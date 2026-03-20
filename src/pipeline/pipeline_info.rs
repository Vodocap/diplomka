/// Informacie o nakonfigurovanom pipeline — model, procesor, selektor, eval mode.
/// Pouziva sa pre logovanie a diagnostiku.
#[derive(Debug, Clone)]
pub struct PipelineInfo
{
    pub model_name: String,
    pub model_type: String,
    pub processors: Vec<String>,
    pub selector: Option<String>,
    pub evaluation_mode: String,
}

impl PipelineInfo
{
    pub fn print(&self)
    {
        println!("=== ML Pipeline Info ===");
        println!("Model: {} ({})", self.model_name, self.model_type);
        if self.processors.is_empty() {
            println!("Processors: None");
        } else {
            println!("Processors: {}", self.processors.join(", "));
        }
        println!("Feature Selector: {}", self.selector.as_ref().unwrap_or(&"None".to_string()));
        println!("Evaluation Mode: {}", self.evaluation_mode);
        println!("=======================");
    }
}
