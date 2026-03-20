/// Informácie o nakonfigurovanom pipeline
#[derive(Debug, Clone)]
pub struct PipelineInfo
{
    pub model_name: String,
    pub model_type: String,
    pub processor: Option<String>,
    pub selector: Option<String>,
    pub evaluation_mode: String,
}

impl PipelineInfo
{
    pub fn print(&self)
    {
        println!("=== ML Pipeline Info ===");
        println!("Model: {} ({})", self.model_name, self.model_type);
        println!("Processor: {}", self.processor.as_ref().unwrap_or(&"None".to_string()));
        println!("Feature Selector: {}", self.selector.as_ref().unwrap_or(&"None".to_string()));
        println!("Evaluation Mode: {}", self.evaluation_mode);
        println!("=======================");
    }
}
