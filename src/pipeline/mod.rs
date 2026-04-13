/// Modul pre ML pipeline - Builder pattern, model management
pub mod pipeline;
pub mod pipeline_info;
pub mod builder;

pub use pipeline::MLPipeline;
pub use builder::MLPipelineBuilder;
