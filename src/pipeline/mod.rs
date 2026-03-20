/// Modul pre ML pipeline - Builder/Director pattern, kompatibilita.
pub mod pipeline;
pub mod pipeline_info;   // PipelineInfo struct
pub mod builder;
pub mod director;
pub mod compatibility;

pub use pipeline::MLPipeline;
#[allow(unused_imports)]
pub use pipeline_info::PipelineInfo;
pub use builder::MLPipelineBuilder;
pub use director::MLPipelineDirector;
pub use compatibility::CompatibilityRegistry;
