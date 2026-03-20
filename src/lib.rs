// Removed unused import

mod data_loading;
mod evaluation;
mod pipeline;
mod wasm_facade;
mod models;
mod processing;
mod feature_selection_strategies;
mod target_analysis;
mod embedded;
pub mod entropy;

pub use data_loading::{DataLoader, DataLoaderFactory, LoadedData, CsvDataLoader, JsonDataLoader};
pub use models::{IModel, LinRegWrapper, KnnWrapper, LogRegWrapper, TreeWrapper, model_factory::ModelFactory};
pub use processing::{DataProcessor, StandardScaler, Binner, OneHotEncoder, processor_factory::ProcessorFactory, TimeConverter,
    CommaToDotProcessor, ThousandsSeparatorRemover, OrdinalEncoder, FrequencyEncoder, TargetEncoder};
pub use feature_selection_strategies::{
    FeatureSelector,
    VarianceSelector,
    ChiSquareSelector,
    MutualInformationSelector,
    feature_selector_factory::FeatureSelectorFactory
};
pub use target_analysis::{
    TargetAnalyzer,
    TargetCandidate,
    CorrelationAnalyzer as TargetCorrelationAnalyzer,
    MutualInformationAnalyzer as TargetMIAnalyzer,
    TargetAnalyzerFactory,
};
pub use evaluation::{ModelEvaluator, EvaluationReport};
pub use pipeline::{MLPipeline, MLPipelineBuilder, MLPipelineDirector, CompatibilityRegistry};

// WASM API exports
pub use wasm_facade::{WasmMLPipeline, WasmDataLoader, WasmFactory};
