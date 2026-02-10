// Removed unused import

mod data_loading;
mod evaluation;
mod pipeline;
mod wasm_api;
mod models;
mod processing;
mod feature_selection_strategies;

pub use data_loading::csv_loader::CsvLoader;
pub use data_loading::{DataLoader, DataLoaderFactory, LoadedData, CsvDataLoader, JsonDataLoader};
pub use models::{IModel, LinRegWrapper, KnnWrapper, LogRegWrapper, TreeWrapper, factory::ModelFactory};
pub use processing::{DataProcessor, StandardScaler, Binner, OneHotEncoder, factory::ProcessorFactory};
pub use feature_selection_strategies::{
    FeatureSelector, 
    VarianceSelector, 
    CorrelationSelector,
    ChiSquareSelector,
    InformationGainSelector,
    MutualInformationSelector,
    factory::FeatureSelectorFactory
};
pub use evaluation::{ModelEvaluator, EvaluationReport};
pub use pipeline::{MLPipeline, MLPipelineBuilder, MLPipelineDirector, CompatibilityRegistry};

// WASM API exports
pub use wasm_api::{WasmMLPipeline, WasmDataLoader, WasmFactory};
