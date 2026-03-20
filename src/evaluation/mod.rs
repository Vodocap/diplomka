/// Modul pre evaluaciu nauceneho modelu - metriky pre klasifikaciu aj regresiu.
pub mod evaluator;
pub mod metrics;

pub use evaluator::ModelEvaluator;
pub use metrics::EvaluationReport;
