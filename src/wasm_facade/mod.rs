/// WASM fasady - exportovane struktury pre JavaScript cez wasm_bindgen.
/// Kazda fasada obaluje Rust logiku a konvertuje medzi Rust typmi a JsValue.
pub mod ml_pipeline_facade;
pub mod data_loader_facade;
pub mod factory_facade;

pub use ml_pipeline_facade::WasmMLPipeline;
pub use data_loader_facade::WasmDataLoader;
pub use factory_facade::WasmFactory;
