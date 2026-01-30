use wasm_bindgen::prelude::*;

mod data_loading;

#[cfg(not(target_arch = "wasm32"))]
mod models;

pub use data_loading::csv_loader::CsvLoader;

#[cfg(not(target_arch = "wasm32"))]
pub use models::{IModel, LinRegWrapper, KnnWrapper, LogRegWrapper, TreeWrapper};