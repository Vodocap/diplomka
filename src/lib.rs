use wasm_bindgen::prelude::*;

mod csv_loader;
mod model;

pub use csv_loader::CsvLoader;
pub use model::LinearRegression;