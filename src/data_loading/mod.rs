pub mod csv_loader; // Original WASM loader
pub mod data_loader; // Strategy trait
pub mod csv_data_loader; // CSV implementation of strategy
pub mod json_data_loader; // JSON implementation of strategy
pub mod factory; // Factory for loaders

// CsvLoader removed - not used
pub use data_loader::{DataLoader, LoadedData};
pub use csv_data_loader::CsvDataLoader;
pub use json_data_loader::JsonDataLoader;
pub use factory::DataLoaderFactory;
