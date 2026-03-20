pub mod loaded_data;     // LoadedData struct
pub mod data_loader;     // DataLoader trait
pub mod csv_data_loader; // CSV implementation of strategy
pub mod json_data_loader; // JSON implementation of strategy
pub mod data_loader_factory;       // Factory for loaders

pub use loaded_data::LoadedData;
pub use data_loader::DataLoader;
pub use csv_data_loader::CsvDataLoader;
pub use json_data_loader::JsonDataLoader;
pub use data_loader_factory::DataLoaderFactory;
