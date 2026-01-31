use wasm_bindgen::prelude::*;
use serde::{Serialize, Deserialize};

#[cfg(not(target_arch = "wasm32"))]
use crate::data_loading::{DataLoader, DataLoaderFactory};

#[derive(Serialize, Deserialize)]
pub struct LoadedDataInfo {
    pub num_samples: usize,
    pub num_features: usize,
    pub feature_names: Vec<String>,
    pub target_column: String,
}

#[wasm_bindgen]
pub struct WasmDataLoader {
    #[cfg(not(target_arch = "wasm32"))]
    loader: Option<Box<dyn DataLoader>>,
    format: String,
}

#[wasm_bindgen]
impl WasmDataLoader {
    #[wasm_bindgen(constructor)]
    pub fn new(format: String) -> Result<WasmDataLoader, JsValue> {
        console_error_panic_hook::set_once();
        
        #[cfg(not(target_arch = "wasm32"))]
        {
            let loader = DataLoaderFactory::create(&format)
                .map_err(|e| JsValue::from_str(&e))?;
            
            Ok(WasmDataLoader {
                loader: Some(loader),
                format,
            })
        }
        
        #[cfg(target_arch = "wasm32")]
        Ok(WasmDataLoader { format })
    }

    /// Automaticky detekuje formát
    #[wasm_bindgen(js_name = createAuto)]
    pub fn create_auto(data: &str) -> Result<WasmDataLoader, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let loader = DataLoaderFactory::create_auto(data)
                .map_err(|e| JsValue::from_str(&e))?;
            
            let format = loader.get_name().to_string();
            
            Ok(WasmDataLoader {
                loader: Some(loader),
                format,
            })
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Získa dostupné stĺpce z dát
    #[wasm_bindgen(js_name = getAvailableColumns)]
    pub fn get_available_columns(&self, data: &str) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref loader) = self.loader {
                let columns = loader.get_available_columns(data)
                    .map_err(|e| JsValue::from_str(&e))?;
                Ok(serde_wasm_bindgen::to_value(&columns).unwrap())
            } else {
                Err(JsValue::from_str("Loader not initialized"))
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Validuje formát dát
    #[wasm_bindgen(js_name = validateFormat)]
    pub fn validate_format(&self, data: &str) -> Result<(), JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref loader) = self.loader {
                loader.validate_format(data)
                    .map_err(|e| JsValue::from_str(&e))
            } else {
                Err(JsValue::from_str("Loader not initialized"))
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }

    /// Načíta dáta (vracia info o dátach)
    #[wasm_bindgen(js_name = loadData)]
    pub fn load_data(&mut self, data: &str, target_column: &str) -> Result<JsValue, JsValue> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Some(ref mut loader) = self.loader {
                let loaded = loader.load_from_string(data, target_column)
                    .map_err(|e| JsValue::from_str(&e))?;
                
                let info = LoadedDataInfo {
                    num_samples: loaded.num_samples(),
                    num_features: loaded.num_features(),
                    feature_names: loaded.headers.clone(),
                    target_column: target_column.to_string(),
                };
                
                Ok(serde_wasm_bindgen::to_value(&info).unwrap())
            } else {
                Err(JsValue::from_str("Loader not initialized"))
            }
        }
        
        #[cfg(target_arch = "wasm32")]
        Err(JsValue::from_str("Not supported in WASM"))
    }
}
