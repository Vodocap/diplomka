use wasm_bindgen::prelude::*;
use csv::ReaderBuilder;
use std::collections::HashMap;
use wasm_bindgen::JsValue;
use wasm_bindgen_futures::spawn_local;
use serde_wasm_bindgen::to_value;

#[wasm_bindgen]
pub struct CsvLoader {
    headers: Vec<String>,
    records: Vec<HashMap<String, String>>,
}

#[wasm_bindgen]
impl CsvLoader {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CsvLoader {
        CsvLoader {
            headers: vec![],
            records: vec![],
        }
    }

    pub async fn load_csv_async(&mut self, csv_text: String) -> Result<(), JsValue> {
        self.load_csv(&csv_text)
    }

    pub fn load_csv(&mut self, csv_text: &str) -> Result<(), JsValue> {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(csv_text.as_bytes());

        self.headers = rdr
            .headers()
            .map_err(|e| JsValue::from_str(&format!("CSV Error: {}", e)))?
            .iter()
            .map(|s| s.to_string())
            .collect();

        let records: Result<Vec<_>, _> = rdr
            .records()
            .map(|r| {
                r.map(|record| {
                    record
                        .iter()
                        .enumerate()
                        .map(|(i, val)| (self.headers[i].clone(), val.to_string()))
                        .collect::<HashMap<_, _>>()
                })
            })
            .collect();

        self.records = records.map_err(|e| JsValue::from_str(&format!("CSV Error: {}", e)))?;

        Ok(())
    }

    pub fn get_training_data(&self, target_header: &str) -> Result<JsValue, JsValue> {
        let mut x_data = Vec::new();
        let mut y_data = Vec::new();

        for record in &self.records {
            let mut row = Vec::new();
            for header in &self.headers {
                let val_str = record.get(header).ok_or("Missing value")?;
                let val_f64 = val_str.parse::<f64>().unwrap_or(0.0);
                
                if header == target_header {
                    y_data.push(val_f64);
                } else {
                    row.push(val_f64);
                }
            }
            x_data.push(row);
        }
        
        Ok(to_value(&(x_data, y_data)).unwrap())
    }

    pub fn get_headers(&self) -> JsValue {
        to_value(&self.headers).unwrap()
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }
}
