use super::data_loader::{DataLoader, LoadedData};
use std::collections::HashMap;
use smartcore::linalg::basic::matrix::DenseMatrix;
use serde_json::Value;

/// JSON Data Loader - implementácia Strategy pattern pre JSON súbory
pub struct JsonDataLoader;

impl JsonDataLoader {
    pub fn new() -> Self {
        Self
    }

    /// Parsuje JSON array of objects formát
    /// Príklad: [{"feature1": 1.0, "feature2": 2.0, "target": 0}, ...]
    fn parse_json_array(&self, json_text: &str) -> Result<(Vec<String>, Vec<HashMap<String, String>>), String> {
        let parsed: Value = serde_json::from_str(json_text)
            .map_err(|e| format!("Chyba pri parsovaní JSON: {}", e))?;

        let array = parsed.as_array()
            .ok_or("JSON musí byť array objektov")?;

        if array.is_empty() {
            return Err("JSON array je prázdny".to_string());
        }

        // Získať headers z prvého objektu
        let first_obj = array[0].as_object()
            .ok_or("Prvý element musí byť objekt")?;
        
        let headers: Vec<String> = first_obj.keys()
            .map(|k| k.to_string())
            .collect();

        // Konvertovať objekty na HashMap
        let mut records = Vec::new();
        for (idx, item) in array.iter().enumerate() {
            let obj = item.as_object()
                .ok_or_else(|| format!("Element {} nie je objekt", idx))?;

            let mut record = HashMap::new();
            for header in &headers {
                let value = obj.get(header)
                    .ok_or_else(|| format!("Chýba kľúč '{}' v elemente {}", header, idx))?;

                let value_str = match value {
                    Value::Number(n) => n.to_string(),
                    Value::String(s) => s.clone(),
                    Value::Bool(b) => if *b { "1" } else { "0" }.to_string(),
                    _ => return Err(format!("Nepodporovaný typ hodnoty pre kľúč '{}'", header)),
                };

                record.insert(header.clone(), value_str);
            }
            records.push(record);
        }

        Ok((headers, records))
    }
}

impl DataLoader for JsonDataLoader {
    fn get_name(&self) -> &str {
        "JSON Data Loader"
    }

    fn load_from_string(&mut self, data: &str, target_column: &str) -> Result<LoadedData, String> {
        self.validate_format(data)?;

        let (headers, records) = self.parse_json_array(data)?;

        // Kontrola existencie target stĺpca
        if !headers.contains(&target_column.to_string()) {
            return Err(format!(
                "Target stĺpec '{}' sa nenachádza v dátach. Dostupné stĺpce: {:?}",
                target_column, headers
            ));
        }

        // Extrakcia X a y dát
        let mut x_rows: Vec<Vec<f64>> = Vec::new();
        let mut y_data: Vec<f64> = Vec::new();

        for (row_idx, record) in records.iter().enumerate() {
            let mut row = Vec::new();
            
            for header in &headers {
                let val_str = record.get(header)
                    .ok_or_else(|| format!("Chýbajúca hodnota pre kľúč '{}'", header))?;
                
                let val = val_str.parse::<f64>()
                    .map_err(|_| format!(
                        "Hodnota '{}' pre kľúč '{}' (riadok {}) nie je číslo",
                        val_str, header, row_idx + 1
                    ))?;
                
                if header == target_column {
                    y_data.push(val);
                } else {
                    row.push(val);
                }
            }
            
            if !row.is_empty() {
                x_rows.push(row);
            }
        }

        if x_rows.is_empty() || y_data.is_empty() {
            return Err("Nepodarilo sa extrahovať trénovacie dáta".to_string());
        }

        let x_data = DenseMatrix::from_2d_vec(&x_rows).unwrap();

        let feature_headers: Vec<String> = headers
            .into_iter()
            .filter(|h| h != target_column)
            .collect();

        Ok(LoadedData::new(feature_headers, x_data, y_data, records))
    }

    fn get_available_columns(&self, data: &str) -> Result<Vec<String>, String> {
        let (headers, _) = self.parse_json_array(data)?;
        Ok(headers)
    }

    fn validate_format(&self, data: &str) -> Result<(), String> {
        if data.trim().is_empty() {
            return Err("JSON dáta sú prázdne".to_string());
        }

        // Základná validácia JSON formátu
        let trimmed = data.trim();
        if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
            return Err("JSON musí byť array (začínať '[' a končiť ']')".to_string());
        }

        Ok(())
    }
}

impl Default for JsonDataLoader {
    fn default() -> Self {
        Self::new()
    }
}
