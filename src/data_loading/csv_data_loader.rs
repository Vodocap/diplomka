use super::data_loader::{DataLoader, LoadedData};
use csv::ReaderBuilder;
use std::collections::HashMap;
use smartcore::linalg::basic::matrix::DenseMatrix;

/// CSV Data Loader - implementácia Strategy pattern pre CSV súbory
pub struct CsvDataLoader;

impl CsvDataLoader {
    pub fn new() -> Self {
        Self
    }

    /// Helper pre parsovanie CSV
    fn parse_csv(&self, csv_text: &str) -> Result<(Vec<String>, Vec<HashMap<String, String>>), String> {
        let mut rdr = ReaderBuilder::new()
            .has_headers(true)
            .flexible(false)
            .trim(csv::Trim::All)
            .from_reader(csv_text.as_bytes());

        // Načítať headers
        let headers: Vec<String> = rdr
            .headers()
            .map_err(|e| format!("Chyba pri čítaní CSV hlavičiek: {}", e))?
            .iter()
            .map(|s| s.trim().to_string())
            .collect();

        if headers.is_empty() {
            return Err("CSV nemá žiadne stĺpce".to_string());
        }

        // Načítať záznamy
        let records: Result<Vec<_>, _> = rdr
            .records()
            .enumerate()
            .map(|(idx, r)| {
                r.map(|record| {
                    if record.len() != headers.len() {
                        return Err(format!(
                            "Riadok {} má {} stĺpcov, očakávaných {}",
                            idx + 1,
                            record.len(),
                            headers.len()
                        ));
                    }
                    Ok(record
                        .iter()
                        .enumerate()
                        .map(|(i, val)| (headers[i].clone(), val.trim().to_string()))
                        .collect::<HashMap<_, _>>())
                })
                .map_err(|e| format!("Chyba pri čítaní riadku {}: {}", idx + 1, e))?
            })
            .collect();

        let records = records.map_err(|e: String| e)?;

        if records.is_empty() {
            return Err("CSV neobsahuje žiadne dáta".to_string());
        }

        Ok((headers, records))
    }

    /// Konvertuje string hodnoty na f64 s lepším error handlingom
    fn parse_numeric_value(&self, val: &str, _column: &str, _row: usize) -> Result<f64, String> {
        let trimmed = val.trim();
        if trimmed.is_empty() {
            return Ok(0.0);
        }
        trimmed.parse::<f64>()
            .or_else(|_| trimmed.replace(',', ".").parse::<f64>())
            .map_err(|_| format!(
                "Hodnota '{}' v stĺpci '{}' (riadok {}) nie je číslo",
                val, _column, _row + 1
            ))
    }
}

impl DataLoader for CsvDataLoader {
    fn get_name(&self) -> &str {
        "CSV Data Loader"
    }

    fn load_from_string(&mut self, data: &str, target_column: &str) -> Result<LoadedData, String> {
        // Validácia
        self.validate_format(data)?;

        // Parsovanie CSV
        let (headers, records) = self.parse_csv(data)?;

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
                    .ok_or_else(|| format!("Chýbajúca hodnota v stĺpci '{}'", header))?;
                
                let val = self.parse_numeric_value(val_str, header, row_idx)?;
                
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

        // Validácia že máme dáta
        if x_rows.is_empty() || y_data.is_empty() {
            return Err("Nepodarilo sa extrahovať trénovacie dáta".to_string());
        }

        if x_rows.len() != y_data.len() {
            return Err(format!(
                "Nesúlad v počte vzoriek: X má {}, y má {}",
                x_rows.len(),
                y_data.len()
            ));
        }

        // Vytvorenie DenseMatrix
        let x_data = DenseMatrix::from_2d_vec(&x_rows).unwrap();

        // Filtrovanie headers (bez target stĺpca)
        let feature_headers: Vec<String> = headers
            .into_iter()
            .filter(|h| h != target_column)
            .collect();

        Ok(LoadedData::new(feature_headers, x_data, y_data, records))
    }

    fn get_available_columns(&self, data: &str) -> Result<Vec<String>, String> {
        let (headers, _) = self.parse_csv(data)?;
        Ok(headers)
    }

    fn validate_format(&self, data: &str) -> Result<(), String> {
        if data.trim().is_empty() {
            return Err("CSV dáta sú prázdne".to_string());
        }

        // Základná validácia CSV formátu
        let lines: Vec<&str> = data.lines().collect();
        if lines.len() < 2 {
            return Err("CSV musí obsahovať aspoň header a jeden riadok dát".to_string());
        }

        Ok(())
    }
}

impl Default for CsvDataLoader {
    fn default() -> Self {
        Self::new()
    }
}
