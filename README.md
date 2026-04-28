# Diplomka

Toto je moja diplomová práca implementujúca webovú aplikáciu pre analýzu dát a strojové učenie pomocou Rust (WASM) a JavaScript.

## Štruktúra kódu

Projekt je rozdelený na nasledujúce hlavné časti:

### Rust backend (WASM)
- **`src/lib.rs`**: Hlavný vstupný bod WASM modulu
- **`src/data_loading/`**: Moduly pre načítanie dát z CSV a JSON súborov
  - `csv_data_loader.rs`, `json_data_loader.rs`, `data_loader_factory.rs`
- **`src/feature_selection_strategies/`**: Implementácie stratégií výberu vlastností
  - `chi_square_selector.rs`, `mutual_information_selector.rs`, `variance_selector.rs`, atď.
- **`src/models/`**: Implementácie strojových učebných modelov
  - `knn.rs`, `svm.rs`, `random_forest.rs`, `logreg.rs`, atď.
- **`src/processing/`**: Spracovanie a transformácia dát
  - `minmax_scaler.rs`, `one_hot_encoder.rs`, `null_handler.rs`, atď.
- **`src/target_analysis/`**: Analýza vzťahov s cieľovou premennou
  - `correlation_analyzer.rs`, `mutual_information_analyzer.rs`
- **`src/pipeline/`**: Pipeline pre spracovanie dát
- **`src/wasm_facade/`**: Rozhranie medzi Rust a JavaScript

### JavaScript frontend
- **`js/app.js`**: Hlavná logika aplikácie
- **`js/data.js`**: Správa dát a komunikácia s WASM
- **`js/editor.js`**: Editor pre konfiguráciu pipeline
- **`js/heatmap.js`**: Vizualizácia heatmap
- **`js/selectors.js`**: Komponenty pre výber vlastností
- **`js/state.js`**: Správa stavu aplikácie
- **`js/utils.js`**: Pomocné funkcie
- **`js/main.js`**: Inicializácia aplikácie

### Štýly
- **`css/layout.css`**: Základné rozloženie
- **`css/components.css`**: Štýly komponentov
- **`css/editor-heatmap.css`**: Štýly pre editor a heatmap
- **`css/feature-selection.css`**: Štýly pre výber vlastností

### Testy a dáta
- **`tests/`**: Python testy pomocou pytest
- **`test_data/`**: Testovacie CSV súbory

## Ako spustiť

### Predpoklady
- Rust s wasm-pack
- Node.js a npm (pre development server)
- Python s pytest (pre testy)

### Lokálne spustenie
1. Naklonujte repozitár
2. Spustite `./build-and-run.sh` pre build a spustenie development servera
3. Alebo `./serve.sh` pre spustenie iba servera (po builde)

### Testy
Spustite `./run-tests.sh` pre spustenie všetkých testov.

### Build
- `./build-and-run.sh` - buildí WASM a spúšťa server
- PowerShell verzie sú dostupné pre Windows

## Deploy

Projekt je automaticky deploynutý na GitHub Pages: [https://vodocap.github.io/diplomka/](https://vodocap.github.io/diplomka/)

Deploy sa vykonáva cez GitHub Actions pri push do main vetvy.