# ML Pipeline — WebAssembly Aplikácia

## Technická dokumentácia

**Verzia:** 0.1.0  
**Technológie:** Rust, WebAssembly, HTML/CSS/JS  
**Autor:** Matúš  
**Dátum:** Marec 2026

---

## Obsah

1. [Úvod a motivácia](#1-úvod-a-motivácia)
2. [Architektúra systému](#2-architektúra-systému)
3. [Návrhové vzory](#3-návrhové-vzory)
4. [Modely strojového učenia](#4-modely-strojového-učenia)
5. [Procesory dát (preprocessing)](#5-procesory-dát-preprocessing)
6. [Selektory príznakov (feature selection)](#6-selektory-príznakov-feature-selection)
7. [Analýza cieľovej premennej (target analysis)](#7-analýza-cieľovej-premennej-target-analysis)
8. [Embedded selektory](#8-embedded-selektory)
9. [Mutual Information — KSG estimátor](#9-mutual-information--ksg-estimátor)
10. [Evaluácia modelov](#10-evaluácia-modelov)
11. [Pipeline — životný cyklus](#11-pipeline--životný-cyklus)
12. [WASM API rozhranie](#12-wasm-api-rozhranie)
13. [Načítavanie dát](#13-načítavanie-dát)
14. [Frontendová aplikácia](#14-frontendová-aplikácia)
15. [Konfigurácia a nasadenie](#15-konfigurácia-a-nasadenie)
16. [Automatizované testovanie](#16-automatizované-testovanie)
17. [Presety pipeline](#17-presety-pipeline)
18. [Používateľská príručka](#18-používateľská-príručka)

---

## 1. Úvod a motivácia

Táto aplikácia predstavuje **kompletný ML pipeline**, ktorý beží priamo v prehliadači vďaka technológii **WebAssembly (WASM)**. Celá výpočtová logika je napísaná v jazyku **Rust** a skompilovaná do WASM modulu, čo umožňuje:

- **Výkon blízky natívnemu** — Rust kód beží takmer rovnako rýchlo ako natívna aplikácia
- **Žiadny server** — všetky výpočty prebiehajú na strane klienta, nevyžaduje sa žiadny backend
- **Portabilita** — aplikácia funguje v akomkoľvek modernom prehliadači
- **Bezpečnosť dát** — dáta nikdy neopúšťajú zariadenie používateľa

### Kľúčové funkcie

| Funkcia | Popis |
|---------|-------|
| Trénovanie modelov | Logistická/lineárna regresia, KNN, Decision Tree |
| Preprocessing dát | 15+ procesorov (scaling, encoding, transformácie) |
| Selekcia príznakov | 8 metód vrátane metaheuristík (VNS, SA) |
| Analýza targetu | 4 metódy analýzy cieľovej premennej |
| Vizualizácie | Heatmapa korelácií/MI, interaktívna mapa príznakov |
| Editor dát | In-browser editácia dátovej sady |
| Evaluácia | 18+ metrík pre klasifikáciu a regresiu |

### Technologický stack

```
┌──────────────────────────────────────────────┐
│            Frontend (index.html)             │
│         HTML / CSS / JavaScript              │
│    Single-Page Application, 3600+ riadkov    │
├──────────────────────────────────────────────┤
│          WebAssembly Bridge                   │
│         wasm-bindgen + serde                 │
├──────────────────────────────────────────────┤
│           Rust Core Engine                   │
│   SmartCore ML · KD-tree · Statrs            │
│   ~5000 riadkov Rust kódu                    │
└──────────────────────────────────────────────┘
```

---

## 2. Architektúra systému

### Štruktúra projektu

```
src/
├── lib.rs                    # Vstupný bod WASM modulu
├── mi_estimator.rs           # Zdieľaný KSG MI estimátor
├── data_loading/             # Načítavanie dát (CSV, JSON)
│   ├── csv_data_loader.rs
│   ├── json_data_loader.rs
│   ├── factory.rs
│   └── mod.rs
├── models/                   # ML modely
│   ├── IModel.rs             # Trait definícia
│   ├── linreg.rs             # Lineárna regresia
│   ├── logreg.rs             # Logistická regresia
│   ├── knn.rs                # K-Nearest Neighbors
│   ├── tree.rs               # Decision Tree
│   └── factory.rs
├── processing/               # Preprocessing procesorov
│   ├── scaler.rs             # Standard Scaler
│   ├── minmax_scaler.rs      # MinMax Scaler
│   ├── robust_scaler.rs      # Robust Scaler
│   ├── binner.rs             # Diskretizácia
│   ├── ohencoder.rs          # One-Hot Encoding
│   ├── label_encoder.rs      # Label Encoding
│   ├── null_handler.rs       # Ošetrenie chýbajúcich hodnôt
│   ├── outlier_clipper.rs    # Orezávanie odľahlých hodnôt
│   ├── log_transformer.rs    # Log transformácia
│   ├── power_transformer.rs  # Box-Cox / Yeo-Johnson
│   ├── time_converter.rs     # Konverzia času
│   ├── selective_processor.rs # Automatická detekcia typov stĺpcov
│   ├── processor_decorator.rs # Reťazenie procesorov
│   └── factory.rs
├── feature_selection_strategies/  # Selektory príznakov
│   ├── variance_selector.rs
│   ├── correlation_selector.rs
│   ├── chi_square_selector.rs
│   ├── information_gain_selector.rs
│   ├── mutual_information_selector.rs
│   ├── smc_selector.rs
│   ├── synergy_vns_selector.rs    # VNS metaheuristika
│   ├── synergy_sa_selector.rs     # SA metaheuristika
│   └── factory.rs
├── target_analysis/          # Analýza cieľovej premennej
│   ├── correlation_analyzer.rs
│   ├── entropy_analyzer.rs
│   ├── mutual_information_analyzer.rs
│   ├── smc_analyzer.rs
│   └── factory.rs
├── embedded/                 # Embedded selektory
│   ├── random_forest_selector.rs
│   ├── ridge_selector.rs
│   └── factory.rs
├── evaluation/               # Evaluácia modelov
│   ├── evaluator.rs
│   └── metrics.rs
├── pipeline/                 # Jadro pipeline
│   ├── pipeline.rs           # Fasáda (Facade)
│   ├── builder.rs            # Builder pattern
│   ├── director.rs           # Presety
│   └── compatibility.rs      # Kompatibilita komponentov
└── wasm_api/                 # WASM API vrstva
    ├── ml_pipeline_api.rs    # Hlavný API endpoint (~3000 riadkov)
    ├── factory_api.rs        # Factory API
    └── data_loader_api.rs    # Data Loader API
```

### Vrstvy systému

```
┌─────────────────────────────────────────────────────────┐
│                   WASM API vrstva                       │
│  WasmMLPipeline · WasmFactory · WasmDataLoader          │
│  (serde-wasm-bindgen, JsValue konverzie)                │
├─────────────────────────────────────────────────────────┤
│                  Pipeline vrstva                        │
│  MLPipeline (Facade) · MLPipelineBuilder · Director      │
│  CompatibilityRegistry (Singleton)                      │
├─────────────────────────────────────────────────────────┤
│              Doménová logika                            │
│  Modely · Procesory · Selektory · Analyzéry             │
│  (Strategy pattern — trait implementácie)                │
├─────────────────────────────────────────────────────────┤
│             Spoločné utility                            │
│  MI Estimátor (KSG + KD-tree) · Evaluátor               │
│  DataLoader Factory · ProcessorChain                    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Návrhové vzory

Projekt využíva **6 klasických návrhových vzorov** (Design Patterns):

### 3.1 Builder Pattern — `MLPipelineBuilder`

Umožňuje postupné skladanie pipeline s fluent API:

```rust
let pipeline = MLPipelineBuilder::new()
    .model("knn")                       // Výber modelu
    .model_param("k", "5")              // Parameter modelu
    .processor("scaler")                // Preprocessing
    .feature_selector("variance")       // Selekcia príznakov
    .selector_param("threshold", "0.01") // Parameter selektora
    .evaluation_mode("classification")  // Mód evaluácie
    .build()?;                          // Zostavenie + validácia
```

**Validácia pri `build()`:**
1. Overí, že model je nastavený
2. Skontroluje kompatibilitu cez `CompatibilityRegistry`
3. Vytvorí komponenty cez príslušné Factory triedy
4. Automaticky detekuje evaluation mode, ak nie je zadaný

### 3.2 Factory Pattern

Každá skupina komponentov má vlastnú Factory:

| Factory | Metóda | Vstup → Výstup |
|---------|--------|----------------|
| `ModelFactory` | `create(name)` | `"knn"` → `Box<dyn IModel>` |
| `ProcessorFactory` | `create(name)` | `"scaler"` → `Box<dyn DataProcessor>` |
| `FeatureSelectorFactory` | `create(name)` | `"variance"` → `Box<dyn FeatureSelector>` |
| `TargetAnalyzerFactory` | `create(name)` | `"correlation"` → `Box<dyn TargetAnalyzer>` |
| `DataLoaderFactory` | `create(format)` | `"csv"` → `Box<dyn DataLoader>` |
| `EmbeddedSelectorFactory` | `create(name)` | `"ridge"` → `Box<dyn EmbeddedFeatureSelector>` |

Všetky Factory triedy poskytujú aj metódy `available()`, `get_description()` a `get_supported_params()`.

### 3.3 Strategy Pattern

Správanie je zapuzdrené za spoločným trait rozhraním. Klient pracuje s `Box<dyn Trait>` — konkrétna implementácia je zameniteľná:

```rust
pub trait IModel {
    fn get_name(&self) -> &str;
    fn train(&mut self, x: &DenseMatrix<f64>, y: &[f64]) -> Result<(), String>;
    fn predict(&self, x: &DenseMatrix<f64>) -> Result<Vec<f64>, String>;
    fn get_supported_params(&self) -> Vec<&str>;
    fn set_param(&mut self, key: &str, value: &str) -> Result<(), String>;
}
```

Rovnako pre `DataProcessor`, `FeatureSelector`, `TargetAnalyzer`, `DataLoader`, `EmbeddedFeatureSelector`.

### 3.4 Facade Pattern — `MLPipeline`

`MLPipeline` je hlavný vstupný bod, ktorý orchestruje celý ML workflow:

```
Vstupná matica X, vektor y
        │
        ▼
┌─── Feature Selection ───┐
│   Výber relevantných     │
│   príznakov               │
└──────────┬───────────────┘
           ▼
┌─── Preprocessing ────────┐
│   Fit + Transform na     │
│   trénovacej sade         │
└──────────┬───────────────┘
           ▼
┌─── Model Training ───────┐
│   Trénovanie modelu       │
│   na predspracovaných dátach│
└──────────┬───────────────┘
           ▼
┌─── Evaluation ───────────┐
│   Metriky na testovacej   │
│   sade (ak je split)      │
└──────────────────────────┘
```

### 3.5 Decorator Pattern — `ProcessorChain`

Umožňuje reťazenie viacerých procesorov:

```rust
// Vytvorí reťazec: Outlier Clipper → Standard Scaler
let chain = ProcessorFactory::create_chain(vec!["outlier_clipper", "scaler"]);
```

`SelectiveProcessor` je ďalší dekorátor, ktorý automaticky detekuje typy stĺpcov (Numeric, Categorical, Discrete) a aplikuje procesor len na vhodné stĺpce.

### 3.6 Singleton Pattern — `CompatibilityRegistry`

Jediná inštancia registra kompatibility, chránená `Mutex-om`:

```rust
static REGISTRY: Lazy<Mutex<CompatibilityRegistry>> = Lazy::new(|| {
    Mutex::new(CompatibilityRegistry::new())
});
```

Definuje, ktoré modely sú kompatibilné s ktorými procesormi a selektormi.

---

## 4. Modely strojového učenia

Všetky modely implementujú trait `IModel` a používajú knižnicu **SmartCore**.

### 4.1 Lineárna Regresia (`linreg`)

| Vlastnosť | Hodnota |
|-----------|---------|
| **Typ** | Regresia |
| **Implementácia** | `smartcore::linear::linear_regression::LinearRegression` |
| **Parametre** | `solver`: voľba riešiča — `"qr"` (QR dekompozícia) alebo `"svd"` (SVD dekompozícia) |
| **Predvolená hodnota** | `solver = "qr"` |
| **Popis** | Predikuje spojité hodnoty minimalizáciou súčtu štvorcov reziduálov |

**Matematický model:**

$$\hat{y} = X \cdot \beta + \epsilon$$

kde $\beta = (X^T X)^{-1} X^T y$ (metóda najmenších štvorcov).

### 4.2 Logistická Regresia (`logreg`)

| Vlastnosť | Hodnota |
|-----------|---------|
| **Typ** | Klasifikácia |
| **Implementácia** | `smartcore::linear::logistic_regression::LogisticRegression` |
| **Parametre** | `alpha`: regularizačný parameter (L2 penalizácia, f64) |
| **Predvolená hodnota** | `alpha = 1.0` |
| **Popis** | Binárna klasifikácia modelujúca pravdepodobnosť triedy |

**Sigmoid funkcia:**

$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta^T x)}}$$

Cieľová premenná `y` sa automaticky konvertuje na `u32` triedy.

### 4.3 K-Nearest Neighbors (`knn`)

| Vlastnosť | Hodnota |
|-----------|---------|
| **Typ** | Regresia aj klasifikácia |
| **Implementácia** | `smartcore::neighbors::knn_regressor::KNNRegressor` |
| **Parametre** | `k`: počet susedov (usize) |
| **Predvolená hodnota** | `k = 5` |
| **Popis** | Predikuje priemer k najbližších susedov (euklidovská vzdialenosť) |

**Vzdialenostná metrika:**

$$d(x, x') = \sqrt{\sum_{i=1}^{n} (x_i - x'_i)^2}$$

### 4.4 Decision Tree (`tree`)

| Vlastnosť | Hodnota |
|-----------|---------|
| **Typ** | Regresia aj klasifikácia |
| **Implementácia** | `smartcore::tree::decision_tree_regressor::DecisionTreeRegressor` |
| **Parametre** | `max_depth` (u16), `min_samples_split` (u16) |
| **Predvolené hodnoty** | `max_depth = 10`, `min_samples_split = 2` |
| **Popis** | Rekurzívne delí priestor príznakov podľa najlepšieho rozdeľovacieho kritéria |

---

## 5. Procesory dát (preprocessing)

Procesory implementujú trait `DataProcessor` a sú automaticky obalené v `SelectiveProcessor`, ktorý detekuje typ stĺpca:

- **Numeric** — stĺpce s reálnymi číslami (≤ 20 unikátnych hodnôt = Discrete)
- **Categorical** — stĺpce s textovými hodnotami
- **Discrete** — stĺpce s malým počtom unikátnych číselných hodnôt

### 5.1 Numerické procesory

#### Standard Scaler (`scaler`)

Normalizácia na priemer 0 a štandardnú odchýlku 1:

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

| Parameter | Typ | Popis |
|-----------|-----|-------|
| — | — | Žiadne konfigurovateľné parametre |

#### MinMax Scaler (`minmax_scaler`)

Normalizácia do intervalu [min, max]:

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}} \cdot (max - min) + min$$

| Parameter | Typ | Predvolená hodnota | Popis |
|-----------|-----|---------------------|-------|
| `min` | f64 | 0.0 | Dolná hranica normalizácie |
| `max` | f64 | 1.0 | Horná hranica normalizácie |

#### Robust Scaler (`robust_scaler`)

Škálovanie pomocou mediánu a IQR, odolné voči odľahlým hodnotám:

$$x_{scaled} = \frac{x - \text{medián}}{IQR}$$

kde $IQR = Q_3 - Q_1$ (interkvartilové rozpätie).

#### Outlier Clipper (`outlier_clipper`)

Orezáva odľahlé hodnoty. Podporuje tri metódy:
- **IQR metóda** — orezáva mimo $[Q_1 - 1.5 \cdot IQR, \; Q_3 + 1.5 \cdot IQR]$
- **Percentilová metóda** — orezáva mimo zadaného percentilu
- **Z-score metóda** — orezáva hodnoty s |z| > prahová hodnota

#### Log Transformer (`log_transformer`)

Logaritmická transformácia pre normalizáciu rozdelenia:

$$x_{transformed} = \ln(x + \text{offset})$$

| Parameter | Typ | Predvolená hodnota | Popis |
|-----------|-----|---------------------|-------|
| `offset` | f64 | 1.0 | Konštanta pridaná pred logaritmom (pre nulové hodnoty) |

#### Power Transformer (`power_transformer`)

Box-Cox alebo Yeo-Johnson transformácia na normalizáciu rozdelenia.

### 5.2 Diskretizačné procesory

#### Binner (`binner`)

Diskretizuje spojité hodnoty do binov (intervalov):

| Parameter | Typ | Predvolená hodnota | Popis |
|-----------|-----|---------------------|-------|
| `bins` | usize | 10 | Počet diskretizačných intervalov |

**Dôležité:** Vyžadovaný pre `Information Gain` selektor.

### 5.3 Kódovacie procesory (pre kategorické dáta)

#### One-Hot Encoder (`onehot`)

Vytvára binárne stĺpce pre každú kategóriu. Napr. farba ∈ {červená, modrá, zelená} → 3 binárne stĺpce.

#### Label Encoder (`label_encoder`)

Kóduje kategórie na celé čísla: A → 0, B → 1, C → 2.

#### Ordinal Encoder (`ordinal_encoder`)

Ako Label Encoder, ale zachováva poradie hodnôt.

#### Frequency Encoder (`frequency_encoder`)

Kóduje kategórie podľa frekvencie výskytu v dátach.

#### Target Encoder (`target_encoder`)

Kóduje kategórie priemerom cieľovej premennej pre danú kategóriu.

### 5.4 Špeciálne procesory

#### Null Handler (`null_handler`)

Ošetruje chýbajúce hodnoty (NaN, null, prázdne reťazce):

| Parameter | Typ | Popis |
|-----------|-----|-------|
| `null_repr` | String | Reprezentácia null hodnôt (predvolená: `""`) |
| `strategy` | String | Stratégia nahradenia: `"mean"`, `"median"`, `"mode"` |

#### Time Converter (`time_converter`)

Konvertuje časové hodnoty medzi formátmi (sekundy, minúty, hodiny).

#### Comma to Dot (`comma_to_dot`)

Nahrádza desatinné čiarky bodkami (`"3,14"` → `"3.14"`).

#### Thousands Separator Remover (`thousands_separator_remover`)

Odstraňuje oddeľovače tisícov (`"1,000,000"` → `"1000000"`).

### 5.5 Reťazenie procesorov

Procesory je možné reťaziť pomocou `ProcessorChain` (Decorator pattern):

```rust
let chain = ProcessorFactory::create_chain(vec![
    "null_handler",
    "outlier_clipper",
    "scaler"
]);
// Výsledok: NullHandler → OutlierClipper → StandardScaler
```

---

## 6. Selektory príznakov (feature selection)

Feature selection je **kľúčový krok** v ML pipeline, ktorý redukuje počet príznakov na najrelevantnejšie.

### Prehľad selektorov

| Selektor | Typ | Popis | Metrika |
|----------|-----|-------|---------|
| Variance Threshold | Filter | Odstráni konštantné features | Variancia |
| Correlation | Filter | Výber podľa korelácie s targetom | Pearsonova korelácia |
| Chi-Square Test | Filter | Test nezávislosti (len klasifikácia) | χ² štatistika |
| Information Gain | Filter | Redukcia entropie (vyžaduje Binner!) | IG/Entropia |
| Mutual Information | Filter | KSG estimátor (kontinuálne dáta) | MI score |
| SMC | Filter | Príspevok k R² predikcii | R² drop |
| **Synergy VNS** | **Wrapper (metaheuristika)** | **Variable Neighborhood Search** | **Fitness** |
| **Synergy SA** | **Wrapper (metaheuristika)** | **Simulated Annealing** | **Fitness** |

### 6.1 Variance Threshold

Odstráni príznaky s varianciou pod prahovou hodnotou — typicky konštantné alebo takmer konštantné stĺpce.

| Parameter | Predvolená | Popis |
|-----------|-----------|-------|
| `threshold` | 0.01 | Minimálna variancia |

### 6.2 Correlation Selector

Vyberie príznaky s najvyššou absolútnou Pearsonovou koreláciou k cieľovej premennej.

| Parameter | Predvolená | Popis |
|-----------|-----------|-------|
| `threshold` | 0.95 | Prahová hodnota korelácie |

### 6.3 Chi-Square Test

Test nezávislosti medzi príznakmi a targetom. **Len pre klasifikáciu.**

$$\chi^2 = \sum \frac{(O_i - E_i)^2}{E_i}$$

| Parameter | Predvolená | Popis |
|-----------|-----------|-------|
| `num_features` | 5 | Počet vybraných príznakov |

### 6.4 Information Gain

Meria redukciu entropie — koľko informácie príznak prinesie o targete.

$$IG(T, A) = H(T) - H(T|A)$$

**Upozornenie:** Vyžaduje predchádzajúce spracovanie procesorom `Binner`!

| Parameter | Predvolená | Popis |
|-----------|-----------|-------|
| `num_features` | 5 | Počet vybraných príznakov |

### 6.5 Mutual Information (KSG)

Odhaduje vzájomnú informáciu medzi príznakom a targetom pomocou KSG estimátora. Funguje na **spojitých dátach** bez nutnosti diskretizácie.

| Parameter | Predvolená | Popis |
|-----------|-----------|-------|
| `num_features` | 5 | Počet vybraných príznakov |

### 6.6 SMC (Squared Multiple Correlation)

Meria príspevok každého príznaku k multi-lineárnej predikcii targetu cez pokles v R²:

$$SMC_i = R^2_{all} - R^2_{all \setminus i}$$

| Parameter | Predvolená | Popis |
|-----------|-----------|-------|
| `num_features` | 5 | Počet vybraných príznakov |

### 6.7 Synergy VNS (Variable Neighborhood Search)

Metaheuristický selektor, ktorý optimalizuje **kombinovanú fitness funkciu** prehľadávaním rôznych typov susedstiev.

#### Fitness funkcia

$$F(S) = \alpha \cdot \overline{MI}(S) + \beta \cdot \text{Diverzita}(S) - \gamma \cdot \text{Redundancia}(S)$$

kde:
- $\overline{MI}(S)$ — priemerná MI vybraných features s targetom (relevancia)
- $\text{Diverzita}(S)$ — priemerná inverzná MI medzi features: $\frac{1}{1 + MI(f_i, f_j)}$ (synergia)
- $\text{Redundancia}(S)$ — priemerná MI medzi features (penalizácia)

#### Typy susedstiev (neighborhoods)

| k | Typ | Popis |
|---|-----|-------|
| 1 | Swap | Vymeň 1 príznak za iný |
| 2 | Flip-k | Vymeň k príznakov naraz (k=2) |
| 3 | Add | Pridaj 1 príznak |
| 4 | Remove | Odober 1 príznak |

#### Parametre

| Parameter | Predvolená | Popis |
|-----------|-----------|-------|
| `num_features` | 5 | Cieľový počet príznakov |
| `max_iterations` | 100 | Počet iterácií algoritmu |
| `k_max` | 4 | Počet typov neighborhoods (1-4) |
| `alpha` | 0.7 | Váha relevancie (MI s targetom) |
| `beta` | 0.2 | Váha synergie (diverzita features) |
| `gamma` | 0.3 | Penalizácia redundancie |
| `initial_solution` | `"greedy"` | Počiatočné riešenie (`"greedy"` / `"random"`) |

#### Algoritmus VNS

```
1. Inicializácia: S₀ = greedy/random riešenie
2. Pre každú iteráciu:
   a. k = 1
   b. Kým k ≤ k_max:
      i.   Generuj suseda S' v k-tom neighborhood
      ii.  Lokálne prehľadávanie: S'' = local_search(S')
      iii. Ak F(S'') > F(S*): aktualizuj najlepšie, k = 1
      iv.  Inak: k++
3. Diversifikácia: každých 30 iterácií reštart
```

### 6.8 Synergy SA (Simulated Annealing)

Metaheuristický selektor využívajúci **simulované žíhanie** s **Boltzmannovým akceptačným kritériom**. Na rozdiel od VNS akceptuje aj horšie riešenia s pravdepodobnosťou závislou od teploty, čo mu umožňuje efektívnejšie unikať z lokálnych optím.

#### Boltzmannovo akceptačné kritérium

$$P(\text{accept}) = \begin{cases} 1 & \text{ak } \Delta F \geq 0 \\ e^{\Delta F / T} & \text{ak } \Delta F < 0 \end{cases}$$

kde $\Delta F = F(S') - F(S)$ je zmena fitness a $T$ je aktuálna teplota.

#### Chladiaci rozvrh

$$T_{i+1} = T_i \cdot \text{cooling\_rate}$$

S **reheating** mechanizmom: po `reheat_interval` iteráciách bez zlepšenia sa teplota čiastočne zvýši a algoritmus sa reštartuje z najlepšieho riešenia.

#### Parametre

| Parameter | Predvolená | Popis |
|-----------|-----------|-------|
| `num_features` | 5 | Cieľový počet príznakov |
| `max_iterations` | 200 | Maximálny počet iterácií |
| `initial_temp` | 1.0 | Počiatočná teplota (vyššia = viac explorácie) |
| `cooling_rate` | 0.95 | Rýchlosť chladnutia (0.5 – 0.9999) |
| `min_temp` | 0.001 | Zastavenie pod touto teplotou |
| `reheat_interval` | 50 | Po koľkých iteráciách reheating |
| `alpha` | 0.7 | Váha relevancie (MI s targetom) |
| `beta` | 0.2 | Váha synergie |
| `gamma` | 0.3 | Penalizácia redundancie |
| `initial_solution` | `"greedy"` | Počiatočné riešenie (`"greedy"` / `"random"`) |

#### Porovnanie VNS vs SA

| Aspekt | VNS | SA |
|--------|-----|-----|
| Prehľadávanie | Deterministické zmeny neighborhoods | Stochastické, pravdepodobnostné |
| Akceptačné kritérium | Len lepšie riešenia | Aj horšie (Boltzmann) |
| Únik z lokálnych optím | Cez zmenu susedstva | Cez teplotu → akceptovanie horších |
| Parametrizácia | k_max, neighborhoods | Teplota, cooling_rate, reheating |
| Konvergencia | Typicky rýchlejšia | Pomalšia ale dôkladnejšia |

---

## 7. Analýza cieľovej premennej (target analysis)

Analýza pomáha používateľovi identifikovať **najvhodnejší stĺpec ako cieľovú premennú**.

### 7.1 Correlation Analyzer

Vypočíta sumu štvorcov Pearsonových korelácií medzi kandidátom a ostatnými stĺpcami:

$$\text{Score} = \sum_{i=1}^{n} r_i^2$$

Stĺpec s najvyšším Σr² je najlepší kandidát — „vysvetľuje" najviac z ostatných premenných.

### 7.2 Mutual Information Analyzer

Používa KSG estimátor na výpočet sumy MI medzi kandidátom a všetkými ostatnými stĺpcami:

$$\text{Score} = \sum_{i=1}^{n} MI(Y, X_i)$$

Zachytáva aj **nelineárne závislosti**, ktoré korelácia nezachytí.

### 7.3 SMC Analyzer (Squared Multiple Correlation)

Meria, koľko variability kandidáta je vysveteľných ostatnými premennými pomocou multi-lineárneho modelu.

### 7.4 Entropy Analyzer

Kombinuje entropiu a Information Gain:
- **Entropia** — miera neistoty: $H(Y) = -\sum p_i \log_2 p_i$
- **Information Gain** — redukcia entropie pri znalosti iného príznaku

### Výstup analyzérov

Každý analyzér vracia štruktúru `TargetCandidate`:

```rust
pub struct TargetCandidate {
    pub column_index: usize,
    pub column_name: String,
    pub score: f64,
    pub unique_values: usize,
    pub variance: f64,
    pub suggested_type: String,    // "classification" / "regression"
    pub extra_metrics: HashMap<String, f64>,
}
```

Typ (klasifikácia vs regresia) sa automaticky heuristicky odhadne: < 20 unikátnych hodnôt → klasifikácia.

---

## 8. Embedded selektory

Na rozdiel od filter/wrapper metód, embedded selektory extrahujú dôležitosť príznakov **priamo z trénovaného modelu**.

### 8.1 Random Forest Selector

| Vlastnosť | Hodnota |
|-----------|---------|
| **Podpora** | Klasifikácia aj regresia |
| **Metóda** | Dôležitosť príznakov zo stromového modelu |
| **Kľúče** | `tree`, `random_forest`, `tree_importance` |

### 8.2 Ridge Selector

| Vlastnosť | Hodnota |
|-----------|---------|
| **Podpora** | Len regresia |
| **Metóda** | L2 regularizácia — veľkosť koeficientov indikuje dôležitosť |
| **Kľúče** | `ridge`, `l2` |
| **Parameter** | `alpha` (regularizačná sila, predvolená: 1.0) |

Automatický výber: pre klasifikáciu → `RandomForestSelector`, pre regresiu → `RidgeSelector`.

---

## 9. Mutual Information — KSG estimátor

### Algoritmus KSG (Kraskov-Stögbauer-Grassberger)

Zdieľaný modul `mi_estimator.rs` implementuje **KSG algoritmus** pre odhad vzájomnej informácie medzi dvoma premennými bez nutnosti diskretizácie.

#### Vzorec

$$MI(X, Y) = \psi(k) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle + \psi(N)$$

kde:
- $\psi$ — digamma funkcia
- $k$ — počet najbližších susedov
- $n_x, n_y$ — počet bodov v marginom intervale
- $N$ — celkový počet vzoriek

#### Optimalizácie

Modul využíva dve kľúčové optimalizácie pre efektívny výpočet:

**1. KD-tree s Chebyshevovou vzdialenosťou**

$$d_\infty(a, b) = \max_i |a_i - b_i|$$

KD-tree (`kdtree` crate) umožňuje nájsť k najbližších susedov v $O(k \log n)$ namiesto $O(n)$ pri naivnom prehľadávaní.

**2. Zoradené polia + binárne vyhľadávanie**

Pre počítanie marginálnych susedov v 1D priestore sa používa zoradený index s binárnym hľadaním. `SortedIndex` štruktúra umožňuje count_within operáciu v $O(\log n)$ namiesto $O(n)$.

**Celková zložitosť:**

$$O(n \cdot k \cdot \log n) \quad \text{namiesto} \quad O(n^2)$$

#### Adaptívne k

Pre veľké datasety sa automaticky redukuje k:
- > 1000 vzoriek: k ≤ 2
- > 500 vzoriek: k ≤ 3
- Inak: k = 3 (predvolené)

#### API

```rust
// Jedna dvojica premenných
pub fn estimate_mi_ksg(x: &[f64], y: &[f64], k: usize) -> f64;

// Celá matica (všetky páry stĺpcov)
pub fn compute_mi_matrix(columns: &[Vec<f64>], k: usize) -> Vec<Vec<f64>>;
```

---

## 10. Evaluácia modelov

`ModelEvaluator` automaticky rozpozná typ úlohy (klasifikácia / regresia) a nasadí príslušné metriky.

### 10.1 Klasifikačné metriky

| Metrika | Vzorec | Popis |
|---------|--------|-------|
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Podiel správnych predikcií |
| **Precision** | $\frac{TP}{TP + FP}$ | Presnosť pozitívnych predikcií |
| **Recall** | $\frac{TP}{TP + FN}$ | Úplnosť zachytenia pozitívnych |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ | Harmonický priemer precision a recall |
| **Specificity** | $\frac{TN}{TN + FP}$ | Schopnosť identifikovať negatívne |
| **FPR** | $\frac{FP}{FP + TN}$ | Miera falošných poplachov |
| **FNR** | $\frac{FN}{FN + TP}$ | Miera chýbajúcich detekcií |
| **MCC** | $\frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$ | Matthews Correlation — vhodný pre nevyvážené dáta |

### 10.2 Regresné metriky

| Metrika | Vzorec | Popis |
|---------|--------|-------|
| **MSE** | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Stredná kvadratická chyba |
| **RMSE** | $\sqrt{MSE}$ | Odmocnina MSE |
| **MAE** | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Stredná absolútna chyba |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Koeficient determinácie |
| **MAPE** | $\frac{100}{n}\sum\left|\frac{y_i - \hat{y}_i}{y_i}\right|$ | Stredná absolútna percentuálna chyba |
| **MASE** | $\frac{MAE}{MAE_{naive}}$ | Škálovaná chyba vs naivná predikcia |
| **MdAE** | $\text{medián}(|y_i - \hat{y}_i|)$ | Medián absolútnej chyby |
| **Explained Var** | $1 - \frac{Var(y - \hat{y})}{Var(y)}$ | Vysvetlená variancia |
| **Pearson r** | $\frac{Cov(y, \hat{y})}{\sigma_y \sigma_{\hat{y}}}$ | Pearsonova korelácia predikcií |

---

## 11. Pipeline — životný cyklus

### 11.1 Vytvorenie pipeline

```
Používateľ → buildFromConfig(config) → MLPipelineBuilder
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ Validácia    │
                                    │ kompatibility│
                                    └──────┬───────┘
                                           │
                              ┌────────────┼────────────┐
                              ▼            ▼            ▼
                         ModelFactory  ProcessorFactory  SelectorFactory
                              │            │            │
                              ▼            ▼            ▼
                         Box<IModel>  Box<Processor>  Box<Selector>
                              │            │            │
                              └────────────┼────────────┘
                                           ▼
                                     MLPipeline
```

### 11.2 Trénovanie

```
loadData(csv, target)
        │
        ▼
  ┌─────────────┐      ┌──────────────┐
  │ Parse CSV    │─────→│ X: DenseMatrix│
  │ Extract y    │      │ y: Vec<f64>   │
  └─────────────┘      └──────┬────────┘
                               │
                    trainWithSplit(ratio)
                               │
                ┌──────────────┼──────────────┐
                ▼                             ▼
         Train Split (80%)              Test Split (20%)
                │                             │
    ┌───────────┴───────────┐                 │
    ▼                       ▼                 │
Feature Selection    (vybrané indexy)          │
    │                       │                 │
    ▼                       │                 │
Processor.fit()             │                 │
    │                       │                 │
    ▼                       │                 │
Processor.transform()       │                 │
    │                       │                 │
    ▼                       │                 │
Model.train()               │                 │
    │                       │                 │
    └───────────────────────┤                 │
                            ▼                 ▼
                    Model.predict(test)  ←── Apply same
                            │               feature indices
                            │               + processor
                            ▼
                     Evaluate(y_true, y_pred)
                            │
                            ▼
                    EvaluationReport
```

### 11.3 Deterministický split

Pipeline používa **deterministický split** namiesto náhodného. To zabezpečuje:
- Rovnaký train/test split pre všetky selektory pri porovnávaní
- Reprodukovateľné výsledky
- Férové porovnanie metód selekcie

Split indexy sa cachujú v `split_cache` a opätovne používajú.

### 11.4 Cachovací systém

| Cache | Obsah | Invalidácia |
|-------|-------|------------|
| `data_cache` | X matica + y vektor | Pri `loadData()` |
| `analysis_cache` | Parsované CSV stĺpce + hlavičky | Pri `loadData()` |
| `matrices_cache` | Korelačná + MI + SMC matica | Pri `loadData()` |
| `split_cache` | Train/test indexy + data splity | Pri `loadData()` |

---

## 12. WASM API rozhranie

Aplikácia exportuje **3 WASM triedy** do JavaScriptu:

### 12.1 `WasmMLPipeline` — hlavný pipeline

Najväčšia trieda (~3000 riadkov), poskytuje kompletný ML workflow:

| Metóda | Popis |
|--------|-------|
| `new()` | Vytvorí novú inštanciu pipeline |
| `buildFromConfig(config)` | Zostaví pipeline z JSON konfigurácie |
| `buildFromPreset(name, model, ...)` | Zostaví z predpripraveného presetu |
| `loadData(data, target, format)` | Načíta dáta a pripraví X, y |
| `train()` | Trénuje model na celom datasete |
| `trainWithSplit(ratio)` | Trénuje s train/test splitom |
| `trainWithFeatureIndices(ratio, indices)` | Trénuje s explicitnými feature indexami |
| `predict(input)` | Predikcia na jednom vstupe |
| `getInfo()` | Info o aktuálnom pipeline |
| `inspectData(maxRows)` | Náhľad na surové dáta |
| `inspectProcessedData(maxRows)` | Náhľad na predspracované dáta |
| `getFeatureSelectionInfo()` | Detaily o selekcii features |
| `compareSelectors(data, target, format, selectors)` | Porovnanie selektorov |
| `analyzeTargetCandidates(data, format)` | Analýza kandidátov na target |
| `analyzeTargetWith(data, format, method)` | Analýza špecifickou metódou |
| `compareTargetAnalyzers(data, format, methods)` | Porovnanie analyzérov |
| `getFeatureMatrices(data, format)` | Korelačná + MI matica pre heatmapu |
| `checkFeatureRedundancy(...)` | Kontrola redundancie features |
| `getEditableData(data, format)` | Dáta pre editor |
| `applyProcessorToColumn(...)` | Aplikácia procesora na stĺpec |
| `deleteColumn(data, column)` | Zmazanie stĺpca |
| `setCellValue(...)` | Úprava bunky |
| `replaceAllInColumn(...)` | Nahradenie hodnôt v stĺpci |
| `getEmbeddedFeatureRanking(...)` | Embedded feature importance |

### 12.2 `WasmFactory` — factory pre frontend

| Metóda | Popis |
|--------|-------|
| `new()` | Vytvorí factory |
| `getAvailableOptions()` | Všetky modely, procesory, selektory, formáty, presety |
| `getCompatibleProcessors(model)` | Kompatibilné procesory pre model |
| `getCompatibleSelectors(model)` | Kompatibilné selektory pre model |
| `getModelParams(model)` | Parametre modelu |
| `getSelectorParams(selector)` | Parametre selektora |
| `getProcessorParamDefinitions(proc)` | Detailné definície parametrov (typ, min, max, popis) |
| `getPresetDetails(preset)` | Detail presetu |

### 12.3 `WasmDataLoader` — načítavač dát

| Metóda | Popis |
|--------|-------|
| `new(format)` | Loader pre špecifický formát |
| `createAuto(data)` | Auto-detekcia formátu |
| `getAvailableColumns(data)` | Zoznam stĺpcov |
| `validateFormat(data)` | Validácia formátu |
| `loadData(data, target)` | Načítanie dát s targetom |

---

## 13. Načítavanie dát

### Podporované formáty

| Formát | Popis | Auto-detekcia |
|--------|-------|---------------|
| **CSV** | Comma-Separated Values, štandardný tabuľkový formát | Obsahuje `,` a `\n` |
| **JSON** | Array of Objects `[{"col1": val1, ...}, ...]` | Začína `[{` |

### Proces načítania

```
Vstupný reťazec (CSV/JSON)
        │
        ▼
  DataLoaderFactory::create(format)
        │
        ▼
  loader.load_from_string(data, target_column)
        │
        ├── Parse reťazca na stĺpce
        ├── Identifikácia target stĺpca
        ├── Konverzia na numerické hodnoty
        │   (ne-numerické → NaN, potom nahradenie priemerom)
        ├── Vytvorenie DenseMatrix<f64> pre X
        └── Extrakcia Vec<f64> pre y
        │
        ▼
  LoadedData { x_data, y_data, headers }
```

### Ukážkový dataset

Aplikácia obsahuje predpripravený vzorový dataset s 30 riadkami a 14 stĺpcami pre úlohu schvaľovania úverov:

| Stĺpec | Typ | Popis |
|---------|-----|-------|
| `age` | Numeric | Vek žiadateľa |
| `income` | Numeric | Príjem |
| `credit_score` | Numeric | Kreditné skóre |
| `loan_amount` | Numeric | Výška pôžičky |
| `employment_years` | Numeric | Roky zamestnania |
| `debt_to_income` | Numeric | Pomer dlhu k príjmu |
| `num_credit_lines` | Discrete | Počet úverových liniek |
| `late_payments` | Discrete | Počet neskorých platieb |
| `home_ownership` | Numeric (encoded) | Vlastníctvo nehnuteľnosti |
| `education_level` | Numeric (encoded) | Úroveň vzdelania |
| `marital_status` | Numeric (encoded) | Rodinný stav |
| `has_cosigner` | Binary | Má ručiteľa |
| `loan_purpose` | Numeric (encoded) | Účel pôžičky |
| **`approved`** | **Binary target** | **Schválenie (0/1)** |

---

## 14. Frontendová aplikácia

### Technológia

Celá aplikácia je jednostránková (SPA) v jedinom súbore `index.html` (~3600 riadkov):
- **CSS**: ~1050 riadkov — vlastný design, responzívny layout
- **HTML body**: ~350 riadkov — štruktúra UI
- **JavaScript**: ~2200 riadkov — logika aplikácie (ES modules)

### Hlavné sekcie UI

```
┌──────────────────────────────────────────────────────────────┐
│                    ML Pipeline Application                    │
├───────────────────────────┬──────────────────────────────────┤
│                           │                                  │
│   1. VÝBER PIPELINE       │   2. NAČÍTANIE DÁT              │
│                           │                                  │
│   ├── Model dropdown      │   ├── Formát (CSV/JSON)         │
│   ├── Eval mode           │   ├── Text/File toggle          │
│   ├── Model params        │   ├── Target dropdown           │
│   └── [Vytvoriť Pipeline] │   ├── Train/Test split slider   │
│                           │   ├── Target Analysis           │
│                           │   │   ├── Analyzer cards        │
│                           │   │   └── [Porovnať]            │
│                           │   └── [Inspect] [Editor]        │
├───────────────────────────┴──────────────────────────────────┤
│                                                              │
│   3. PRIESKUM A VÝBER PRÍZNAKOV (full width)                 │
│                                                              │
│   ├── Selector comparison grid                               │
│   │   ├── Variance    ├── Correlation   ├── Chi-Square      │
│   │   ├── Info Gain   ├── MI            ├── SMC             │
│   │   ├── Synergy VNS ├── Synergy SA                        │
│   ├── [Porovnať selektory]                                   │
│   ├── Interactive feature map (checkboxes)                   │
│   ├── [Show Heatmap]                                         │
│   └── Training results comparison                            │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   4. DOSTUPNÉ MOŽNOSTI (full width)                          │
│                                                              │
│   ├── Models badges    ├── Processors badges                 │
│   ├── Selectors badges ├── Current pipeline info             │
│   └── Action buttons                                         │
└──────────────────────────────────────────────────────────────┘
```

### Modálne okná

| Modal | Funkcia |
|-------|---------|
| **Heatmap Modal** | Fullscreen korelačná/MI matica s tooltipom, sortovaním, legendou |
| **Analyzer Details** | Detailná analýza cieľovej premennej |
| **Data Editor** | Editácia dát — aplikácia procesorov na stĺpce, mazanie, úprava buniek |
| **Data Inspector** | Tabuľkový náhľad na surové/predspracované dáta |

### JavaScript architektúra

```javascript
// Inicializácia
async function initApp() {
    const wasm = await import('./pkg/wasm.js');
    await wasm.default();           // Init WASM module
    
    factory = new wasm.WasmFactory();
    pipeline = new wasm.WasmMLPipeline();
    availableOptions = convertWasmResult(factory.getAvailableOptions());
    
    populateOptions();              // Naplní UI dropdowny
    setupEventListeners();          // Event handlery
}
```

### Heatmapa

Interaktívna vizualizácia matice korelácií alebo MI medzi všetkými pármi príznakov:

- **Farebná škála**: Modrá (záporná korelácia) → Biela (0) → Červená (kladná korelácia)
- **MI škála**: Biela (0) → Tmavo červená (vysoká MI)
- **Funkcie**: Prepínanie Correlation/MI, zobrazenie/skrytie hodnôt, zoradenie, tooltip s detailmi
- **Ovládanie**: Escape/tlačidlo na zatvorenie, fullscreen overlay

---

## 15. Konfigurácia a nasadenie

### Požiadavky na build

| Nástroj | Verzia | Použitie |
|---------|--------|----------|
| Rust | Edition 2021 | Jadro aplikácie |
| wasm-pack | 0.14+ | Kompilácia do WASM |
| wasm-bindgen | 0.2 | JS ↔ Rust bridge |

### Závislosti (Cargo.toml)

| Crate | Verzia | Popis |
|-------|--------|-------|
| `smartcore` | 0.4 | ML algoritmy (regresia, klasifikácia, stromy, KNN) |
| `kdtree` | 0.7 | KD-tree pre KSG MI estimátor |
| `statrs` | 0.16 | Štatistické funkcie (digamma) |
| `serde` + `serde_json` | 1.0 | Serializácia/deserializácia |
| `csv` | 1.2 | Parsovanie CSV |
| `ndarray` | 0.15 | N-rozmerné polia |
| `once_cell` | 1.19 | Lazy static (Singleton) |
| `getrandom` | 0.2 + `js` feature | Generátor náhodných čísel pre WASM |
| `web-sys` | 0.3 | Web API prístup (console) |

### Buildovanie

```bash
# Kompilácia do WASM
wasm-pack build --target web --out-dir pkg

# Spustenie servera (vyžaduje COOP/COEP hlavičky)
./serve.sh
# alebo
python3 -c "
import http.server, socketserver

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

with socketserver.TCPServer(('', 3333), Handler) as httpd:
    httpd.serve_forever()
"
```

### Build a Run skripty

| Skript | Popis |
|--------|-------|
| `build-and-run.sh` | Skompiluje WASM a spustí server |
| `serve.sh` | Spustí HTTP server na porte 3333 |
| `run-tests.sh` | Spustí automatizované testy |

---

## 16. Automatizované testovanie

### Framework

Projekt používa **Playwright** s **pytest** pre E2E (end-to-end) testovanie.

| Nástroj | Verzia |
|---------|--------|
| Playwright | 1.58+ |
| pytest | 7.4+ |
| pytest-playwright | 0.7+ |

### Testové súbory

| Súbor | Počet testov | Pokrytie |
|-------|-------------|----------|
| `test_01_init.py` | 7 | Načítanie stránky, WASM init, zobrazenie modelov/procesorov/selektorov |
| `test_02_data_loading.py` | 9 | Parsovanie CSV, target dropdown, inspect dát, split slider |
| `test_03_pipeline.py` | 7 | Build pipeline pre každý model, parametre, eval mode |
| `test_04_target_analysis.py` | 5 | Analyzer karty, detaily, porovnanie analyzérov |
| `test_05_selectors.py` | 5 | Selector karty, select all, porovnanie selektorov |
| `test_06_heatmap.py` | 12 | Heatmapa, prepínanie matíc, tooltip, legenda, zoradenie |
| `test_07_editor.py` | 4 | Editor dát, výber stĺpca, procesory |

### Spustenie testov

```bash
# Všetky testy (headed)
python3 -m pytest tests/ --browser chromium -v

# Headless
python3 -m pytest tests/ --browser chromium --headed false

# Filtrovanie
python3 -m pytest tests/ -k heatmap

# Cez skript
./run-tests.sh
./run-tests.sh --headless
./run-tests.sh -k "pipeline"
```

### Testová infraštruktúra

**Fixtures** (`conftest.py`):
- `server` — auto-start HTTP servera na porte 3333
- `app_page` — navigácia + čakanie na WASM init
- `loaded_page` — build pipeline + načítanie dát + potvrdenie targetu (kompletný setup)
- `SAMPLE_CSV` — testovací dataset s 30 riadkami

---

## 17. Presety pipeline

Aplikácia obsahuje ~20 predpripravených konfigurácií pipeline:

### Klasifikačné presety

| Preset | Model | Procesor | Selektor |
|--------|-------|----------|----------|
| `basic_classification` | logreg | scaler | variance |
| `advanced_classification` | logreg | scaler + outlier_clipper | mi |
| `logreg_minmax_chisquare` | logreg | minmax_scaler | chi_square |
| `tree_binner_infogain` | tree | binner | information_gain |
| `knn_classifier` | knn | scaler | variance |
| `knn_robust_mi` | knn | robust_scaler | mi |
| `logreg_scaler_only` | logreg | scaler | — |
| `tree_labelenc_chisquare` | tree | label_encoder | chi_square |
| `logreg_labelenc_variance` | logreg | label_encoder | variance |
| `decision_tree` | tree | — | — |

### Regresné presety

| Preset | Model | Procesor | Selektor |
|--------|-------|----------|----------|
| `basic_regression` | linreg | scaler | correlation |
| `advanced_regression` | linreg | scaler + outlier_clipper | mi |
| `knn_regressor` | knn | scaler | variance |
| `linreg_minmax_correlation` | linreg | minmax_scaler | correlation |
| `linreg_robust_mi` | linreg | robust_scaler | mi |
| `linreg_log_correlation` | linreg | log_transformer | correlation |
| `knn_power_variance` | knn | power_transformer | variance |
| `linreg_outlier_correlation` | linreg | outlier_clipper | correlation |
| `knn_outlier_variance` | knn | outlier_clipper | variance |
| `linreg_minmax_only` | linreg | minmax_scaler | — |
| `knn_robust_only` | knn | robust_scaler | — |

### Špeciálne

| Preset | Popis |
|--------|-------|
| `minimal` | Žiadny preprocessing ani selekcia |

---

## 18. Používateľská príručka

### Krok 1: Vytvorenie Pipeline

1. Vyberte **model** z dropdown menu (logistická regresia, lineárna regresia, KNN, Decision Tree)
2. Nastavte **evaluation mode** (Auto / Classification / Regression)
3. Upravte **parametre modelu** (napr. k pre KNN, max_depth pre Tree)
4. Kliknite **„Vytvoriť Pipeline"**

### Krok 2: Načítanie dát

1. Vyberte **formát** (CSV alebo JSON)
2. Vložte dáta do textového poľa alebo nahrajte súbor
   - Môžete použiť **„Stiahnuť ukážkový CSV"** pre rýchly test
3. Kliknite **„Načítať dáta"**
4. Z dropdown menu vyberte **cieľovú premennú** (target)
5. Upravte **pomer train/test** sliderom (predvolené 80/20)
6. Kliknite **„Potvrdiť target"**

### Krok 3: Analýza targetu (voliteľné)

1. V sekcii **Target Analysis** vyberte analyzéry (korelácia, MI, SMC, entropia)
2. Kliknite **„Porovnať"** pre porovnanie metód
3. Kliknite na kartu analyzéra pre detailný rozklad

### Krok 4: Selekcia príznakov

1. V sekcii **Prieskum a výber príznakov** zaškrtnite požadované selektory
2. Nastavte parametre (počet features, prahy, váhy α/β/γ pre VNS/SA)
3. Kliknite **„Porovnať vybrané selektory"**
4. Prezrite si výsledky — interaktívna mapa príznakov s checkboxami
5. Kliknite **„Heatmap"** pre korelačnú/MI maticu

### Krok 5: Tréning a evaluácia

1. Po porovnaní selektorov kliknite **„Trénovať všetky"**
2. Aplikácia natrénuje model s features od každého selektora
3. Porovnajte výsledky — metriky pre každú variantu selekcie

### Krok 6: Editor dát (voliteľné)

1. Kliknite **„Editor"** pre otvorenie data editora
2. Vyberte stĺpec a aplikujte processor (encoding, scaling, transformácie)
3. Zmeny sa okamžite premietnu do dát

---

*Dokumentácia generovaná pre ML Pipeline WASM Application v0.1.0*
