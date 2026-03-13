// Shared app state – loaded first
// WASM instances
let factory = null, pipeline = null, dataLoader = null;
let availableOptions = null;
let WasmDataLoaderClass = null;

// Heatmap state (shared between heatmap.js and selectors.js)
let _heatmapData = null;
let _heatmapMode = 'correlation';
let _heatmapShowValues = false;
let _heatmapSorted = false;
let _heatmapThreshold = 0.0;
let _heatmapMaxVal = 1.0;
