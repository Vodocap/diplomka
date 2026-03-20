// Utilities: helpers, status, loaders, color functions

function convertWasmResult(result) {
    if (result instanceof Map) {
        const obj = {};
        result.forEach((val, key) => {
            obj[key] = convertWasmResult(val);
        });
        return obj;
    }
    if (Array.isArray(result)) {
        return result.map(item => convertWasmResult(item));
    }
    return result;
}

function showStatus(type, message, elementId) {
    const statusDiv = document.getElementById(elementId);
    statusDiv.className = `status ${type}`;
    statusDiv.textContent = message;
}

function showLoading(show) {
    document.getElementById('loadingOverlay').className = show ? 'show' : '';
}

// Data Inspection Functions
function heatmapColor(value, mode) {
    // Correlation: -1..1 → red (negative) → white (0) → blue (positive)
    // MI: 0..max → white (0) → purple (max)
    if (mode === 'correlation') {
        const v = Math.max(-1, Math.min(1, value));
        const absV = Math.abs(v);
        if (v >= 0) {
            // White → Blue
            const r = Math.round(255 * (1 - absV));
            const g = Math.round(255 * (1 - absV * 0.6));
            const b = 255;
            return `rgb(${r},${g},${b})`;
        } else {
            // White → Red
            const r = 255;
            const g = Math.round(255 * (1 - absV * 0.7));
            const b = Math.round(255 * (1 - absV * 0.8));
            return `rgb(${r},${g},${b})`;
        }
    } else {
        // MI: 0 → white, max → deep purple
        const v = Math.max(0, Math.min(1, value));
        const r = Math.round(255 - v * 175);
        const g = Math.round(255 - v * 215);
        const b = Math.round(255 - v * 75);
        return `rgb(${r},${g},${b})`;
    }
}

function textColor(value, mode) {
    if (mode === 'correlation') {
        return Math.abs(value) > 0.6 ? '#fff' : '#333';
    } else {
        return value > 0.6 ? '#fff' : '#333';
    }
}

