# Simple HTTP Server script for WASM development
# Runs Python HTTP server on port 8000

Write-Host "🚀 Starting HTTP Server for WASM App..." -ForegroundColor Green
Write-Host "📂 Serving from: $PWD" -ForegroundColor Cyan
Write-Host "🌐 Open browser at: http://localhost:8000" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

try {
    python -m http.server 8000
} catch {
    Write-Host "❌ Error: Python not found. Please install Python first." -ForegroundColor Red
    Write-Host "Or use: npm install -g http-server && http-server -p 8000" -ForegroundColor Yellow
}
