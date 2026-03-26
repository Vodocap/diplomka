# Build and Run WASM Application

Write-Host "Building WASM module..." -ForegroundColor Green

# Set PATH
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
$env:Path += ";$env:USERPROFILE\.cargo\bin"

# Build WASM (SIMD128 pre wide crate zrychlenie)
$env:RUSTFLAGS = "-C target-feature=+simd128"
wasm-pack build --target web --out-dir pkg

if ($LASTEXITCODE -eq 0) {
    Write-Host "Build successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Starting server..." -ForegroundColor Cyan
    
    # Start server
    .\serve.ps1
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}
