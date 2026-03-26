#!/bin/bash

echo "Building WASM module..."

export PATH=$PATH:$HOME/.cargo/bin

# SIMD128 pre wide crate zrychlenie
export RUSTFLAGS="-C target-feature=+simd128"
wasm-pack build --target web --out-dir pkg

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo ""
    echo "Starting server..."
    ./serve.sh
else
    echo "Build failed!"
    exit 1
fi