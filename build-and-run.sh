#!/bin/bash

echo "Building WASM module..."

export PATH=$PATH:$HOME/.cargo/bin

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