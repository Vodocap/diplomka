#!/bin/bash

echo "Starting HTTP Server for WASM App..."
echo "Serving from: $PWD"
echo "Open browser at: http://localhost:3333"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Use custom handler to disable caching and set correct MIME types
python3 -c "
import http.server
import socketserver

class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        '.wasm': 'application/wasm',
        '.js': 'application/javascript',
    }
    
    def end_headers(self):
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(('', 3333), NoCacheHandler) as httpd:
    httpd.serve_forever()
" || {
    echo "Error: Python not found. Please install Python first."
    echo "Or use: npm install -g http-server && http-server -p 3000"
}