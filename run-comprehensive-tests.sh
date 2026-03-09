#!/bin/bash
# Run comprehensive model tests in headless mode with HTML report
# Usage:
#   ./run-comprehensive-tests.sh           - Run all comprehensive tests
#   ./run-comprehensive-tests.sh -k "e2e"  - Run only E2E tests
#   ./run-comprehensive-tests.sh -k "editor" - Run only editor tests

set -e
cd "$(dirname "$0")"

echo "══════════════════════════════════════════════════════════════"
echo "  ML Pipeline – Comprehensive Model Tests (Headless)"
echo "══════════════════════════════════════════════════════════════"

# Step 1: Build WASM module
echo ""
echo "🔨 Building WASM module..."
export PATH=$PATH:$HOME/.cargo/bin
wasm-pack build --target web --out-dir pkg
if [ $? -ne 0 ]; then
    echo "WASM build failed!"
    exit 1
fi
echo "WASM build successful."

# Step 2: Start server if not already running
if ! curl -s http://localhost:3333 > /dev/null 2>&1; then
    echo ""
    echo "🌐 Spúšťam server na pozadí..."
    bash serve.sh &
    SERVER_PID=$!
    sleep 3
    echo "   Server PID: $SERVER_PID"
    trap "kill $SERVER_PID 2>/dev/null; echo '   Server zastavený'" EXIT
else
    echo "Server už beží na porte 3333"
fi

# Create report directory
mkdir -p tests/reports

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="tests/reports/test_report_${TIMESTAMP}.html"

echo ""
echo "🧪 Spúšťam testy v headless mode..."
echo "   Log: tests/output/test_report.log"
echo "   HTML report: ${REPORT_FILE}"
echo ""

# Run only the comprehensive test file, headless, with HTML report
python3 -m pytest tests/test_08_all_models_comprehensive.py \
    --browser chromium \
    -v \
    --timeout 180 \
    --html="${REPORT_FILE}" --self-contained-html \
    --tb=short \
    "$@" 2>&1 | tee tests/output/test_output.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "══════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  VŠETKY TESTY PREŠLI"
else
    echo "  NIEKTORÉ TESTY ZLYHALI (exit code: $EXIT_CODE)"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Detailný log: tests/test_report.log"
echo "HTML report:  ${REPORT_FILE}"
echo "Konzolový výstup: tests/output/test_output.log"
echo ""

exit $EXIT_CODE
