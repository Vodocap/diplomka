#!/bin/bash
# Run Playwright tests in headless mode with HTML report
#
# Usage:
#   ./run-comprehensive-tests.sh                                     - Všetky testy (test_08 + test_09)
#   ./run-comprehensive-tests.sh tests/test_09_aggressive.py         - Len aggressive testy
#   ./run-comprehensive-tests.sh tests/test_01_init.py               - Len init testy
#   ./run-comprehensive-tests.sh tests/test_09_aggressive.py::TestWrongOrder  - Jedna trieda
#   ./run-comprehensive-tests.sh tests/test_09_aggressive.py::TestWrongOrder::test_empty_string  - Jeden test
#   ./run-comprehensive-tests.sh -k "editor"                         - Podľa názvu (pytest -k filter)
#   ./run-comprehensive-tests.sh --no-build                          - Preskočiť WASM build
#   ./run-comprehensive-tests.sh --headed                            - S oknom browsera
#
# Kombinácie:
#   ./run-comprehensive-tests.sh tests/test_09_aggressive.py -k "heatmap" --headed
#   ./run-comprehensive-tests.sh --no-build tests/test_01_init.py tests/test_02_data_loading.py

set -e
cd "$(dirname "$0")"

# ── Parse arguments ──────────────────────────────────────────────
DO_BUILD=true
HEADED=""
TEST_TARGETS=()
PYTEST_EXTRA=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build)
            DO_BUILD=false
            shift
            ;;
        --headed)
            HEADED="--headed"
            shift
            ;;
        -k|-m|--timeout|--tb|--html)
            # pytest flags that take a value
            PYTEST_EXTRA+=("$1" "$2")
            shift 2
            ;;
        -v|--verbose|-x|--exitfirst|-s|--capture=no)
            PYTEST_EXTRA+=("$1")
            shift
            ;;
        tests/*)
            TEST_TARGETS+=("$1")
            shift
            ;;
        *)
            PYTEST_EXTRA+=("$1")
            shift
            ;;
    esac
done

# Default: run test_08 + test_09 if no specific targets given
if [ ${#TEST_TARGETS[@]} -eq 0 ]; then
    TEST_TARGETS=("tests/test_08_all_models_comprehensive.py" "tests/test_09_aggressive.py")
fi

echo "══════════════════════════════════════════════════════════════"
echo "  ML Pipeline – Playwright Tests (Headless)"
echo "══════════════════════════════════════════════════════════════"
echo "  Targets: ${TEST_TARGETS[*]}"
[ ${#PYTEST_EXTRA[@]} -gt 0 ] && echo "  Extra:   ${PYTEST_EXTRA[*]}"
[ -n "$HEADED" ] && echo "  Mode:    HEADED (s oknom)"

# ── Step 1: Build WASM ───────────────────────────────────────────
if $DO_BUILD; then
    echo ""
    echo "🔨 Building WASM module..."
    export PATH=$PATH:$HOME/.cargo/bin
    wasm-pack build --target web --out-dir pkg
    if [ $? -ne 0 ]; then
        echo "❌ WASM build failed!"
        exit 1
    fi
    echo "✅ WASM build successful."
else
    echo ""
    echo "⏭  Preskakujem WASM build (--no-build)"
fi

# ── Step 2: Start server if needed ───────────────────────────────
if ! curl -s http://localhost:3333 > /dev/null 2>&1; then
    echo ""
    echo "🌐 Spúšťam server na pozadí..."
    bash serve.sh &
    SERVER_PID=$!
    sleep 3
    echo "   Server PID: $SERVER_PID"
    trap "kill $SERVER_PID 2>/dev/null; echo '   Server zastavený'" EXIT
else
    echo "✅ Server už beží na porte 3333"
fi

# ── Step 3: Run tests ────────────────────────────────────────────
mkdir -p tests/reports tests/output

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
REPORT_FILE="tests/reports/test_report_${TIMESTAMP}.html"

echo ""
echo "🧪 Spúšťam testy..."
echo "   HTML report: ${REPORT_FILE}"
echo ""

python3 -m pytest "${TEST_TARGETS[@]}" \
    --browser chromium \
    $HEADED \
    -v \
    --timeout 180 \
    --html="${REPORT_FILE}" --self-contained-html \
    --tb=short \
    "${PYTEST_EXTRA[@]}" 2>&1 | tee tests/output/test_output.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "══════════════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  ✅ VŠETKY TESTY PREŠLI"
else
    echo "  ❌ NIEKTORÉ TESTY ZLYHALI (exit code: $EXIT_CODE)"
fi
echo "══════════════════════════════════════════════════════════════"
echo ""
echo "Detailný log: tests/test_report.log"
echo "HTML report:  ${REPORT_FILE}"
echo "Konzolový výstup: tests/output/test_output.log"
echo ""

exit $EXIT_CODE
