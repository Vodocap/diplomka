#!/bin/bash
# Run all Playwright tests for the Aplikácia na podporu rozhodavnia pri tréningu predikčných modelov app
# Usage:
#   ./run-tests.sh              - Run all tests (headed - vidíš browser)
#   ./run-tests.sh --headless   - Run headless (bez okna)

set -e
cd "$(dirname "$0")"

EXTRA_ARGS=""

# Parse arguments
for arg in "$@"; do
    case $arg in
        --headless)
            EXTRA_ARGS="$EXTRA_ARGS --browser chromium"
            # Override headed mode
            export HEADED=""
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $arg"
            ;;
    esac
done

# Check if server is running
if ! curl -s http://localhost:3333 > /dev/null 2>&1; then
    echo " Server nie je spustený na porte 3333"
    echo " Spúšťam server na pozadí..."
    bash serve.sh &
    SERVER_PID=$!
    sleep 2
    echo "   Server PID: $SERVER_PID"
    trap "kill $SERVER_PID 2>/dev/null; echo '   Server zastavený'" EXIT
fi

echo " Spúšťam Playwright testy..."
echo ""

if [ -n "$HEADED" ] || [[ ! " $@ " =~ " --headless " ]]; then
    python3 -m pytest --headed --browser chromium $EXTRA_ARGS
else
    python3 -m pytest --browser chromium --timeout 120 $EXTRA_ARGS
fi
