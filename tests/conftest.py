"""
Playwright test configuration for ML Pipeline WASM application.
Starts the local server automatically before tests and stops it after.
"""
import pytest
import subprocess
import time
import socket
import signal
import os

BASE_URL = "http://localhost:3333"
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Sample CSV data used in tests (same as downloadSampleCSV in index.html)
SAMPLE_CSV = """age,income,credit_score,debt_ratio,employment_years,education_level,num_accounts,account_balance,loan_amount,monthly_payment,savings,investment_score,property_value,approved
25,50000,700,0.35,3,2,4,12500,15000,450,8000,65,0,1
35,75000,750,0.28,8,3,6,28000,22000,680,25000,78,180000,1
45,60000,650,0.42,15,2,3,8500,28000,820,15000,58,150000,0
22,30000,600,0.48,1,1,2,3500,12000,380,2500,42,0,0
50,90000,800,0.22,20,4,8,45000,35000,920,55000,88,320000,1
30,55000,680,0.38,5,3,5,15000,18000,520,12000,68,0,1
40,80000,720,0.30,12,3,7,32000,25000,750,30000,75,220000,1
28,45000,640,0.44,4,2,3,7500,20000,610,6000,55,0,0
55,95000,790,0.20,25,4,9,58000,40000,980,68000,92,450000,1
32,62000,710,0.33,7,3,6,18500,21000,640,16000,72,125000,1
38,72000,730,0.29,10,3,7,26000,24000,710,28000,76,195000,1
27,48000,660,0.40,3,2,4,9500,17000,550,9500,60,0,0
52,88000,780,0.24,22,4,8,42000,32000,890,52000,86,380000,1
29,51000,670,0.37,4,2,5,11500,19000,570,11000,63,0,1
43,68000,700,0.32,13,3,6,21000,23000,690,22000,74,175000,1
24,42000,630,0.45,2,2,3,6000,14000,480,5000,50,0,0
48,82000,760,0.26,18,3,7,35000,28000,820,38000,82,280000,1
31,58000,690,0.36,6,3,5,14000,20000,600,13500,66,110000,1
36,70000,725,0.31,9,3,6,24000,26000,730,26000,77,200000,1
26,46000,655,0.41,3,2,4,8000,16000,520,7500,58,0,0
53,92000,795,0.21,24,4,9,50000,38000,950,62000,90,410000,1
33,64000,705,0.34,8,3,6,19500,22000,660,18000,70,140000,1
41,76000,740,0.27,14,3,7,30000,27000,780,32000,80,240000,1
23,38000,620,0.47,1,1,2,4500,13000,420,3500,45,0,0
49,85000,770,0.25,19,4,8,40000,33000,870,48000,84,350000,1
34,66000,715,0.30,9,3,6,22000,24000,700,24000,73,160000,1
39,74000,735,0.28,11,3,7,28000,26000,760,29000,78,210000,1
28,49000,665,0.39,4,2,4,10000,18000,560,10500,62,0,0
54,94000,785,0.23,23,4,9,55000,36000,930,65000,89,430000,1
30,56000,685,0.35,5,3,5,13500,19000,590,14500,67,95000,1""".strip()


def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0


@pytest.fixture(scope="session")
def server():
    """Start the HTTP server for the WASM app if not already running."""
    if is_port_in_use(3333):
        # Server already running externally
        yield BASE_URL
        return

    # Start the server
    proc = subprocess.Popen(
        ["bash", "serve.sh"],
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
    )

    # Wait for server to be ready
    for _ in range(30):
        if is_port_in_use(3333):
            break
        time.sleep(0.5)
    else:
        proc.kill()
        raise RuntimeError("Server failed to start on port 3333")

    yield BASE_URL

    # Cleanup
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def browser_context_args():
    """Extra browser context args."""
    return {
        "viewport": {"width": 1400, "height": 900},
    }


@pytest.fixture
def app_page(page, server):
    """Navigate to the app page and wait for WASM to initialize."""
    page.goto(server, wait_until="networkidle")
    # Wait for WASM initialization - modelsInfo should be populated
    page.wait_for_function(
        "document.getElementById('modelsInfo') && document.getElementById('modelsInfo').children.length > 0",
        timeout=30000,
    )
    return page


@pytest.fixture
def loaded_page(app_page):
    """Page with pipeline built, sample data parsed and target selected."""
    page = app_page

    # Build pipeline first
    page.select_option("#modelSelect", "logreg")
    page.click("#buildPipelineBtn")
    page.wait_for_selector("#currentPipelineInfo", state="visible", timeout=15000)

    # Fill data
    page.fill("#dataInput", SAMPLE_CSV)
    page.click("#parseDataBtn")

    # Wait for target selection area
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    # Select target
    page.select_option("#targetColumnSelect", "approved")
    page.click("#confirmTargetBtn")

    # Wait for data to be loaded
    page.wait_for_function(
        "document.getElementById('dataStatus').textContent.length > 0",
        timeout=15000,
    )

    return page
