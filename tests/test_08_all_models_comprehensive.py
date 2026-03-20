"""
Comprehensive Playwright tests for all model types with real cycling data.

Tests all 4 model types (linreg, logreg, knn, tree) in both regression and
classification modes, including data editor transformations, feature selectors,
target analysis, heatmaps, and training with metric validation.

For classification models, data is first transformed via the editor (binner)
to create a categorical target column.

Runs headless. Outputs log + test report.
"""
import pytest
import os
import time
import json
import logging
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
# Logging setup – writes detailed log to tests/test_report.log
# ──────────────────────────────────────────────────────────────────
LOG_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(LOG_DIR, "test_report.log")

logger = logging.getLogger("ml_pipeline_tests")
logger.setLevel(logging.DEBUG)
# Clear old handlers
if logger.handlers:
    logger.handlers.clear()
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(fh)
# Also console
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.addHandler(ch)

# ──────────────────────────────────────────────────────────────────
# Data helpers
# ──────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DATA_PATH = os.path.join(PROJECT_DIR, "test_data", "upravene_data(1).csv")

def load_test_csv():
    """Load the cycling dataset CSV."""
    with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()

TEST_CSV = load_test_csv()

# Regression targets (continuous numeric columns in the cycling data)
REGRESSION_TARGETS = [
    "Kalórie",               # Calories burned – good for regression
    "Priemerná rýchlosť",    # Average speed
]

# Column to bin for classification target
CLASSIFICATION_BIN_TARGET = "Aeróbny TE"  # Aerobic Training Effect (small discrete set 0.4–5.0)

# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clean_page(page, server):
    """Fresh app page with WASM initialized, collecting all JS errors."""
    js_errors = []
    page.on("console", lambda msg: js_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: js_errors.append(str(exc)))

    page.goto(server, wait_until="networkidle")
    page.wait_for_function(
        "document.getElementById('modelsInfo') && document.getElementById('modelsInfo').children.length > 0",
        timeout=30000,
    )
    page._js_errors = js_errors
    logger.info("Page loaded, WASM initialized.")
    return page


def _build_pipeline(page, model: str, eval_mode: str = "", knn_k: int = 5, tree_depth: int = 10):
    """Build pipeline with given model and eval mode."""
    page.select_option("#modelSelect", model)
    logger.info(f"Selected model: {model}")

    # Set eval mode if provided
    if eval_mode:
        page.select_option("#evalModeSelect", eval_mode)
        logger.info(f"Eval mode set to: {eval_mode}")

    # Model-specific params
    if model == "knn":
        page.wait_for_selector("#knnKGroup", state="visible", timeout=3000)
        page.fill("#knnK", str(knn_k))
        logger.info(f"KNN K = {knn_k}")
    elif model == "tree":
        page.wait_for_selector("#treeMaxDepthGroup", state="visible", timeout=3000)
        page.fill("#treeMaxDepth", str(tree_depth))
        logger.info(f"Tree max_depth = {tree_depth}")

    page.click("#buildPipelineBtn")
    page.wait_for_selector("#currentPipelineInfo", state="visible", timeout=15000)

    status = page.inner_text("#pipelineStatus")
    logger.info(f"Pipeline build status: {status}")
    assert "error" not in status.lower() and "chyba" not in status.lower(), f"Pipeline build error: {status}"
    return status


def _load_data_and_select_target(page, csv_data: str, target_col: str, split_pct: int = 80):
    """Load CSV data and confirm target column."""
    page.fill("#dataInput", csv_data)
    page.click("#parseDataBtn")
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)
    logger.info("Data parsed, target selection visible.")

    # Set split
    page.evaluate(f"document.getElementById('trainTestSplit').value = {split_pct}")
    page.evaluate("document.getElementById('trainTestSplit').dispatchEvent(new Event('input'))")
    split_txt = page.inner_text("#splitValue")
    logger.info(f"Train/test split: {split_txt}")

    # Select target
    page.select_option("#targetColumnSelect", target_col)
    page.click("#confirmTargetBtn")
    page.wait_for_function(
        "document.getElementById('dataStatus').textContent.length > 0",
        timeout=15000,
    )
    status = page.inner_text("#dataStatus")
    logger.info(f"Target confirm status: {status}")
    assert "chyba" not in status.lower() and "error" not in status.lower(), f"Confirm target error: {status}"
    return status


def _prepare_classification_data_via_editor(page, target_col: str):
    """
    Open the data editor, apply binner to the target column
    to create categorical data suitable for classification.
    Then close editor and re-parse.
    """
    logger.info(f"Opening editor to bin column '{target_col}' for classification...")
    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)
    page.wait_for_selector("#editorTableContainer table", timeout=10000)
    logger.info("Editor opened with table.")

    # Click the target column header to select it
    col_headers = page.locator("#editorTableContainer table .col-header").all()
    target_header = None
    for hdr in col_headers:
        if target_col in hdr.inner_text():
            target_header = hdr
            break

    assert target_header is not None, f"Column header '{target_col}' not found in editor"
    target_header.click()
    page.wait_for_timeout(500)
    assert page.is_visible("#editorProcessorPanel"), "Processor panel not shown"
    logger.info(f"Selected column '{target_col}' in editor.")

    # Select binner processor
    page.select_option("#editorProcessorSelect", "binner")
    page.wait_for_timeout(300)
    logger.info("Selected 'binner' processor.")

    # Set binner params if available (num_bins)
    bins_input = page.locator("#editor_param_num_bins")
    if bins_input.count() > 0:
        bins_input.fill("3")
        logger.info("Set num_bins = 3")

    # Apply processor
    page.click("#editorApplyProcessorBtn")
    page.wait_for_timeout(1000)

    editor_status = page.inner_text("#editorStatusText")
    logger.info(f"Editor status after binner: {editor_status}")
    assert "chyba" not in editor_status.lower() and "error" not in editor_status.lower(), \
        f"Binner apply error: {editor_status}"

    # Apply changes and close
    page.click("button:has-text('Použiť zmeny')")
    page.wait_for_timeout(1500)
    logger.info("Editor closed, changes applied.")


def _compare_selectors(page, selector_names: list, timeout: int = 90000):
    """Select and compare given selectors. Returns comparison status text."""
    # First deselect all
    page.evaluate("""
        document.querySelectorAll('#selectorCompareGrid .selector-compare-card.active input[type=checkbox]')
            .forEach(cb => { cb.checked = false; cb.closest('.selector-compare-card').classList.remove('active'); });
    """)

    for sel in selector_names:
        card = page.locator(f"#compare_card_{sel}")
        if card.count() > 0:
            cb = card.locator("input[type='checkbox']")
            cb.check()
            # Also toggle active class
            page.evaluate(f"document.getElementById('compare_card_{sel}').classList.add('active')")
            logger.info(f"Selected selector: {sel}")

    page.click("#compareSelectorBtn")
    # Wait for modal with results
    page.wait_for_selector("#dataModal.show", timeout=timeout)
    page.wait_for_timeout(500)

    modal_text = page.inner_text("#modalBody")
    logger.info(f"Selector comparison modal opened, content length: {len(modal_text)}")
    assert len(modal_text) > 50, "Comparison modal too short"
    return modal_text


def _train_from_comparison(page, timeout: int = 120000):
    """Click train button inside comparison modal and wait for results."""
    train_btn = page.locator("#trainFromComparisonBtn")
    if train_btn.count() > 0:
        train_btn.click()
        logger.info("Clicked 'Train from comparison' button inside modal.")
    else:
        # Close modal, try trainAllSelectors
        page.click(".modal-close")
        page.wait_for_timeout(500)
        page.click("#trainAllSelectorsBtn")
        logger.info("Clicked 'Train all selectors' button.")

    # Wait for training results
    page.wait_for_function(
        """(() => {
            const el = document.getElementById('comparisonTrainingResults') || document.getElementById('selectorTrainingResults');
            return el && (el.innerHTML.includes('ACC') || el.innerHTML.includes('R²') || el.innerHTML.includes('RMSE') || el.innerHTML.includes('Accuracy'));
        })()""",
        timeout=timeout,
    )
    page.wait_for_timeout(500)
    logger.info("Training results rendered.")


def _get_training_metrics(page):
    """Extract training metrics from the results table."""
    results_el = page.locator("#comparisonTrainingResults")
    if results_el.count() == 0:
        results_el = page.locator("#selectorTrainingResults")

    html = results_el.inner_html()
    text = results_el.inner_text()
    logger.info(f"Training results text (first 500 chars): {text[:500]}")
    return text, html


def _check_no_js_errors(page):
    """Assert no critical JS errors occurred."""
    errors = getattr(page, '_js_errors', [])
    critical = [e for e in errors if "favicon" not in e.lower() and "404" not in e]
    if critical:
        logger.warning(f"JS errors found: {critical}")
    # Don't assert hard – log them. Some internal WASM warnings are non-fatal.
    return critical


# ══════════════════════════════════════════════════════════════════
# TEST: Linear Regression (regression on real data)
# ══════════════════════════════════════════════════════════════════
class TestLinRegRegression:
    """Linear Regression with cycling data – natural regression target."""

    def test_linreg_full_pipeline(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: LinReg full pipeline (regression)")
        logger.info("=" * 60)

        # 1) Build pipeline
        _build_pipeline(page, "linreg", eval_mode="Regression")

        # 2) Load data with regression target
        target = "Kalórie"
        _load_data_and_select_target(page, TEST_CSV, target)

        # 3) Compare fast selectors
        _compare_selectors(page, ["variance", "smc"])

        # 4) Train
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)

        # 5) Validate regression metrics
        assert "R²" in html or "r2" in html.lower(), "Missing R² metric for regression"
        assert "RMSE" in html, "Missing RMSE metric"
        assert "MAE" in html, "Missing MAE metric"
        logger.info("LinReg regression metrics validated.")

        # 6) No JS errors
        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors: {errors}"
        logger.info("TEST PASSED: LinReg full pipeline")

    def test_linreg_with_all_selectors(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: LinReg with all fast selectors")
        logger.info("=" * 60)

        _build_pipeline(page, "linreg", eval_mode="Regression")
        _load_data_and_select_target(page, TEST_CSV, "Priemerná rýchlosť")

        # Select all selectors
        page.click("#selectAllSelectorsBtn")
        active = page.locator("#selectorCompareGrid .selector-compare-card.active").count()
        total = page.locator("#selectorCompareGrid .selector-compare-card").count()
        logger.info(f"All selectors toggled: {active}/{total}")
        assert active == total, f"Not all selectors active: {active}/{total}"

        page.click("#compareSelectorBtn")
        page.wait_for_selector("#dataModal.show", timeout=120000)
        page.wait_for_timeout(500)

        _train_from_comparison(page)
        text, html = _get_training_metrics(page)
        assert "R²" in html or "r2" in html.lower(), "Missing R² in results"
        logger.info("TEST PASSED: LinReg all selectors")


# ══════════════════════════════════════════════════════════════════
# TEST: Logistic Regression (classification – needs binned target)
# ══════════════════════════════════════════════════════════════════
class TestLogRegClassification:
    """Logistic Regression – classification on binned cycling data."""

    def test_logreg_with_binned_target(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: LogReg classification with binned target")
        logger.info("=" * 60)

        # 1) Build classification pipeline
        _build_pipeline(page, "logreg", eval_mode="Classification")

        # 2) Parse data first (needed for editor)
        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)
        logger.info("Data parsed for logreg.")

        # 3) Open editor and bin the target column for classification
        _prepare_classification_data_via_editor(page, CLASSIFICATION_BIN_TARGET)

        # 4) Now select target and confirm
        page.select_option("#targetColumnSelect", CLASSIFICATION_BIN_TARGET)
        page.click("#confirmTargetBtn")
        page.wait_for_function(
            "document.getElementById('dataStatus').textContent.length > 0",
            timeout=15000,
        )
        status = page.inner_text("#dataStatus")
        logger.info(f"Target confirm (logreg): {status}")

        # 5) Compare selectors
        _compare_selectors(page, ["variance", "smc"])

        # 6) Train
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)

        # 7) Validate classification metrics
        assert "ACC" in html or "Accuracy" in html or "accuracy" in text.lower(), \
            "Missing accuracy/ACC metric for classification"
        logger.info("LogReg classification metrics validated.")

        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors: {errors}"
        logger.info("TEST PASSED: LogReg classification with binned target")


# ══════════════════════════════════════════════════════════════════
# TEST: KNN (both regression and classification)
# ══════════════════════════════════════════════════════════════════
class TestKNN:
    """KNN model – test with different K values in regression and classification."""

    def test_knn_regression(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: KNN regression (K=3)")
        logger.info("=" * 60)

        _build_pipeline(page, "knn", eval_mode="Regression", knn_k=3)
        _load_data_and_select_target(page, TEST_CSV, "Kalórie")

        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)

        assert "R²" in html or "r2" in html.lower(), "Missing R² for KNN regression"
        assert "RMSE" in html, "Missing RMSE for KNN regression"
        logger.info("KNN regression metrics validated.")

        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors: {errors}"
        logger.info("TEST PASSED: KNN regression")

    def test_knn_regression_k7(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: KNN regression (K=7)")
        logger.info("=" * 60)

        _build_pipeline(page, "knn", eval_mode="Regression", knn_k=7)
        _load_data_and_select_target(page, TEST_CSV, "Priemerná rýchlosť")

        _compare_selectors(page, ["mutual_information", "variance"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)

        assert "R²" in html or "r2" in html.lower(), "Missing R² for KNN regression K=7"
        logger.info("TEST PASSED: KNN regression K=7")

    def test_knn_classification(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: KNN classification with binned target")
        logger.info("=" * 60)

        _build_pipeline(page, "knn", eval_mode="Classification", knn_k=5)

        # Parse data
        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        # Bin the target for classification
        _prepare_classification_data_via_editor(page, CLASSIFICATION_BIN_TARGET)

        # Select binned target and confirm
        page.select_option("#targetColumnSelect", CLASSIFICATION_BIN_TARGET)
        page.click("#confirmTargetBtn")
        page.wait_for_function(
            "document.getElementById('dataStatus').textContent.length > 0",
            timeout=15000,
        )

        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)

        assert "ACC" in html or "Accuracy" in html or "accuracy" in text.lower(), \
            "Missing accuracy for KNN classification"
        logger.info("TEST PASSED: KNN classification with binned target")


# ══════════════════════════════════════════════════════════════════
# TEST: Decision Tree (both regression and classification)
# ══════════════════════════════════════════════════════════════════
class TestDecisionTree:
    """Decision Tree – test classification and regression with different depths."""

    def test_tree_regression(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: Decision Tree regression (depth=5)")
        logger.info("=" * 60)

        _build_pipeline(page, "tree", eval_mode="Regression", tree_depth=5)
        _load_data_and_select_target(page, TEST_CSV, "Kalórie")

        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)

        assert "R²" in html or "r2" in html.lower(), "Missing R² for Tree regression"
        assert "RMSE" in html, "Missing RMSE for Tree regression"
        logger.info("TEST PASSED: Tree regression depth=5")

    def test_tree_regression_deep(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: Decision Tree regression (depth=15)")
        logger.info("=" * 60)

        _build_pipeline(page, "tree", eval_mode="Regression", tree_depth=15)
        _load_data_and_select_target(page, TEST_CSV, "Priemerná rýchlosť")

        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)

        assert "R²" in html or "r2" in html.lower(), "Missing R² for Tree regression depth=15"
        logger.info("TEST PASSED: Tree regression depth=15")

    def test_tree_classification(self, clean_page):
        page = clean_page
        logger.info("=" * 60)
        logger.info("TEST: Decision Tree classification with binned target")
        logger.info("=" * 60)

        _build_pipeline(page, "tree", eval_mode="Classification", tree_depth=8)

        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        _prepare_classification_data_via_editor(page, CLASSIFICATION_BIN_TARGET)

        page.select_option("#targetColumnSelect", CLASSIFICATION_BIN_TARGET)
        page.click("#confirmTargetBtn")
        page.wait_for_function(
            "document.getElementById('dataStatus').textContent.length > 0",
            timeout=15000,
        )

        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)

        assert "ACC" in html or "Accuracy" in html or "accuracy" in text.lower(), \
            "Missing accuracy for Tree classification"
        logger.info("TEST PASSED: Tree classification with binned target")


# ══════════════════════════════════════════════════════════════════
# TEST: Data Editor – all processor types
# ══════════════════════════════════════════════════════════════════
class TestDataEditor:
    """Test the data editor with various processors."""

    def _open_editor_with_data(self, page):
        """Load data and open editor."""
        _build_pipeline(page, "linreg")
        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)
        page.click("#editDataBtn")
        page.wait_for_selector("#dataEditorModal.show", timeout=15000)
        page.wait_for_selector("#editorTableContainer table", timeout=10000)

    def _select_column_in_editor(self, page, col_name: str):
        """Click on a column header in the editor."""
        col_headers = page.locator("#editorTableContainer table .col-header").all()
        for hdr in col_headers:
            if col_name in hdr.inner_text():
                hdr.click()
                page.wait_for_timeout(500)
                return True
        return False

    def test_editor_scaler(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – scaler processor")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, "Kalórie")

        page.select_option("#editorProcessorSelect", "scaler")
        page.wait_for_timeout(300)
        page.click("#editorApplyProcessorBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#editorStatusText")
        logger.info(f"Scaler applied: {status}")
        assert "chyba" not in status.lower(), f"Scaler error: {status}"
        assert "aplikovaný" in status.lower() or "applied" in status.lower() or "riadkov" in status.lower()
        logger.info("TEST PASSED: Editor scaler")

    def test_editor_minmax_scaler(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – minmax_scaler")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, "Vzdialenosť")

        page.select_option("#editorProcessorSelect", "minmax_scaler")
        page.wait_for_timeout(300)
        page.click("#editorApplyProcessorBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#editorStatusText")
        logger.info(f"MinMax scaler applied: {status}")
        assert "chyba" not in status.lower(), f"MinMax scaler error: {status}"
        logger.info("TEST PASSED: Editor minmax_scaler")

    def test_editor_robust_scaler(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – robust_scaler")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, "Priemerný tep")

        page.select_option("#editorProcessorSelect", "robust_scaler")
        page.wait_for_timeout(300)
        page.click("#editorApplyProcessorBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#editorStatusText")
        logger.info(f"Robust scaler applied: {status}")
        assert "chyba" not in status.lower(), f"Robust scaler error: {status}"
        logger.info("TEST PASSED: Editor robust_scaler")

    def test_editor_binner(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – binner")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, CLASSIFICATION_BIN_TARGET)

        page.select_option("#editorProcessorSelect", "binner")
        page.wait_for_timeout(300)
        bins_input = page.locator("#editor_param_num_bins")
        if bins_input.count() > 0:
            bins_input.fill("4")
        page.click("#editorApplyProcessorBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#editorStatusText")
        logger.info(f"Binner applied: {status}")
        assert "chyba" not in status.lower(), f"Binner error: {status}"
        logger.info("TEST PASSED: Editor binner")

    def test_editor_log_transformer(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – log_transformer")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, "Kalórie")

        page.select_option("#editorProcessorSelect", "log_transformer")
        page.wait_for_timeout(300)
        page.click("#editorApplyProcessorBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#editorStatusText")
        logger.info(f"Log transformer applied: {status}")
        assert "chyba" not in status.lower(), f"Log transformer error: {status}"
        logger.info("TEST PASSED: Editor log_transformer")

    def test_editor_outlier_clipper(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – outlier_clipper")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, "Maximálna rýchlosť")

        page.select_option("#editorProcessorSelect", "outlier_clipper")
        page.wait_for_timeout(300)
        page.click("#editorApplyProcessorBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#editorStatusText")
        logger.info(f"Outlier clipper applied: {status}")
        assert "chyba" not in status.lower(), f"Outlier clipper error: {status}"
        logger.info("TEST PASSED: Editor outlier_clipper")

    def test_editor_null_handler(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – null_handler on column with missing values")
        self._open_editor_with_data(page)
        # Maximálny výkon has missing values in the test data
        assert self._select_column_in_editor(page, "Maximálny výkon")

        page.select_option("#editorProcessorSelect", "null_handler")
        page.wait_for_timeout(300)
        page.click("#editorApplyProcessorBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#editorStatusText")
        logger.info(f"Null handler applied: {status}")
        assert "chyba" not in status.lower(), f"Null handler error: {status}"
        logger.info("TEST PASSED: Editor null_handler")

    def test_editor_delete_column(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – delete column")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, "Minimálna teplota")

        # Handle confirm dialog
        page.on("dialog", lambda d: d.accept())
        page.click("#editorDeleteColumnBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#editorStatusText")
        logger.info(f"Delete column status: {status}")
        assert "vymazaný" in status.lower() or "deleted" in status.lower() or "zostáva" in status.lower(), \
            f"Delete column failed: {status}"
        logger.info("TEST PASSED: Editor delete column")

    def test_editor_find_replace(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – find & replace")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, "Aeróbny TE")

        # Find 3.0, replace with 3
        search_input = page.locator("#editorSearchValue")
        replace_input = page.locator("#editorReplaceValue")
        if search_input.count() > 0 and replace_input.count() > 0:
            search_input.fill("3.0")
            replace_input.fill("3")
            page.click("button:has-text('Nahradiť')")
            page.wait_for_timeout(1000)
            status = page.inner_text("#editorStatusText")
            logger.info(f"Find & replace status: {status}")
        logger.info("TEST PASSED: Editor find & replace")

    def test_editor_apply_and_close(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – apply changes and close")
        self._open_editor_with_data(page)
        assert self._select_column_in_editor(page, "Kalórie")

        page.select_option("#editorProcessorSelect", "scaler")
        page.wait_for_timeout(300)
        page.click("#editorApplyProcessorBtn")
        page.wait_for_timeout(1000)

        # Apply and close
        page.click("button:has-text('Použiť zmeny')")
        page.wait_for_timeout(1500)
        assert not page.is_visible("#dataEditorModal"), "Editor should have closed"

        # Verify parseStatus shows update
        parse_status = page.inner_text("#parseStatus")
        logger.info(f"Parse status after editor close: {parse_status}")
        assert "aktualizované" in parse_status.lower() or "stĺpcov" in parse_status.lower(), \
            f"Data not updated after editor close: {parse_status}"
        logger.info("TEST PASSED: Editor apply and close")

    def test_editor_download_data(self, clean_page):
        page = clean_page
        logger.info("TEST: Editor – download button exists")
        self._open_editor_with_data(page)
        download_btn = page.locator("button:has-text('Stiahnuť')")
        assert download_btn.count() > 0, "Download button not found"
        logger.info("TEST PASSED: Editor download button exists")


# ══════════════════════════════════════════════════════════════════
# TEST: Target Analysis
# ══════════════════════════════════════════════════════════════════
class TestTargetAnalysis:
    """Test target analysis functionality with real data."""

    def _setup_data(self, page):
        """Build pipeline and load data."""
        _build_pipeline(page, "linreg")
        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    def test_target_analyzers_exist(self, clean_page):
        page = clean_page
        logger.info("TEST: Target analyzers populated")
        self._setup_data(page)

        # Open analysis details/summary
        summary = page.locator("details summary")
        if summary.count() > 0:
            summary.first.click()
            page.wait_for_timeout(500)

        page.wait_for_selector("#targetAnalyzerGrid", state="visible", timeout=5000)
        cards = page.locator("#targetAnalyzerGrid .selector-compare-card").count()
        logger.info(f"Target analyzer cards: {cards}")
        assert cards >= 2, f"Expected at least 2 analyzer cards, got {cards}"
        logger.info("TEST PASSED: Target analyzers populated")

    def test_select_all_and_compare_analyzers(self, clean_page):
        page = clean_page
        logger.info("TEST: Compare all target analyzers")
        self._setup_data(page)

        summary = page.locator("details summary")
        if summary.count() > 0:
            summary.first.click()
            page.wait_for_timeout(500)

        page.wait_for_selector("#targetAnalyzerGrid", state="visible", timeout=5000)
        page.click("#selectAllAnalyzersBtn")

        page.click("#compareTargetAnalyzersBtn")
        page.wait_for_selector("#targetComparisonResultsArea", state="visible", timeout=120000)
        page.wait_for_timeout(1000)

        map_content = page.inner_text("#targetVariableMap")
        logger.info(f"Target variable map length: {len(map_content)}")
        assert len(map_content) > 0, "Target variable map is empty"
        logger.info("TEST PASSED: Compare all target analyzers")

    def test_analyzer_details_modal(self, clean_page):
        page = clean_page
        logger.info("TEST: Analyzer details modal")
        self._setup_data(page)

        summary = page.locator("details summary")
        if summary.count() > 0:
            summary.first.click()
            page.wait_for_timeout(500)

        page.wait_for_selector("#targetAnalyzerGrid", state="visible", timeout=5000)
        detail_btns = page.locator("#targetAnalyzerGrid button:has-text('detaily')")
        if detail_btns.count() > 0:
            detail_btns.first.click()
            page.wait_for_selector("#analyzerDetailsModal.show", timeout=30000)
            content = page.inner_text("#analyzerDetailsContent")
            logger.info(f"Analyzer details content length: {len(content)}")
            assert len(content.strip()) > 0, "Analyzer details empty"

            # Close modal
            close_btn = page.locator("#analyzerDetailsModal button:has-text('Zavrieť')")
            if close_btn.count() > 0:
                close_btn.click()
                page.wait_for_timeout(500)
        logger.info("TEST PASSED: Analyzer details modal")


# ══════════════════════════════════════════════════════════════════
# TEST: Heatmap Matrix
# ══════════════════════════════════════════════════════════════════
class TestHeatmap:
    """Test heatmap functionality with real data."""

    def _setup_with_target(self, page):
        """Full setup with target confirmed."""
        _build_pipeline(page, "linreg")
        _load_data_and_select_target(page, TEST_CSV, "Kalórie")

    def test_heatmap_correlation_matrix(self, clean_page):
        page = clean_page
        logger.info("TEST: Heatmap correlation matrix")
        self._setup_with_target(page)

        page.click("#showMatrixBtn")
        page.wait_for_selector("#heatmapModal.show", timeout=30000)
        page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

        rows = page.locator("#heatmapContainer .heatmap-table tbody tr").count()
        logger.info(f"Heatmap rows: {rows}")
        assert rows >= 5, f"Too few heatmap rows: {rows}"

        # Check correlation button is active
        corr_class = page.get_attribute("#heatmapBtnCorr", "class") or ""
        assert "active" in corr_class, "Correlation button not active"
        logger.info("TEST PASSED: Heatmap correlation matrix")

    def test_heatmap_mi_switch(self, clean_page):
        page = clean_page
        logger.info("TEST: Heatmap MI switch")
        self._setup_with_target(page)

        page.click("#showMatrixBtn")
        page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

        # Switch to MI
        page.click("#heatmapBtnMI")
        page.wait_for_timeout(1000)

        mi_class = page.get_attribute("#heatmapBtnMI", "class") or ""
        assert "active" in mi_class, "MI button not active after click"

        title = page.inner_text("#heatmapTitle")
        assert "mutual" in title.lower() or "mi" in title.lower() or "information" in title.lower()
        logger.info("TEST PASSED: Heatmap MI switch")

    def test_heatmap_toggle_values(self, clean_page):
        page = clean_page
        logger.info("TEST: Heatmap toggle values")
        self._setup_with_target(page)

        page.click("#showMatrixBtn")
        page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

        page.click("#heatmapBtnValues")
        page.wait_for_timeout(300)

        total = page.locator("#heatmapContainer .heatmap-table td").count()
        assert total > 0, "No cells in heatmap"
        logger.info("TEST PASSED: Heatmap toggle values")

    def test_heatmap_toggle_sort(self, clean_page):
        page = clean_page
        logger.info("TEST: Heatmap toggle sort")
        self._setup_with_target(page)

        page.click("#showMatrixBtn")
        page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

        page.click("#heatmapBtnSort")
        page.wait_for_timeout(500)
        logger.info("TEST PASSED: Heatmap toggle sort")

    def test_heatmap_close_with_escape(self, clean_page):
        page = clean_page
        logger.info("TEST: Heatmap close with Escape")
        self._setup_with_target(page)

        page.click("#showMatrixBtn")
        page.wait_for_selector("#heatmapModal.show", timeout=30000)

        page.keyboard.press("Escape")
        page.wait_for_timeout(500)

        # Check modal closed
        has_show = page.evaluate("document.getElementById('heatmapModal').classList.contains('show')")
        assert not has_show, "Heatmap modal not closed by Escape"
        logger.info("TEST PASSED: Heatmap close with Escape")

    def test_heatmap_legend_rendered(self, clean_page):
        page = clean_page
        logger.info("TEST: Heatmap legend")
        self._setup_with_target(page)

        page.click("#showMatrixBtn")
        page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

        legend = page.locator("#heatmapLegend")
        if legend.count() > 0:
            assert legend.is_visible(), "Legend not visible"
        logger.info("TEST PASSED: Heatmap legend")


# ══════════════════════════════════════════════════════════════════
# TEST: Feature Selectors - all types individually
# ══════════════════════════════════════════════════════════════════
class TestFeatureSelectors:
    """Test each feature selector individually."""

    def _setup_for_selectors(self, page):
        """Pipeline + data loaded."""
        _build_pipeline(page, "linreg", eval_mode="Regression")
        _load_data_and_select_target(page, TEST_CSV, "Kalórie")

    def test_variance_selector(self, clean_page):
        page = clean_page
        logger.info("TEST: Variance selector")
        self._setup_for_selectors(page)
        _compare_selectors(page, ["variance"])
        logger.info("TEST PASSED: Variance selector")

    def test_smc_selector_individual(self, clean_page):
        page = clean_page
        logger.info("TEST: SMC selector (individual)")
        self._setup_for_selectors(page)
        _compare_selectors(page, ["smc"], timeout=120000)
        logger.info("TEST PASSED: SMC selector (individual)")

    def test_mutual_information_selector(self, clean_page):
        page = clean_page
        logger.info("TEST: Mutual information selector")
        self._setup_for_selectors(page)
        _compare_selectors(page, ["mutual_information"], timeout=120000)
        logger.info("TEST PASSED: Mutual information selector")

    def test_chi_square_selector(self, clean_page):
        page = clean_page
        logger.info("TEST: Chi-square selector")
        self._setup_for_selectors(page)
        _compare_selectors(page, ["chi_square"], timeout=120000)
        logger.info("TEST PASSED: Chi-square selector")

    def test_smc_selector(self, clean_page):
        page = clean_page
        logger.info("TEST: SMC selector")
        self._setup_for_selectors(page)
        _compare_selectors(page, ["smc"], timeout=120000)
        logger.info("TEST PASSED: SMC selector")


# ══════════════════════════════════════════════════════════════════
# TEST: Data Inspection Modals
# ══════════════════════════════════════════════════════════════════
class TestDataInspection:
    """Test data inspection modals."""

    def test_inspect_raw_data(self, clean_page):
        page = clean_page
        logger.info("TEST: Inspect raw data modal")
        _build_pipeline(page, "linreg")
        _load_data_and_select_target(page, TEST_CSV, "Kalórie")

        page.click("#inspectDataBtn")
        page.wait_for_function(
            "document.getElementById('dataModal').classList.contains('show')",
            timeout=15000,
        )
        modal_text = page.inner_text("#modalBody")
        assert "Vzdialenosť" in modal_text or "Kalórie" in modal_text, "Modal missing data columns"
        logger.info(f"Raw data modal content length: {len(modal_text)}")

        page.click(".modal-close")
        page.wait_for_timeout(500)
        logger.info("TEST PASSED: Inspect raw data modal")

    # test_inspect_processed_data removed: #inspectProcessedBtn not in main index.html


# ══════════════════════════════════════════════════════════════════
# TEST: UI State & Buttons
# ══════════════════════════════════════════════════════════════════
class TestUIState:
    """Test UI state transitions and button behavior."""

    def test_initial_state(self, clean_page):
        page = clean_page
        logger.info("TEST: Initial UI state")

        # Buttons should be disabled
        assert page.get_attribute("#inspectDataBtn", "disabled") is not None
        assert page.get_attribute("#editDataBtn", "disabled") is not None

        # Sections hidden
        assert not page.is_visible("#featureExplorationSection")
        assert not page.is_visible("#targetSelectionArea")
        assert not page.is_visible("#currentPipelineInfo")
        logger.info("TEST PASSED: Initial UI state correct")

    def test_model_params_visibility(self, clean_page):
        page = clean_page
        logger.info("TEST: Model params visibility")

        page.select_option("#modelSelect", "knn")
        assert page.is_visible("#knnKGroup")
        assert not page.is_visible("#treeMaxDepthGroup")

        page.select_option("#modelSelect", "tree")
        assert not page.is_visible("#knnKGroup")
        assert page.is_visible("#treeMaxDepthGroup")

        page.select_option("#modelSelect", "linreg")
        assert not page.is_visible("#knnKGroup")
        assert not page.is_visible("#treeMaxDepthGroup")

        page.select_option("#modelSelect", "logreg")
        assert not page.is_visible("#knnKGroup")
        assert not page.is_visible("#treeMaxDepthGroup")
        logger.info("TEST PASSED: Model params visibility")

    def test_eval_mode_has_options(self, clean_page):
        page = clean_page
        logger.info("TEST: Eval mode options")
        options = page.locator("#evalModeSelect option").all()
        texts = [o.inner_text() for o in options]
        logger.info(f"Eval mode options: {texts}")
        assert len(options) >= 3, f"Expected at least 3 eval mode options, got {len(options)}"
        logger.info("TEST PASSED: Eval mode options")

    def test_data_input_method_toggle(self, clean_page):
        page = clean_page
        logger.info("TEST: Data input method toggle")

        assert page.is_visible("#textInputContainer")
        assert not page.is_visible("#fileInputContainer")

        page.click("#dataInputFile")
        assert not page.is_visible("#textInputContainer")
        assert page.is_visible("#fileInputContainer")

        page.click("#dataInputText")
        assert page.is_visible("#textInputContainer")
        assert not page.is_visible("#fileInputContainer")
        logger.info("TEST PASSED: Data input method toggle")

    def test_train_test_split_slider(self, clean_page):
        page = clean_page
        logger.info("TEST: Train/test split slider")

        _build_pipeline(page, "linreg")
        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        # Set to 70%
        page.evaluate("document.getElementById('trainTestSplit').value = 70")
        page.evaluate("document.getElementById('trainTestSplit').dispatchEvent(new Event('input'))")
        split_text = page.inner_text("#splitValue")
        assert "70" in split_text
        logger.info(f"Split slider: {split_text}")

        # Set to 90%
        page.evaluate("document.getElementById('trainTestSplit').value = 90")
        page.evaluate("document.getElementById('trainTestSplit').dispatchEvent(new Event('input'))")
        split_text = page.inner_text("#splitValue")
        assert "90" in split_text
        logger.info("TEST PASSED: Train/test split slider")

    def test_pipeline_info_shows_after_build(self, clean_page):
        page = clean_page
        logger.info("TEST: Pipeline info after build")

        _build_pipeline(page, "knn", knn_k=5, eval_mode="Regression")
        assert page.is_visible("#currentPipelineInfo")

        model_text = page.inner_text("#currentModel")
        logger.info(f"Current model info: {model_text}")
        assert len(model_text) > 0, "Model info empty"
        logger.info("TEST PASSED: Pipeline info shows after build")

    def test_selector_select_all_toggle(self, clean_page):
        page = clean_page
        logger.info("TEST: Selector select all/deselect all")

        _build_pipeline(page, "linreg")
        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        # Select all
        page.click("#selectAllSelectorsBtn")
        active = page.locator("#selectorCompareGrid .selector-compare-card.active").count()
        total = page.locator("#selectorCompareGrid .selector-compare-card").count()
        assert active == total, f"Select all: {active}/{total}"

        # Deselect all
        page.click("#selectAllSelectorsBtn")
        active = page.locator("#selectorCompareGrid .selector-compare-card.active").count()
        assert active == 0, f"Deselect all: {active}/0"
        logger.info("TEST PASSED: Selector select all toggle")

    def test_empty_data_error(self, clean_page):
        page = clean_page
        logger.info("TEST: Empty data parse error")

        page.fill("#dataInput", "")
        page.click("#parseDataBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#parseStatus")
        logger.info(f"Empty data status: {status}")
        assert len(status.strip()) > 0, "No error for empty data"
        logger.info("TEST PASSED: Empty data error")

    def test_compare_without_selection_error(self, clean_page):
        page = clean_page
        logger.info("TEST: Compare selectors without selection")

        _build_pipeline(page, "linreg")
        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        # Deselect all
        page.evaluate("""
            document.querySelectorAll('#selectorCompareGrid .selector-compare-card.active input[type=checkbox]')
                .forEach(cb => { cb.checked = false; cb.closest('.selector-compare-card').classList.remove('active'); });
        """)

        page.click("#compareSelectorBtn")
        page.wait_for_timeout(1000)
        status = page.inner_text("#compareSelectorStatus")
        logger.info(f"No-selection compare status: {status}")
        assert len(status.strip()) > 0
        logger.info("TEST PASSED: Compare without selection error")


# ══════════════════════════════════════════════════════════════════
# TEST: File Upload
# ══════════════════════════════════════════════════════════════════
class TestFileUpload:
    """Test file upload method with real CSV file."""

    def test_file_upload_loads_data(self, clean_page):
        page = clean_page
        logger.info("TEST: File upload with cycling CSV")

        _build_pipeline(page, "linreg")

        # Switch to file input
        page.click("#dataInputFile")
        assert page.is_visible("#fileInputContainer")

        # Upload test data file
        page.set_input_files("#fileInput", TEST_DATA_PATH)
        page.wait_for_timeout(500)

        file_info = page.inner_text("#fileInfo")
        logger.info(f"File info: {file_info}")
        assert "upravene_data" in file_info or "KB" in file_info, f"File info unexpected: {file_info}"

        # Parse
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        options = page.locator("#targetColumnSelect option").all()
        logger.info(f"Target columns from file: {len(options)}")
        assert len(options) >= 20, f"Expected 20+ columns from file, got {len(options)}"
        logger.info("TEST PASSED: File upload loads data")


# TestEmbeddedComparison removed: #compareEmbeddedBtn, #compareMatrixR2Btn,
# #embeddedComparisonResults, #matrixR2Results not in main index.html


# ══════════════════════════════════════════════════════════════════
# TEST: Metric Values Sanity Check
# ══════════════════════════════════════════════════════════════════
class TestMetricSanity:
    """Validate that metric values are within sane ranges."""

    def _extract_metrics_from_html(self, html: str, is_classification: bool):
        """Extract numeric metric values from training results HTML."""
        import re
        metrics = {}
        # Find table cells with numeric values (e.g. 0.85, 95.2%, etc.)
        # Results are in table td elements
        numbers = re.findall(r'(\d+\.?\d*)', html)
        return numbers  # Just check that there are numbers

    def test_regression_metrics_sanity(self, clean_page):
        page = clean_page
        logger.info("TEST: Regression metrics sanity")

        _build_pipeline(page, "linreg", eval_mode="Regression")
        _load_data_and_select_target(page, TEST_CSV, "Kalórie")
        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)

        text, html = _get_training_metrics(page)

        # Check that we have numeric values
        import re
        numbers = re.findall(r'[\d]+\.[\d]+', text)
        logger.info(f"Numeric values found in metrics: {len(numbers)}")
        assert len(numbers) >= 3, "Too few numeric values in regression results"

        # R² should appear with a reasonable value
        r2_match = re.search(r'R².*?([\-]?\d+\.?\d*)', text)
        if r2_match:
            r2_val = float(r2_match.group(1))
            logger.info(f"R² value: {r2_val}")
            # R² can be negative for bad fits, but should be roughly -5 to 1
            assert -10 < r2_val <= 1.01, f"R² value out of range: {r2_val}"

        logger.info("TEST PASSED: Regression metrics sanity")

    def test_classification_metrics_sanity(self, clean_page):
        page = clean_page
        logger.info("TEST: Classification metrics sanity")

        _build_pipeline(page, "logreg", eval_mode="Classification")

        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        _prepare_classification_data_via_editor(page, CLASSIFICATION_BIN_TARGET)

        page.select_option("#targetColumnSelect", CLASSIFICATION_BIN_TARGET)
        page.click("#confirmTargetBtn")
        page.wait_for_function(
            "document.getElementById('dataStatus').textContent.length > 0",
            timeout=15000,
        )

        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)

        text, html = _get_training_metrics(page)

        import re
        numbers = re.findall(r'[\d]+\.[\d]+', text)
        logger.info(f"Numeric values in classification metrics: {len(numbers)}")
        assert len(numbers) >= 3, "Too few numeric values in classification results"

        # ACC should be 0-1 range
        acc_match = re.search(r'ACC.*?([\d]+\.[\d]+)', text)
        if acc_match:
            acc_val = float(acc_match.group(1))
            logger.info(f"ACC value: {acc_val}")
            assert 0 <= acc_val <= 1.01, f"Accuracy out of range: {acc_val}"

        logger.info("TEST PASSED: Classification metrics sanity")


# ══════════════════════════════════════════════════════════════════
# TEST: Full end-to-end pipeline for each model
# ══════════════════════════════════════════════════════════════════
class TestFullPipelineE2E:
    """End-to-end test: build → load → edit → select → compare → train → verify."""

    def test_e2e_linreg_regression(self, clean_page):
        """Full E2E: LinReg regression."""
        page = clean_page
        logger.info("=" * 60)
        logger.info("E2E: LinReg regression full flow")
        logger.info("=" * 60)

        # Build
        _build_pipeline(page, "linreg", eval_mode="Regression")

        # Load data
        _load_data_and_select_target(page, TEST_CSV, "Kalórie", split_pct=75)

        # Open heatmap
        page.click("#showMatrixBtn")
        page.wait_for_selector("#heatmapModal.show", timeout=30000)
        page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)
        rows = page.locator("#heatmapContainer .heatmap-table tbody tr").count()
        assert rows >= 5
        page.keyboard.press("Escape")
        page.wait_for_timeout(500)
        logger.info("Heatmap opened and closed.")

        # Compare selectors
        _compare_selectors(page, ["variance", "smc"])

        # Train
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)
        assert "R²" in html or "RMSE" in html

        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors in E2E: {errors}"
        logger.info("E2E PASSED: LinReg regression")

    def test_e2e_logreg_classification(self, clean_page):
        """Full E2E: LogReg classification with data editing."""
        page = clean_page
        logger.info("=" * 60)
        logger.info("E2E: LogReg classification full flow with editor")
        logger.info("=" * 60)

        _build_pipeline(page, "logreg", eval_mode="Classification")

        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        # Editor: bin target, apply null_handler
        _prepare_classification_data_via_editor(page, CLASSIFICATION_BIN_TARGET)

        # Select target
        page.select_option("#targetColumnSelect", CLASSIFICATION_BIN_TARGET)
        page.click("#confirmTargetBtn")
        page.wait_for_function(
            "document.getElementById('dataStatus').textContent.length > 0",
            timeout=15000,
        )

        # Compare
        _compare_selectors(page, ["variance", "smc"])

        # Train
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)
        assert "ACC" in html or "Accuracy" in html or "F1" in html

        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors in E2E: {errors}"
        logger.info("E2E PASSED: LogReg classification")

    def test_e2e_knn_regression(self, clean_page):
        """Full E2E: KNN regression."""
        page = clean_page
        logger.info("=" * 60)
        logger.info("E2E: KNN regression full flow")
        logger.info("=" * 60)

        _build_pipeline(page, "knn", eval_mode="Regression", knn_k=5)
        _load_data_and_select_target(page, TEST_CSV, "Priemerná rýchlosť")
        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)
        assert "R²" in html or "RMSE" in html

        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors: {errors}"
        logger.info("E2E PASSED: KNN regression")

    def test_e2e_tree_regression(self, clean_page):
        """Full E2E: Decision Tree regression."""
        page = clean_page
        logger.info("=" * 60)
        logger.info("E2E: Tree regression full flow")
        logger.info("=" * 60)

        _build_pipeline(page, "tree", eval_mode="Regression", tree_depth=10)
        _load_data_and_select_target(page, TEST_CSV, "Kalórie", split_pct=80)
        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)
        assert "R²" in html or "RMSE" in html

        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors: {errors}"
        logger.info("E2E PASSED: Tree regression")

    def test_e2e_tree_classification(self, clean_page):
        """Full E2E: Decision Tree classification with binning."""
        page = clean_page
        logger.info("=" * 60)
        logger.info("E2E: Tree classification full flow with editor")
        logger.info("=" * 60)

        _build_pipeline(page, "tree", eval_mode="Classification", tree_depth=6)

        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        _prepare_classification_data_via_editor(page, CLASSIFICATION_BIN_TARGET)

        page.select_option("#targetColumnSelect", CLASSIFICATION_BIN_TARGET)
        page.click("#confirmTargetBtn")
        page.wait_for_function(
            "document.getElementById('dataStatus').textContent.length > 0",
            timeout=15000,
        )

        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)
        assert "ACC" in html or "Accuracy" in html or "F1" in html

        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors: {errors}"
        logger.info("E2E PASSED: Tree classification")

    def test_e2e_knn_classification(self, clean_page):
        """Full E2E: KNN classification with binning."""
        page = clean_page
        logger.info("=" * 60)
        logger.info("E2E: KNN classification full flow with editor")
        logger.info("=" * 60)

        _build_pipeline(page, "knn", eval_mode="Classification", knn_k=3)

        page.fill("#dataInput", TEST_CSV)
        page.click("#parseDataBtn")
        page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

        _prepare_classification_data_via_editor(page, CLASSIFICATION_BIN_TARGET)

        page.select_option("#targetColumnSelect", CLASSIFICATION_BIN_TARGET)
        page.click("#confirmTargetBtn")
        page.wait_for_function(
            "document.getElementById('dataStatus').textContent.length > 0",
            timeout=15000,
        )

        _compare_selectors(page, ["variance", "smc"])
        _train_from_comparison(page)
        text, html = _get_training_metrics(page)
        assert "ACC" in html or "Accuracy" in html or "F1" in html

        errors = _check_no_js_errors(page)
        assert len(errors) == 0, f"JS errors: {errors}"
        logger.info("E2E PASSED: KNN classification")
