"""
Tests for data loading, parsing, and target selection.

Tests:
- Paste CSV data and parse
- Target column dropdown populated
- Target selection and confirmation
- Data format switching
- Train/test split slider
- File upload method toggle
"""
from conftest import SAMPLE_CSV


def test_parse_csv_data(app_page):
    """Parse sample CSV data and verify status."""
    page = app_page

    page.fill("#dataInput", SAMPLE_CSV)
    page.click("#parseDataBtn")

    # Wait for target selection area to appear
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    # Check that target dropdown has columns
    options = page.locator("#targetColumnSelect option").all()
    # First is placeholder + columns from CSV
    assert len(options) >= 14, f"Expected 14+ options in target select, got {len(options)}"


def test_target_dropdown_contains_all_columns(app_page):
    """All CSV columns appear as target options."""
    page = app_page

    page.fill("#dataInput", SAMPLE_CSV)
    page.click("#parseDataBtn")
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    option_texts = [opt.inner_text() for opt in page.locator("#targetColumnSelect option").all()]
    expected_cols = ["age", "income", "credit_score", "approved"]
    for col in expected_cols:
        assert any(col in t for t in option_texts), f"Column '{col}' not found in dropdown"


def test_confirm_target(app_page):
    """Selecting and confirming target succeeds."""
    page = app_page

    # Need pipeline built first
    page.select_option("#modelSelect", "logreg")
    page.click("#buildPipelineBtn")
    page.wait_for_selector("#currentPipelineInfo", state="visible", timeout=15000)

    page.fill("#dataInput", SAMPLE_CSV)
    page.click("#parseDataBtn")
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    page.select_option("#targetColumnSelect", "approved")
    page.click("#confirmTargetBtn")

    # Wait for status to appear and contain success text
    page.wait_for_function(
        "document.getElementById('dataStatus').textContent.length > 0",
        timeout=15000
    )
    status_text = page.inner_text("#dataStatus")
    assert "úspešne" in status_text.lower() or "success" in status_text.lower() \
        or "načítan" in status_text.lower() or "train" in status_text.lower(), \
        f"Unexpected status: {status_text}"


def test_inspect_data_enabled_after_parse(app_page):
    """Inspect data button becomes enabled after parsing."""
    page = app_page

    page.fill("#dataInput", SAMPLE_CSV)
    page.click("#parseDataBtn")
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    # inspectDataBtn should now be enabled
    assert page.get_attribute("#inspectDataBtn", "disabled") is None


def test_feature_exploration_visible_after_parse(app_page):
    """Feature exploration section appears after data is parsed."""
    page = app_page

    page.fill("#dataInput", SAMPLE_CSV)
    page.click("#parseDataBtn")
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    assert page.is_visible("#featureExplorationSection")


def test_train_test_split_slider(app_page):
    """Train/test split slider updates display."""
    page = app_page

    page.fill("#dataInput", SAMPLE_CSV)
    page.click("#parseDataBtn")
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    # Change slider to 70%
    page.evaluate("document.getElementById('trainTestSplit').value = 70")
    page.evaluate("document.getElementById('trainTestSplit').dispatchEvent(new Event('input'))")

    split_text = page.inner_text("#splitValue")
    assert "70" in split_text


def test_data_input_method_toggle(app_page):
    """Switching between text and file input methods."""
    page = app_page

    # Text input should be visible by default
    assert page.is_visible("#textInputContainer")
    assert not page.is_visible("#fileInputContainer")

    # Switch to file
    page.click("#dataInputFile")
    assert not page.is_visible("#textInputContainer")
    assert page.is_visible("#fileInputContainer")

    # Switch back to text
    page.click("#dataInputText")
    assert page.is_visible("#textInputContainer")
    assert not page.is_visible("#fileInputContainer")


def test_empty_data_shows_error(app_page):
    """Parsing without data shows error status."""
    page = app_page

    page.fill("#dataInput", "")
    page.click("#parseDataBtn")

    page.wait_for_selector("#parseStatus", timeout=10000)
    # Should show some error
    status = page.inner_text("#parseStatus")
    assert len(status.strip()) > 0, "Expected error message for empty data"


def test_inspect_data_modal(app_page):
    """Inspect data button opens modal with data table."""
    page = app_page

    # Need pipeline + confirmed target for inspectData to work
    page.select_option("#modelSelect", "logreg")
    page.click("#buildPipelineBtn")
    page.wait_for_selector("#currentPipelineInfo", state="visible", timeout=15000)

    page.fill("#dataInput", SAMPLE_CSV)
    page.click("#parseDataBtn")
    page.wait_for_selector("#targetSelectionArea", state="visible", timeout=15000)

    page.select_option("#targetColumnSelect", "approved")
    page.click("#confirmTargetBtn")
    page.wait_for_function(
        "document.getElementById('dataStatus').textContent.length > 0",
        timeout=15000
    )

    page.click("#inspectDataBtn")
    # Modal uses .show class
    page.wait_for_function(
        "document.getElementById('dataModal').classList.contains('show')",
        timeout=15000
    )

    modal_text = page.inner_text("#modalBody")
    assert "age" in modal_text or "income" in modal_text, "Modal doesn't contain data columns"

    # Close modal
    page.click(".modal-close")
    page.wait_for_timeout(500)
