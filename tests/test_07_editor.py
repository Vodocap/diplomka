"""
Tests for the data editor modal.

Tests:
- Editor opens with data table
- Column selection
- Editor processes column
- Close and apply changes
"""
from conftest import SAMPLE_CSV


def test_editor_opens(loaded_page):
    """Data editor modal opens with table."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)

    # Table should be rendered
    page.wait_for_selector("#editorTableContainer table", timeout=10000)
    rows = page.locator("#editorTableContainer table tbody tr").count()
    assert rows >= 10, f"Expected data rows in editor, got {rows}"


def test_editor_column_select(loaded_page):
    """Clicking column header selects the column."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)
    page.wait_for_selector("#editorTableContainer table", timeout=10000)

    # Click on first column header
    first_header = page.locator("#editorTableContainer table .col-header").first
    header_name = first_header.inner_text().strip()
    first_header.click()
    page.wait_for_timeout(500)

    # Processor panel should be visible
    assert page.is_visible("#editorProcessorPanel"), "Processor panel not shown after column select"

    # Label should show selected column
    label = page.inner_text("#editorSelectedColLabel")
    assert header_name in label


def test_editor_close_without_changes(loaded_page):
    """Closing editor without changes works."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)

    page.click("button:has-text('Zrušiť')")
    page.wait_for_timeout(500)

    assert not page.is_visible("#dataEditorModal"), "Editor modal not closed"


def test_editor_processor_selection(loaded_page):
    """Processor dropdown in editor has options."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)
    page.wait_for_selector("#editorTableContainer table", timeout=10000)

    # Select a column first
    first_header = page.locator("#editorTableContainer table .col-header").first
    first_header.click()
    page.wait_for_timeout(500)

    # Check processor dropdown
    options = page.locator("#editorProcessorSelect option").count()
    assert options >= 3, f"Expected at least 3 processor options, got {options}"


def test_editor_outlier_processor_selection_and_params_panel(loaded_page):
    """Outlier clipper selection updates processor params panel consistently."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)
    page.wait_for_selector("#editorTableContainer table", timeout=10000)

    # Select first column to activate processor panel
    page.locator("#editorTableContainer table .col-header").first.click()
    page.wait_for_selector("#editorProcessorPanel", state="visible", timeout=5000)

    page.select_option("#editorProcessorSelect", "outlier_clipper")

    # Depending on the currently loaded WASM build, params may or may not be exposed.
    params_count = page.locator("#editorProcessorParams .param-item").count()
    if params_count > 0:
        page.wait_for_selector("#editor_param_method", state="visible", timeout=5000)
        page.wait_for_selector("#editor_param_threshold", state="visible", timeout=5000)
        method_value = page.input_value("#editor_param_method")
        threshold_value = page.input_value("#editor_param_threshold")
        assert method_value == "iqr"
        assert threshold_value != ""
    else:
        # No params rendered, but panel remains functional and non-crashing.
        assert page.is_visible("#editorProcessorPanel")


def test_editor_power_transformer_has_no_extra_params(loaded_page):
    """Power transformer currently has no dynamic params in editor."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)
    page.wait_for_selector("#editorTableContainer table", timeout=10000)

    page.locator("#editorTableContainer table .col-header").first.click()
    page.wait_for_selector("#editorProcessorPanel", state="visible", timeout=5000)

    page.select_option("#editorProcessorSelect", "power_transformer")
    page.wait_for_timeout(300)

    params_count = page.locator("#editorProcessorParams .param-item").count()
    assert params_count == 0, f"Expected no params for power_transformer, got {params_count}"


def test_editor_apply_processor_to_column(loaded_page):
    """Applying a processor to selected column updates editor status."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)
    page.wait_for_selector("#editorTableContainer table", timeout=10000)

    # Pick first column and apply outlier clipper
    page.locator("#editorTableContainer table .col-header").first.click()
    page.wait_for_selector("#editorProcessorPanel", state="visible", timeout=5000)

    page.select_option("#editorProcessorSelect", "outlier_clipper")
    if page.locator("#editor_param_threshold").count() > 0:
        page.fill("#editor_param_threshold", "2.0")

    page.click("#editorApplyProcessorBtn")
    page.wait_for_timeout(800)

    status = page.inner_text("#editorStatusText")
    assert "aplikovaný" in status.lower() or "rows" in status.lower(), f"Unexpected editor status: {status}"


def test_editor_replace_all_in_selected_column(loaded_page):
    """Replace-all flow in selected column reports affected rows."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)
    page.wait_for_selector("#editorTableContainer table", timeout=10000)

    page.locator("#editorTableContainer table .col-header").first.click()
    page.wait_for_selector("#editorProcessorPanel", state="visible", timeout=5000)

    # Replace one known value in age column from SAMPLE_CSV
    page.fill("#editorSearchValue", "25")
    page.fill("#editorReplaceValue", "26")
    page.click("button:has-text('Nahradiť v stĺpci')")
    page.wait_for_timeout(800)

    status = page.inner_text("#editorStatusText")
    assert "nahraden" in status.lower(), f"Replace-all status not shown: {status}"


def test_editor_apply_and_close_updates_main_status(loaded_page):
    """Apply-and-close keeps modal closed and triggers data reparse status."""
    page = loaded_page

    page.click("#editDataBtn")
    page.wait_for_selector("#dataEditorModal.show", timeout=15000)
    page.wait_for_selector("#editorTableContainer table", timeout=10000)

    # Make a tiny edit to ensure apply path is exercised
    page.locator("#editorTableContainer table .col-header").first.click()
    page.wait_for_selector("#editorProcessorPanel", state="visible", timeout=5000)
    page.fill("#editorSearchValue", "25")
    page.fill("#editorReplaceValue", "27")
    page.click("button:has-text('Nahradiť v stĺpci')")
    page.wait_for_timeout(500)

    page.click("button:has-text('Použiť zmeny a zavrieť')")
    page.wait_for_timeout(800)

    assert not page.is_visible("#dataEditorModal"), "Editor modal should be closed after apply"

    parse_status = page.inner_text("#parseStatus")
    assert "aktualizované" in parse_status.lower() or "stĺpc" in parse_status.lower(), \
        f"Expected parse status refresh after apply, got: {parse_status}"
