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
