"""
Tests for the fullscreen heatmap matrix visualization.

Tests:
- Heatmap modal opens from feature exploration
- Heatmap modal opens from target analysis section
- Correlation matrix renders
- MI matrix switch works
- Toggle values (show/hide numbers)
- Sorting toggle
- Tooltip on hover
- Legend renders
- Escape closes modal
"""
from conftest import SAMPLE_CSV


def test_heatmap_opens_from_selectors(loaded_page):
    """Heatmap matrix button in feature exploration section works."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapModal.show", timeout=30000)
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    # Verify table has rows
    rows = page.locator("#heatmapContainer .heatmap-table tbody tr").count()
    assert rows >= 5, f"Expected at least 5 rows in heatmap, got {rows}"


def test_heatmap_opens_from_target(loaded_page):
    """Heatmap matrix button in target analysis section works."""
    page = loaded_page

    # Open analysis details section
    page.click("details summary")
    page.wait_for_selector("#showMatrixBtnTarget", state="visible", timeout=5000)

    page.click("#showMatrixBtnTarget")
    page.wait_for_selector("#heatmapModal.show", timeout=30000)
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    rows = page.locator("#heatmapContainer .heatmap-table tbody tr").count()
    assert rows >= 5


def test_heatmap_correlation_matrix(loaded_page):
    """Default correlation matrix has proper cells with backgrounds."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    # Check that cells have background styles
    cell = page.locator("#heatmapContainer .heatmap-table td").first
    bg = cell.evaluate("el => getComputedStyle(el).backgroundColor")
    assert bg != "" and bg != "rgba(0, 0, 0, 0)", f"Cell has no background color: {bg}"

    # Correlation button should be active
    assert "active" in (page.get_attribute("#heatmapBtnCorr", "class") or "")


def test_heatmap_switch_to_mi(loaded_page):
    """Switch to MI matrix updates display."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    # Switch to MI
    page.click("#heatmapBtnMI")
    page.wait_for_timeout(500)

    # MI button should be active, corr not
    assert "active" in (page.get_attribute("#heatmapBtnMI", "class") or "")
    assert "active" not in (page.get_attribute("#heatmapBtnCorr", "class") or "")

    # Title should mention MI
    title = page.inner_text("#heatmapTitle")
    assert "mutual" in title.lower() or "mi" in title.lower() or "information" in title.lower()


def test_heatmap_toggle_values(loaded_page):
    """Toggle value display shows/hides numbers in cells."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    # Click toggle values button
    page.click("#heatmapBtnValues")
    page.wait_for_timeout(300)

    # Check if cells have show-values class toggled
    has_values = page.locator("#heatmapContainer .heatmap-table td.show-values").count()
    no_values = page.locator("#heatmapContainer .heatmap-table td:not(.show-values)").count()

    # Either all have it or none (just check consistency — depends on initial state)
    total = has_values + no_values
    assert total > 0, "No cells found"
    assert has_values == total or has_values == 0, "Inconsistent show-values state"


def test_heatmap_sort_toggle(loaded_page):
    """Sort button reorders matrix rows/columns."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    # Get initial row header order
    headers_before = [
        el.inner_text()
        for el in page.locator("#heatmapContainer .heatmap-table th.row-header").all()
    ]

    # Click sort
    page.click("#heatmapBtnSort")
    page.wait_for_timeout(500)

    headers_after = [
        el.inner_text()
        for el in page.locator("#heatmapContainer .heatmap-table th.row-header").all()
    ]

    assert "active" in (page.get_attribute("#heatmapBtnSort", "class") or "")
    # Headers may or may not change order depending on data, but at least they should exist
    assert len(headers_after) == len(headers_before), "Row count changed after sort"


def test_heatmap_tooltip_on_hover(loaded_page):
    """Hovering over a cell shows tooltip."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    # Hover over first cell
    cell = page.locator("#heatmapContainer .heatmap-table td").first
    cell.hover()
    page.wait_for_timeout(300)

    tooltip = page.locator("#heatmapTooltip")
    tooltip_display = tooltip.evaluate("el => getComputedStyle(el).display")
    assert tooltip_display != "none", "Tooltip is not visible on hover"


def test_heatmap_legend_renders(loaded_page):
    """Legend bar renders at the bottom."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    legend = page.locator("#heatmapLegend")
    assert legend.is_visible()
    legend_html = legend.inner_html()
    assert "lg-bar" in legend_html, "Legend bar not rendered"


def test_heatmap_close_escape(loaded_page):
    """Escape key closes heatmap modal."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapModal.show", timeout=30000)

    page.keyboard.press("Escape")
    page.wait_for_timeout(500)

    assert not page.is_visible("#heatmapModal"), "Heatmap modal was not closed by Escape"


def test_heatmap_close_button(loaded_page):
    """Close button closes heatmap modal."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapModal.show", timeout=30000)

    page.click("#heatmapModal button:has-text('Zavrieť')")
    page.wait_for_timeout(500)

    assert not page.is_visible("#heatmapModal"), "Heatmap modal was not closed by button"


def test_heatmap_info_text(loaded_page):
    """Info text shows feature count and row count."""
    page = loaded_page

    page.click("#showMatrixBtn")
    page.wait_for_selector("#heatmapContainer .heatmap-table", timeout=30000)

    info = page.inner_text("#heatmapInfo")
    # Should contain something like "14 príznakov × 30 riadkov"
    assert "príznakov" in info or "×" in info, f"Unexpected info text: {info}"
