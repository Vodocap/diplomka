"""
Tests for target analysis functionality.

Tests:
- Target analyzer cards populate
- Analyzer details modal opens
- Fullscreen matrix button in analyzer details
- Target comparison runs
- Target variable map displays
"""
from conftest import SAMPLE_CSV


def test_target_analyzers_populated(loaded_page):
    """Target analyzer cards are generated after data load."""
    page = loaded_page

    # Open the analysis details section
    page.click("details summary")
    page.wait_for_selector("#targetAnalyzerGrid", state="visible", timeout=5000)

    cards = page.locator("#targetAnalyzerGrid .selector-compare-card").count()
    assert cards >= 2, f"Expected at least 2 analyzer cards, got {cards}"


def test_analyzer_details_modal(loaded_page):
    """Clicking 'Zobraziť detaily' opens analyzer details modal."""
    page = loaded_page

    # Open analysis section
    page.click("details summary")
    page.wait_for_selector("#targetAnalyzerGrid", state="visible", timeout=5000)

    # Click the first "Zobraziť detaily" button
    detail_btns = page.locator("#targetAnalyzerGrid button:has-text('detaily')")
    if detail_btns.count() > 0:
        detail_btns.first.click()

        page.wait_for_selector("#analyzerDetailsModal.show", timeout=30000)

        # Modal should have content
        content = page.inner_text("#analyzerDetailsContent")
        assert len(content.strip()) > 0, "Analyzer details modal is empty"

        # Close modal
        page.click("#analyzerDetailsModal button:has-text('Zavrieť')")
        page.wait_for_timeout(500)


def test_compare_target_analyzers(loaded_page):
    """Compare target analyzers and verify results."""
    page = loaded_page

    # Open analysis section
    page.click("details summary")
    page.wait_for_selector("#targetAnalyzerGrid", state="visible", timeout=5000)

    # Select all analyzers
    page.click("#selectAllAnalyzersBtn")

    # Click compare button
    page.click("#compareTargetAnalyzersBtn")

    # Wait for comparison results
    page.wait_for_selector("#targetComparisonResultsArea", state="visible", timeout=60000)

    # Target variable map should have content
    map_content = page.inner_text("#targetVariableMap")
    assert len(map_content.strip()) > 0, "Target variable map is empty"


def test_select_target_from_map(loaded_page):
    """Clicking a variable in the target map selects it."""
    page = loaded_page

    page.click("details summary")
    page.wait_for_selector("#targetAnalyzerGrid", state="visible", timeout=5000)

    page.click("#selectAllAnalyzersBtn")
    page.click("#compareTargetAnalyzersBtn")
    page.wait_for_selector("#targetComparisonResultsArea", state="visible", timeout=60000)

    # Find a clickable variable badge in the map
    badges = page.locator("#targetVariableMap [onclick], #targetVariableMap .target-badge")
    if badges.count() > 0:
        first_badge = badges.first
        badge_text = first_badge.inner_text()
        first_badge.click()
        page.wait_for_timeout(500)

        # Target select should have changed
        selected = page.evaluate("document.getElementById('targetColumnSelect').value")
        assert len(selected) > 0, "Target was not selected from map"


def test_matrix_button_in_analysis(loaded_page):
    """Matrix button next to target analyzers works."""
    page = loaded_page

    page.click("details summary")
    page.wait_for_selector("#showMatrixBtnTarget", state="visible", timeout=5000)

    page.click("#showMatrixBtnTarget")
    page.wait_for_selector("#heatmapModal.show", timeout=30000)

    # Heatmap container should have a table
    page.wait_for_selector("#heatmapContainer table", timeout=30000)

    table = page.locator("#heatmapContainer .heatmap-table")
    assert table.count() > 0, "Heatmap table was not rendered"

    # Close
    page.keyboard.press("Escape")
    page.wait_for_timeout(500)
