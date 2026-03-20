"""
Tests for the feature selector comparison and training.

Tests:
- Selector cards populated
- Select all / toggle selectors
- Compare selectors shows modal with results
- Train from comparison
"""
from conftest import SAMPLE_CSV


def test_selector_cards_populated(loaded_page):
    """Selector comparison cards are generated."""
    page = loaded_page

    cards = page.locator("#selectorCompareGrid .selector-compare-card").count()
    assert cards >= 4, f"Expected at least 4 selector cards, got {cards}"


def test_select_all_selectors(loaded_page):
    """'Vybrať všetky' toggles all selector cards active."""
    page = loaded_page

    page.click("#selectAllSelectorsBtn")

    active_cards = page.locator("#selectorCompareGrid .selector-compare-card.active").count()
    total_cards = page.locator("#selectorCompareGrid .selector-compare-card").count()
    assert active_cards == total_cards, f"Not all cards active: {active_cards}/{total_cards}"

    # Toggle again - should deselect all
    page.click("#selectAllSelectorsBtn")
    active_cards = page.locator("#selectorCompareGrid .selector-compare-card.active").count()
    assert active_cards == 0, f"Cards should be deselected, got {active_cards}"


def test_compare_selectors(loaded_page):
    """Compare selectors and view results in modal."""
    page = loaded_page

    # Select 2 fast selectors
    fast_selectors = ["variance", "smc"]
    for sel in fast_selectors:
        card = page.locator(f"#compare_card_{sel}")
        if card.count() > 0:
            card.locator("input[type='checkbox']").check()

    page.click("#compareSelectorBtn")

    # Wait for modal
    page.wait_for_selector("#dataModal.show", timeout=60000)

    modal_text = page.inner_text("#modalBody")
    assert len(modal_text.strip()) > 100, "Comparison modal has too little content"

    # Close modal
    page.click(".modal-close")
    page.wait_for_timeout(500)


def test_compare_no_selection_error(loaded_page):
    """Comparing with no selectors selected shows error."""
    page = loaded_page

    # Make sure none selected
    page.evaluate("""
        document.querySelectorAll('#selectorCompareGrid .selector-compare-card.active input[type=checkbox]')
            .forEach(cb => { cb.checked = false; cb.closest('.selector-compare-card').classList.remove('active'); });
    """)

    page.click("#compareSelectorBtn")

    page.wait_for_selector("#compareSelectorStatus", timeout=10000)
    status = page.inner_text("#compareSelectorStatus")
    assert "aspoň" in status.lower() or "error" in status.lower() or "vyberte" in status.lower(), \
        f"Expected error for no selectors: {status}"


def test_train_all_selectors(loaded_page):
    """Train with all selector variants (after comparison)."""
    page = loaded_page

    # First compare with a couple selectors
    for sel in ["variance", "smc"]:
        card = page.locator(f"#compare_card_{sel}")
        if card.count() > 0:
            card.locator("input[type='checkbox']").check()

    page.click("#compareSelectorBtn")
    page.wait_for_selector("#dataModal.show", timeout=60000)

    # Click Train from comparison inside the modal
    train_btn = page.locator("#trainFromComparisonBtn")
    if train_btn.count() > 0:
        train_btn.click()

    # Wait for training results
    page.wait_for_function(
        """(() => {
            const el = document.getElementById('comparisonTrainingResults') || document.getElementById('selectorTrainingResults');
            return el && el.innerText.length > 50;
        })()""",
        timeout=120000,
    )

    results_text = (page.locator("#comparisonTrainingResults").inner_text()
                    if page.locator("#comparisonTrainingResults").count() > 0
                    else page.inner_text("#selectorTrainingResults"))
    assert len(results_text.strip()) > 50, "Training results too short"
