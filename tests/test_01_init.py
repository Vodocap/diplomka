"""
Tests for WASM initialization and basic page load.

Tests:
- Page loads correctly
- WASM module initializes
- Available options (models, processors, selectors) are displayed
- No JS console errors during initialization
"""
import re


def test_page_loads(app_page):
    """Page title and header render correctly."""
    page = app_page
    assert "ML Pipeline" in page.title() or "ML Pipeline" in page.inner_text("h1")
    assert page.is_visible("h1")


def test_wasm_initializes_no_errors(page, server):
    """WASM loads without JS console errors."""
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

    page.goto(server, wait_until="networkidle")
    page.wait_for_function(
        "document.getElementById('modelsInfo') && document.getElementById('modelsInfo').children.length > 0",
        timeout=30000,
    )

    # Filter out non-critical errors (e.g. favicon 404)
    critical = [e for e in errors if "favicon" not in e.lower() and "404" not in e]
    assert len(critical) == 0, f"JS console errors: {critical}"


def test_models_displayed(app_page):
    """Available models are populated after WASM init."""
    page = app_page
    models_div = page.locator("#modelsInfo")
    count = models_div.locator(".option-item, span, div").count()
    assert count >= 3, f"Expected at least 3 models, got {count}"


def test_processors_displayed(app_page):
    """Processors section is populated."""
    page = app_page
    text = page.inner_text("#processorsInfo")
    assert len(text.strip()) > 0, "Processors section is empty"


def test_selectors_displayed(app_page):
    """Selectors section is populated."""
    page = app_page
    text = page.inner_text("#selectorsInfo")
    assert len(text.strip()) > 0, "Selectors section is empty"


def test_model_select_has_options(app_page):
    """Model select dropdown has options after WASM init."""
    page = app_page
    options = page.locator("#modelSelect option").all()
    # At least placeholder + some models
    assert len(options) >= 4, f"Expected at least 4 model options, got {len(options)}"


def test_initial_ui_state(app_page):
    """Buttons and sections have correct initial state."""
    page = app_page

    # Inspect buttons should be disabled
    assert page.get_attribute("#inspectDataBtn", "disabled") is not None
    assert page.get_attribute("#inspectProcessedBtn", "disabled") is not None
    assert page.get_attribute("#editDataBtn", "disabled") is not None

    # Feature exploration section should be hidden
    assert not page.is_visible("#featureExplorationSection")

    # Target selection area should be hidden
    assert not page.is_visible("#targetSelectionArea")

    # Pipeline info should be hidden
    assert not page.is_visible("#currentPipelineInfo")
