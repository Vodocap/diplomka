"""
Tests for pipeline building functionality.

Tests:
- Build pipeline with each model type
- Model parameter visibility toggling
- Pipeline status message
- Eval mode selection
"""


def test_build_pipeline_linreg(app_page):
    """Build pipeline with linear regression."""
    page = app_page

    page.select_option("#modelSelect", "linreg")
    page.click("#buildPipelineBtn")

    page.wait_for_selector("#pipelineStatus", timeout=10000)
    status = page.inner_text("#pipelineStatus")
    assert "linreg" in status.lower() or "úspešne" in status.lower() or "vytvor" in status.lower(), \
        f"Pipeline build failed: {status}"

    # Pipeline info should be visible
    page.wait_for_selector("#currentPipelineInfo", state="visible", timeout=5000)


def test_build_pipeline_logreg(app_page):
    """Build pipeline with logistic regression."""
    page = app_page

    page.select_option("#modelSelect", "logreg")
    page.click("#buildPipelineBtn")

    page.wait_for_selector("#pipelineStatus", timeout=10000)
    status = page.inner_text("#pipelineStatus")
    assert "error" not in status.lower() and "chyba" not in status.lower(), \
        f"Pipeline build error: {status}"


def test_build_pipeline_knn(app_page):
    """Build KNN pipeline and check K parameter UI."""
    page = app_page

    page.select_option("#modelSelect", "knn")

    # KNN params should be visible
    page.wait_for_selector("#knnKGroup", state="visible", timeout=5000)

    page.fill("#knnK", "7")
    page.select_option("#evalModeSelect", "regression")
    page.click("#buildPipelineBtn")

    page.wait_for_selector("#pipelineStatus", timeout=10000)
    status = page.inner_text("#pipelineStatus")
    assert "error" not in status.lower() and "chyba" not in status.lower(), \
        f"Pipeline build error: {status}"


def test_build_pipeline_tree(app_page):
    """Build decision tree pipeline and check max_depth UI."""
    page = app_page

    page.select_option("#modelSelect", "tree")

    # Tree params should be visible
    page.wait_for_selector("#treeMaxDepthGroup", state="visible", timeout=5000)

    page.fill("#treeMaxDepth", "5")
    page.select_option("#evalModeSelect", "regression")
    page.click("#buildPipelineBtn")

    page.wait_for_selector("#pipelineStatus", timeout=10000)
    status = page.inner_text("#pipelineStatus")
    assert "error" not in status.lower() and "chyba" not in status.lower()


def test_model_params_toggle(app_page):
    """Switching models shows/hides their parameter groups."""
    page = app_page

    # Select KNN - should show knnKGroup
    page.select_option("#modelSelect", "knn")
    assert page.is_visible("#knnKGroup")
    assert not page.is_visible("#treeMaxDepthGroup")

    # Switch to tree - should show treeMaxDepthGroup
    page.select_option("#modelSelect", "tree")
    assert not page.is_visible("#knnKGroup")
    assert page.is_visible("#treeMaxDepthGroup")

    # Switch to linreg - should hide both
    page.select_option("#modelSelect", "linreg")
    assert not page.is_visible("#knnKGroup")
    assert not page.is_visible("#treeMaxDepthGroup")


def test_eval_mode_selection(app_page):
    """Evaluation mode can be set."""
    page = app_page

    page.select_option("#modelSelect", "logreg")
    page.select_option("#evalModeSelect", "classification")
    page.click("#buildPipelineBtn")

    page.wait_for_selector("#currentPipelineInfo", state="visible", timeout=10000)
    eval_text = page.inner_text("#currentEvalMode")
    assert "classification" in eval_text.lower()


def test_no_model_selected_error(app_page):
    """Building pipeline without selecting model shows error."""
    page = app_page

    # Don't select any model - leave placeholder
    page.click("#buildPipelineBtn")

    page.wait_for_selector("#pipelineStatus", timeout=10000)
    status = page.inner_text("#pipelineStatus")
    # Should be an error
    assert len(status.strip()) > 0, "Expected error for no model selected"
