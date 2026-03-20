// Entry point – global event handlers, app start

// Close modals when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('dataModal');
    if (event.target === modal) {
        closeModal();
    }
    const editorModal = document.getElementById('dataEditorModal');
    if (event.target === editorModal) {
        closeDataEditor();
    }
};

// Start the application
setupBasicListeners();
initApp();
