use llmd::state::AppState;

#[tokio::test]
async fn test_optional_embeddings() {
    // This should not panic when embeddings are missing
    let result = AppState::init().await;
    assert!(result.is_ok(), "AppState initialization should succeed even without embeddings");

    let state = result.unwrap();
    assert!(!state.embeddings_available, "Embeddings should be marked as unavailable");

    // Test that inference fails gracefully
    let inference_result = state.model.infer("test").await;
    assert!(inference_result.is_err(), "Inference should fail when embeddings are not available");
    assert!(inference_result.unwrap_err().to_string().contains("Embedding model not available"));
}

#[tokio::test]
async fn test_embeddings_available() {
    // Restore the model temporarily for this test
    std::fs::rename("models/minilm.test", "models/minilm").ok();

    let result = AppState::init().await;
    assert!(result.is_ok());

    let state = result.unwrap();
    assert!(state.embeddings_available, "Embeddings should be available when model exists");

    // Restore the test state
    std::fs::rename("models/minilm", "models/minilm.test").ok();
}
