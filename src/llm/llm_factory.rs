use std::sync::Arc;
use anyhow::Result;

/// Supported LLM backends
#[derive(Clone, Debug)]
pub enum LlmBackend {
    /// Local models using Candle (GPU/CPU inference)
    Candle,
    /// Remote models via Ollama HTTP API
    Ollama,
}

/// Build an LLM backend instance from environment variables
///
/// Required environment variables:
/// - LLMD_BACKEND: "candle" or "ollama"
/// - LLMD_MODEL_PATH: path to model directory (required for candle backend)
/// - LLMD_OLLAMA_MODEL: model name (required for ollama backend)
/// - LLMD_OLLAMA_URL: server URL (optional, defaults to http://localhost:11434)
///
/// Returns an error if configuration is invalid or missing required variables.
pub fn build_llm_from_env() -> Result<Arc<dyn super::llm::LlmModelTrait>> {
    let backend_str = std::env::var("LLMD_LLM_BACKEND")
                    .map_err(|_| anyhow::anyhow!(
                        "LLMD_LLM_BACKEND is required. Set to 'candle' or 'ollama'"
                    ))?;

    let backend = match backend_str.to_lowercase().as_str() {
        "candle" => LlmBackend::Candle,
        "ollama" => LlmBackend::Ollama,
        _ => return Err(anyhow::anyhow!("LLMD_BACKEND must be 'candle' or 'ollama', got: {}", backend_str)),
    };

    tracing::info!("Selected LLM backend: {:?}", backend);

    match backend {
        LlmBackend::Candle => {
            let model_path = std::env::var("LLMD_MODEL_PATH")
                .map_err(|_| anyhow::anyhow!("LLMD_MODEL_PATH is required when using candle backend"))?;

            tracing::info!("Loading Candle model from: {}", model_path);
            let llm = super::llm::CandleLlm::new(&model_path)?;
            Ok(Arc::new(llm))
        }
        LlmBackend::Ollama => {
            let ollama_model = std::env::var("LLMD_OLLAMA_MODEL")
                .map_err(|_| anyhow::anyhow!("LLMD_OLLAMA_MODEL is required when using ollama backend"))?;

            let ollama_url = std::env::var("LLMD_OLLAMA_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());

            tracing::info!("Connecting to Ollama at {} with model {}", ollama_url, ollama_model);
            let llm = super::llm::OllamaLlm::new(ollama_url, ollama_model);
            Ok(Arc::new(llm))
        }
    }
}
