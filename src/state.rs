use crate::executor::Executor;
use crate::llm::LlmModelTrait;
use crate::llm_factory::build_llm_from_env;
use crate::llm_registry::LlmRegistry;
use crate::model::Model;
use crate::persona_memory::IntelligentMemory;
use crate::storage::Storage;
use crate::system_prompt::SystemPromptManager;
use crate::toolport::ToolRegistry;
use crate::vector_store::VectorStore;

use model_loader_core::plan::{LoadPlan, LoadStep};

use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct AppState {
    /// Embedding model only (never used for text generation)
    pub model: Arc<Model>,
    pub embeddings_available: bool,

    /// Registered LLMs (Candle or Ollama)
    pub llms: Arc<LlmRegistry>,

    pub storage: Arc<Mutex<Storage>>,
    pub store: Arc<Mutex<VectorStore>>,

    pub system_prompt: SystemPromptManager,
    pub persona_memory: Arc<Mutex<IntelligentMemory>>,

    /// Tool registry for external capabilities
    pub tools: Arc<ToolRegistry>,

    /// Executor owns execution authority and memory mutation
    pub executor: Arc<Executor>,
}

impl AppState {
    // --------------------------------------------------
    // Candle / local backend (requires LoadPlan)
    // --------------------------------------------------
    pub async fn init_from_plan(
        plan: LoadPlan,
        tool_registry: Arc<ToolRegistry>,
    ) -> anyhow::Result<Self> {
        let (llm_model, model_id) = Self::execute_plan(plan).await?;

        let mut llm_registry = LlmRegistry::new();
        llm_registry.register(model_id.clone(), llm_model)?;
        tracing::info!("Registered Candle model: {}", model_id);

        Self::init_common(Arc::new(llm_registry), tool_registry).await
    }

    // --------------------------------------------------
    // Ollama / remote backend (no LoadPlan)
    // --------------------------------------------------
    pub async fn init_remote_llm(
        tool_registry: Arc<ToolRegistry>,
    ) -> anyhow::Result<Self> {
        let llm_model = build_llm_from_env()?;

        let model_id = std::env::var("LLMD_OLLAMA_MODEL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "ollama".to_string());

        let mut llm_registry = LlmRegistry::new();
        llm_registry.register(model_id.clone(), llm_model)?;
        tracing::info!("Registered Ollama model: {}", model_id);

        Self::init_common(Arc::new(llm_registry), tool_registry).await
    }

    // --------------------------------------------------
    // Execute LoadPlan → ensure Candle can boot
    // --------------------------------------------------
    async fn execute_plan(
        plan: LoadPlan,
    ) -> anyhow::Result<(Arc<dyn LlmModelTrait>, String)> {
        tracing::info!("Executing LoadPlan with {} steps", plan.steps.len());

        let model_dir = Self::extract_model_dir_from_plan(&plan)?;
        let model_id = Self::extract_model_id_from_dir(&model_dir);

        // 🔑 Critical fix:
        // If using Candle and LLMD_MODEL_PATH is not set,
        // derive it automatically from the LoadPlan.
        let backend = std::env::var("LLMD_LLM_BACKEND")
            .unwrap_or_else(|_| "candle".to_string())
            .to_lowercase();

            if backend == "candle" && std::env::var("LLMD_MODEL_PATH").is_err() {
                tracing::info!(
                    "Setting LLMD_MODEL_PATH from LoadPlan: {}",
                    model_dir
                );
            
                // SAFETY:
                // This is executed once at startup, before any worker threads are spawned.
                // llmd does not mutate environment variables after initialization.
                unsafe {
                    std::env::set_var("LLMD_MODEL_PATH", &model_dir);
                }
            }

        let model = build_llm_from_env()?;
        Ok((model, model_id))
    }

    fn extract_model_dir_from_plan(plan: &LoadPlan) -> anyhow::Result<String> {
        for step in &plan.steps {
            match step {
                LoadStep::LoadConfig { path }
                | LoadStep::LoadTokenizer { path }
                | LoadStep::LoadShard { path, .. } => {
                    if let Some(dir) = std::path::Path::new(path).parent() {
                        if let Some(dir_str) = dir.to_str() {
                            return Ok(dir_str.to_string());
                        }
                    }
                }
            }
        }
        Err(anyhow::anyhow!("No valid model directory found in LoadPlan"))
    }

    fn extract_model_id_from_dir(model_dir: &str) -> String {
        std::path::Path::new(model_dir)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown-model")
            .to_string()
    }

    // --------------------------------------------------
    // Shared initialization (storage, memory, executor)
    // --------------------------------------------------
    async fn init_common(
        llms: Arc<LlmRegistry>,
        tool_registry: Arc<ToolRegistry>,
    ) -> anyhow::Result<Self> {
        // ---- Optional embedding model
        let (model, embeddings_available) = match Model::load("models/minilm").await {
            Ok(m) => match m.infer("hello world").await {
                Ok(_) => (m, true),
                Err(e) => {
                    tracing::warn!("Embedding inference failed: {}", e);
                    (Model::dummy(), false)
                }
            },
            Err(e) => {
                tracing::warn!("MiniLM not available: {}", e);
                (Model::dummy(), false)
            }
        };

        let db_path = "data.db";

        let storage = Storage::new(db_path)?;
        let existing = storage.load_all_entries();

        let dim = if embeddings_available {
            if let Some(first) = existing.first() {
                first.embedding.len()
            } else {
                model.infer("dim probe").await.map(|v| v.len()).unwrap_or(384)
            }
        } else {
            384
        };

        let mut vector_store = VectorStore::new(dim, Storage::new(db_path)?);
        if embeddings_available {
            vector_store.rebuild_index();
        }

        let persona_memory = Arc::new(Mutex::new(IntelligentMemory::new(db_path)?));

        tracing::info!("Vector memory enabled: {}", embeddings_available);

        let model_arc = Arc::new(model);
        let system_prompt = SystemPromptManager::new();

        let executor = Arc::new(Executor::new(
            model_arc.clone(),
            embeddings_available,
            system_prompt.clone(),
            persona_memory.clone(),
            tool_registry.clone(),
        ));

        Ok(Self {
            model: model_arc,
            embeddings_available,
            llms,
            storage: Arc::new(Mutex::new(storage)),
            store: Arc::new(Mutex::new(vector_store)),
            system_prompt,
            persona_memory,
            tools: tool_registry,
            executor,
        })
    }
}