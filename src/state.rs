use crate::model::Model;
use std::sync::Arc;
use crate::vector_store::VectorStore;
use std::sync::Mutex;
use crate::storage::Storage;
use crate::system_prompt::SystemPromptManager;
use crate::persona_memory::IntelligentMemory;
use crate::llm_registry::LlmRegistry;
use crate::llm::LlmModelTrait;
use crate::executor::Executor;


#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Model>,
    pub embeddings_available: bool,
    pub llms: Arc<LlmRegistry>,
    pub storage: Arc<Mutex<Storage>>,
    pub store: Arc<Mutex<VectorStore>>,
    pub system_prompt: SystemPromptManager,
    pub persona_memory: Arc<Mutex<IntelligentMemory>>,
    /// Executor owns execution authority and agent state mutations
    pub executor: Arc<Executor>,
}

use model_loader_core::plan::{LoadPlan, LoadStep};

impl AppState {
    pub async fn init_from_plan(plan: LoadPlan) -> Result<Self, anyhow::Error> {
        // Execute the LoadPlan to get an LLM instance
        let (llm_model, model_id) = Self::execute_plan(plan).await?;

        // Create registry and register the loaded model
        let mut registry = LlmRegistry::new();
        registry.register(model_id.clone(), llm_model)?;
        tracing::info!("Registered model: {}", model_id);

        // Continue with rest of initialization using the registry
        Self::init_with_registry(Arc::new(registry)).await
    }

    async fn execute_plan(plan: LoadPlan) -> anyhow::Result<(Arc<dyn LlmModelTrait>, String)> {
        tracing::info!("Executing LoadPlan with {} steps", plan.steps.len());

        // Extract model directory from LoadPlan steps
        let model_dir = Self::extract_model_dir_from_plan(&plan)?;
        let model_id = Self::extract_model_id_from_dir(&model_dir);

        tracing::info!("Loading model from directory: {}", model_dir);

        // Load real CandleLlm
        let candle_llm = crate::llm::CandleLlm::new(&model_dir)?;
        let model = Arc::new(candle_llm);

        tracing::info!("Successfully loaded CandleLlm: {}", model_id);

        Ok((model, model_id))
    }

    fn extract_model_dir_from_plan(plan: &LoadPlan) -> anyhow::Result<String> {
        // Find the first path in LoadPlan steps and extract directory
        for step in &plan.steps {
            match step {
                LoadStep::LoadConfig { path } |
                LoadStep::LoadTokenizer { path } |
                LoadStep::LoadShard { path, .. } => {
                    // Extract directory from path (e.g., "models/test/config.json" -> "models/test")
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
        // Extract model name from directory path (e.g., "models/test" -> "test")
        std::path::Path::new(model_dir)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown-model")
            .to_string()
    }

    async fn init_with_registry(llms: Arc<LlmRegistry>) -> anyhow::Result<Self> {
        // Try to load embedding model
        let (model, embeddings_available) = match Model::load("models/minilm").await {
            Ok(m) => {
                // Test if embeddings work
                let test_result = m.infer("hello world").await;
                match test_result {
                    Ok(_) => (m, true),
                    Err(e) => {
                        tracing::warn!("Embedding model loaded but inference failed: {}", e);
                        tracing::warn!("Vector memory features will be disabled.");
                        (Model::dummy(), false)
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Could not load embedding model from models/minilm: {}", e);
                tracing::warn!("Vector memory features will be disabled. Using keyword-based memory only.");
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
                match model.infer("dimension probe").await {
                    Ok(emb) => emb.len(),
                    Err(_) => 384, // Default BERT dimension
                }
            }
        } else {
            384 // Default dimension when embeddings not available
        };

        let mut vector_store = VectorStore::new(dim, Storage::new(db_path)?);
        if embeddings_available {
            vector_store.rebuild_index();
        }

        let persona_memory = Arc::new(Mutex::new(IntelligentMemory::new(db_path)?));

        // Debug: show embedding status
        eprintln!("Vector memory: embeddings_available = {}", embeddings_available);

        let model_arc = Arc::new(model);
        let executor = Arc::new(Executor::new(
            model_arc.clone(),
            embeddings_available,
            SystemPromptManager::new(),
            persona_memory.clone(),
        ));

        Ok(Self {
            model: model_arc,
            embeddings_available,
            llms,
            storage: Arc::new(Mutex::new(storage)),
            store: Arc::new(Mutex::new(vector_store)),
            system_prompt: SystemPromptManager::new(),
            persona_memory,
            executor,
        })
    }
}


impl AppState {
    pub async fn init() -> anyhow::Result<Self> {
        // Try to load embedding model
        let (model, embeddings_available) = match Model::load("models/minilm").await {
            Ok(m) => {
                // Test if embeddings work
                let test_result = m.infer("hello world").await;
                match test_result {
                    Ok(_) => (m, true),
                    Err(e) => {
                        tracing::warn!("Embedding model loaded but inference failed: {}", e);
                        tracing::warn!("Vector memory features will be disabled.");
                        (Model::dummy(), false)
                    }
                }
            }
            Err(e) => {
                tracing::warn!("Could not load embedding model from models/minilm: {}", e);
                tracing::warn!("Vector memory features will be disabled. Using keyword-based memory only.");
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
                match model.infer("dimension probe").await {
                    Ok(emb) => emb.len(),
                    Err(_) => 384, // Default BERT dimension
                }
            }
        } else {
            384 // Default dimension when embeddings not available
        };

        let mut vector_store = VectorStore::new(dim, Storage::new(db_path)?);
        if embeddings_available {
            vector_store.rebuild_index();
        }

        let llms = Arc::new(LlmRegistry::discover_models("models/llm").await?);
        let persona_memory = Arc::new(Mutex::new(IntelligentMemory::new(db_path)?));

        // Debug: show embedding status
        eprintln!("Vector memory: embeddings_available = {}", embeddings_available);

        let model_arc = Arc::new(model);
        let executor = Arc::new(Executor::new(
            model_arc.clone(),
            embeddings_available,
            SystemPromptManager::new(),
            persona_memory.clone(),
        ));

        Ok(Self {
            model: model_arc,
            embeddings_available,
            llms,
            storage: Arc::new(Mutex::new(storage)),
            store: Arc::new(Mutex::new(vector_store)),
            system_prompt: SystemPromptManager::new(),
            persona_memory,
            executor,
        })
    }

    #[allow(dead_code)]
    pub fn new_with_llms(
        model: Model,
        embeddings_available: bool,
        llms: LlmRegistry,
        storage: Storage,
        vector_store: VectorStore,
        system_prompt: SystemPromptManager,
        persona_memory: IntelligentMemory,
    ) -> Self {
        let model_arc = Arc::new(model);
        let persona_memory_arc = Arc::new(Mutex::new(persona_memory));
        let executor = Arc::new(Executor::new(
            model_arc.clone(),
            embeddings_available,
            system_prompt.clone(),
            persona_memory_arc.clone(),
        ));

        Self {
            model: model_arc,
            embeddings_available,
            llms: Arc::new(llms),
            storage: Arc::new(Mutex::new(storage)),
            store: Arc::new(Mutex::new(vector_store)),
            system_prompt,
            persona_memory: persona_memory_arc,
            executor,
        }
    }
}