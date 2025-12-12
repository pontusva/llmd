use crate::model::Model;
use std::sync::Arc;
use crate::vector_store::VectorStore;
use std::sync::Mutex;
use crate::storage::Storage;
use crate::system_prompt::SystemPromptManager;
use crate::persona_memory::IntelligentMemory;
use crate::llm_registry::LlmRegistry;


#[derive(Clone)]
pub struct AppState {
    pub model: Arc<Model>,
    pub embeddings_available: bool,
    pub llms: Arc<LlmRegistry>,
    pub storage: Arc<Mutex<Storage>>,
    pub store: Arc<Mutex<VectorStore>>,
    pub system_prompt: SystemPromptManager,
    pub persona_memory: Arc<Mutex<IntelligentMemory>>,
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
                        eprintln!("Warning: Embedding model loaded but inference failed: {}", e);
                        eprintln!("Vector memory features will be disabled.");
                        (m, false)
                    }
                }
            }
            Err(e) => {
                eprintln!("Warning: Could not load embedding model from models/minilm: {}", e);
                eprintln!("Vector memory features will be disabled. Using keyword-based memory only.");
                // Create a dummy model that will fail gracefully
                // For now, we'll panic here since the system expects a model
                return Err(e);
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

        Ok(Self {
            model: Arc::new(model),
            embeddings_available,
            llms,
            storage: Arc::new(Mutex::new(storage)),
            store: Arc::new(Mutex::new(vector_store)),
            system_prompt: SystemPromptManager::new(),
            persona_memory,
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
        Self {
            model: Arc::new(model),
            embeddings_available,
            llms: Arc::new(llms),
            storage: Arc::new(Mutex::new(storage)),
            store: Arc::new(Mutex::new(vector_store)),
            system_prompt,
            persona_memory: Arc::new(Mutex::new(persona_memory)),
        }
    }
}