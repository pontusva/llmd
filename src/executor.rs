use std::sync::Arc;
use crate::llm::{LlmModelTrait, GenerateOptions, ChatMessage};
use crate::persona_memory::{IntelligentMemory, MemoryConfig, MemoryType, MemoryMode, MemoryPolicy};
use crate::system_prompt::SystemPromptManager;
use crate::model::Model;
use futures::Stream;

/// Executor owns execution authority and agent state mutations.
/// It decides when inference runs and is the only component that mutates memory state.
pub struct Executor {
    /// Reference to embedding model (for memory operations)
    embedding_model: Arc<Model>,
    /// Whether embeddings are available
    embeddings_available: bool,
    /// Reference to system prompt manager
    system_prompt: SystemPromptManager,
    /// Reference to persona memory (MUTATION AUTHORITY)
    persona_memory: Arc<std::sync::Mutex<IntelligentMemory>>,
}

impl Executor {
    pub fn new(
        embedding_model: Arc<Model>,
        embeddings_available: bool,
        system_prompt: SystemPromptManager,
        persona_memory: Arc<std::sync::Mutex<IntelligentMemory>>,
    ) -> Self {
        Self {
            embedding_model,
            embeddings_available,
            system_prompt,
            persona_memory,
        }
    }

    /// Execute a chat completion request (non-streaming).
    /// Returns the generated response and updates memory if enabled.
    pub async fn execute_chat(
        &self,
        llm: Arc<dyn LlmModelTrait>,
        messages: Vec<ChatMessage>,
        persona: Option<&str>,
        system_prompt_override: Option<&str>,
        memory_update: Option<&str>,
        options: GenerateOptions,
    ) -> anyhow::Result<String> {
        let persona_name = persona.unwrap_or("default");

        // Build memory config from request
        let memory_config = self.build_memory_config(memory_update);

        // Get memory context
        let memory_context = self.retrieve_memory_context(&messages, persona_name, &memory_config).await;

        // Build full messages with memory
        let full_messages = self.system_prompt.build_chat_messages(
            &messages,
            system_prompt_override,
            persona,
            Some(&memory_context),
        );

        // Execute inference
        let mut opts = options;
        opts.messages = full_messages;
        let reply = llm.generate_with_options(opts).await?;

        // Update memory (this is the ONLY place memory gets mutated)
        if !matches!(memory_update, Some("disable")) {
            self.update_memory(persona_name, &messages, &reply, &memory_config).await;
        }

        Ok(reply)
    }

    /// Execute a streaming chat completion request.
    /// Returns a token stream and handles memory updates when streaming completes.
    pub async fn execute_stream(
        &self,
        llm: Arc<dyn LlmModelTrait>,
        messages: Vec<ChatMessage>,
        persona: Option<&str>,
        system_prompt_override: Option<&str>,
        memory_update: Option<&str>,
        options: GenerateOptions,
    ) -> anyhow::Result<(Box<dyn Stream<Item = String> + Send + Unpin>, MemoryUpdateTask)> {
        let persona_name = persona.unwrap_or("default").to_string();

        // Build memory config from request
        let memory_config = self.build_memory_config(memory_update);

        // Get memory context
        let memory_context = self.retrieve_memory_context(&messages, &persona_name, &memory_config).await;

        // Build full messages with memory
        let full_messages = self.system_prompt.build_chat_messages(
            &messages,
            system_prompt_override,
            Some(&persona_name),
            Some(&memory_context),
        );

        // Execute streaming inference
        let mut opts = options;
        opts.messages = full_messages;
        let token_stream = llm.stream_generate(opts).await?;

        // Create memory update task for when streaming completes
        let memory_task = if !matches!(memory_update, Some("disable")) {
            let messages_clone = messages.clone();
            let memory_config_clone = memory_config.clone();
            MemoryUpdateTask::Enabled {
                persona_name,
                messages: messages_clone,
                memory_config: memory_config_clone,
            }
        } else {
            MemoryUpdateTask::Disabled
        };

        Ok((Box::new(token_stream), memory_task))
    }

    /// Complete a streaming execution by updating memory.
    /// This is called after the stream finishes to perform memory mutations.
    pub async fn complete_stream(&self, memory_task: MemoryUpdateTask, collected_response: &str) {
        match memory_task {
            MemoryUpdateTask::Enabled { persona_name, messages, memory_config } => {
                self.update_memory(&persona_name, &messages, collected_response, &memory_config).await;
            }
            MemoryUpdateTask::Disabled => {
                // No memory update needed
            }
        }
    }

    // Helper methods (private - not part of public API)

    fn build_memory_config(&self, memory_update: Option<&str>) -> MemoryConfig {
        MemoryConfig {
            mode: if matches!(memory_update, Some("disable")) {
                MemoryMode::Read
            } else {
                MemoryMode::ReadWrite
            },
            policy: MemoryPolicy::Auto,
            debug: false,
            vector_threshold: 0.78,
            vector_top_k: 3,
            vector_types: vec![MemoryType::Persona, MemoryType::Conversation],
        }
    }

    async fn retrieve_memory_context(
        &self,
        messages: &[ChatMessage],
        persona_name: &str,
        memory_config: &MemoryConfig,
    ) -> String {
        // Extract last user message for memory retrieval
        let last_user_msg = messages.last()
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "".to_string());

        // Compute embedding if needed
        let embedding = if !memory_config.vector_types.is_empty() && self.embeddings_available {
            match self.embedding_model.infer(&last_user_msg).await {
                Ok(emb) => Some(emb),
                Err(e) => {
                    if memory_config.debug {
                        println!("[MEMORY] Failed to compute embedding: {}", e);
                    }
                    None
                }
            }
        } else {
            None
        };

        let pm = self.persona_memory.lock().unwrap();

        // Get keyword-based memory
        let keyword_memory = pm.build_retrieved_memory_context(persona_name, &last_user_msg, memory_config)
            .unwrap_or_default();

        // Get vector-based memory
        let vector_memory = if let Some(ref emb) = embedding {
            pm.build_vector_memory_context(persona_name, emb, memory_config)
                .unwrap_or_default()
        } else {
            String::new()
        };

        // Combine memories
        if keyword_memory.is_empty() && vector_memory.is_empty() {
            String::new()
        } else if keyword_memory.is_empty() {
            vector_memory
        } else if vector_memory.is_empty() {
            keyword_memory
        } else {
            format!("{}\n\n{}", keyword_memory, vector_memory)
        }
    }

    async fn update_memory(
        &self,
        persona_name: &str,
        messages: &[ChatMessage],
        response: &str,
        memory_config: &MemoryConfig,
    ) {
        // Extract last user message for memory storage
        let last_user_msg = messages.last()
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "".to_string());

        // Compute embedding from user's input (not AI response)
        let embedding = if self.embeddings_available {
            match self.embedding_model.infer(&last_user_msg).await {
                Ok(emb) => Some(emb),
                Err(_e) => None, // Fail silently for memory updates
            }
        } else {
            None
        };

        // Update memory with embedding
        let _ = self.persona_memory.lock().unwrap()
            .update_memory_with_embedding_sync(
                MemoryType::Conversation,
                persona_name,
                &last_user_msg,
                response,
                memory_config,
                embedding.as_deref()
            );
    }
}

/// Memory update task for streaming completions.
/// Created during stream execution, executed after stream completion.
pub enum MemoryUpdateTask {
    Enabled {
        persona_name: String,
        messages: Vec<ChatMessage>,
        memory_config: MemoryConfig,
    },
    Disabled,
}
