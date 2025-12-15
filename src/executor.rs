use std::sync::Arc;
use crate::llm::{LlmModelTrait, GenerateOptions, ChatMessage};
use crate::persona_memory::{IntelligentMemory, MemoryConfig, MemoryType, MemoryMode, MemoryPolicy};
use crate::system_prompt::SystemPromptManager;
use crate::model::Model;
use crate::toolport::{ToolRegistry, parse_tool_call};
use futures::Stream;
use serde_json;

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
    /// Reference to tool registry (EXECUTION AUTHORITY)
    tool_registry: Arc<ToolRegistry>,
}

impl Executor {
    /// Check if text looks like JSON (starts with { or ```json)
    fn looks_like_json(text: &str) -> bool {
        let trimmed = text.trim();
        trimmed.starts_with('{') || trimmed.starts_with("```json") || trimmed.starts_with("```")
    }

    /// Build enhanced system prompt that includes available tools
    fn build_enhanced_system_prompt(&self, override_prompt: Option<&str>) -> Option<String> {
        if let Some(prompt) = override_prompt {
            // If there's an override, enhance it with tool information
            let available_tools = self.tool_registry.list_tools();
            if available_tools.is_empty() {
                return Some(prompt.to_string());
            }

            let tools_list = available_tools.join(", ");
            let enhanced = format!(
                "{}\n\nAvailable tools: {}",
                prompt, tools_list
            );
            Some(enhanced)
        } else {
            // No override, check if we should add tool information to default prompt
            let available_tools = self.tool_registry.list_tools();
            if available_tools.is_empty() {
                None
            } else {
                let tools_list = available_tools.join(", ");
                let enhanced = format!(
                    "You are running inside the llmd runtime.\n\nYou do NOT execute tools yourself.\nYou only decide WHETHER a tool should be used.\n\nIf a user request requires an external capability, you MUST respond with a single JSON object describing a tool call.\nIf no tool is needed, respond normally in natural language.\n\nWhen calling a tool, respond with ONLY valid JSON.\nDo NOT include explanations, prose, markdown, or code fences.\n\nThe JSON MUST have this exact shape:\n{{\n  \"type\": \"tool_call\",\n  \"name\": \"<tool_name>\",\n  \"arguments\": {{ ... }}\n}}\n\nAvailable tools: {}\n\nRules:\n- Output must be valid JSON\n- No trailing text\n- No partial JSON\n- No extra fields\n- \"type\" MUST equal \"tool_call\"\n\nIf unsure, DO NOT call a tool.",
                    tools_list
                );
                Some(enhanced)
            }
        }
    }

    /// Strip ChatML artifacts from response
    fn strip_chatml(text: &str) -> String {
        text.replace("<|im_start|>", "").replace("<|im_end|>", "").trim().to_string()
    }

    /// Validate that response is safe for memory and user
    fn validate_response(&self, response: &str, is_tool_result: bool) -> bool {
        // Tool results are always safe (they come from validated tool execution)
        if is_tool_result {
            return true;
        }

        // Plain text responses: reject if they look like JSON but aren't valid tool calls
        if Self::looks_like_json(response) {
            // If it looks like JSON but doesn't parse as a valid tool call, it's unsafe
            return parse_tool_call(response).is_some();
        }

        // Plain text is safe
        true
    }
    pub fn new(
        embedding_model: Arc<Model>,
        embeddings_available: bool,
        system_prompt: SystemPromptManager,
        persona_memory: Arc<std::sync::Mutex<IntelligentMemory>>,
        tool_registry: Arc<ToolRegistry>,
    ) -> Self {
        Self {
            embedding_model,
            embeddings_available,
            system_prompt,
            persona_memory,
            tool_registry,
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

        // Build enhanced system prompt with available tools listed
        let enhanced_system_prompt = self.build_enhanced_system_prompt(system_prompt_override);

        // Build full messages with memory
        let full_messages = self.system_prompt.build_chat_messages(
            &messages,
            enhanced_system_prompt.as_deref(),
            persona,
            Some(&memory_context),
        );

        tracing::info!(
            "🧠 FINAL messages sent to LLM:\n{}",
            serde_json::to_string_pretty(&full_messages).unwrap()
        );

        // Execute inference
        let mut opts = options.clone();
        opts.messages = full_messages;
        let reply = llm.generate_with_options(opts).await?;

        // Parse model output for tool calls
        if let Some(tool_call) = parse_tool_call(&reply) {
            // Tool call detected - validate against whitelist
            tracing::info!("Tool call detected: {} with arguments {:?}", tool_call.name, tool_call.arguments);

            // Tool execution whitelisting: only execute registered tools
            if let Some(tool) = self.tool_registry.get(&tool_call.name) {
                // Tool is whitelisted, execute it
                let tool_input = crate::toolport::ToolInput {
                    payload: tool_call.arguments.args,
                    metadata: crate::toolport::ToolMetadata {
                        tool_name: tool_call.name.clone(),
                    },
                };

                tracing::info!("Executing whitelisted tool: {}", tool_call.name);
                match tool.execute(tool_input) {
                    Ok(tool_output) => {
                        tracing::info!("Tool execution succeeded: {}", tool_call.name);
                        // Tool results are always safe - strip ChatML and return
                        let result = serde_json::to_string_pretty(&tool_output.payload)
                            .unwrap_or_else(|_| "{\"error\":\"invalid tool output\"}".to_string());
                        let safe_result = Self::strip_chatml(&result);
                        return Ok(safe_result);
                    }
                    Err(tool_error) => {
                        tracing::warn!("Tool execution failed: {} - {:?}", tool_call.name, tool_error);
                        // Jail retry for failed tool execution
                        let jail_reply = self.execute_jail_retry(llm, &messages, persona, &options).await?;
                        let safe_reply = Self::strip_chatml(&jail_reply);
                        return Ok(safe_reply);
                    }
                }
            } else {
                // Tool not whitelisted - reject and jail retry
                tracing::warn!("Rejected unknown tool call: {} (not in registry)", tool_call.name);
                let jail_reply = self.execute_jail_retry(llm, &messages, persona, &options).await?;
                let safe_reply = Self::strip_chatml(&jail_reply);
                return Ok(safe_reply);
            }
        }

        // No tool call detected - validate response safety
        if Self::looks_like_json(&reply) {
            // JSON-like output that's not a valid tool call is unsafe - jail retry
            tracing::warn!("Rejected unsafe JSON output (not a valid tool call)");
            let jail_reply = self.execute_jail_retry(llm, &messages, persona, &options).await?;
            let safe_reply = Self::strip_chatml(&jail_reply);
            return Ok(safe_reply);
        }

        // Plain text response - validate and return
        let safe_reply = Self::strip_chatml(&reply);
        if self.validate_response(&safe_reply, false) {
            // Safe response - update memory
            if !matches!(memory_update, Some("disable")) {
                self.update_memory(persona_name, &messages, &safe_reply, &memory_config).await;
            }
            Ok(safe_reply)
        } else {
            // Unexpected unsafe response - fallback
            tracing::warn!("Rejected unsafe plain text response");
            Ok("I apologize, but I encountered an error processing your request.".to_string())
        }
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

    /// Execute a jail retry when tool hallucination or invalid JSON is detected.
    /// Forces the model to respond in plain natural language without tool calls or JSON.
    async fn execute_jail_retry(
        &self,
        llm: Arc<dyn LlmModelTrait>,
        messages: &[ChatMessage],
        persona: Option<&str>,
        options: &GenerateOptions,
    ) -> anyhow::Result<String> {
        tracing::info!("🔒 Executing tool hallucination jail retry");

        // Use strict jail prompt to force plain text response
        let jail_prompt = "Respond ONLY in plain text. Do NOT output JSON.";

        // Build jail messages with strict system prompt override
        let jail_messages = self.system_prompt.build_chat_messages(
            messages,
            Some(jail_prompt),
            persona,
            None, // No memory context for jail retry
        );

        // Execute jail inference
        let mut jail_opts = options.clone();
        jail_opts.messages = jail_messages;
        let jail_reply = llm.generate_with_options(jail_opts).await?;

        // Check if jail response still contains tool calls or looks like JSON (violation)
        if parse_tool_call(&jail_reply).is_some() || Self::looks_like_json(&jail_reply) {
            tracing::warn!("🚨 Jail violation: Model still output tool call or JSON in jail retry");
            // Return fixed fallback message
            return Ok("I apologize, but I encountered an error processing your request. Please try rephrasing your question.".to_string());
        }

        // Jail retry succeeded - return plain text response
        tracing::info!("✅ Jail retry successful: Model responded with plain text");
        Ok(jail_reply)
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
