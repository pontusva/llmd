use std::sync::Arc;
use crate::memory::embedding_observability::EmbeddingStats;
use crate::llm::{LlmModelTrait, GenerateOptions, ChatMessage};
use crate::memory::persona_memory::{IntelligentMemory, MemoryConfig, MemoryType, MemoryMode, MemoryPolicy};
use crate::llm::system_prompt::SystemPromptManager;
use crate::core::model::Model;
use crate::runtime::toolport::{ToolRegistry, parse_tool_call, ToolEligibilityContext};
use crate::memory::embedding_decision::{EmbeddingDecisionMatrix, MemoryEventKind, DecisionContext, DecisionResult};
use crate::tools::graphql::NameResolutionRegistry;
use futures::Stream;
use serde_json;

/// Prompt role determines which type of system prompt to use
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PromptRole {
    IntentCompiler,
    Conversational,
}

/// Static intent compiler prompt - immutable and tool-agnostic
pub fn intent_compiler_prompt() -> &'static str {
    include_str!("../prompts/intent_compiler.txt")
}

/// Executor context containing per-request dependencies
pub struct ExecutorContext {
    pub name_registry: Arc<dyn NameResolutionRegistry + Send + Sync>,
}

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
    /// Name resolution registry for building/real estate names
    name_registry: Arc<dyn NameResolutionRegistry + Send + Sync>,
    /// Embedding observability and metrics
    embedding_stats: Arc<EmbeddingStats>,
}

impl Executor {
    /// Check if text looks like JSON (starts with { or ```json)
    fn looks_like_json(text: &str) -> bool {
        let trimmed = text.trim();
        trimmed.starts_with('{') || trimmed.starts_with("```json") || trimmed.starts_with("```")
    }



    /// Tool eligibility gate - determines if a specific tool call is appropriate for the conversation
    fn is_tool_eligible(tool_name: &str, user_text: &str, assistant_text: &str) -> bool {
        match tool_name {
            "echo" => Self::is_echo_tool_eligible(user_text),
            // Add other tool-specific eligibility checks here as needed
            _ => true, // Other tools are eligible by default
        }
    }

    /// Check if echo tool usage is appropriate for the given user input
    fn is_echo_tool_eligible(user_text: &str) -> bool {
        let trimmed = user_text.trim().to_lowercase();

        // Echo is ONLY eligible if user explicitly requests tool usage
        const EXPLICIT_REQUESTS: &[&str] = &[
            "use the echo tool",
            "call echo",
            "echo this",
            "echo:",
        ];

        for &request in EXPLICIT_REQUESTS {
            if trimmed.contains(request) {
                return true;
            }
        }

        false
    }

    /// Check if a tool was explicitly requested in the user message
    fn is_tool_explicitly_requested(tool_name: &str, user_text: &str) -> bool {
        match tool_name {
            "echo" => Self::is_echo_tool_eligible(user_text),
            // Add other tools here as needed
            _ => false, // Default to not explicitly requested for unknown tools
        }
    }

    /// Validate echo tool arguments - must be {"message": "<exact user-provided text>"}
    fn validate_echo_tool_arguments(arguments: &serde_json::Value) -> bool {
        // Must be an object
        let obj = match arguments.as_object() {
            Some(obj) => obj,
            None => return false,
        };

        // Must have exactly one key: "message"
        if obj.len() != 1 {
            return false;
        }

        // Must have "message" key
        let message_value = match obj.get("message") {
            Some(val) => val,
            None => return false,
        };

        // Must be a non-empty string
        match message_value.as_str() {
            Some(s) => !s.trim().is_empty(),
            None => false,
        }
    }

    /// Create a DecisionContext for the embedding decision matrix
    fn create_decision_context<'a>(
        &self,
        persona: &'a str,
        memory_update: Option<&'a str>,
        user_text: &'a str,
        assistant_text: &'a str,
        is_streaming: bool,
    ) -> DecisionContext<'a> {
        DecisionContext {
            persona,
            memory_update,
            embeddings_available: self.embeddings_available,
            has_vector_types: !self.build_memory_config(memory_update).vector_types.is_empty(),
            user_text,
            assistant_text,
            is_streaming,
        }
    }

    /// Build enhanced system prompt that includes available tools
    fn build_chat_system_prompt(&self, override_prompt: Option<&str>) -> Option<String> {
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
            // For chat mode, provide enhanced prompt with tools
            let available_tools = self.tool_registry.list_tools();
            if available_tools.is_empty() {
                None
            } else {
                let tools_list = available_tools.join(", ");
                let enhanced = format!(
                    "You are a helpful assistant with access to tools.\n\nAvailable tools: {}",
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
        name_registry: Arc<dyn NameResolutionRegistry + Send + Sync>,
        embedding_stats: Arc<EmbeddingStats>,
    ) -> Self {
        Self {
            embedding_model,
            embeddings_available,
            system_prompt,
            persona_memory,
            tool_registry,
            name_registry,
            embedding_stats,
        }
    }

    /// Execute a chat completion request (non-streaming).
    /// Returns the generated response and updates memory if enabled.
    pub async fn execute_chat(
        &self,
        llm: Arc<dyn LlmModelTrait>,
        messages: Vec<ChatMessage>,
        persona: Option<&str>,
        prompt_role: PromptRole,
        system_prompt_override: Option<&str>,
        memory_update: Option<&str>,
        options: GenerateOptions,
        allow_tools: bool,
    ) -> anyhow::Result<String> {
        let persona_name = persona.unwrap_or("default");

        // Extract last user message for decision making
        let last_user_msg = messages.last()
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "".to_string());

        // Build memory config from request
        let memory_config = self.build_memory_config(memory_update);

        // Make decision for user message
        let user_decision_ctx = self.create_decision_context(
            persona_name,
            memory_update,
            &last_user_msg,
            "", // no assistant text yet
            false, // not streaming
        );
        let user_decision = EmbeddingDecisionMatrix::decide(
            MemoryEventKind::UserMessage,
            &user_decision_ctx,
            &memory_config,
            &self.embedding_stats,
        );

        // Log decision if debug enabled
        if memory_config.debug {
            tracing::info!("🧠 User message decision: {} ({})", user_decision.reason, user_decision.tags.join(","));
        }

        // Get memory context (respect user input embedding decision)
        let memory_context = self.retrieve_memory_context_with_decision(&messages, persona_name, &memory_config, &user_decision).await;

        // Build system prompt based on role
        let system_prompt = match prompt_role {
            PromptRole::IntentCompiler => {
                // Compiler prompt is immutable and tool-agnostic
                debug_assert!(
                    !intent_compiler_prompt().contains("Available tools"),
                    "Compiler prompt must not contain tool information"
                );
                Some(intent_compiler_prompt().to_string())
            }
            PromptRole::Conversational => {
                // Chat prompts can be enhanced with tool information
                self.build_chat_system_prompt(system_prompt_override)
            }
        };

        // Build full messages with memory
        let full_messages = self.system_prompt.build_chat_messages(
            &messages,
            system_prompt.as_deref(),
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
        if allow_tools {
            if let Some(tool_call) = parse_tool_call(&reply) {
            // Tool call detected - validate tool registry and eligibility
            tracing::info!("Tool call detected: {} with arguments {:?}", tool_call.name, tool_call.arguments);

            // Tool execution whitelisting: only execute registered tools
            let tool = match self.tool_registry.get(&tool_call.name) {
                Some(t) => t,
                None => {
                    // Unknown tool → jail retry (hallucinated tool name)
                    tracing::warn!("Rejected unknown tool call: {} (not in registry)", tool_call.name);
                    let jail_reply = self.execute_jail_retry(llm, &messages, persona, &options).await?;
                    let safe_reply = Self::strip_chatml(&jail_reply);
                    return Ok(safe_reply);
                }
            };

            // Parse Intent for eligibility checking (only for query_intent tool)
            let parsed_intent = if tool_call.name == "query_intent" {
                // Parse the intent from tool arguments
                match crate::tools::graphql::IntentQueryTool::parse_intent(&tool_call.arguments.args) {
                    Ok(mut intent) => {
                        // EXECUTOR GUARDRAIL: Validate intent shape BEFORE any processing
                        // Extract the intent value from the payload and validate it
                        let payload_obj = match tool_call.arguments.args.as_object() {
                            Some(obj) => obj,
                            None => {
                                tracing::warn!("Invalid tool arguments for '{}' - not a JSON object", tool_call.name);
                                return Ok(serde_json::json!({
                                    "error": "Invalid tool arguments",
                                    "details": "Tool arguments must be a JSON object",
                                    "tool": tool_call.name
                                }).to_string());
                            }
                        };

                        let intent_value = match payload_obj.get("intent") {
                            Some(intent) => intent,
                            None => {
                                tracing::warn!("Missing 'intent' field in tool arguments for '{}'", tool_call.name);
                                return Ok(serde_json::json!({
                                    "error": "Invalid tool arguments",
                                    "details": "Missing required 'intent' field in tool arguments",
                                    "tool": tool_call.name
                                }).to_string());
                            }
                        };

                        if let Err(e) = crate::tools::graphql::IntentQueryTool::validate_intent_shape(intent_value) {
                            tracing::warn!(
                                "Intent shape validation failed for tool call '{}' - rejecting: {}",
                                tool_call.name, e
                            );
                            return Ok(serde_json::json!({
                                "error": "Invalid intent structure",
                                "details": format!("{}", e),
                                "tool": tool_call.name
                            }).to_string());
                        }

                        // Apply time and status filter normalization from user message
                        crate::tools::graphql::IntentQueryTool::normalize_time_and_status_filters(
                            &mut intent,
                            &last_user_msg
                        );

                        // Normalize and lower intent from logical to physical representation
                        if let Err(e) = crate::tools::graphql::intent_lowering::normalize_intent(&mut intent, &*self.name_registry) {
                            tracing::warn!(
                                "Lowered intent validation failed for tool call '{}' - rejecting: {}",
                                tool_call.name, e
                            );
                            return Ok(serde_json::json!({
                                "error": "Invalid lowered intent",
                                "details": format!("{}", e),
                                "tool": tool_call.name
                            }).to_string());
                        }

                        Some(intent)
                    },
                    Err(_) => {
                        tracing::warn!(
                            "Invalid intent payload in tool call '{}' - rejecting without jail retry",
                            tool_call.name
                        );
                        // Invalid intent = hard stop, no jail retry for malformed intents
                        let safe_reply = Self::strip_chatml(&reply);
                        return Ok(safe_reply);
                    }
                }
            } else {
                None // Other tools don't use Intent
            };

            // Debug log filters before eligibility check
            if let Some(ref intent) = parsed_intent {
                if let Some(ref filters) = intent.filters {
                    tracing::info!("🔍 Intent filters before eligibility: {:?}", filters);
                }
            }

            // Check tool eligibility FIRST - this is the single source of truth for intent
            // EXCEPTION: query_intent is the compiler output channel, not a user tool
            // It is implicitly eligible if the intent passes validation + normalization
            // This ensures the LLM can always emit valid intents without explicit user requests
            let is_eligible = if tool_call.name == "query_intent" {
                // Compiler tool is always eligible (intent validation happens elsewhere)
                true
            } else {
                let explicitly_requested = Self::is_tool_explicitly_requested(&tool_call.name, &last_user_msg);
                let eligibility_ctx = ToolEligibilityContext {
                    user_message: &last_user_msg,
                    assistant_message: &reply,
                    explicitly_requested,
                    persona: persona_name,
                    intent: parsed_intent.as_ref(),
                };
                tool.is_eligible(&eligibility_ctx)
            };

            if !is_eligible {
                tracing::info!(
                    "Tool call ignored: '{}' not eligible for message '{}' - treating as plain text",
                    tool_call.name,
                    last_user_msg
                );
                // Tool call not eligible - treat model output as plain text, no jail retry
                // Fall through to plain text processing below
            } else {
                // Tool is eligible - proceed directly to validation and execution
                // NO further intent checks by executor - ToolEligibility is authoritative

                // Additional validation for echo tool arguments
                if tool_call.name == "echo" && !Self::validate_echo_tool_arguments(&tool_call.arguments.args) {
                    tracing::warn!("Echo tool rejected: invalid arguments {:?}", tool_call.arguments.args);
                    let jail_reply = self.execute_jail_retry(llm, &messages, persona, &options).await?;
                    let safe_reply = Self::strip_chatml(&jail_reply);
                    return Ok(safe_reply);
                }

                // Tool is eligible and arguments are valid, execute it
                let tool_input = crate::runtime::toolport::ToolInput {
                    payload: tool_call.arguments.args,
                    metadata: crate::runtime::toolport::ToolMetadata {
                        tool_name: tool_call.name.clone(),
                    },
                    user_message: last_user_msg.clone(),
                    parsed_intent: if tool_call.name == "query_intent" {
                        parsed_intent.clone() // Pass the normalized intent for query_intent tool
                    } else {
                        None // Other tools don't use pre-parsed intents
                    },
                };

                tracing::info!("Executing eligible tool: {}", tool_call.name);

                // Create executor context with injected name registry
                let executor_ctx = ExecutorContext {
                    name_registry: self.name_registry.clone(),
                };

                match tool.execute(tool_input, &executor_ctx) {
                    Ok(tool_output) => {
                        tracing::info!("Tool execution succeeded: {}", tool_call.name);
                        // Tool results are always safe - strip ChatML and return
                        let result = serde_json::to_string_pretty(&tool_output.payload)
                            .unwrap_or_else(|_| "{\"error\":\"invalid tool output\"}".to_string());
                        let safe_result = Self::strip_chatml(&result);

                        // Make decision for tool result
                        let tool_result_decision_ctx = self.create_decision_context(
                            persona_name,
                            memory_update,
                            &last_user_msg,
                            &safe_result,
                            false,
                        );
                        let tool_result_decision = EmbeddingDecisionMatrix::decide(
                            MemoryEventKind::ToolResult,
                            &tool_result_decision_ctx,
                            &memory_config,
                            &self.embedding_stats,
                        );

                        if memory_config.debug {
                            tracing::info!("🧠 Tool result decision: {} ({})", tool_result_decision.reason, tool_result_decision.tags.join(","));
                        }

                        // Update memory for tool result if decision allows
                        if tool_result_decision.should_store_memory {
                            self.update_memory_with_decision(
                                persona_name,
                                &messages,
                                &safe_result,
                                &memory_config,
                                &user_decision,
                                &tool_result_decision,
                            ).await;
                        }

                        return Ok(safe_result);
                    }
                    Err(tool_error) => {
                        tracing::warn!("Tool execution failed: {} - {:?}", tool_call.name, tool_error);
                        // Jail retry for failed tool execution (actual execution error, not intent)
                        let jail_reply = self.execute_jail_retry(llm, &messages, persona, &options).await?;
                        let safe_reply = Self::strip_chatml(&jail_reply);
                        return Ok(safe_reply);
                    }
                }
            }
        }

        // No tool call detected - validate response safety
        if Self::looks_like_json(&reply) {
            // JSON-like output that's not a valid tool call is unsafe - jail retry
            tracing::warn!("Rejected unsafe JSON output (not a valid tool call)");
            let jail_reply = self.execute_jail_retry(llm, &messages, persona, &options).await?;
            let safe_reply = Self::strip_chatml(&jail_reply);

            // Make decision for jail retry violation (since this was unsafe JSON)
            let jail_violation_decision_ctx = self.create_decision_context(
                persona_name,
                memory_update,
                &last_user_msg,
                &safe_reply,
                false,
            );
            let jail_violation_decision = EmbeddingDecisionMatrix::decide(
                MemoryEventKind::JailRetryViolation,
                &jail_violation_decision_ctx,
                &memory_config,
                &self.embedding_stats,
            );

            if memory_config.debug {
                tracing::info!("🧠 Jail violation decision: {} ({})", jail_violation_decision.reason, jail_violation_decision.tags.join(","));
            }

            // Jail retry violations should not store memory (decision will reflect this)
            if jail_violation_decision.should_store_memory {
                self.update_memory_with_decision(
                    persona_name,
                    &messages,
                    &safe_reply,
                    &memory_config,
                    &user_decision,
                    &jail_violation_decision,
                ).await;
            }

            return Ok(safe_reply);
        }

        // Plain text response - make decision for assistant response
        let safe_reply = Self::strip_chatml(&reply);
        let assistant_decision_ctx = self.create_decision_context(
            persona_name,
            memory_update,
            &last_user_msg,
            &safe_reply,
            false,
        );
        let assistant_decision = EmbeddingDecisionMatrix::decide(
            MemoryEventKind::AssistantPlainText,
            &assistant_decision_ctx,
            &memory_config,
            &self.embedding_stats,
        );

        if memory_config.debug {
            tracing::info!("🧠 Assistant response decision: {} ({})", assistant_decision.reason, assistant_decision.tags.join(","));
        }

        if self.validate_response(&safe_reply, false) {
            // Safe response - update memory if decision allows
            if assistant_decision.should_store_memory {
                self.update_memory_with_decision(
                    persona_name,
                    &messages,
                    &safe_reply,
                    &memory_config,
                    &user_decision,
                    &assistant_decision,
                ).await;
            }
            Ok(safe_reply)
        } else {
            // Unexpected unsafe response - fallback (no memory update)
            tracing::warn!("Rejected unsafe plain text response");
            Ok("I apologize, but I encountered an error processing your request.".to_string())
        }
        } else {
            // Tools disabled - return response as-is if safe
            let safe_reply = Self::strip_chatml(&reply);
            if self.validate_response(&safe_reply, false) {
                Ok(safe_reply)
            } else {
                tracing::warn!("Rejected unsafe plain text response (tools disabled)");
                Ok("I apologize, but I encountered an error processing your request.".to_string())
            }
        }
    }

    /// Execute a streaming chat completion request.
    /// Returns a token stream and handles memory updates when streaming completes.
    pub async fn execute_stream(
        &self,
        llm: Arc<dyn LlmModelTrait>,
        messages: Vec<ChatMessage>,
        persona: Option<&str>,
        prompt_role: PromptRole,
        system_prompt_override: Option<&str>,
        memory_update: Option<&str>,
        options: GenerateOptions,
        allow_tools: bool,
    ) -> anyhow::Result<(Box<dyn Stream<Item = String> + Send + Unpin>, MemoryUpdateTask)> {
        let persona_name = persona.unwrap_or("default").to_string();

        // Extract last user message for decision making
        let last_user_msg = messages.last()
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "".to_string());

        // Build memory config from request
        let memory_config = self.build_memory_config(memory_update);

        // Make decision for user message
        let user_decision_ctx = self.create_decision_context(
            &persona_name,
            memory_update,
            &last_user_msg,
            "", // no assistant text yet
            true, // streaming
        );
        let user_decision = EmbeddingDecisionMatrix::decide(
            MemoryEventKind::UserMessage,
            &user_decision_ctx,
            &memory_config,
            &self.embedding_stats,
        );

        if memory_config.debug {
            tracing::info!("🧠 Stream user decision: {} ({})", user_decision.reason, user_decision.tags.join(","));
        }

        // Get memory context respecting user input embedding decision
        let memory_context = self.retrieve_memory_context_with_decision(&messages, &persona_name, &memory_config, &user_decision).await;

        // Build system prompt based on role
        let system_prompt = match prompt_role {
            PromptRole::IntentCompiler => {
                // Compiler prompt is immutable and tool-agnostic
                debug_assert!(
                    !intent_compiler_prompt().contains("Available tools"),
                    "Compiler prompt must not contain tool information"
                );
                Some(intent_compiler_prompt().to_string())
            }
            PromptRole::Conversational => {
                // Chat prompts can be enhanced with tool information
                self.build_chat_system_prompt(system_prompt_override)
            }
        };

        // Build full messages with memory
        let full_messages = self.system_prompt.build_chat_messages(
            &messages,
            system_prompt.as_deref(),
            Some(&persona_name),
            Some(&memory_context),
        );

        // Execute streaming inference
        let mut opts = options;
        opts.messages = full_messages;
        let token_stream = llm.stream_generate(opts).await?;

        // Create memory update task for when streaming completes
        let memory_task = if !matches!(memory_update, Some("disable")) && user_decision.should_store_memory {
            let messages_clone = messages.clone();
            let memory_config_clone = memory_config.clone();
            MemoryUpdateTask::Enabled {
                persona_name,
                messages: messages_clone,
                memory_config: memory_config_clone,
                user_decision: user_decision.clone(),
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
            MemoryUpdateTask::Enabled { persona_name, messages, memory_config, user_decision } => {
                // Extract last user message for decision making
                let last_user_msg = messages.last()
                    .map(|m| m.content.clone())
                    .unwrap_or_else(|| "".to_string());

                // Make decision for streaming response
                let stream_decision_ctx = self.create_decision_context(
                    &persona_name,
                    None, // memory_update not available in complete_stream
                    &last_user_msg,
                    collected_response,
                    true, // was streaming
                );
                let stream_decision = EmbeddingDecisionMatrix::decide(
                    MemoryEventKind::AssistantPlainText,
                    &stream_decision_ctx,
                    &memory_config,
                    &self.embedding_stats,
                );

                if memory_config.debug {
                    tracing::info!("🧠 Stream completion decision: {} ({})", stream_decision.reason, stream_decision.tags.join(","));
                }

                // Update memory only if streaming decision allows
                if stream_decision.should_store_memory {
                    self.update_memory_with_decision(
                        &persona_name,
                        &messages,
                        collected_response,
                        &memory_config,
                        &user_decision,
                        &stream_decision,
                    ).await;
                }
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

    /// Retrieve memory context respecting embedding decision
    async fn retrieve_memory_context_with_decision(
        &self,
        messages: &[ChatMessage],
        persona_name: &str,
        memory_config: &MemoryConfig,
        user_decision: &DecisionResult,
    ) -> String {
        // Extract last user message for memory retrieval
        let last_user_msg = messages.last()
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "".to_string());

        // Compute embedding only if decision allows
        let embedding = if user_decision.should_embed_input && !memory_config.vector_types.is_empty() && self.embeddings_available {
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

        // Get keyword-based memory (always available for reading)
        let keyword_memory = pm.build_retrieved_memory_context(persona_name, &last_user_msg, memory_config)
            .unwrap_or_default();

        // Get vector-based memory only if embedding was computed
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

    /// Update memory respecting embedding and storage decisions
    async fn update_memory_with_decision(
        &self,
        persona_name: &str,
        messages: &[ChatMessage],
        response: &str,
        memory_config: &MemoryConfig,
        user_decision: &DecisionResult,
        response_decision: &DecisionResult,
    ) {
        // Extract last user message for memory storage
        let last_user_msg = messages.last()
            .map(|m| m.content.clone())
            .unwrap_or_else(|| "".to_string());

        // Compute embedding only if both decisions allow embedding
        let embedding = if user_decision.should_embed_input || response_decision.should_embed_output {
            if self.embeddings_available {
                let start = std::time::Instant::now();
                match self.embedding_model.infer(&last_user_msg).await {
                    Ok(emb) => {
                        let latency_ms = start.elapsed().as_millis() as u64;
                        let vector_dim = emb.len();
                        self.embedding_stats.embedded_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        tracing::info!("🧮 embedding computed: latency_ms={}, vector_dim={}, ttl_seconds={:?}", latency_ms, vector_dim, response_decision.ttl_seconds);
                        Some(emb)
                    }
                    Err(_e) => None, // Fail silently for memory updates
                }
            } else {
                None
            }
        } else {
            None
        };

        // Update memory with the decided memory type and embedding
        let _ = self.persona_memory.lock().unwrap()
            .update_memory_with_embedding_sync(
                response_decision.memory_type,
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
        user_decision: DecisionResult,
    },
    Disabled,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_validate_echo_tool_arguments_valid() {
        let valid_args = json!({"message": "hello world"});
        assert!(Executor::validate_echo_tool_arguments(&valid_args));
    }

    #[test]
    fn test_validate_echo_tool_arguments_wrong_key() {
        let invalid_args = json!({"input": "hello world"});
        assert!(!Executor::validate_echo_tool_arguments(&invalid_args));
    }

    #[test]
    fn test_validate_echo_tool_arguments_empty_object() {
        let invalid_args = json!({});
        assert!(!Executor::validate_echo_tool_arguments(&invalid_args));
    }

    #[test]
    fn test_validate_echo_tool_arguments_wrong_type() {
        let invalid_args = json!({"message": 123});
        assert!(!Executor::validate_echo_tool_arguments(&invalid_args));
    }

    #[test]
    fn test_validate_echo_tool_arguments_empty_string() {
        let invalid_args = json!({"message": ""});
        assert!(!Executor::validate_echo_tool_arguments(&invalid_args));
    }

    #[test]
    fn test_validate_echo_tool_arguments_whitespace_string() {
        let invalid_args = json!({"message": "   "});
        assert!(!Executor::validate_echo_tool_arguments(&invalid_args));
    }

    #[test]
    fn test_validate_echo_tool_arguments_extra_keys() {
        let invalid_args = json!({"message": "hello", "extra": "field"});
        assert!(!Executor::validate_echo_tool_arguments(&invalid_args));
    }


    #[test]
    fn test_is_tool_eligible_echo_explicit_requests() {
        // Echo should be eligible when user explicitly requests it
        assert!(Executor::is_tool_eligible("echo", "use the echo tool", ""));
        assert!(Executor::is_tool_eligible("echo", "call echo", ""));
        assert!(Executor::is_tool_eligible("echo", "echo this message", ""));
        assert!(Executor::is_tool_eligible("echo", "please echo:", ""));
    }

    #[test]
    fn test_is_tool_eligible_echo_inappropriate_usage() {
        // Echo should NOT be eligible for casual conversation
        assert!(!Executor::is_tool_eligible("echo", "hello", ""));
        assert!(!Executor::is_tool_eligible("echo", "what do you think?", ""));
        assert!(!Executor::is_tool_eligible("echo", "tell me about yourself", ""));
        assert!(!Executor::is_tool_eligible("echo", "I love coding", ""));
    }

    #[test]
    fn test_is_tool_eligible_other_tools() {
        // Other tools should be eligible by default (for now)
        assert!(Executor::is_tool_eligible("some_other_tool", "any message", ""));
        assert!(Executor::is_tool_eligible("graphql_query", "hello", ""));
    }
}
