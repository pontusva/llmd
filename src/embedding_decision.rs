use crate::embedding_observability::EmbeddingStats;
use crate::persona_memory::{MemoryConfig, MemoryMode, MemoryType};
use std::sync::Arc;

/// Events that trigger embedding/memory decisions in the Executor
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryEventKind {
    /// User input message received
    UserMessage,
    /// Assistant response as plain text (no tool call)
    AssistantPlainText,
    /// Assistant response that parses as valid tool call JSON
    AssistantToolCallJson,
    /// Result from executing a tool
    ToolResult,
    /// Plain text response from jail retry (tool hallucination recovery)
    JailRetryPlainText,
    /// Jail retry that still contained violations (tool calls/JSON)
    JailRetryViolation,
}

/// Context information provided to the decision matrix
#[derive(Debug, Clone)]
pub struct DecisionContext<'a> {
    /// Persona name
    pub persona: &'a str,
    /// Memory update command (e.g., Some("disable"))
    pub memory_update: Option<&'a str>,
    /// Whether embeddings are available in the runtime
    pub embeddings_available: bool,
    /// Whether memory_config.vector_types is non-empty
    pub has_vector_types: bool,
    /// Last user message text (for UserMessage events)
    pub user_text: &'a str,
    /// Assistant response or tool result text
    pub assistant_text: &'a str,
    /// Whether this is a streaming response
    pub is_streaming: bool,
}

/// Decision result from the embedding decision matrix
#[derive(Debug, Clone)]
pub struct DecisionResult {
    /// Whether to create embeddings for input (user_text)
    pub should_embed_input: bool,
    /// Whether to create embeddings for output (assistant_text)
    pub should_embed_output: bool,
    /// Whether to store any memory at all
    pub should_store_memory: bool,
    /// Type of memory to store (if storing)
    pub memory_type: MemoryType,
    /// TTL for stored memory in seconds (None = no expiration)
    pub ttl_seconds: Option<u64>,
    /// Human-readable reason for this decision (for logs/debugging)
    pub reason: &'static str,
    /// Optional tags for categorization/debugging
    pub tags: Vec<&'static str>,
}

/// Embedding decision matrix - single source of truth for memory/embed decisions
pub struct EmbeddingDecisionMatrix;

impl EmbeddingDecisionMatrix {
    /// Main decision function - returns embedding/memory decisions for any event
    pub fn decide(
        kind: MemoryEventKind,
        ctx: &DecisionContext,
        memory_config: &MemoryConfig,
        stats: &Arc<EmbeddingStats>,
    ) -> DecisionResult {
        // Track decision metrics
        stats.decisions_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Global gates - apply to all events

        // 1. Memory disabled globally
        if matches!(ctx.memory_update, Some("disable")) {
            return DecisionResult {
                should_embed_input: false,
                should_embed_output: false,
                should_store_memory: false,
                memory_type: MemoryType::Conversation, // irrelevant when not storing
                ttl_seconds: None,
                reason: "memory disabled",
                tags: vec!["global", "disabled"],
            };
        }

        // 2. Read-only mode
        if memory_config.mode == MemoryMode::Read {
            return DecisionResult {
                should_embed_input: false,
                should_embed_output: false,
                should_store_memory: false,
                memory_type: MemoryType::Conversation,
                ttl_seconds: None,
                reason: "read-only mode",
                tags: vec!["global", "read-only"],
            };
        }

        // 3. Embedding requires BOTH embeddings_available AND has_vector_types
        let can_embed = ctx.embeddings_available && ctx.has_vector_types;

        // Apply event-specific logic
        let result = match kind {
            MemoryEventKind::UserMessage => Self::decide_user_message(ctx, can_embed),
            MemoryEventKind::AssistantPlainText => Self::decide_assistant_plain_text(ctx, can_embed),
            MemoryEventKind::AssistantToolCallJson => Self::decide_assistant_tool_call_json(ctx),
            MemoryEventKind::ToolResult => Self::decide_tool_result(ctx, can_embed),
            MemoryEventKind::JailRetryPlainText => Self::decide_jail_retry_plain_text(ctx, can_embed),
            MemoryEventKind::JailRetryViolation => Self::decide_jail_retry_violation(),
        };

        // Track allowed/denied metrics
        if result.should_store_memory {
            stats.allowed_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        } else {
            stats.denied_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::debug!("🧠 embedding denied: {} ({})", result.reason, result.tags.join(","));
        }

        result
    }

    /// Decide for user message events
    fn decide_user_message(ctx: &DecisionContext, can_embed: bool) -> DecisionResult {
        // Check denylist for user input
        if Self::is_denied_by_content_rules(ctx.user_text) {
            return DecisionResult {
                should_embed_input: false,
                should_embed_output: false,
                should_store_memory: false,
                memory_type: MemoryType::Conversation,
                ttl_seconds: None,
                reason: "user input matches denylist",
                tags: vec!["denylist", "user-input"],
            };
        }

        // Normal user message - store as conversation, embed if available
        DecisionResult {
            should_embed_input: can_embed,
            should_embed_output: false, // user messages don't have output
            should_store_memory: true,
            memory_type: MemoryType::Conversation,
            ttl_seconds: Some(604800), // 7 days
            reason: "normal user message",
            tags: vec!["conversation", "user-input"],
        }
    }

    /// Decide for plain text assistant responses
    fn decide_assistant_plain_text(ctx: &DecisionContext, can_embed: bool) -> DecisionResult {
        // Check for echo-like responses (assistant text similar to user text)
        if Self::is_echo_like(ctx.user_text, ctx.assistant_text) {
            return DecisionResult {
                should_embed_input: false,
                should_embed_output: false,
                should_store_memory: false,
                memory_type: MemoryType::Conversation,
                ttl_seconds: None,
                reason: "echo-like response",
                tags: vec!["denylist", "echo"],
            };
        }

        // Check denylist for assistant output
        if Self::is_denied_by_content_rules(ctx.assistant_text) {
            return DecisionResult {
                should_embed_input: false,
                should_embed_output: false,
                should_store_memory: false,
                memory_type: MemoryType::Conversation,
                ttl_seconds: None,
                reason: "assistant output matches denylist",
                tags: vec!["denylist", "assistant-output"],
            };
        }

        // Normal assistant response - store as conversation, embed both input and output if available
        DecisionResult {
            should_embed_input: can_embed,
            should_embed_output: can_embed,
            should_store_memory: true,
            memory_type: MemoryType::Conversation,
            ttl_seconds: Some(604800), // 7 days
            reason: "normal assistant response",
            tags: vec!["conversation", "assistant-output"],
        }
    }

    /// Decide for tool call JSON responses
    fn decide_assistant_tool_call_json(_ctx: &DecisionContext) -> DecisionResult {
        // Tool call JSON should NOT be stored or embedded
        DecisionResult {
            should_embed_input: false,
            should_embed_output: false,
            should_store_memory: false,
            memory_type: MemoryType::Conversation,
            ttl_seconds: None,
            reason: "tool call - do not store tool json",
            tags: vec!["tool-call", "json"],
        }
    }

    /// Decide for tool execution results
    fn decide_tool_result(ctx: &DecisionContext, can_embed: bool) -> DecisionResult {
        const MEMORY_SAFE_TOOLS: &[&str] = &["graphql_query"];

        // Extract tool name from context - assume it's embedded in assistant_text or we need to parse it
        // For now, we'll check if any memory-safe tool name appears in the result
        let is_memory_safe = MEMORY_SAFE_TOOLS
            .iter()
            .any(|tool_name| ctx.assistant_text.contains(tool_name));

        // Size limit for tool results
        let within_size_limit = ctx.assistant_text.len() <= 2000;

        if is_memory_safe && within_size_limit {
            DecisionResult {
                should_embed_input: false, // don't embed tool inputs
                should_embed_output: false, // store as keyword only, no embedding
                should_store_memory: true,
                memory_type: MemoryType::Conversation,
                ttl_seconds: Some(3600), // 1 hour
                reason: "memory-safe tool result",
                tags: vec!["tool-result", "memory-safe"],
            }
        } else {
            DecisionResult {
                should_embed_input: false,
                should_embed_output: false,
                should_store_memory: false,
                memory_type: MemoryType::Conversation,
                ttl_seconds: None,
                reason: "tool result - not memory safe or too large",
                tags: vec!["tool-result", "denied"],
            }
        }
    }

    /// Decide for jail retry plain text responses
    fn decide_jail_retry_plain_text(ctx: &DecisionContext, can_embed: bool) -> DecisionResult {
        // Treat like normal assistant text but apply denylist
        if Self::is_denied_by_content_rules(ctx.assistant_text) {
            return DecisionResult {
                should_embed_input: false,
                should_embed_output: false,
                should_store_memory: false,
                memory_type: MemoryType::Conversation,
                ttl_seconds: None,
                reason: "jail retry matches denylist",
                tags: vec!["jail-retry", "denylist"],
            };
        }

        DecisionResult {
            should_embed_input: can_embed,
            should_embed_output: can_embed,
            should_store_memory: true,
            memory_type: MemoryType::Conversation,
            ttl_seconds: Some(604800), // 7 days
            reason: "jail retry plain text",
            tags: vec!["jail-retry", "conversation"],
        }
    }

    /// Decide for jail retry violations
    fn decide_jail_retry_violation() -> DecisionResult {
        DecisionResult {
            should_embed_input: false,
            should_embed_output: false,
            should_store_memory: false,
            memory_type: MemoryType::Conversation,
            ttl_seconds: None,
            reason: "jail retry violation",
            tags: vec!["jail-retry", "violation"],
        }
    }

    /// Check if text matches denylist patterns (greetings, smalltalk, meta questions, etc.)
    fn is_denied_by_content_rules(text: &str) -> bool {
        let normalized = text.to_lowercase();

        // Very short texts are likely greetings/smalltalk
        if normalized.len() < 5 {
            return true;
        }

        // Greetings
        if Self::starts_with_greeting(&normalized) {
            return true;
        }

        // Acknowledgements/confirmations
        if Self::is_acknowledgement(&normalized) {
            return true;
        }

        // Meta/tooling questions
        if Self::is_meta_question(&normalized) {
            return true;
        }

        false
    }

    /// Check if text starts with greeting patterns
    fn starts_with_greeting(text: &str) -> bool {
        const GREETINGS: &[&str] = &[
            "hi", "hello", "hey", "yo", "sup", "waddap", "wazzup", "hola", "hej", "tjena"
        ];

        GREETINGS.iter().any(|&g| {
            text.starts_with(&format!("{} ", g)) ||
            text.starts_with(&format!("{}, ", g)) ||
            text == g
        })
    }

    /// Check if text is an acknowledgement/confirmation
    fn is_acknowledgement(text: &str) -> bool {
        const ACKS: &[&str] = &[
            "ok", "okay", "thanks", "thx", "nice", "cool", "great", "perfect",
            "no thanks", "yep", "yup", "nah"
        ];

        ACKS.iter().any(|&ack| {
            text.starts_with(&format!("{} ", ack)) ||
            text.starts_with(&format!("{}, ", ack)) ||
            text == ack ||
            text.starts_with(&format!("{}!", ack)) ||
            text.starts_with(&format!("{}.", ack))
        })
    }

    /// Check if text contains meta/tooling questions
    fn is_meta_question(text: &str) -> bool {
        const META_PHRASES: &[&str] = &[
            "why did you", "tools", "system prompt", "jail", "tool call", "registry",
            "how do you", "what are your", "tell me about your"
        ];

        META_PHRASES.iter().any(|&phrase| text.contains(phrase))
    }

    /// Check if assistant text is an echo/paraphrase of user text
    fn is_echo_like(user_text: &str, assistant_text: &str) -> bool {
        // Normalize both texts: lowercase, trim whitespace, remove punctuation
        let normalize = |s: &str| {
            s.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect::<String>()
        };

        let user_norm = normalize(user_text);
        let assistant_norm = normalize(assistant_text);

        // Skip if either is too short
        if user_norm.len() < 10 || assistant_norm.len() < 10 {
            return false;
        }

        // Check length similarity (assistant should be within ±15% of user length)
        let user_len = user_norm.len() as f32;
        let assistant_len = assistant_norm.len() as f32;
        let length_ratio = assistant_len / user_len;

        if !(0.85..=1.15).contains(&length_ratio) {
            return false;
        }

        // Check if assistant text contains significant portions of user text
        // Simple heuristic: if assistant contains 80% of user words (in order)
        let user_words: Vec<&str> = user_norm.split_whitespace().collect();
        let assistant_words: Vec<&str> = assistant_norm.split_whitespace().collect();

        if user_words.is_empty() || assistant_words.is_empty() {
            return false;
        }

        // Count how many user words appear in assistant (in any order for simplicity)
        let matching_words = user_words.iter()
            .filter(|&&word| assistant_words.contains(&word))
            .count();

        let match_ratio = matching_words as f32 / user_words.len() as f32;
        match_ratio >= 0.8
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persona_memory::{MemoryMode, MemoryPolicy};

    fn create_test_config() -> MemoryConfig {
        MemoryConfig {
            mode: MemoryMode::ReadWrite,
            policy: MemoryPolicy::Auto,
            debug: false,
            vector_threshold: 0.78,
            vector_top_k: 3,
            vector_types: vec![MemoryType::Conversation],
        }
    }

    #[test]
    fn test_greeting_denial() {
        let config = create_test_config();
        let ctx = DecisionContext {
            persona: "test",
            memory_update: None,
            embeddings_available: true,
            has_vector_types: true,
            user_text: "hello",
            assistant_text: "",
            is_streaming: false,
        };

        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::UserMessage, &ctx, &config, &stats);
        assert!(!decision.should_store_memory);
        assert_eq!(decision.reason, "user input matches denylist");
        assert!(decision.tags.contains(&"denylist"));
    }

    #[test]
    fn test_tool_call_json_exclusion() {
        let config = create_test_config();
        let ctx = DecisionContext {
            persona: "test",
            memory_update: None,
            embeddings_available: true,
            has_vector_types: true,
            user_text: "run a tool",
            assistant_text: r#"{"type": "tool_call", "name": "echo", "arguments": {"text": "hello"}}"#,
            is_streaming: false,
        };

        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::AssistantToolCallJson, &ctx, &config, &stats);
        assert!(!decision.should_store_memory);
        assert!(!decision.should_embed_input);
        assert!(!decision.should_embed_output);
        assert_eq!(decision.reason, "tool call - do not store tool json");
    }

    #[test]
    fn test_tool_result_memory_safe() {
        let config = create_test_config();
        let ctx = DecisionContext {
            persona: "test",
            memory_update: None,
            embeddings_available: true,
            has_vector_types: true,
            user_text: "query something",
            assistant_text: r#"{"result": "some graphql data from graphql_query"}"#,
            is_streaming: false,
        };

        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::ToolResult, &ctx, &config, &stats);
        assert!(decision.should_store_memory);
        assert!(!decision.should_embed_output); // keyword only
        assert_eq!(decision.ttl_seconds, Some(3600)); // 1 hour
        assert_eq!(decision.reason, "memory-safe tool result");
    }

    #[test]
    fn test_tool_result_not_memory_safe() {
        let config = create_test_config();
        let ctx = DecisionContext {
            persona: "test",
            memory_update: None,
            embeddings_available: true,
            has_vector_types: true,
            user_text: "run echo",
            assistant_text: r#"{"result": "echo output that's too long to store...................................................."}"#,
            is_streaming: false,
        };

        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::ToolResult, &ctx, &config, &stats);
        assert!(!decision.should_store_memory);
        assert_eq!(decision.reason, "tool result - not memory safe or too large");
    }

    #[test]
    fn test_echo_detection() {
        let config = create_test_config();
        let ctx = DecisionContext {
            persona: "test",
            memory_update: None,
            embeddings_available: true,
            has_vector_types: true,
            user_text: "Hello world",
            assistant_text: "Hello world", // Exact echo
            is_streaming: false,
        };

        // This should be detected as echo-like (exact match)
        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::AssistantPlainText, &ctx, &config, &stats);
        assert!(!decision.should_store_memory);
        assert_eq!(decision.reason, "echo-like response");
    }

    #[test]
    fn test_memory_disable() {
        let config = create_test_config();
        let ctx = DecisionContext {
            persona: "test",
            memory_update: Some("disable"),
            embeddings_available: true,
            has_vector_types: true,
            user_text: "hello world",
            assistant_text: "response",
            is_streaming: false,
        };

        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::UserMessage, &ctx, &config, &stats);
        assert!(!decision.should_store_memory);
        assert!(!decision.should_embed_input);
        assert!(!decision.should_embed_output);
        assert_eq!(decision.reason, "memory disabled");
    }

    #[test]
    fn test_normal_conversation() {
        let config = create_test_config();
        let ctx = DecisionContext {
            persona: "test",
            memory_update: None,
            embeddings_available: true,
            has_vector_types: true,
            user_text: "What is machine learning?",
            assistant_text: "Machine learning is a subset of artificial intelligence...",
            is_streaming: false,
        };

        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::AssistantPlainText, &ctx, &config, &stats);
        assert!(decision.should_store_memory);
        assert!(decision.should_embed_input);
        assert!(decision.should_embed_output);
        assert_eq!(decision.memory_type, MemoryType::Conversation);
        assert_eq!(decision.ttl_seconds, Some(604800)); // 7 days
        assert_eq!(decision.reason, "normal assistant response");
    }

    #[test]
    fn test_read_only_mode() {
        let mut config = create_test_config();
        config.mode = MemoryMode::Read;
        let ctx = DecisionContext {
            persona: "test",
            memory_update: None,
            embeddings_available: true,
            has_vector_types: true,
            user_text: "hello",
            assistant_text: "",
            is_streaming: false,
        };

        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::UserMessage, &ctx, &config, &stats);
        assert!(!decision.should_store_memory);
        assert_eq!(decision.reason, "read-only mode");
    }

    #[test]
    fn test_jail_retry_violation() {
        let config = create_test_config();
        let ctx = DecisionContext {
            persona: "test",
            memory_update: None,
            embeddings_available: true,
            has_vector_types: true,
            user_text: "malicious input",
            assistant_text: "jailbreak response",
            is_streaming: false,
        };

        let stats = Arc::new(EmbeddingStats::new());
        let decision = EmbeddingDecisionMatrix::decide(MemoryEventKind::JailRetryViolation, &ctx, &config, &stats);
        assert!(!decision.should_store_memory);
        assert_eq!(decision.reason, "jail retry violation");
    }
}
