use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Context for tool eligibility decisions
#[derive(Debug)]
pub struct ToolEligibilityContext<'a> {
    /// The user's message that triggered the tool call
    pub user_message: &'a str,
    /// The assistant's message containing the tool call
    pub assistant_message: &'a str,
    /// Whether the user explicitly requested this tool by name
    pub explicitly_requested: bool,
    /// The persona being used
    pub persona: &'a str,
    /// The parsed intent (if successfully parsed)
    pub intent: Option<&'a crate::tools::graphql::Intent>,
}

/// Trait for tools to determine their own eligibility
pub trait ToolEligibility: Send + Sync {
    fn is_eligible(&self, ctx: &ToolEligibilityContext) -> bool;
}

/// Metadata for tool execution
#[derive(Debug, Clone)]
pub struct ToolMetadata {
    /// Name of the tool to execute
    pub tool_name: String,
}

/// Input for tool execution
#[derive(Debug, Clone)]
pub struct ToolInput {
    /// Tool payload as structured data
    pub payload: Value,
    /// Tool execution metadata
    pub metadata: ToolMetadata,
    pub user_message: String,
    /// Pre-parsed and normalized intent (for query_intent tool)
    pub parsed_intent: Option<crate::tools::graphql::Intent>,
}

/// Output from tool execution
#[derive(Debug, Clone)]
pub struct ToolOutput {
    /// Tool result payload as structured data
    pub payload: Value,
}

/// Error types for tool execution
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("Tool not found: {0}")]
    ToolNotFound(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Tool execution failed: {0}")]
    ExecutionFailed(String),
}

/// ToolPort represents a single external capability
///
/// Tools are stateless external interfaces that perform specific operations.
/// They do NOT decide when to run - that authority belongs to the Executor.
/// They do NOT access inference or memory - those are separate concerns.
///
/// The Executor is the only caller of ToolPort implementations.
pub trait ToolPort: ToolEligibility {
    /// Returns the name of this tool
    fn name(&self) -> &str;

    /// Returns whether this tool is read-only (safe for mixed-intent messages)
    fn is_read_only(&self) -> bool {
        false // Default implementation - tools are not read-only unless explicitly marked
    }

    /// Execute the tool with the given input
    ///
    /// # Arguments
    /// * `input` - The tool input containing name and parameters
    /// * `ctx` - Executor context containing per-request dependencies
    ///
    /// # Returns
    /// ToolOutput on success, ToolError on failure
    fn execute(&self, input: ToolInput, ctx: &crate::runtime::executor::ExecutorContext) -> Result<ToolOutput, ToolError>;
}

/// ToolRegistry maps tool names to ToolPort implementations
///
/// The registry owns tool instances and provides access to them.
/// Only the Executor should access the registry to execute tools.
pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn ToolPort + Send + Sync>>,
}

impl ToolRegistry {
    /// Create a new empty tool registry
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    /// Register a tool in the registry
    pub fn register<T: ToolPort + Send + Sync + 'static>(&mut self, tool: T) {
        let name = tool.name().to_string();
        self.tools.insert(name, Box::new(tool));
    }

    /// Get a tool by name
    pub fn get(&self, name: &str) -> Option<&(dyn ToolPort + Send + Sync)> {
        self.tools.get(name).map(|t| t.as_ref())
    }

    /// List all registered tool names
    pub fn list_tools(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
}

/// Arguments for a tool call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallArguments {
    #[serde(flatten)]
    pub args: serde_json::Value,
}


/// Represents a parsed tool call request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub r#type: String,
    pub name: String,
    pub arguments: ToolCallArguments,
}

/// Parses a tool call from model output.
/// Returns Some(ToolCall) if valid tool JSON with type === "tool_call", None otherwise.
/// Safe and strict - never panics, returns None on any parsing failure.
pub fn parse_tool_call(output: &str) -> Option<ToolCall> {
    let trimmed = output.trim();

    // HARD SIZE LIMIT: Prevent parsing of pathological inputs
    if trimmed.len() > 10_000 {
        return None;
    }

    // Skip if empty or doesn't look like JSON
    if trimmed.is_empty() || (!trimmed.starts_with('{') && !trimmed.starts_with("```")) {
        return None;
    }

    // STRICT CODE-FENCE HANDLING: Reject JSON with trailing/leading text or multiple blocks
    // This is intentional safety - any ambiguity results in "no tool call"
    let json_str = if trimmed.starts_with("```json") && trimmed.ends_with("```") {
        let start = trimmed.find('\n').unwrap_or(7) + 1;
        let end = trimmed.len() - 3;
        if start >= end {
            return None;
        }
        &trimmed[start..end]
    } else if trimmed.starts_with("```") && trimmed.ends_with("```") {
        let start = trimmed.find('\n').unwrap_or(3) + 1;
        let end = trimmed.len() - 3;
        if start >= end {
            return None;
        }
        &trimmed[start..end]
    } else {
        trimmed
    };

    // Parse as JSON - return None on failure
    let value: serde_json::Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return None,
    };

    // Must be an object
    let obj = value.as_object()?;

    // TYPE FIELD HARDENING: Must exist, be string, equal exactly "tool_call"
    let type_value = obj.get("type")?;
    let type_str = type_value.as_str()?;
    if type_str != "tool_call" {
        return None;
    }

    // Extract name - must be non-empty string
    let name_value = obj.get("name")?;
    let name = name_value.as_str()?;
    if name.is_empty() {
        return None;
    }

    // ARGUMENTS FIELD VALIDATION: Must exist and be a JSON object
    let arguments_value = obj.get("arguments")?;
    // Reject null, arrays, strings, numbers, booleans - must be object
    let _arguments_obj = arguments_value.as_object()?;
    let arguments = match serde_json::from_value(arguments_value.clone()) {
        Ok(args) => args,
        Err(_) => return None,
    };

    Some(ToolCall {
        r#type: type_str.to_string(),
        name: name.to_string(),
        arguments,
    })
}
