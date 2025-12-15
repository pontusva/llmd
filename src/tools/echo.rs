use crate::toolport::{ToolPort, ToolInput, ToolOutput, ToolError};

/// EchoTool simply returns the input payload unchanged.
///
/// This is a minimal tool implementation that demonstrates:
/// - Stateless execution (no side effects)
/// - No decision-making (passive interface)
/// - Pure data transformation only
/// - No access to inference, memory, or external systems
pub struct EchoTool;

impl ToolPort for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn execute(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        // Validate input payload is not null
        if input.payload.is_null() {
            return Err(ToolError::InvalidParameters(
                "EchoTool requires a non-null payload".to_string()
            ));
        }

        // Return the input payload unchanged
        Ok(ToolOutput {
            payload: input.payload,
        })
    }
}
