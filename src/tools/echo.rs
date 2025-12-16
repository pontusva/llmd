use crate::runtime::toolport::{ToolPort, ToolInput, ToolOutput, ToolError, ToolEligibility, ToolEligibilityContext};

/// EchoTool simply returns the input payload unchanged.
///
/// This is a minimal tool implementation that demonstrates:
/// - Stateless execution (no side effects)
/// - No decision-making (passive interface)
/// - Pure data transformation only
/// - No access to inference, memory, or external systems
pub struct EchoTool;

impl ToolEligibility for EchoTool {
    fn is_eligible(&self, ctx: &ToolEligibilityContext) -> bool {
        ctx.explicitly_requested
    }
}

impl ToolPort for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    fn execute(&self, input: ToolInput, _ctx: &crate::runtime::executor::ExecutorContext) -> Result<ToolOutput, ToolError> {
        // EchoTool expects a string payload (validated by executor)
        // Return the input payload unchanged
        Ok(ToolOutput {
            payload: input.payload,
        })
    }
}
