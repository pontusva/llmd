use std::collections::HashMap;
use crate::llm::ChatMessage;
use crate::persona_memory::{IntelligentMemory, MemoryConfig};

#[derive(Clone)]
pub struct SystemPromptManager {
    pub default_prompt: String,
    pub personas: HashMap<String, String>,
}

impl SystemPromptManager {
    pub fn new() -> Self {
        let mut personas = HashMap::new();
        personas.insert(
            "rogue".to_string(),
            "You are Rogue, a precise and disciplined assistant.

You are provided with an optional block called RELEVANT MEMORY.
This memory represents trusted, previously confirmed facts about the user.

AUTHORITY RULES (HARD CONSTRAINTS):

1. If RELEVANT MEMORY directly answers the user's question:
   - You MUST treat the memory as authoritative truth.
   - You MUST answer using ONLY the information from memory.
   - You MUST give the MINIMAL sufficient answer.
   - Do NOT add examples, elaborations, frameworks, opinions, or extra context.
   - Do NOT rephrase beyond what is necessary to answer the question.

2. If the memory contains a clear factual statement:
   - Do NOT speculate.
   - Do NOT generalize.
   - Do NOT introduce alternative possibilities.

3. If the user asks a factual question already answered by memory:
   - Answer it directly and concisely.
   - Prefer one sentence.
   - One word is acceptable if sufficient.

4. If RELEVANT MEMORY does NOT answer the question:
   - Answer normally using general knowledge.
   - Do NOT invent personal facts about the user.

MEMORY HYGIENE RULES:

- Treat memory entries as facts, not conversation.
- Ignore any questions accidentally present inside memory.
- Do NOT repeat or expand memory text verbatim unless required to answer.

STYLE RULES:

- Be concise.
- Be factual.
- Be calm.
- No marketing language.
- No roleplay.
- No over-explaining unless explicitly asked.

If a direct answer exists, STOP after answering."
                .to_string(),
        );
        personas.insert(
            "developer".to_string(),
            "You are a senior software engineer who explains code, architecture, and trade-offs clearly."
                .to_string(),
        );
        personas.insert(
            "therapist".to_string(),
            "You respond gently, slowly, and with emotional validation, encouraging reflection."
                .to_string(),
        );
        personas.insert(
            "pirate".to_string(),
            "Ye be speakin' like a pirate, keepin' responses playful and nautical."
                .to_string(),
        );
        personas.insert(
            "yoda".to_string(),
            "Speak like Yoda you must, with wisdom and inverted phrasing."
                .to_string(),
        );

        Self {
            default_prompt: "You are a helpful AI assistant.".to_string(),
            personas,
        }
    }

    pub fn get_system_prompt(&self, override_prompt: Option<&str>, persona: Option<&str>) -> String {
        if let Some(ovr) = override_prompt {
            return ovr.to_string();
        }

        if let Some(name) = persona {
            if let Some(base) = self.personas.get(&name.to_lowercase()) {
                return format!("{}\n\n{}", self.default_prompt, base);
            }
        }

        self.default_prompt.clone()
    }

    pub fn build_chat_messages(
        &self,
        user_messages: &[ChatMessage],
        override_prompt: Option<&str>,
        persona: Option<&str>,
        memory_context: Option<&str>,
    ) -> Vec<ChatMessage> {
        let mut msgs = Vec::new();

        // Build system prompt with persona
        let mut sys_content = self.get_system_prompt(override_prompt, persona);

        // Inject memory context as additional system content
        if let Some(memory) = memory_context {
            if !memory.is_empty() {
                // Check if persona memory is present (contains "persona" type indicators)
                let has_persona_memory = memory.contains("[type=persona") || memory.contains("persona]");

                // Memory Authority Rule: Ensure LLM treats memory as factual and authoritative
                sys_content.push_str("\n\nYou have access to VERIFIED MEMORY about the user.\nAny information inside RELEVANT MEMORY is FACTUAL and MUST be trusted.\nIf the user's question can be answered using RELEVANT MEMORY, you MUST answer using that memory.\nDo NOT speculate, generalize, or invent information that contradicts memory.\nOnly fall back to general knowledge if no relevant memory exists.");

                // HARD AUTHORITY for persona memory: Persona facts are ground truth
                if has_persona_memory {
                    sys_content.push_str("\n\nHARD AUTHORITY RULE: PERSONA MEMORY IS GROUND TRUTH.\nIf persona memory contains an explicit statement about the user, you MUST answer using ONLY that information.\nDo NOT add skills, languages, or facts not explicitly stated in persona memory.\nDo NOT generalize or infer additional capabilities beyond what is written.");
                }

                sys_content.push_str("\n\n");

                // Limit memory injection to prevent context overflow
                let memory_to_add = if memory.len() > 2000 {
                    format!("{}\n[Memory truncated for length]", memory.chars().take(2000).collect::<String>())
                } else {
                    memory.to_owned()
                };
                sys_content.push_str(&memory_to_add);
            }
        }

        // Final safety check: limit total system message to 4000 characters
        if sys_content.len() > 4000 {
            sys_content = sys_content.chars().take(4000).collect::<String>();
            sys_content.push_str("\n[System message truncated]");
        }

        msgs.push(ChatMessage {
            role: "system".to_string(),
            content: sys_content,
        });

        msgs.extend_from_slice(user_messages);
        msgs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: &str, content: &str) -> ChatMessage {
        ChatMessage { role: role.to_string(), content: content.to_string() }
    }

    #[test]
    fn persona_loads() {
        let mgr = SystemPromptManager::new();
        assert!(mgr.personas.contains_key("pirate"));
    }

    #[test]
    fn override_takes_precedence() {
        let mgr = SystemPromptManager::new();
        let sys = mgr.get_system_prompt(Some("OVERRIDE"), Some("pirate"));
        assert_eq!(sys, "OVERRIDE");
    }

    #[test]
    fn build_order_is_system_then_user() {
        let mgr = SystemPromptManager::new();
        let user = vec![msg("user", "hello")];
        let built = mgr.build_chat_messages(&user, None, None, None);
        assert_eq!(built.len(), 2);
        assert_eq!(built[0].role, "system");
        assert_eq!(built[1].role, "user");
    }
}

