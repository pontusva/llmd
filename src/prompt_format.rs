use crate::llm::ChatMessage;

#[allow(dead_code)]
pub enum PromptFormat {
    ChatML,
    OpenAI,
    Llama2,
    Phi,
}

#[allow(dead_code)]
#[allow(dead_code)]
pub fn format_chatml(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for m in messages {
        let role = match m.role.as_str() {
            "system" => "system",
            "assistant" => "assistant",
            "user" => "user",
            _ => "user",
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(&m.content);
        out.push('\n');
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

#[allow(dead_code)]
pub fn format_llama2(messages: &[ChatMessage]) -> String {
    let mut system_txt = String::new();
    let mut body = String::new();

    for m in messages {
        match m.role.as_str() {
            "system" => {
                if !system_txt.is_empty() {
                    system_txt.push('\n');
                }
                system_txt.push_str(&m.content);
            }
            "assistant" => {
                body.push_str("### Assistant:\n");
                body.push_str(&m.content);
                body.push_str("\n\n");
            }
            _ => {
                body.push_str("### Human:\n");
                body.push_str(&m.content);
                body.push_str("\n\n");
            }
        }
    }

    let mut prompt = String::new();
    if !system_txt.is_empty() {
        prompt.push_str("<<SYS>>\n");
        prompt.push_str(&system_txt);
        prompt.push_str("\n<</SYS>>\n\n");
    }
    prompt.push_str(&body);
    prompt.push_str("### Assistant:\n");
    prompt
}

#[allow(dead_code)]
pub fn format_phi(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for m in messages {
        match m.role.as_str() {
            "assistant" => {
                prompt.push_str("### Assistant:\n");
                prompt.push_str(&m.content);
                prompt.push_str("\n\n");
            }
            _ => {
                prompt.push_str("### Human:\n");
                prompt.push_str(&m.content);
                prompt.push_str("\n\n");
            }
        }
    }
    prompt.push_str("### Assistant:\n");
    prompt
}

#[allow(dead_code)]
pub fn format_openai(messages: &[ChatMessage]) -> String {
    let mut prompt = String::new();
    for m in messages {
        prompt.push_str(&format!("{}: {}\n", m.role, m.content));
    }
    prompt.push_str("assistant:");
    prompt
}

