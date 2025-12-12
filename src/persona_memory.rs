use anyhow::Result;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use tracing::info;

/// Trait for models that can compute embeddings
#[async_trait::async_trait]
pub trait EmbeddingModel {
    async fn compute_embedding(&self, text: &str) -> Result<Vec<f32>>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    Persona,      // Long-term identity & preferences
    Conversation, // Rolling context of current chat
    Fact,         // Extracted facts (optional, short)
}

impl MemoryType {
    pub fn as_str(&self) -> &'static str {
        match self {
            MemoryType::Persona => "persona",
            MemoryType::Conversation => "conversation",
            MemoryType::Fact => "fact",
        }
    }
}

#[derive(Debug, Clone)]
pub struct MemoryEntry {
    pub memory_type: MemoryType,
    pub persona: String,
    pub content: String,
    pub timestamp: i64,
}

#[derive(Debug, Clone)]
pub struct ScoredMemory {
    pub entry: MemoryEntry,
    pub score: usize,
}

#[derive(Debug, Clone)]
pub struct MemoryConfig {
    pub mode: MemoryMode,
    pub policy: MemoryPolicy,
    pub debug: bool,
    pub vector_threshold: f32,
    pub vector_top_k: usize,
    pub vector_types: Vec<MemoryType>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryMode {
    Off,      // No memory operations
    Read,     // Read memory but don't write
    Write,    // Write memory but don't read
    ReadWrite, // Read and write memory
}

impl MemoryMode {
    pub fn should_read(&self) -> bool {
        matches!(self, MemoryMode::Read | MemoryMode::ReadWrite)
    }

    pub fn should_write(&self) -> bool {
        matches!(self, MemoryMode::Write | MemoryMode::ReadWrite)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MemoryPolicy {
    Append,  // Always append to existing memory
    Replace, // Replace existing memory
    Auto,    // Use intelligent heuristics
}

pub struct IntelligentMemory {
    pub conn: Connection,
}

impl IntelligentMemory {
    pub fn new(path: &str) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY,
                memory_type TEXT NOT NULL,
                persona TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                UNIQUE(memory_type, persona)
            );",
            [],
        )?;

        // Create memory vectors table for semantic search
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_vectors (
                id INTEGER PRIMARY KEY,
                memory_id INTEGER NOT NULL,
                memory_type TEXT NOT NULL,
                persona TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                timestamp INTEGER NOT NULL,
                FOREIGN KEY(memory_id) REFERENCES memories(id) ON DELETE CASCADE,
                UNIQUE(memory_id)
            );",
            [],
        )?;

        // Create index for faster vector searches
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memory_vectors_persona_type
             ON memory_vectors(persona, memory_type);",
            [],
        )?;

        // Migrate old persona_memory table if it exists
        let exists = conn.query_row(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='persona_memory'",
            [],
            |_| Ok(()),
        ).is_ok();

        if exists {
            // Migrate existing data
            let mut stmt = conn.prepare("SELECT persona, memory FROM persona_memory")?;
            let rows = stmt.query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;

            for row in rows {
                let (persona, memory) = row?;
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as i64;

                conn.execute(
                    "INSERT OR IGNORE INTO memories (memory_type, persona, content, timestamp)
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![MemoryType::Persona.as_str(), persona, memory, timestamp],
                )?;
            }

            // Drop old table
            conn.execute("DROP TABLE persona_memory", [])?;
        }

        Ok(Self { conn })
    }

    pub fn get_memory(&self, memory_type: MemoryType, persona: &str) -> Result<String> {
        let mut stmt = self.conn.prepare(
            "SELECT content FROM memories WHERE memory_type = ?1 AND persona = ?2"
        )?;
        let res: Option<String> = stmt.query_row(
            [memory_type.as_str(), persona],
            |row| row.get(0)
        ).optional()?;
        Ok(res.unwrap_or_else(|| "".to_string()))
    }

    pub fn set_memory(&self, memory_type: MemoryType, persona: &str, content: &str) -> Result<i64> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        self.conn.execute(
            "INSERT INTO memories (memory_type, persona, content, timestamp)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(memory_type, persona) DO UPDATE SET
             content=excluded.content, timestamp=excluded.timestamp",
            rusqlite::params![memory_type.as_str(), persona, content, timestamp],
        )?;

        // Get the memory_id (either the inserted row or the updated row)
        let memory_id = self.conn.query_row(
            "SELECT id FROM memories WHERE memory_type = ?1 AND persona = ?2",
            [memory_type.as_str(), persona],
            |row| row.get(0)
        )?;

        Ok(memory_id)
    }

    pub fn append_memory(&self, memory_type: MemoryType, persona: &str, new_info: &str) -> Result<i64> {
        let current = self.get_memory(memory_type, persona)?;
        let combined = if current.is_empty() {
            new_info.to_string()
        } else {
            format!("{}\n{}", current, new_info)
        };
        self.set_memory(memory_type, persona, &combined)
    }

    /// Reset (wipe) memory for a specific persona or all personas
    #[allow(dead_code)]
    pub fn reset_memory(&self, persona: Option<&str>) -> Result<()> {
        if let Some(persona_name) = persona {
            // Reset memory for specific persona
            self.conn.execute(
                "DELETE FROM memories WHERE persona = ?1",
                rusqlite::params![persona_name],
            )?;
            info!("Reset memory for persona: {}", persona_name);
        } else {
            // Reset all memory
            self.conn.execute("DELETE FROM memories", [])?;
            info!("Reset all memory");
        }
        Ok(())
    }


    pub fn should_write_memory(user_message: &str, assistant_response: &str, config: &MemoryConfig) -> bool {
        let user_lower = user_message.to_lowercase();
        let response_lower = assistant_response.to_lowercase();

        // HARD STOP: Never write memory for greetings (expanded check)
        if user_lower.contains("hello") || user_lower.contains("hi") || user_lower.contains("hey") ||
           user_lower == "hi" || user_lower == "hello" || user_lower == "hey" {
            if config.debug {
                println!("[MEMORY] Skipping - greeting message");
            }
            return false;
        }

        // HARD STOP: Never write memory for empty or whitespace-only content
        if user_message.trim().is_empty() {
            if config.debug {
                println!("[MEMORY] Skipping - empty message");
            }
            return false;
        }

        // HARD STOP: Never write memory for messages shorter than 15 characters
        if user_message.len() < 15 {
            if config.debug {
                println!("[MEMORY] Skipping - message too short (<15 chars)");
            }
            return false;
        }

        // HARD STOP: Never write memory for generic/templated responses
        if response_lower.contains("i'm sorry") || response_lower.contains("i apologize") ||
           response_lower.contains("unfortunately") || response_lower.starts_with("sorry") ||
           (response_lower.len() < 20 && (response_lower.contains("ok") || response_lower.contains("sure"))) {
            if config.debug {
                println!("[MEMORY] Skipping - generic/templated response");
            }
            return false;
        }

        // Skip memory for basic acknowledgements
        if user_lower.starts_with("yes") || user_lower.starts_with("no") ||
           user_lower.starts_with("ok") || user_lower == "thanks" ||
           user_lower == "thank you" {
            if config.debug {
                println!("[MEMORY] Skipping - basic acknowledgement");
            }
            return false;
        }

        // Write memory for preferences, identity, goals, dislikes
        if user_lower.contains("i like") || user_lower.contains("i love") ||
           user_lower.contains("i hate") || user_lower.contains("i dislike") ||
           user_lower.contains("i prefer") || user_lower.contains("i want") ||
           user_lower.contains("i need") || user_lower.contains("i'm ") ||
           user_lower.contains("my name is") || user_lower.contains("i am") ||
           user_lower.contains("i work") || user_lower.contains("i live") ||
           user_lower.contains("i have") || user_lower.contains("i don't") ||
           user_lower.contains("i can't") || user_lower.contains("i won't") ||
           user_lower.contains("never") || user_lower.contains("always") {

            if config.debug {
                println!("[MEMORY] Writing due to preference/identity expression");
            }
            return true;
        }

        // Write memory for corrections
        if user_lower.contains("no, i meant") || user_lower.contains("actually") ||
           user_lower.contains("correction") || user_lower.contains("wrong") ||
           user_lower.contains("not that") || user_lower.contains("instead") {

            if config.debug {
                println!("[MEMORY] Writing due to correction");
            }
            return true;
        }

        // Write memory for long explanations or emotional signals
        if user_message.len() > 100 ||
           user_lower.contains("because") || user_lower.contains("since") ||
           response_lower.contains("remember") || response_lower.contains("noted") ||
           response_lower.contains("understood") || response_lower.contains("got it") {

            if config.debug {
                println!("[MEMORY] Writing due to explanation/emotion");
            }
            return true;
        }

        if config.debug {
            println!("[MEMORY] Skipping - no significant content");
        }
        false
    }

    pub fn summarize_memory(content: &str) -> String {
        if content.len() <= 2000 {
            return content.to_string();
        }

        // Extract key information using simple heuristics
        let mut bullets = Vec::new();
        let sentences: Vec<&str> = content.split_terminator(|c| c == '.' || c == '!' || c == '?' || c == '\n')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty() && s.len() > 10)
            .collect();

        // Keep most recent sentences with preference/identity keywords
        let mut priority_sentences = Vec::new();
        let mut other_sentences = Vec::new();

        for sentence in sentences.iter().rev() {
            let lower = sentence.to_lowercase();
            if lower.contains("i like") || lower.contains("i love") ||
               lower.contains("i hate") || lower.contains("i prefer") ||
               lower.contains("my name") || lower.contains("i am") ||
               lower.contains("i work") || lower.contains("i live") ||
               lower.contains("never") || lower.contains("always") {
                priority_sentences.push(*sentence);
            } else {
                other_sentences.push(*sentence);
            }
        }

        // Add priority sentences (up to 5)
        for sentence in priority_sentences.iter().take(5) {
            if bullets.len() >= 10 { break; }
            bullets.push(format!("- {}", sentence));
        }

        // Add other recent sentences (up to remaining slots)
        for sentence in other_sentences.iter().take(10 - bullets.len()) {
            bullets.push(format!("- {}", sentence));
        }

        if bullets.is_empty() {
            // Fallback: take first 2000 chars
            content.chars().take(2000).collect()
        } else {
            bullets.join("\n")
        }
    }


    // INVARIANT: If should_write_memory == false, no memory mutation may occur.
    // Returns true if memory was actually written, false if skipped.
    pub fn update_memory(&self, memory_type: MemoryType, persona: &str, user_message: &str, assistant_response: &str, config: &MemoryConfig) -> Result<Option<i64>> {
        // HARD STOP 1: Check if memory writing is enabled at all
        if !config.mode.should_write() {
            return Ok(None);
        }

        // HARD STOP 2: Core memory writing decision (includes all guards)
        let should_write = match config.policy {
            MemoryPolicy::Auto => Self::should_write_memory(user_message, assistant_response, config),
            _ => true, // Always write for Append/Replace policies
        };

        if !should_write {
            // should_write_memory already logged the reason
            return Ok(None);   // HARD STOP — NO FALLTHROUGH
        }

        let memory_id = match config.policy {
            MemoryPolicy::Append => {
                // Extract content based on the actual memory type
                let memory_entry = Self::extract_memory_content(memory_type, user_message, assistant_response);
                if config.debug {
                    println!("[MEMORY] Extracted content ({} chars): '{}...'",
                            memory_entry.len(),
                            &memory_entry[..memory_entry.len().min(50)]);
                }
                let id = self.append_memory(memory_type, persona, &memory_entry)?;
                if config.debug {
                    println!("[MEMORY] Appended to memory id: {}", id);
                }
                // Check if we need to summarize
                let current = self.get_memory(memory_type, persona)?;
                if current.len() > 2000 {
                    let summarized = Self::summarize_memory(&current);
                    self.set_memory(memory_type, persona, &summarized)?;
                }
                id
            }
            MemoryPolicy::Replace => {
                // Extract content based on the actual memory type
                let memory_entry = Self::extract_memory_content(memory_type, user_message, assistant_response);
                if config.debug {
                    println!("[MEMORY] Extracted content ({} chars): '{}...'",
                            memory_entry.len(),
                            &memory_entry[..memory_entry.len().min(50)]);
                }
                let summarized = Self::summarize_memory(&memory_entry);
                let id = self.set_memory(memory_type, persona, &summarized)?;
                if config.debug {
                    println!("[MEMORY] Replaced memory id: {}", id);
                }
                id
            }
            MemoryPolicy::Auto => {
                // Persona memory is atomic and replace-only (represents stable identity/facts)
                // Conversation memory uses append (represents ongoing dialogue)
                let (target_type, use_replace) = if Self::is_identity_content(user_message) && !Self::is_question(user_message) {
                    // Identity statements: ALWAYS replace (never append) to maintain atomic persona memory
                    (MemoryType::Persona, true)
                } else {
                    // Questions or non-identity: use conversation memory with append semantics
                    (MemoryType::Conversation, false)
                };

                // Extract content based on the determined target type
                let memory_entry = Self::extract_memory_content(target_type, user_message, assistant_response);
                if config.debug {
                    println!("[MEMORY] Extracted content ({} chars): '{}...'",
                            memory_entry.len(),
                            &memory_entry[..memory_entry.len().min(50)]);
                }

                let id = if use_replace {
                    // Persona memory: always replace for atomic updates
                    if config.debug {
                        println!("[MEMORY] Persona memory using replace semantics (atomic identity)");
                    }
                    self.set_memory(target_type, persona, &memory_entry)?
                } else {
                    // Conversation memory: append for ongoing dialogue
                    self.append_memory(target_type, persona, &memory_entry)?
                };

                if config.debug {
                    println!("[MEMORY] Auto-mode stored as {} memory id: {} ({})",
                            target_type.as_str(), id,
                            if use_replace { "replaced" } else { "appended" });
                }
                id
            }
        };

        // Note: Embedding storage is handled separately by update_memory_with_embedding

        Ok(Some(memory_id))
    }

    /// Update memory with pre-computed embedding
    pub fn update_memory_with_embedding_sync(
        &self,
        memory_type: MemoryType,
        persona: &str,
        user_message: &str,
        assistant_response: &str,
        config: &MemoryConfig,
        embedding: Option<&[f32]>
    ) -> Result<bool> {
        if config.debug {
            println!("[MEMORY] update_memory_with_embedding_sync called: type={}, persona={}, embedding_present={}",
                    memory_type.as_str(), persona, embedding.is_some());
        }

        // Use a transaction for atomic memory + vector writes
        let tx = self.conn.unchecked_transaction()?;

        // First do the regular memory update
        let memory_result = self.update_memory(memory_type, persona, user_message, assistant_response, config);

        let (memory_written, memory_id) = match memory_result {
            Ok(Some(id)) => (true, id),
            Ok(None) => (false, 0),
            Err(e) => {
                tx.rollback()?;
                return Err(e);
            }
        };

        if config.debug {
            println!("[MEMORY] Memory write result: {} (id: {})", memory_written, memory_id);
        }

        if memory_written && embedding.is_some() && memory_id > 0 {
            if config.debug {
                println!("[MEMORY] Attempting to store embedding...");
            }

            // Determine the actual memory type that was written (important for Auto policy)
            let actual_memory_type = match config.policy {
                MemoryPolicy::Auto => {
                    if Self::is_identity_content(user_message) {
                        MemoryType::Persona
                    } else {
                        MemoryType::Conversation
                    }
                }
                _ => memory_type,
            };

            if config.debug {
                println!("[MEMORY] Storing embedding for type: {}, id: {}", actual_memory_type.as_str(), memory_id);
            }

            // Store the embedding within the transaction
            if let Err(e) = self.store_memory_vector_in_tx(&tx, memory_id, actual_memory_type, persona, embedding.unwrap()) {
                if config.debug {
                    println!("[MEMORY] Failed to store embedding: {}", e);
                }
                tx.rollback()?;
                return Err(e);
            } else {
                if config.debug {
                    println!("[MEMORY] Embedding stored successfully");
                }
            }
        } else if config.debug {
            println!("[MEMORY] Skipping embedding storage: memory_written={}, embedding_present={}, memory_id={}",
                    memory_written, embedding.is_some(), memory_id);
        }

        // Commit the transaction
        tx.commit()?;
        if config.debug {
            println!("[MEMORY] Transaction committed successfully");
        }

        Ok(memory_written)
    }

    /// Update memory and compute/store embeddings
    pub async fn update_memory_with_embedding<M>(
        &self,
        memory_type: MemoryType,
        persona: &str,
        user_message: &str,
        assistant_response: &str,
        config: &MemoryConfig,
        model: &M
    ) -> Result<bool>
    where
        M: EmbeddingModel,
    {
        // Extract the content that will be stored (for embedding invariant)
        let stored_content = Self::extract_memory_content(memory_type, user_message, assistant_response);

        // First do the regular memory update and get the memory_id
        let memory_result = self.update_memory(memory_type, persona, user_message, assistant_response, config)?;
        let (memory_written, memory_id) = match memory_result {
            Some(id) => (true, id),
            None => (false, 0),
        };

        // Compute embedding from the SAME content that will be stored (embedding invariant)
        let embedding = if memory_written && config.vector_types.contains(&memory_type) {
            // Embed the SAME content that gets stored for semantic fidelity
            if config.debug {
                println!("[MEMORY] Computing embedding from stored content ({} chars): '{}...'",
                        stored_content.len(),
                        &stored_content[..stored_content.len().min(50)]);
            }

            if !stored_content.is_empty() {
                match model.compute_embedding(&stored_content).await {
                    Ok(emb) => {
                        if config.debug {
                            println!("[MEMORY] Computed embedding with {} dimensions from stored content", emb.len());
                        }
                        Some(emb)
                    }
                    Err(e) => {
                        if config.debug {
                            println!("[MEMORY] Failed to compute embedding from user message: {}", e);
                        }
                        None
                    }
                }
            } else {
                if config.debug {
                    println!("[MEMORY] User message is empty, skipping embedding");
                }
                None
            }
        } else {
            if config.debug && memory_written {
                println!("[MEMORY] Skipping embedding - type {} not in vector_types {:?}",
                        memory_type.as_str(), config.vector_types);
            }
            None
        };

        // Store the embedding if we have one
        if let Some(emb) = embedding {
            if memory_id > 0 {
                // Determine the actual memory type that was written (important for Auto policy)
                let actual_memory_type = match config.policy {
                    MemoryPolicy::Auto => {
                        if Self::is_identity_content(user_message) {
                            MemoryType::Persona
                        } else {
                            MemoryType::Conversation
                        }
                    }
                    _ => memory_type,
                };

                if config.debug {
                    println!("[MEMORY] Storing embedding for {} memory id {}", actual_memory_type.as_str(), memory_id);
                }
                if let Err(e) = self.store_memory_vector(memory_id, actual_memory_type, persona, &emb) {
                    if config.debug {
                        println!("[MEMORY] Failed to store embedding: {}", e);
                    }
                }
            } else {
                if config.debug {
                    println!("[MEMORY] No memory_id available for embedding storage");
                }
            }
        } else if config.debug && config.vector_types.contains(&memory_type) {
            println!("[MEMORY] No embedding computed for {} memory", memory_type.as_str());
        }

        Ok(memory_written)
    }

    /// Retrieve vector-based memory context
    pub async fn retrieve_vector_memory<M>(
        &self,
        persona: &str,
        query_text: &str,
        config: &MemoryConfig,
        model: &M
    ) -> Result<String>
    where
        M: EmbeddingModel,
    {
        if !config.mode.should_read() || config.vector_types.is_empty() {
            return Ok(String::new());
        }

        match model.compute_embedding(query_text).await {
            Ok(query_embedding) => {
                self.build_vector_memory_context(persona, &query_embedding, config)
            }
            Err(e) => {
                if config.debug {
                    println!("[MEMORY] Failed to compute query embedding: {}", e);
                }
                Ok(String::new())
            }
        }
    }

    pub fn is_identity_content(message: &str) -> bool {
        let lower = message.to_lowercase();
        lower.contains("i like") || lower.contains("i love") ||
        lower.contains("i hate") || lower.contains("i prefer") ||
        lower.contains("my name") || lower.contains("i am") ||
        lower.contains("i work") || lower.contains("i live") ||
        lower.contains("never") || lower.contains("always")
    }

    /// Check if message appears to be a question
    fn is_question(message: &str) -> bool {
        let trimmed = message.trim();
        trimmed.ends_with('?') ||
        trimmed.to_lowercase().starts_with("what") ||
        trimmed.to_lowercase().starts_with("how") ||
        trimmed.to_lowercase().starts_with("why") ||
        trimmed.to_lowercase().starts_with("when") ||
        trimmed.to_lowercase().starts_with("where") ||
        trimmed.to_lowercase().starts_with("who") ||
        trimmed.to_lowercase().starts_with("which") ||
        trimmed.to_lowercase().starts_with("do you") ||
        trimmed.to_lowercase().starts_with("can you") ||
        trimmed.to_lowercase().starts_with("would you") ||
        trimmed.to_lowercase().starts_with("could you")
    }

    /// Clean and truncate memory content for safe injection
    fn clean_memory_content(content: &str) -> String {
        let mut cleaned = content
            // Remove EOS tokens
            .replace("</s>", "")
            // Clean up excessive whitespace
            .split_whitespace()
            .collect::<Vec<&str>>()
            .join(" ")
            // Limit length to prevent context overflow (150 chars max per memory)
            .chars()
            .take(150)
            .collect::<String>();

        // If we truncated, add ellipsis
        if cleaned.len() < content.len() {
            cleaned.push_str("...");
        }

        cleaned
    }

    /// Extract focused memory content based on memory type
    pub fn extract_memory_content(memory_type: MemoryType, user_message: &str, assistant_response: &str) -> String {
        match memory_type {
            MemoryType::Persona => {
                // For persona memory, extract preferences and identity statements
                Self::extract_persona_facts(user_message, assistant_response)
            }
            MemoryType::Conversation => {
                // For conversation memory, extract key decisions or facts discussed
                Self::extract_conversation_keypoints(user_message, assistant_response)
            }
            MemoryType::Fact => {
                // For fact memory, extract factual information
                Self::extract_facts(user_message, assistant_response)
            }
        }
    }

    fn extract_persona_facts(user_message: &str, assistant_response: &str) -> String {
        let user_lower = user_message.to_lowercase();
        let response_lower = assistant_response.to_lowercase();

        // For vector memory semantic fidelity: store the actual user content that was embedded
        // Instead of generic labels, store semantically faithful content

        if user_lower.contains("my name is") || user_lower.contains("i am") ||
           user_lower.contains("i work") || user_lower.contains("i live") ||
           user_lower.contains("i'm a") || user_lower.contains("i'm an") ||
           user_lower.contains("i'm from") || user_lower.contains("i do") {
            // Identity statements: store the actual user message for semantic fidelity
            return user_message.to_string();
        }

        if user_lower.contains("i like") || user_lower.contains("i love") ||
           user_lower.contains("i prefer") || user_lower.contains("i enjoy") ||
           user_lower.contains("i'm into") || user_lower.contains("i'm a fan of") {
            // Preferences: store the actual user message for semantic fidelity
            return user_message.to_string();
        }

        if user_lower.contains("i know") || user_lower.contains("i can") ||
           user_lower.contains("i'm good at") || user_lower.contains("i'm experienced") ||
           user_lower.contains("i work with") || user_lower.contains("i use") {
            // Skills/expertise: store the actual user message for semantic fidelity
            return user_message.to_string();
        }

        // Fallback: extract structured facts only for non-semantic content
        let mut facts = Vec::new();

        if user_lower.contains("i hate") || user_lower.contains("i dislike") ||
           user_lower.contains("i don't like") || user_lower.contains("i can't stand") {
            facts.push(format!("User dislikes: {}", Self::extract_preference(user_message)));
        }

        if user_lower.contains("never") || user_lower.contains("always") ||
           user_lower.contains("usually") || user_lower.contains("typically") {
            facts.push(format!("User habits: {}", Self::extract_habit(user_message)));
        }

        // Extract assistant confirmations of user facts
        if response_lower.contains("noted") || response_lower.contains("understood") ||
           response_lower.contains("remember") || response_lower.contains("got it") ||
           response_lower.contains("interesting") || response_lower.contains("good to know") {
            if let Some(user_fact) = Self::find_user_fact_in_response(user_message, assistant_response) {
                facts.push(format!("Confirmed: {}", user_fact));
            }
        }

        if facts.is_empty() {
            // Fallback: store a cleaned version of the interaction
            format!("User: {}\nAssistant: {}", Self::clean_memory_content(user_message), Self::clean_memory_content(assistant_response))
        } else {
            facts.join("; ")
        }
    }

    fn extract_conversation_keypoints(user_message: &str, assistant_response: &str) -> String {
        let mut keypoints = Vec::new();
        let user_lower = user_message.to_lowercase();
        let response_lower = assistant_response.to_lowercase();

        // Look for decisions or choices
        if user_lower.contains("choose") || user_lower.contains("decide") ||
           user_lower.contains("prefer") || user_lower.contains("select") {
            keypoints.push("Made a decision/choice".to_string());
        }

        // Look for agreements or disagreements
        if response_lower.contains("agree") || response_lower.contains("yes") ||
           response_lower.contains("good idea") || response_lower.contains("sounds good") {
            keypoints.push("Reached agreement".to_string());
        }

        // Look for questions answered
        if user_message.contains("?") && !response_lower.contains("sorry") {
            keypoints.push("Question answered".to_string());
        }

        // Look for plans or next steps
        if response_lower.contains("next") || response_lower.contains("plan") ||
           response_lower.contains("then") || response_lower.contains("after") ||
           response_lower.contains("following") {
            keypoints.push("Discussed next steps".to_string());
        }

        // Look for explanations or information sharing
        if response_lower.contains("here") || response_lower.contains("this is") ||
           response_lower.contains("you can") || response_lower.contains("to do this") ||
           response_lower.len() > 100 {
            keypoints.push("Shared information/explanation".to_string());
        }

        // Look for personal information or preferences
        if user_lower.contains("i am") || user_lower.contains("i like") ||
           user_lower.contains("i work") || user_lower.contains("my") {
            keypoints.push("Shared personal information".to_string());
        }

        if keypoints.is_empty() {
            // Extract a summary of the main topic - improved version
            Self::summarize_interaction(user_message, assistant_response)
        } else {
            keypoints.join("; ")
        }
    }

    fn extract_facts(user_message: &str, assistant_response: &str) -> String {
        // For now, facts are explicitly stated information
        // This could be expanded with better fact extraction
        let combined = format!("{} {}", user_message, assistant_response);
        let lower = combined.to_lowercase();

        if lower.contains("fact") || lower.contains("actually") ||
           lower.contains("true") || lower.contains("correct") {
            format!("Fact: {}", Self::clean_memory_content(&combined))
        } else {
            // No clear facts, don't store
            String::new()
        }
    }

    fn extract_preference(message: &str) -> String {
        // Simple extraction of what comes after preference words
        let words: Vec<&str> = message.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let w_lower = word.to_lowercase();
            if w_lower == "like" || w_lower == "love" || w_lower == "prefer" ||
               w_lower == "enjoy" || w_lower == "hate" || w_lower == "dislike" {
                if i + 1 < words.len() {
                    return words[i + 1..].join(" ");
                }
            }
        }
        message.to_string()
    }

    fn extract_identity(message: &str) -> String {
        let lower = message.to_lowercase();

        if lower.contains("my name is") {
            if let Some(start) = message.to_lowercase().find("my name is") {
                let after = &message[start + 11..];
                if let Some(end) = after.find(|c: char| !c.is_alphanumeric() && c != ' ') {
                    return after[..end].trim().to_string();
                }
                return after.trim().to_string();
            }
        }

        if lower.contains("i am") {
            if let Some(start) = message.to_lowercase().find("i am") {
                let after = &message[start + 5..];
                if let Some(end) = after.find(|c: char| c == '.' || c == '!' || c == '?') {
                    return after[..end].trim().to_string();
                }
                return after.trim().to_string();
            }
        }

        message.to_string()
    }

    fn extract_habit(message: &str) -> String {
        let _lower = message.to_lowercase();
        if _lower.contains("never") {
            "Never does certain things".to_string()
        } else if _lower.contains("always") {
            "Always does certain things".to_string()
        } else if _lower.contains("usually") {
            "Usually does certain things".to_string()
        } else if _lower.contains("typically") {
            "Typically does certain things".to_string()
        } else {
            message.to_string()
        }
    }

    fn extract_skills(message: &str) -> String {
        // Extract skills and expertise from user statements
        let _lower = message.to_lowercase();
        let words: Vec<&str> = message.split_whitespace().collect();

        for (i, word) in words.iter().enumerate() {
            let w_lower = word.to_lowercase();
            if (w_lower == "know" || w_lower == "can" || w_lower == "good" ||
                w_lower == "experienced" || w_lower == "work" || w_lower == "use") &&
               i + 1 < words.len() {
                return words[i + 1..].join(" ");
            }
            if w_lower == "i'm" && i + 1 < words.len() && i + 2 < words.len() {
                let next_word = words[i + 1].to_lowercase();
                if next_word == "good" || next_word == "experienced" {
                    return words[i + 2..].join(" ");
                }
            }
        }
        message.to_string()
    }

    fn find_user_fact_in_response(user_message: &str, assistant_response: &str) -> Option<String> {
        // Look for user statements that the assistant acknowledges
        let user_lower = user_message.to_lowercase();
        let response_lower = assistant_response.to_lowercase();

        if (user_lower.contains("i like") || user_lower.contains("i love")) &&
           (response_lower.contains("noted") || response_lower.contains("good to know")) {
            Some(Self::extract_preference(user_message))
        } else if user_lower.contains("my name is") &&
                  (response_lower.contains("nice to meet") || response_lower.contains("hello")) {
            Some(Self::extract_identity(user_message))
        } else {
            None
        }
    }

    fn summarize_interaction(user_message: &str, assistant_response: &str) -> String {
        // Create a more meaningful summary of the interaction
        let user_words: Vec<&str> = user_message.split_whitespace().take(10).collect();
        let response_words: Vec<&str> = assistant_response.split_whitespace().take(10).collect();

        // Try to extract meaningful content
        let user_preview = if user_words.len() > 3 {
            format!("{} {} {}", user_words[0], user_words[1], user_words[2])
        } else {
            user_words.join(" ")
        };

        let response_preview = if response_words.len() > 3 {
            format!("{} {} {}", response_words[0], response_words[1], response_words[2])
        } else {
            response_words.join(" ")
        };

        format!("User: {}... Assistant: {}...", user_preview, response_preview)
    }

    /// Retrieve individual memory entries for a persona and memory type
    pub fn get_memory_entries(&self, memory_type: MemoryType, persona: &str) -> Result<Vec<MemoryEntry>> {
        let mut stmt = self.conn.prepare(
            "SELECT memory_type, persona, content, timestamp FROM memories
             WHERE memory_type = ?1 AND persona = ?2
             ORDER BY timestamp DESC"
        )?;

        let rows = stmt.query_map([memory_type.as_str(), persona], |row| {
            Ok(MemoryEntry {
                memory_type: match row.get::<_, String>(0)?.as_str() {
                    "persona" => MemoryType::Persona,
                    "conversation" => MemoryType::Conversation,
                    "fact" => MemoryType::Fact,
                    _ => MemoryType::Conversation, // fallback
                },
                persona: row.get(1)?,
                content: row.get(2)?,
                timestamp: row.get(3)?,
            })
        })?;

        let mut entries = Vec::new();
        for row in rows {
            entries.push(row?);
        }

        Ok(entries)
    }

    /// Calculate relevance score based on keyword overlap (excluding stop words)
    pub fn calculate_relevance_score(memory_content: &str, user_message: &str) -> usize {
        // Define common stop words to ignore
        let stop_words: std::collections::HashSet<&str> = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "could", "should", "may", "might", "must", "can", "shall",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my",
            "your", "his", "its", "our", "their", "this", "that", "these", "those", "what", "when",
            "where", "why", "how", "who", "which", "all", "any", "both", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
            "than", "too", "very", "just", "also", "as", "if", "then", "else", "while", "about",
            "above", "after", "again", "against", "all", "am", "because", "before", "below",
            "between", "during", "from", "into", "over", "through", "under", "until", "up",
            "down", "out", "off", "over", "under", "again", "further", "once"
        ].into();

        // Extract meaningful words from user message
        let user_words: std::collections::HashSet<String> = user_message
            .to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2) // Skip very short words
            .filter(|word| !stop_words.contains(word))
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty())
            .collect();

        if user_words.is_empty() {
            return 0;
        }

        // Extract meaningful words from memory content
        let memory_words: std::collections::HashSet<String> = memory_content
            .to_lowercase()
            .split_whitespace()
            .filter(|word| word.len() > 2)
            .filter(|word| !stop_words.contains(word))
            .map(|word| word.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
            .filter(|word| !word.is_empty())
            .collect();

        // Count overlapping words
        user_words.intersection(&memory_words).count()
    }

    /// Retrieve relevant memories based on user message content
    pub fn retrieve_relevant_memories(&self, persona: &str, user_message: &str, config: &MemoryConfig) -> Result<Vec<ScoredMemory>> {
        if !config.mode.should_read() {
            return Ok(Vec::new());
        }

        let mut all_scored = Vec::new();

        // Retrieve and score memories for each type
        for &memory_type in &[MemoryType::Persona, MemoryType::Conversation, MemoryType::Fact] {
            let entries = self.get_memory_entries(memory_type, persona)?;

            for entry in entries {
                let score = Self::calculate_relevance_score(&entry.content, user_message);
                if score > 0 {
                    all_scored.push(ScoredMemory { entry, score });
                }
            }
        }

        // Sort by score (descending), then by timestamp (descending)
        all_scored.sort_by(|a, b| {
            match b.score.cmp(&a.score) {
                std::cmp::Ordering::Equal => b.entry.timestamp.cmp(&a.entry.timestamp),
                other => other,
            }
        });

        Ok(all_scored)
    }

    /// Build retrieved memory context with relevance scoring
    pub fn build_retrieved_memory_context(&self, persona: &str, user_message: &str, config: &MemoryConfig) -> Result<String> {
        let relevant_memories = self.retrieve_relevant_memories(persona, user_message, config)?;

        if relevant_memories.is_empty() {
            if config.debug {
                println!("[MEMORY] No relevant memories found");
            }
            return Ok(String::new());
        }

        // Group by memory type and apply limits
        let mut persona_memories = Vec::new();
        let mut conversation_memories = Vec::new();
        let mut fact_memories = Vec::new();

        for scored in relevant_memories {
            match scored.entry.memory_type {
                MemoryType::Persona if persona_memories.len() < 1 => {
                    persona_memories.push(scored);
                }
                MemoryType::Conversation if conversation_memories.len() < 2 => {
                    conversation_memories.push(scored);
                }
                MemoryType::Fact if fact_memories.len() < 2 => {
                    fact_memories.push(scored);
                }
                _ => {} // Skip if limit reached for this type
            }
        }

        let mut context_parts = Vec::new();

        // Add persona memories
        if !persona_memories.is_empty() {
            context_parts.push("PERSONA MEMORY:".to_string());
            for scored in &persona_memories {
                let cleaned_content = Self::clean_memory_content(&scored.entry.content);
                context_parts.push(format!("- {}", cleaned_content));
                if config.debug {
                    println!("[MEMORY] Retrieved Persona (score: {}): {}", scored.score, cleaned_content.lines().next().unwrap_or(""));
                }
            }
        }

        // Add conversation memories
        if !conversation_memories.is_empty() {
            context_parts.push("RECENT CONVERSATION:".to_string());
            for scored in &conversation_memories {
                let cleaned_content = Self::clean_memory_content(&scored.entry.content);
                context_parts.push(format!("- {}", cleaned_content));
                if config.debug {
                    println!("[MEMORY] Retrieved Conversation (score: {}): {}", scored.score, cleaned_content.lines().next().unwrap_or(""));
                }
            }
        }

        // Add fact memories
        if !fact_memories.is_empty() {
            context_parts.push("KEY FACTS:".to_string());
            for scored in &fact_memories {
                let cleaned_content = Self::clean_memory_content(&scored.entry.content);
                context_parts.push(format!("- {}", cleaned_content));
                if config.debug {
                    println!("[MEMORY] Retrieved Fact (score: {}): {}", scored.score, cleaned_content.lines().next().unwrap_or(""));
                }
            }
        }

        let raw_context = if context_parts.is_empty() {
            String::new()
        } else {
            format!("RELEVANT MEMORY:\n{}", context_parts.join("\n"))
        };

        // Final safety check: limit total memory context to 1000 characters
        let final_context = if raw_context.len() > 1000 {
            let truncated: String = raw_context.chars().take(1000).collect();
            format!("{}...\n[Memory truncated for length]", truncated)
        } else {
            raw_context
        };

        if config.debug && !final_context.is_empty() {
            println!("[MEMORY] Injected memory block ({} chars):\n{}", final_context.len(), final_context);
        }

        Ok(final_context)
    }

    /// Store embedding for a memory entry
    pub fn store_memory_vector(&self, memory_id: i64, memory_type: MemoryType, persona: &str, embedding: &[f32]) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Convert embedding to bytes (little-endian)
        let mut embedding_bytes = Vec::with_capacity(embedding.len() * 4);
        for &value in embedding {
            embedding_bytes.extend_from_slice(&value.to_le_bytes());
        }

        self.conn.execute(
            "INSERT OR REPLACE INTO memory_vectors
             (memory_id, memory_type, persona, embedding, embedding_dim, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                memory_id,
                memory_type.as_str(),
                persona,
                embedding_bytes,
                embedding.len() as i32,
                timestamp
            ],
        )?;

        Ok(())
    }

    /// Store memory vector within a transaction
    pub fn store_memory_vector_in_tx(&self, tx: &rusqlite::Transaction, memory_id: i64, memory_type: MemoryType, persona: &str, embedding: &[f32]) -> Result<()> {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        // Convert embedding to bytes (little-endian)
        let mut embedding_bytes = Vec::with_capacity(embedding.len() * 4);
        for &value in embedding {
            embedding_bytes.extend_from_slice(&value.to_le_bytes());
        }

        tx.execute(
            "INSERT OR REPLACE INTO memory_vectors
             (memory_id, memory_type, persona, embedding, embedding_dim, timestamp)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                memory_id,
                memory_type.as_str(),
                persona,
                embedding_bytes,
                embedding.len() as i32,
                timestamp
            ],
        )?;

        Ok(())
    }

    /// Check if content has concrete entities (not just generic placeholders)
    fn has_concrete_entities(content: &str) -> bool {
        let lower = content.to_lowercase();

        // Skip generic/abstract content
        if lower.contains("shared information") ||
           lower.contains("shared personal") ||
           lower.contains("discussed next steps") ||
           lower.contains("user identity:") ||
           lower.contains("user likes:") ||
           lower.contains("user skills:") ||
           lower.starts_with("confirmed:") {
            return false;
        }

        // Check for concrete entities
        let concrete_indicators = [
            "rust", "python", "java", "javascript", "c++", "c#", "go", "typescript",
            "developer", "engineer", "programmer", "specialize", "expert",
            "senior", "junior", "lead", "architect",
            "web", "mobile", "backend", "frontend", "fullstack", "devops",
            "database", "api", "microservice", "cloud", "aws", "azure", "gcp",
            "linux", "windows", "mac", "docker", "kubernetes",
            "react", "angular", "vue", "node", "django", "flask", "spring",
            "mysql", "postgresql", "mongodb", "redis", "elasticsearch"
        ];

        concrete_indicators.iter().any(|&indicator| lower.contains(indicator))
    }

    /// Search for similar vectors using cosine similarity (dot product since normalized)
    pub fn search_vectors(&self, persona: &str, memory_types: &[MemoryType], query_embedding: &[f32], top_k: usize, threshold: f32) -> Result<Vec<(MemoryType, String, f32, i64)>> {
        let mut results = Vec::new();

        for &memory_type in memory_types {
            let mut stmt = self.conn.prepare(
                "SELECT mv.memory_id, mv.embedding, m.content
                 FROM memory_vectors mv
                 JOIN memories m ON mv.memory_id = m.id
                 WHERE mv.persona = ?1 AND mv.memory_type = ?2"
            )?;

            // Debug: count how many vectors we found for this type
            let count: i64 = self.conn.query_row(
                "SELECT COUNT(*) FROM memory_vectors WHERE persona = ?1 AND memory_type = ?2",
                [persona, memory_type.as_str()],
                |row| row.get(0)
            ).unwrap_or(0);

            println!("[MEMORY] Found {} stored vectors for {} type", count, memory_type.as_str());

            let rows = stmt.query_map([persona, memory_type.as_str()], |row| {
                let memory_id: i64 = row.get(0)?;
                let embedding_bytes: Vec<u8> = row.get(1)?;
                let content: String = row.get(2)?;

                // Convert bytes back to f32 vector
                let mut embedding = Vec::with_capacity(embedding_bytes.len() / 4);
                for chunk in embedding_bytes.chunks_exact(4) {
                    let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    embedding.push(value);
                }

                Ok((memory_id, embedding, content))
            })?;

            for row in rows {
                let (memory_id, stored_embedding, content) = row?;

                // Compute cosine similarity (dot product since vectors are L2 normalized)
                let similarity = stored_embedding.iter()
                    .zip(query_embedding.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();

                if similarity >= threshold {
                    results.push((memory_type, content, similarity, memory_id));
                }
            }
        }

        // Sort by similarity descending and take top_k
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);

        Ok(results)
    }

    /// Build vector-based memory context
    pub fn build_vector_memory_context(&self, persona: &str, query_embedding: &[f32], config: &MemoryConfig) -> Result<String> {
        if !config.mode.should_read() || config.vector_types.is_empty() {
            return Ok(String::new());
        }

        if config.debug {
            println!("[MEMORY] Searching vector memories for persona '{}' with {} types, threshold {:.2}, top_k {}",
                    persona, config.vector_types.len(), config.vector_threshold, config.vector_top_k);
        }

        let vector_results = self.search_vectors(
            persona,
            &config.vector_types,
            query_embedding,
            config.vector_top_k,
            config.vector_threshold
        )?;

        if config.debug {
            println!("[MEMORY] Vector search returned {} results", vector_results.len());
            for (mem_type, content, score, mem_id) in &vector_results {
                println!("[MEMORY] Found {} memory (id: {}, score: {:.3}): {}",
                        mem_type.as_str(), mem_id, score, &content[..content.len().min(50)]);
            }
        }

        if vector_results.is_empty() {
            if config.debug {
                println!("[MEMORY] No relevant vector memories found");
            }
            return Ok(String::new());
        }

        // Filter results to only include content with concrete entities (injection guardrail)
        let mut filtered_results = Vec::new();
        for (memory_type, content, score, memory_id) in &vector_results {
            if Self::has_concrete_entities(content) {
                filtered_results.push((memory_type.clone(), content.clone(), *score, *memory_id));
            } else {
                if config.debug {
                    println!("[MEMORY] Skipping injection of {} memory (id: {}): no concrete entities",
                            memory_type.as_str(), memory_id);
                }
            }
        }

        if filtered_results.is_empty() {
            if config.debug {
                println!("[MEMORY] No vector memories with concrete entities to inject");
            }
            return Ok(String::new());
        }

        let mut context_parts = Vec::new();
        context_parts.push("RELEVANT MEMORY:".to_string());

        for (memory_type, content, score, memory_id) in &filtered_results {
            let type_str = match memory_type {
                MemoryType::Persona => "persona",
                MemoryType::Conversation => "conversation",
                MemoryType::Fact => "fact",
            };

            let cleaned_content = Self::clean_memory_content(content);
            context_parts.push(format!("- [type={} score={:.3} id={}] {}",
                                     type_str, score, memory_id, cleaned_content));

            if config.debug {
                println!("[MEMORY] Retrieved {} (score: {:.3}, id: {}): {}",
                        type_str, score, memory_id, cleaned_content.lines().next().unwrap_or(""));
            }
        }

        let context = context_parts.join("\n");

        if config.debug {
            println!("[MEMORY] Injected vector memory block ({} chars):\n{}", context.len(), context);
        }

        Ok(context)
    }
}

// Backward compatibility alias
pub type PersonaMemory = IntelligentMemory;

/*
PHASE 11: VECTOR MEMORY TEST PLAN
==================================

This implements semantic recall using BERT embeddings + SQLite vector storage.

TEST SCENARIO 1: Basic Vector Memory Storage
-------------------------------------------
1. Start server: ./inference-server --memory-debug --memory-types persona,conversation,fact --memory-k 3 --memory-threshold 0.7
   Or CLI: ./cli --memory-debug --memory-types persona,conversation,fact --memory-k 3 --memory-threshold 0.7
2. Send messages that should be stored:
   curl -X POST http://localhost:3000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "default", "messages": [{"role": "user", "content": "My name is Alice and I love programming"}], "memory_update": "write"}'
3. Check logs for: "[MEMORY] Writing due to preference/identity expression"
4. Verify embeddings stored: sqlite3 data.db "SELECT COUNT(*) FROM memory_vectors;"

TEST SCENARIO 2: Vector Retrieval
---------------------------------
1. Send a related question:
   curl -X POST http://localhost:3000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{"model": "default", "messages": [{"role": "user", "content": "What programming languages do I like?"}]}'
2. Check logs for:
   - "[MEMORY] Retrieved persona (score: 0.85, id: X): User likes: programming"
   - "RELEVANT MEMORY:" section in response

TEST SCENARIO 3: CLI Vector Memory
----------------------------------
1. Run CLI: ./cli --memory-debug --memory-types persona,conversation --memory-threshold 0.8
2. Input: "I am a software engineer from Seattle"
3. Input: "What do I do for work and where am I from?"
4. Check logs show vector memory injection with scores

TEST SCENARIO 4: Threshold Filtering
-----------------------------------
1. Set high threshold: --memory-threshold 0.95
2. Send loosely related question
3. Verify no vector memories retrieved (below threshold)

TEST SCENARIO 5: Multiple Memory Types
--------------------------------------
1. Store different types:
   - Persona: "I prefer Python over JavaScript"
   - Fact: "The Earth orbits the Sun"
   - Conversation: "We discussed machine learning yesterday"
2. Ask: "What programming language do I prefer?"
3. Should retrieve persona memory with high score

TEST SCENARIO 6: Embedding Failure Graceful Degradation
-------------------------------------------------------
1. Remove/disable BERT model
2. Send messages - should still work with keyword-based memory
3. Logs should show: "[MEMORY] Failed to compute embedding..."
4. System startup should show: "Warning: Could not load embedding model..."

TEST SCENARIO 7: Embedding Model Availability Check
---------------------------------------------------
1. Start server with working embedding model
2. Check logs show successful model loading
3. Send vector memory queries - should work with embeddings
4. Temporarily rename models/minilm directory
5. Restart server - should show warning and disable vector features
6. Send same queries - should fall back to keyword-based memory

EXPECTED LOGS:
- Skip: "[MEMORY] Skipping - greeting message"
- Write: "[MEMORY] Writing due to preference/identity expression"
- Retrieve: "[MEMORY] Retrieved persona (score: 0.XX, id: X): content..."
- Inject: "[MEMORY] Injected vector memory block (XX chars)"

DEBUGGING:
- Enable --memory-debug to see all memory operations
- Check SQLite: SELECT * FROM memory_vectors LIMIT 5;
- Monitor embedding dimensions: all should be same size
*/

