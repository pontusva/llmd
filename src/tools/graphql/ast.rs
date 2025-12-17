//! Minimal GraphQL AST for deterministic query construction
//!
//! This AST serves as a compiler IR, making invalid queries unrepresentable
//! and guaranteeing that required filters are never lost.

/// Root GraphQL query
#[derive(Debug, Clone, PartialEq)]
pub struct GqlQuery {
    pub root: GqlField,
}

/// A GraphQL field with optional arguments and selection
#[derive(Debug, Clone, PartialEq)]
pub struct GqlField {
    pub name: String,
    pub arguments: Vec<GqlArgument>,
    pub selection: Vec<GqlField>,
}

/// A GraphQL argument with name and value
#[derive(Debug, Clone, PartialEq)]
pub struct GqlArgument {
    pub name: String,
    pub value: GqlValue,
}

/// GraphQL value types (subset needed for our queries)
#[derive(Debug, Clone, PartialEq)]
pub enum GqlValue {
    String(String),
    Number(i64),
    Boolean(bool),
    Object(Vec<(String, GqlValue)>),
}

impl GqlField {
    /// Create a new field with no arguments or selection
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arguments: Vec::new(),
            selection: Vec::new(),
        }
    }

    /// Add an argument to this field (fluent API)
    pub fn arg(mut self, name: &str, value: GqlValue) -> Self {
        self.arguments.push(GqlArgument {
            name: name.to_string(),
            value,
        });
        self
    }

    /// Add a sub-field to the selection (fluent API)
    pub fn select(mut self, field: GqlField) -> Self {
        self.selection.push(field);
        self
    }
}

impl GqlQuery {
    /// Serialize the query to GraphQL string format
    pub fn to_string(&self) -> String {
        format!("query {{ {} }}", self.root.to_string())
    }
}

impl GqlField {
    /// Serialize a field to GraphQL string format
    fn to_string(&self) -> String {
        let mut result = self.name.clone();

        // Add arguments if present
        if !self.arguments.is_empty() {
            let args_str = self.arguments.iter()
                .map(|arg| format!("{}: {}", arg.name, arg.value.to_string()))
                .collect::<Vec<_>>()
                .join(", ");
            result.push_str(&format!("({})", args_str));
        }

        // Add selection if present
        if !self.selection.is_empty() {
            let selection_str = self.selection.iter()
                .map(|field| field.to_string())
                .collect::<Vec<_>>()
                .join(" ");
            result.push_str(&format!(" {{ {} }}", selection_str));
        }

        result
    }
}

impl GqlValue {
    /// Serialize a GraphQL value to string format
    fn to_string(&self) -> String {
        match self {
            GqlValue::String(s) => format!(r#""{}""#, s),
            GqlValue::Number(n) => n.to_string(),
            GqlValue::Boolean(b) => b.to_string(),
            GqlValue::Object(pairs) => {
                if pairs.is_empty() {
                    "{}".to_string()
                } else {
                    let pairs_str = pairs.iter()
                        .map(|(k, v)| format!("{}: {}", k, v.to_string()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("{{ {} }}", pairs_str)
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn component_window_count_query() {
        let where_clause = GqlValue::Object(vec![
            ("buildingId".into(), GqlValue::String("mock-building-123".into())),
            ("type".into(), GqlValue::String("window".into())),
        ]);

        let query = GqlQuery {
            root: GqlField::new("components")
                .arg("where", where_clause)
                .select(
                    GqlField::new("_count")
                        .select(GqlField::new("id"))
                ),
        };

        let expected = r#"query { components(where: { buildingId: "mock-building-123", type: "window" }) { _count { id } } }"#;
        assert_eq!(query.to_string(), expected);
    }

    #[test]
    fn building_list_query() {
        let where_clause = GqlValue::Object(vec![
            ("teamId".into(), GqlValue::String("team-123".into())),
        ]);

        let query = GqlQuery {
            root: GqlField::new("buildings")
                .arg("where", where_clause)
                .select(GqlField::new("id"))
                .select(GqlField::new("name")),
        };

        let expected = r#"query { buildings(where: { teamId: "team-123" }) { id name } }"#;
        assert_eq!(query.to_string(), expected);
    }

    #[test]
    fn measure_count_with_status() {
        let where_clause = GqlValue::Object(vec![
            ("buildingId".into(), GqlValue::String("mock-building-456".into())),
            ("status".into(), GqlValue::String("completed".into())),
        ]);

        let query = GqlQuery {
            root: GqlField::new("measures")
                .arg("where", where_clause)
                .select(
                    GqlField::new("_count")
                        .select(GqlField::new("id"))
                ),
        };

        let expected = r#"query { measures(where: { buildingId: "mock-building-456", status: "completed" }) { _count { id } } }"#;
        assert_eq!(query.to_string(), expected);
    }

    #[test]
    fn building_aggregate_floor_area() {
        let where_clause = GqlValue::Object(vec![
            ("teamId".into(), GqlValue::String("team-123".into())),
        ]);

        let query = GqlQuery {
            root: GqlField::new("buildings")
                .arg("where", where_clause)
                .select(
                    GqlField::new("_sum")
                        .select(GqlField::new("totalFloorArea"))
                ),
        };

        let expected = r#"query { buildings(where: { teamId: "team-123" }) { _sum { totalFloorArea } } }"#;
        assert_eq!(query.to_string(), expected);
    }

    #[test]
    fn empty_where_clause() {
        let where_clause = GqlValue::Object(vec![]);

        let query = GqlQuery {
            root: GqlField::new("buildings")
                .arg("where", where_clause)
                .select(GqlField::new("id")),
        };

        let expected = r#"query { buildings(where: {}) { id } }"#;
        assert_eq!(query.to_string(), expected);
    }

    #[test]
    fn field_without_arguments_or_selection() {
        let field = GqlField::new("id");
        assert_eq!(field.to_string(), "id");
    }
}
