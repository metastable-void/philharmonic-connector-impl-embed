//! Request model for `embed`.

use crate::error::{Error, Result};

/// One `embed` request payload.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EmbedRequest {
    /// Input texts to embed.
    pub texts: Vec<String>,
}

impl EmbedRequest {
    pub(crate) fn validate(&self, max_batch_size: usize) -> Result<()> {
        if self.texts.is_empty() {
            return Err(Error::InvalidRequest(
                "texts must contain at least one item".to_owned(),
            ));
        }

        let text_count = self.texts.len();
        if text_count > max_batch_size {
            return Err(Error::InvalidRequest(format!(
                "texts length {text_count} exceeds max_batch_size {max_batch_size}"
            )));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn deserialize_valid_shape() {
        let value = json!({
            "texts": ["hello", "world"]
        });

        let request: EmbedRequest = serde_json::from_value(value).unwrap();
        assert_eq!(request.texts, vec!["hello", "world"]);
    }

    #[test]
    fn reject_non_string_text_elements() {
        let value = json!({
            "texts": ["hello", 42]
        });

        let err = serde_json::from_value::<EmbedRequest>(value).unwrap_err();
        assert!(err.to_string().contains("string"));
    }

    #[test]
    fn reject_empty_array() {
        let request = EmbedRequest { texts: Vec::new() };

        let err = request.validate(32).unwrap_err();
        assert_eq!(
            err,
            Error::InvalidRequest("texts must contain at least one item".to_owned())
        );
    }

    #[test]
    fn reject_missing_texts_field() {
        let value = json!({});

        let err = serde_json::from_value::<EmbedRequest>(value).unwrap_err();
        assert!(err.to_string().contains("texts"));
    }
}
