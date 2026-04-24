//! Configuration model for `embed`.

/// Top-level config payload for the `embed` implementation.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EmbedConfig {
    /// Logical model identifier that must match the loaded model.
    pub model_id: String,
    /// Maximum accepted number of texts per request.
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,
    /// Inference timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

pub(crate) const fn default_max_batch_size() -> usize {
    32
}

pub(crate) const fn default_timeout_ms() -> u64 {
    10_000
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value as JsonValue, json};

    #[test]
    fn deserialize_rejects_unknown_fields() {
        let value = json!({
            "model_id": "paraphrase-multilingual-MiniLM-L12-v2",
            "max_batch_size": 16,
            "timeout_ms": 5000,
            "extra": true
        });

        let err = serde_json::from_value::<EmbedConfig>(value).unwrap_err();
        assert!(err.to_string().contains("unknown field"));
    }

    #[test]
    fn defaults_for_batch_size_and_timeout_apply() {
        let value: JsonValue = json!({
            "model_id": "paraphrase-multilingual-MiniLM-L12-v2"
        });

        let config: EmbedConfig = serde_json::from_value(value).unwrap();
        assert_eq!(config.max_batch_size, 32);
        assert_eq!(config.timeout_ms, 10_000);
    }

    #[test]
    fn model_id_is_required() {
        let value = json!({
            "max_batch_size": 32,
            "timeout_ms": 10_000
        });

        let err = serde_json::from_value::<EmbedConfig>(value).unwrap_err();
        assert!(err.to_string().contains("model_id"));
    }
}
