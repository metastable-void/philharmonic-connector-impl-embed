//! Response model for `embed`.

/// One `embed` response payload.
#[derive(Debug, Clone, serde::Deserialize, serde::Serialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EmbedResponse {
    /// Output embeddings, one vector per input text.
    pub embeddings: Vec<Vec<f32>>,
    /// Echoed model identifier.
    pub model: String,
    /// Embedding dimensionality.
    pub dimensions: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_roundtrips_json() {
        let response = EmbedResponse {
            embeddings: vec![vec![0.1_f32, -0.2_f32], vec![0.3_f32, 0.4_f32]],
            model: "paraphrase-multilingual-MiniLM-L12-v2".to_owned(),
            dimensions: 2,
        };

        let encoded = serde_json::to_value(&response).unwrap();
        let decoded: EmbedResponse = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, response);
    }
}
