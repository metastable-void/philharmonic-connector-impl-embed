//! fastembed model wrapper for `embed`.

use crate::error::{Error, Result};
use fastembed::{InitOptionsUserDefined, TextEmbedding, TokenizerFiles, UserDefinedEmbeddingModel};
use std::sync::Mutex;

pub(crate) struct Model {
    embedder: Mutex<Option<TextEmbedding>>,
    model_id: String,
    dimensions: usize,
    max_seq_length: usize,
}

impl Model {
    pub(crate) fn new_from_bytes(
        model_id: impl Into<String>,
        onnx_bytes: Vec<u8>,
        tokenizer_files: TokenizerFiles,
        dimensions: usize,
        max_seq_length: usize,
    ) -> Result<Self> {
        if dimensions == 0 {
            return Err(Error::InvalidConfig("dimensions must be >= 1".to_owned()));
        }

        if max_seq_length == 0 {
            return Err(Error::InvalidConfig(
                "max_seq_length must be >= 1".to_owned(),
            ));
        }

        let model_id = model_id.into();
        let user_defined_model = UserDefinedEmbeddingModel::new(onnx_bytes, tokenizer_files);
        let init_options = InitOptionsUserDefined::new().with_max_length(max_seq_length);

        let embedding = TextEmbedding::try_new_from_user_defined(user_defined_model, init_options)
            .map_err(|err| {
                Error::InvalidConfig(format!(
                    "failed to initialize model '{model_id}' from provided bytes: {err}"
                ))
            })?;

        Ok(Self {
            embedder: Mutex::new(Some(embedding)),
            model_id,
            dimensions,
            max_seq_length,
        })
    }

    pub(crate) fn model_id(&self) -> &str {
        &self.model_id
    }

    pub(crate) fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub(crate) fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut guard = self
            .embedder
            .lock()
            .map_err(|err| Error::Internal(format!("embedder mutex poisoned: {err}")))?;

        let raw_embeddings = match &mut *guard {
            Some(embedder) => embedder
                .embed(texts.to_vec(), None)
                .map_err(|err| Error::Internal(format!("embedding inference failed: {err}")))?,
            None => {
                return Err(Error::Internal(
                    "embedder unavailable in test-only constructor".to_owned(),
                ));
            }
        };

        let embeddings: Vec<Vec<f32>> = raw_embeddings
            .into_iter()
            .map(|embedding| embedding.into_iter().collect::<Vec<f32>>())
            .collect();

        if embeddings.len() != texts.len() {
            return Err(Error::Internal(format!(
                "embedding count mismatch: expected {}, got {}",
                texts.len(),
                embeddings.len()
            )));
        }

        let invalid_index = embeddings
            .iter()
            .position(|embedding| embedding.len() != self.dimensions);
        if let Some(index) = invalid_index {
            return Err(Error::Internal(format!(
                "embedding dimensions mismatch at index {index}: expected {}, got {} (max_seq_length={})",
                self.dimensions,
                embeddings[index].len(),
                self.max_seq_length
            )));
        }

        Ok(embeddings)
    }

    #[cfg(test)]
    pub(crate) fn for_tests(model_id: &str, dimensions: usize, max_seq_length: usize) -> Self {
        Self {
            embedder: Mutex::new(None),
            model_id: model_id.to_owned(),
            dimensions,
            max_seq_length,
        }
    }
}
