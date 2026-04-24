//! Local in-process embedding connector implementation for Philharmonic.
//!
//! `embed` implements the shared
//! [`philharmonic_connector_impl_api::Implementation`] trait using
//! CPU inference from a model bundle that is provided by the deployment
//! binary at startup.
//!
//! ## Architecture
//!
//! This crate does not download model weights, read files, or read
//! environment variables. Deployments bundle ONNX + tokenizer assets
//! (for example with `include_bytes!`) and pass those bytes into
//! [`Embed::new_from_bytes`]. Model loading is eager so malformed bundles
//! fail at startup instead of mid-workflow.
//!
//! ## Runtime behavior
//!
//! - `execute` validates config/request JSON.
//! - Inference runs in `tokio::task::spawn_blocking` because fastembed is
//!   CPU-bound and blocking.
//! - The call is bounded by `timeout_ms`; timeout maps to
//!   `ImplementationError::UpstreamTimeout`.
//! - Successful responses return `{embeddings, model, dimensions}`.

mod config;
mod error;
mod model;
mod request;
mod response;

use crate::error::Error;
use crate::model::Model;
use std::sync::Arc;
use std::time::Duration;

pub use crate::config::EmbedConfig;
pub use crate::request::EmbedRequest;
pub use crate::response::EmbedResponse;
pub use fastembed::TokenizerFiles;
pub use philharmonic_connector_impl_api::{
    ConnectorCallContext, Implementation, ImplementationError, JsonValue, async_trait,
};

const NAME: &str = "embed";

/// `embed` connector implementation backed by one loaded model.
#[derive(Clone)]
pub struct Embed {
    model: Arc<Model>,
}

impl Embed {
    /// Constructs an `Embed` instance from caller-supplied model bytes.
    ///
    /// This eagerly initializes fastembed with `try_new_from_user_defined`
    /// and returns `ImplementationError::InvalidConfig` if the ONNX/tokenizer
    /// bundle is malformed.
    pub fn new_from_bytes(
        model_id: impl Into<String>,
        onnx_bytes: Vec<u8>,
        tokenizer_files: TokenizerFiles,
        dimensions: usize,
        max_seq_length: usize,
    ) -> Result<Self, ImplementationError> {
        let model = Model::new_from_bytes(
            model_id,
            onnx_bytes,
            tokenizer_files,
            dimensions,
            max_seq_length,
        )
        .map_err(ImplementationError::from)?;

        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Returns the loaded model identifier.
    pub fn model_id(&self) -> &str {
        self.model.model_id()
    }

    /// Returns the embedding dimensions for the loaded model.
    pub fn dimensions(&self) -> usize {
        self.model.dimensions()
    }
}

#[async_trait]
impl Implementation for Embed {
    fn name(&self) -> &str {
        NAME
    }

    async fn execute(
        &self,
        config: &JsonValue,
        request: &JsonValue,
        _ctx: &ConnectorCallContext,
    ) -> Result<JsonValue, ImplementationError> {
        let config: EmbedConfig = serde_json::from_value(config.clone())
            .map_err(|err| Error::InvalidConfig(err.to_string()))
            .map_err(ImplementationError::from)?;

        if config.max_batch_size == 0 {
            return Err(ImplementationError::from(Error::InvalidConfig(
                "max_batch_size must be >= 1".to_owned(),
            )));
        }

        if config.timeout_ms == 0 {
            return Err(ImplementationError::from(Error::InvalidConfig(
                "timeout_ms must be >= 1".to_owned(),
            )));
        }

        if config.model_id != self.model.model_id() {
            return Err(ImplementationError::from(Error::InvalidConfig(format!(
                "config model_id '{}' does not match loaded model '{}'",
                config.model_id,
                self.model.model_id()
            ))));
        }

        let request: EmbedRequest = serde_json::from_value(request.clone())
            .map_err(|err| Error::InvalidRequest(err.to_string()))
            .map_err(ImplementationError::from)?;

        request
            .validate(config.max_batch_size)
            .map_err(ImplementationError::from)?;

        let texts = request.texts;
        let model = Arc::clone(&self.model);

        let inference_task = tokio::task::spawn_blocking(move || model.embed(&texts));
        let join_result =
            tokio::time::timeout(Duration::from_millis(config.timeout_ms), inference_task)
                .await
                .map_err(|_| Error::UpstreamTimeout)
                .map_err(ImplementationError::from)?;

        let task_result = join_result.map_err(|err| {
            ImplementationError::from(Error::Internal(format!("embedding task join error: {err}")))
        })?;

        let embeddings = task_result.map_err(ImplementationError::from)?;

        let response = EmbedResponse {
            embeddings,
            model: self.model.model_id().to_owned(),
            dimensions: self.model.dimensions(),
        };

        serde_json::to_value(response)
            .map_err(|err| Error::Internal(err.to_string()))
            .map_err(ImplementationError::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use philharmonic_connector_common::{UnixMillis, Uuid};
    use serde_json::json;

    fn context() -> ConnectorCallContext {
        ConnectorCallContext {
            tenant_id: Uuid::nil(),
            instance_id: Uuid::nil(),
            step_seq: 0,
            config_uuid: Uuid::nil(),
            issued_at: UnixMillis(0),
            expires_at: UnixMillis(1),
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_config_json_maps_to_invalid_config() {
        let embed = Embed {
            model: Arc::new(Model::for_tests("model-a", 384, 512)),
        };

        let err = embed
            .execute(
                &json!({"model_id": 42}),
                &json!({"texts": ["hello"]}),
                &context(),
            )
            .await
            .unwrap_err();

        let ImplementationError::InvalidConfig { detail } = err else {
            panic!("expected InvalidConfig");
        };
        assert!(!detail.is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn model_id_mismatch_maps_to_invalid_config() {
        let embed = Embed {
            model: Arc::new(Model::for_tests("model-a", 384, 512)),
        };

        let err = embed
            .execute(
                &json!({"model_id": "model-b"}),
                &json!({"texts": ["hello"]}),
                &context(),
            )
            .await
            .unwrap_err();

        assert_eq!(
            err,
            ImplementationError::InvalidConfig {
                detail: "config model_id 'model-b' does not match loaded model 'model-a'"
                    .to_owned(),
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_texts_maps_to_invalid_request() {
        let embed = Embed {
            model: Arc::new(Model::for_tests("model-a", 384, 512)),
        };

        let err = embed
            .execute(
                &json!({"model_id": "model-a"}),
                &json!({"texts": []}),
                &context(),
            )
            .await
            .unwrap_err();

        assert_eq!(
            err,
            ImplementationError::InvalidRequest {
                detail: "texts must contain at least one item".to_owned(),
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn oversized_batch_maps_to_invalid_request() {
        let embed = Embed {
            model: Arc::new(Model::for_tests("model-a", 384, 512)),
        };

        let err = embed
            .execute(
                &json!({"model_id": "model-a", "max_batch_size": 2}),
                &json!({"texts": ["a", "b", "c"]}),
                &context(),
            )
            .await
            .unwrap_err();

        assert_eq!(
            err,
            ImplementationError::InvalidRequest {
                detail: "texts length 3 exceeds max_batch_size 2".to_owned(),
            }
        );
    }

    #[test]
    fn constructor_getters_reflect_loaded_model() {
        let embed = Embed {
            model: Arc::new(Model::for_tests("model-a", 384, 512)),
        };

        assert_eq!(embed.name(), "embed");
        assert_eq!(embed.model_id(), "model-a");
        assert_eq!(embed.dimensions(), 384);
    }
}
