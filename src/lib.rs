//! Local in-process embedding connector implementation for Philharmonic.
//!
//! `embed` implements the shared
//! [`philharmonic_connector_impl_api::Implementation`] trait with a
//! pure-Rust inference stack: [`tract-onnx`] loads and runs ONNX
//! sentence-transformer models, [`tokenizers`] parses the single
//! `tokenizer.json` file, and this crate applies mean pooling plus
//! L2 normalization for sentence embeddings.
//!
//! The library never fetches from HuggingFace, reads model paths, or
//! reads environment variables at runtime. Callers either provide all
//! bytes explicitly with [`Embed::new_from_bytes`], or enable the
//! default `bundled-default-model` feature and use [`Embed::new_default`]
//! after `build.rs` has fetched a pinned build-time bundle into the
//! local cache and embedded it with `include_bytes!`.
//!
//! The default bundled model is `BAAI/bge-m3`, pinned by build script
//! revision. It ships ONNX weights as `model.onnx` plus
//! `model.onnx_data`, so [`Embed::new_from_bytes`] accepts optional
//! external-data bytes. Pass `None` for single-file ONNX exports. The
//! build-time bundle can be disabled with `--no-default-features`; it
//! is also skipped on docs.rs.
//!
//! tract validates operator coverage while loading the ONNX graph.
//! BERT-class sentence-transformer exports are the intended target;
//! other architectures should be probed before deployment.

mod config;
mod error;
mod model;
mod pool;
mod request;
mod response;

use crate::error::Error;
use crate::model::Model;
use std::sync::Arc;
use std::time::Duration;

pub use crate::config::EmbedConfig;
pub use crate::request::EmbedRequest;
pub use crate::response::EmbedResponse;
pub use philharmonic_connector_impl_api::{
    ConnectorCallContext, Implementation, ImplementationError, JsonValue, async_trait,
};

const NAME: &str = "embed";

/// `embed` connector implementation backed by one eagerly loaded model.
#[derive(Clone)]
pub struct Embed {
    model: Arc<Model>,
}

impl Embed {
    /// Constructs an `Embed` instance from caller-supplied model bytes.
    ///
    /// Pass `onnx_external_data: Some(bytes)` for ONNX models that ship
    /// external tensor data, such as `model.onnx_data`. Pass `None` for
    /// single-file ONNX exports. Model and tokenizer loading is eager so
    /// malformed bundles fail during startup.
    pub fn new_from_bytes(
        model_id: impl Into<String>,
        onnx_bytes: Vec<u8>,
        onnx_external_data: Option<&[u8]>,
        tokenizer_json_bytes: &[u8],
        dimensions: usize,
        max_seq_length: usize,
    ) -> Result<Self, ImplementationError> {
        let model = Model::new_from_bytes(
            model_id,
            onnx_bytes,
            onnx_external_data,
            tokenizer_json_bytes,
            dimensions,
            max_seq_length,
        )
        .map_err(ImplementationError::from)?;

        Ok(Self {
            model: Arc::new(model),
        })
    }

    /// Constructs an `Embed` instance from the build-time bundled model.
    #[cfg(all(feature = "bundled-default-model", embed_default_bundle))]
    pub fn new_default() -> Result<Self, ImplementationError> {
        let dimensions =
            parse_bundled_usize("EMBED_DEFAULT_DIMENSIONS", env!("EMBED_DEFAULT_DIMENSIONS"))?;
        let max_seq_length = parse_bundled_usize(
            "EMBED_DEFAULT_MAX_SEQ_LENGTH",
            env!("EMBED_DEFAULT_MAX_SEQ_LENGTH"),
        )?;
        #[cfg(embed_default_external_data)]
        let onnx_external_data = Some(BUNDLED_ONNX_EXTERNAL_DATA.as_slice());
        #[cfg(not(embed_default_external_data))]
        let onnx_external_data = None;

        Self::new_from_bytes(
            env!("EMBED_DEFAULT_MODEL_ID"),
            BUNDLED_ONNX_BYTES.to_vec(),
            onnx_external_data,
            BUNDLED_TOKENIZER_JSON_BYTES.as_slice(),
            dimensions,
            max_seq_length,
        )
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

#[cfg(all(feature = "bundled-default-model", embed_default_bundle))]
fn parse_bundled_usize(name: &str, value: &str) -> Result<usize, ImplementationError> {
    value.parse::<usize>().map_err(|err| {
        ImplementationError::from(Error::InvalidConfig(format!(
            "build script emitted invalid {name} value '{value}': {err}"
        )))
    })
}

// The bundled ONNX + tokenizer bytes are placed in `.lrodata.*`
// sections via the `inline_blob::blob!` proc-macro rather than
// the default `.rodata`. Without this, a large default model
// (notably bge-m3 at ~2.27 GB) bloats `.rodata` to the point
// that *other* rodata items (closure vtables, fn pointers)
// land more than 2 GiB from `.text`, and rust-lld's small
// code-model `R_X86_64_PC32` relocations overflow at link
// time. The `.lrodata.*` (large rodata) convention is part of
// the x86_64 SysV ABI; lld places these in a separate large
// segment beyond the 2 GiB-reachable region, and the bytes
// themselves are accessed via 64-bit-friendly addressing
// modes that don't trigger the overflow. See `inline-blob`
// crate's README for details.
#[cfg(all(feature = "bundled-default-model", embed_default_bundle))]
inline_blob::blob!(
    static BUNDLED_ONNX_BYTES,
    concat!(env!("EMBED_DEFAULT_BUNDLE_DIR"), "/model.onnx")
);

#[cfg(all(feature = "bundled-default-model", embed_default_bundle))]
inline_blob::blob!(
    static BUNDLED_TOKENIZER_JSON_BYTES,
    concat!(env!("EMBED_DEFAULT_BUNDLE_DIR"), "/tokenizer.json")
);

#[cfg(all(
    feature = "bundled-default-model",
    embed_default_bundle,
    embed_default_external_data
))]
inline_blob::blob!(
    static BUNDLED_ONNX_EXTERNAL_DATA,
    concat!(env!("EMBED_DEFAULT_BUNDLE_DIR"), "/model.onnx_data")
);

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

        let inference_task = tokio::task::spawn_blocking(move || model.forward(&texts));
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

    fn test_embed() -> Embed {
        Embed {
            model: Arc::new(Model::for_tests("model-a", 384, 128)),
        }
    }

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

    #[test]
    fn name_is_stable() {
        assert_eq!(test_embed().name(), "embed");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn invalid_config_json_maps_to_invalid_config() {
        let err = test_embed()
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
        let err = test_embed()
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
    async fn zero_timeout_maps_to_invalid_config() {
        let err = test_embed()
            .execute(
                &json!({"model_id": "model-a", "timeout_ms": 0}),
                &json!({"texts": ["hello"]}),
                &context(),
            )
            .await
            .unwrap_err();

        assert_eq!(
            err,
            ImplementationError::InvalidConfig {
                detail: "timeout_ms must be >= 1".to_owned(),
            }
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn empty_texts_maps_to_invalid_request() {
        let err = test_embed()
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
}
