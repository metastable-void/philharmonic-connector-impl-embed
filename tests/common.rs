#![cfg(all(feature = "bundled-default-model", embed_default_bundle))]
#![allow(dead_code)]

use philharmonic_connector_common::{UnixMillis, Uuid};
use philharmonic_connector_impl_embed::{
    ConnectorCallContext, Embed, EmbedResponse, Implementation, ImplementationError,
};
use serde_json::json;

pub const MODEL_ID: &str = env!("EMBED_DEFAULT_MODEL_ID");
pub const DIMENSIONS: usize = parse_usize(env!("EMBED_DEFAULT_DIMENSIONS"));
pub const MAX_SEQ_LENGTH: usize = parse_usize(env!("EMBED_DEFAULT_MAX_SEQ_LENGTH"));

const fn parse_usize(value: &str) -> usize {
    let bytes = value.as_bytes();
    let mut index = 0;
    let mut parsed = 0_usize;
    while index < bytes.len() {
        let digit = bytes[index] - b'0';
        parsed = parsed * 10 + digit as usize;
        index += 1;
    }
    parsed
}

pub fn embed() -> Result<Embed, ImplementationError> {
    Embed::new_default()
}

pub fn context() -> ConnectorCallContext {
    ConnectorCallContext {
        tenant_id: Uuid::nil(),
        instance_id: Uuid::nil(),
        step_seq: 0,
        config_uuid: Uuid::nil(),
        issued_at: UnixMillis(0),
        expires_at: UnixMillis(1),
    }
}

pub async fn execute_texts(
    embed: &Embed,
    texts: Vec<&str>,
) -> Result<EmbedResponse, ImplementationError> {
    let config = json!({
        "model_id": MODEL_ID,
        "max_batch_size": 32,
        "timeout_ms": 120_000
    });
    let request = json!({
        "texts": texts
    });
    let value = embed.execute(&config, &request, &context()).await?;
    serde_json::from_value::<EmbedResponse>(value).map_err(|err| ImplementationError::Internal {
        detail: err.to_string(),
    })
}

pub fn cosine(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(l, r)| l * r)
        .sum::<f32>()
}
