#![cfg(all(feature = "bundled-default-model", embed_default_bundle))]

mod common;

use philharmonic_connector_impl_embed::{Implementation, ImplementationError};
use serde_json::json;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_size_boundary_passes_and_overrun_fails() {
    let embed = common::embed().unwrap();

    let config = json!({
        "model_id": common::MODEL_ID,
        "max_batch_size": 2,
        "timeout_ms": 120_000
    });
    let boundary = json!({"texts": ["one", "two"]});
    embed
        .execute(&config, &boundary, &common::context())
        .await
        .unwrap();

    let overrun = json!({"texts": ["one", "two", "three"]});
    let err = embed
        .execute(&config, &overrun, &common::context())
        .await
        .unwrap_err();

    assert_eq!(
        err,
        ImplementationError::InvalidRequest {
            detail: "texts length 3 exceeds max_batch_size 2".to_owned(),
        }
    );
}
