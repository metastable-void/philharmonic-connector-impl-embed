mod common;

use common::{config, context, implementation, maybe_fixture};
use philharmonic_connector_impl_embed::{Implementation, ImplementationError};
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires EMBED_TEST_* env vars and local model files"]
async fn max_batch_size_rejects_oversized_request() {
    let Some(fixture) = maybe_fixture() else {
        return;
    };

    let embed = implementation(&fixture);

    let err = embed
        .execute(
            &config(&fixture.model_id, 2),
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
