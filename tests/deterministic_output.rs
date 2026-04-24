mod common;

use common::{config, context, implementation, maybe_fixture};
use philharmonic_connector_impl_embed::{EmbedResponse, Implementation};
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires EMBED_TEST_* env vars and local model files"]
async fn same_input_produces_identical_vector() {
    let Some(fixture) = maybe_fixture() else {
        return;
    };

    let embed = implementation(&fixture);

    let first = embed
        .execute(
            &config(&fixture.model_id, 32),
            &json!({"texts": ["determinism check"]}),
            &context(),
        )
        .await
        .unwrap();
    let second = embed
        .execute(
            &config(&fixture.model_id, 32),
            &json!({"texts": ["determinism check"]}),
            &context(),
        )
        .await
        .unwrap();

    let first: EmbedResponse = serde_json::from_value(first).unwrap();
    let second: EmbedResponse = serde_json::from_value(second).unwrap();

    assert_eq!(first.embeddings, second.embeddings);
}
