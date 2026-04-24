mod common;

use common::{config, context, implementation, maybe_fixture};
use philharmonic_connector_impl_embed::{EmbedResponse, Implementation};
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires EMBED_TEST_* env vars and local model files"]
async fn inference_produces_correct_shape() {
    let Some(fixture) = maybe_fixture() else {
        return;
    };

    let embed = implementation(&fixture);
    let response = embed
        .execute(
            &config(&fixture.model_id, 32),
            &json!({"texts": ["hello world"]}),
            &context(),
        )
        .await
        .unwrap();

    let response: EmbedResponse = serde_json::from_value(response).unwrap();
    assert_eq!(response.model, fixture.model_id);
    assert_eq!(response.dimensions, fixture.dimensions);
    assert_eq!(response.embeddings.len(), 1);
    assert_eq!(response.embeddings[0].len(), fixture.dimensions);
}
