mod common;

use common::{config, context, implementation, maybe_fixture};
use philharmonic_connector_impl_embed::{EmbedResponse, Implementation};
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires EMBED_TEST_* env vars and local model files"]
async fn batch_inference_returns_one_vector_per_text() {
    let Some(fixture) = maybe_fixture() else {
        return;
    };

    let embed = implementation(&fixture);
    let response = embed
        .execute(
            &config(&fixture.model_id, 32),
            &json!({"texts": ["hello", "hi", "goodbye"]}),
            &context(),
        )
        .await
        .unwrap();

    let response: EmbedResponse = serde_json::from_value(response).unwrap();
    assert_eq!(response.embeddings.len(), 3);
    for embedding in response.embeddings {
        assert_eq!(embedding.len(), fixture.dimensions);
    }
}
