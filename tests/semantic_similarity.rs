mod common;

use common::{config, context, implementation, maybe_fixture};
use philharmonic_connector_impl_embed::{EmbedResponse, Implementation};
use serde_json::json;

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires EMBED_TEST_* env vars and local model files"]
async fn hello_is_closer_to_hi_than_goodbye() {
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

    let hello_to_hi = cosine_similarity(&response.embeddings[0], &response.embeddings[1]);
    let hello_to_goodbye = cosine_similarity(&response.embeddings[0], &response.embeddings[2]);

    assert!(hello_to_hi > hello_to_goodbye);
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> f32 {
    let dot = left
        .iter()
        .zip(right.iter())
        .fold(0.0_f32, |acc, (l, r)| acc + (l * r));

    let left_norm = left
        .iter()
        .fold(0.0_f32, |acc, value| acc + (value * value));
    let right_norm = right
        .iter()
        .fold(0.0_f32, |acc, value| acc + (value * value));

    if left_norm == 0.0_f32 || right_norm == 0.0_f32 {
        return 0.0_f32;
    }

    dot / (left_norm.sqrt() * right_norm.sqrt())
}
