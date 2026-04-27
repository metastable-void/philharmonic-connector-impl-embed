#![cfg(all(feature = "bundled-default-model", embed_default_bundle))]

mod common;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inference_produces_correct_shape() {
    let embed = common::embed().unwrap();
    let response = common::execute_texts(&embed, vec!["hello"]).await.unwrap();

    assert_eq!(response.model, common::MODEL_ID);
    assert_eq!(response.dimensions, common::DIMENSIONS);
    assert_eq!(response.embeddings.len(), 1);
    assert_eq!(response.embeddings[0].len(), common::DIMENSIONS);
}
