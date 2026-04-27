#![cfg(all(feature = "bundled-default-model", embed_default_bundle))]

mod common;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn batch_inference_returns_one_vector_per_text() {
    let embed = common::embed().unwrap();
    let response = common::execute_texts(&embed, vec!["hello", "bonjour", "goodbye"])
        .await
        .unwrap();

    assert_eq!(response.embeddings.len(), 3);
    for embedding in &response.embeddings {
        assert_eq!(embedding.len(), common::DIMENSIONS);
    }
    assert_ne!(response.embeddings[0], response.embeddings[1]);
    assert_ne!(response.embeddings[0], response.embeddings[2]);
}
