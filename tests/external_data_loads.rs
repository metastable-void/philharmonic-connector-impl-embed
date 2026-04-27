#![cfg(all(
    feature = "bundled-default-model",
    embed_default_bundle,
    embed_default_external_data
))]

mod common;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn external_data_default_loads_and_runs_inference() {
    let embed = common::embed().unwrap();
    let response = common::execute_texts(&embed, vec!["external data path"])
        .await
        .unwrap();

    assert_eq!(response.embeddings.len(), 1);
    assert_eq!(response.embeddings[0].len(), common::DIMENSIONS);
}
