#![cfg(all(feature = "bundled-default-model", embed_default_bundle))]

mod common;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn same_input_is_bit_deterministic() {
    let embed = common::embed().unwrap();

    let first = common::execute_texts(&embed, vec!["deterministic text"])
        .await
        .unwrap();
    let second = common::execute_texts(&embed, vec!["deterministic text"])
        .await
        .unwrap();

    assert_eq!(first.embeddings, second.embeddings);
}
