#![cfg(all(feature = "bundled-default-model", embed_default_bundle))]

mod common;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn related_terms_score_higher_than_unrelated_terms() {
    let embed = common::embed().unwrap();
    let response = common::execute_texts(&embed, vec!["hello", "hi", "goodbye"])
        .await
        .unwrap();

    let hello_hi = common::cosine(&response.embeddings[0], &response.embeddings[1]);
    let hello_goodbye = common::cosine(&response.embeddings[0], &response.embeddings[2]);

    assert!(hello_hi > hello_goodbye);
}
