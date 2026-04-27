#![cfg(all(feature = "bundled-default-model", embed_default_bundle))]

mod common;

use approx::assert_abs_diff_eq;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn output_vectors_are_l2_normalized() {
    let embed = common::embed().unwrap();
    let response = common::execute_texts(&embed, vec!["hello", "bonjour"])
        .await
        .unwrap();

    for embedding in response.embeddings {
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert_abs_diff_eq!(norm, 1.0, epsilon = 1.0e-5);
    }
}
