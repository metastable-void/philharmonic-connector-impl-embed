use philharmonic_connector_common::{UnixMillis, Uuid};
use philharmonic_connector_impl_embed::{ConnectorCallContext, Embed, JsonValue, TokenizerFiles};
use serde_json::json;
use std::env;
use std::path::{Path, PathBuf};

#[derive(Clone)]
pub struct TestModelFixture {
    pub onnx_bytes: Vec<u8>,
    pub tokenizer_files: TokenizerFiles,
    pub model_id: String,
    pub dimensions: usize,
    pub max_seq_length: usize,
}

pub fn maybe_fixture() -> Option<TestModelFixture> {
    let onnx_path = env::var("EMBED_TEST_ONNX_PATH").ok();
    let tokenizer_dir = env::var("EMBED_TEST_TOKENIZER_DIR").ok();
    let model_id = env::var("EMBED_TEST_MODEL_ID").ok();
    let dimensions = env::var("EMBED_TEST_DIMENSIONS").ok();
    let max_seq_length = env::var("EMBED_TEST_MAX_SEQ_LENGTH").ok();

    let all_set = onnx_path.is_some()
        && tokenizer_dir.is_some()
        && model_id.is_some()
        && dimensions.is_some()
        && max_seq_length.is_some();

    if !all_set {
        eprintln!(
            "EMBED_TEST_* env vars not fully set; skipping integration test. Required: EMBED_TEST_ONNX_PATH, EMBED_TEST_TOKENIZER_DIR, EMBED_TEST_MODEL_ID, EMBED_TEST_DIMENSIONS, EMBED_TEST_MAX_SEQ_LENGTH"
        );
        return None;
    }

    let onnx_path = PathBuf::from(onnx_path?);
    let tokenizer_dir = PathBuf::from(tokenizer_dir?);
    let model_id = model_id?;

    let dimensions = match dimensions?.parse::<usize>() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("invalid EMBED_TEST_DIMENSIONS value: {err}");
            return None;
        }
    };

    let max_seq_length = match max_seq_length?.parse::<usize>() {
        Ok(value) => value,
        Err(err) => {
            eprintln!("invalid EMBED_TEST_MAX_SEQ_LENGTH value: {err}");
            return None;
        }
    };

    let onnx_bytes = match std::fs::read(&onnx_path) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("failed reading ONNX file {}: {err}", onnx_path.display());
            return None;
        }
    };

    let tokenizer_files = match load_tokenizer_files(&tokenizer_dir) {
        Ok(files) => files,
        Err(err) => {
            eprintln!(
                "failed reading tokenizer files from {}: {err}",
                tokenizer_dir.display()
            );
            return None;
        }
    };

    Some(TestModelFixture {
        onnx_bytes,
        tokenizer_files,
        model_id,
        dimensions,
        max_seq_length,
    })
}

fn load_tokenizer_files(tokenizer_dir: &Path) -> std::io::Result<TokenizerFiles> {
    let tokenizer_file = std::fs::read(tokenizer_dir.join("tokenizer.json"))?;
    let config_file = std::fs::read(tokenizer_dir.join("config.json"))?;
    let special_tokens_map_file = std::fs::read(tokenizer_dir.join("special_tokens_map.json"))?;
    let tokenizer_config_file = std::fs::read(tokenizer_dir.join("tokenizer_config.json"))?;

    Ok(TokenizerFiles {
        tokenizer_file,
        config_file,
        special_tokens_map_file,
        tokenizer_config_file,
    })
}

pub fn implementation(fixture: &TestModelFixture) -> Embed {
    Embed::new_from_bytes(
        fixture.model_id.clone(),
        fixture.onnx_bytes.clone(),
        fixture.tokenizer_files.clone(),
        fixture.dimensions,
        fixture.max_seq_length,
    )
    .expect("test model fixture should build")
}

pub fn context() -> ConnectorCallContext {
    ConnectorCallContext {
        tenant_id: Uuid::nil(),
        instance_id: Uuid::nil(),
        step_seq: 1,
        config_uuid: Uuid::nil(),
        issued_at: UnixMillis(0),
        expires_at: UnixMillis(60_000),
    }
}

pub fn config(model_id: &str, max_batch_size: usize) -> JsonValue {
    json!({
        "model_id": model_id,
        "max_batch_size": max_batch_size,
        "timeout_ms": 10_000
    })
}
