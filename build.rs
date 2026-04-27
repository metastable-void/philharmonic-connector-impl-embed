use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

const HF_BASE: &str = "https://huggingface.co";
const DEFAULT_MODEL: &str = "BAAI/bge-m3";
const DEFAULT_REVISION: &str = "5617a9f61b028005a4858fdac845db406aefb181";
const ONNX_PATH: &str = "onnx/model.onnx";
const ONNX_DATA_PATH: &str = "onnx/model.onnx_data";
const TOKENIZER_PATH: &str = "tokenizer.json";
const CONFIG_PATH: &str = "config.json";
const SENTENCE_BERT_CONFIG_PATH: &str = "sentence_bert_config.json";

#[derive(Debug, Deserialize, Serialize)]
struct Manifest {
    model_id: String,
    revision: String,
    fetched_at: String,
    files: BTreeMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct ModelConfig {
    hidden_size: usize,
    max_position_embeddings: Option<usize>,
}

/// Sentence-transformers ships its canonical sentence-level
/// `max_seq_length` in a separate `sentence_bert_config.json`.
/// For models built on XLM-RoBERTa (e.g. `BAAI/bge-m3`) this is
/// crucial: `config.json`'s `max_position_embeddings` is the
/// position-embedding *table size* (e.g. 8194 for bge-m3),
/// while the effective max sentence length is offset by the
/// padding-token reservation (`max_position_embeddings - 2 =
/// 8192` for XLM-RoBERTa-class encoders). Feeding 8194 into
/// the tokenizer pads inputs past the position table and
/// crashes tract's gather op at inference time.
#[derive(Debug, Deserialize)]
struct SentenceBertConfig {
    max_seq_length: Option<usize>,
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(embed_default_bundle)");
    println!("cargo:rustc-check-cfg=cfg(embed_default_external_data)");
    println!("cargo:rerun-if-env-changed=PHILHARMONIC_EMBED_DEFAULT_MODEL");
    println!("cargo:rerun-if-env-changed=PHILHARMONIC_EMBED_DEFAULT_REVISION");
    println!("cargo:rerun-if-env-changed=PHILHARMONIC_EMBED_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=DOCS_RS");

    if env::var_os("CARGO_FEATURE_BUNDLED_DEFAULT_MODEL").is_none() {
        return;
    }

    if env::var_os("DOCS_RS").is_some() {
        return;
    }

    let (model_id, revision) = model_selection();
    validate_revision(&model_id, &revision);

    let bundle_dir = cache_root().join(format!(
        "{}__{}",
        sanitize_model_id(&model_id),
        revision_prefix(&revision)
    ));

    let bundle = match verify_cache_hit(&bundle_dir, &model_id, &revision) {
        Some(bundle) => bundle,
        None => fetch_bundle(&bundle_dir, &model_id, &revision),
    };

    let config = read_config(&bundle_dir.join("config.json"));
    let sentence_bert_config =
        read_sentence_bert_config(&bundle_dir.join("sentence_bert_config.json"));

    println!("cargo:rustc-cfg=embed_default_bundle");
    if bundle.has_external_data {
        println!("cargo:rustc-cfg=embed_default_external_data");
        println!(
            "cargo:rerun-if-changed={}",
            bundle_dir.join("model.onnx_data").display()
        );
    }
    println!(
        "cargo:rustc-env=EMBED_DEFAULT_BUNDLE_DIR={}",
        bundle_dir.display()
    );
    println!("cargo:rustc-env=EMBED_DEFAULT_MODEL_ID={model_id}");
    println!("cargo:rustc-env=EMBED_DEFAULT_REVISION={revision}");
    println!(
        "cargo:rustc-env=EMBED_DEFAULT_DIMENSIONS={}",
        config.hidden_size
    );
    // Prefer sentence-transformers' canonical `max_seq_length`
    // when present (the right field for sentence-level limits;
    // accounts for XLM-RoBERTa-style position offset). Fall
    // back to the architectural `max_position_embeddings` for
    // raw HF models that don't ship a sentence-bert config,
    // and to a safe 512 if neither is present.
    let max_seq_length = sentence_bert_config
        .and_then(|sbc| sbc.max_seq_length)
        .or(config.max_position_embeddings)
        .unwrap_or(512);
    println!("cargo:rustc-env=EMBED_DEFAULT_MAX_SEQ_LENGTH={max_seq_length}");
    println!(
        "cargo:rerun-if-changed={}",
        bundle_dir.join("manifest.json").display()
    );
}

struct BundleState {
    has_external_data: bool,
}

fn model_selection() -> (String, String) {
    let model = env::var("PHILHARMONIC_EMBED_DEFAULT_MODEL").ok();
    let revision = env::var("PHILHARMONIC_EMBED_DEFAULT_REVISION").ok();

    match (model, revision) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => panic!(
            "PHILHARMONIC_EMBED_DEFAULT_MODEL={model_id} requires PHILHARMONIC_EMBED_DEFAULT_REVISION=<commit-sha>; use --no-default-features to opt out of the bundled embed model"
        ),
        (None, Some(_)) => panic!(
            "PHILHARMONIC_EMBED_DEFAULT_REVISION was set without PHILHARMONIC_EMBED_DEFAULT_MODEL; set both or use --no-default-features to opt out of the bundled embed model"
        ),
        (None, None) => (DEFAULT_MODEL.to_owned(), DEFAULT_REVISION.to_owned()),
    }
}

fn validate_revision(model_id: &str, revision: &str) {
    if revision == "main" || revision == "HEAD" {
        panic!(
            "embed default model {model_id} must be pinned to a commit SHA, got {revision}; use --no-default-features to opt out of the bundled embed model"
        );
    }
    if revision.len() < 12 || !revision.chars().all(|ch| ch.is_ascii_hexdigit()) {
        panic!(
            "embed default model {model_id} revision must look like a commit SHA, got {revision}; use --no-default-features to opt out of the bundled embed model"
        );
    }
}

fn cache_root() -> PathBuf {
    if let Some(path) = env::var_os("PHILHARMONIC_EMBED_CACHE_DIR") {
        return PathBuf::from(path);
    }

    if let Some(path) = dirs::cache_dir() {
        return path.join("philharmonic").join("embed-bundles");
    }

    panic!(
        "could not resolve a cache directory for the bundled embed model; set PHILHARMONIC_EMBED_CACHE_DIR or use --no-default-features"
    );
}

fn sanitize_model_id(model_id: &str) -> String {
    model_id.replace('/', "__")
}

fn revision_prefix(revision: &str) -> &str {
    &revision[..12]
}

fn verify_cache_hit(bundle_dir: &Path, model_id: &str, revision: &str) -> Option<BundleState> {
    let manifest_path = bundle_dir.join("manifest.json");
    if !manifest_path.exists() {
        return None;
    }

    let manifest_bytes = fs::read(&manifest_path).unwrap_or_else(|err| {
        panic!(
            "failed to read bundled embed manifest at {}: {err}",
            manifest_path.display()
        )
    });
    let manifest: Manifest = serde_json::from_slice(&manifest_bytes).unwrap_or_else(|err| {
        panic!(
            "failed to parse bundled embed manifest at {}: {err}",
            manifest_path.display()
        )
    });

    if manifest.model_id != model_id || manifest.revision != revision {
        panic!(
            "cached bundled embed model at {} is for {}@{}, not {}@{}; remove the cache entry or set PHILHARMONIC_EMBED_CACHE_DIR",
            bundle_dir.display(),
            manifest.model_id,
            manifest.revision,
            model_id,
            revision
        );
    }

    for (file, expected_sha) in &manifest.files {
        let path = bundle_dir.join(file);
        let bytes = fs::read(&path).unwrap_or_else(|err| {
            panic!(
                "cached bundled embed file {} is listed in the manifest but could not be read: {err}",
                path.display()
            )
        });
        let actual_sha = sha256_hex(&bytes);
        if actual_sha != *expected_sha {
            panic!(
                "cached bundled embed file {} has sha256 {}, expected {}; remove the cache entry to refetch",
                path.display(),
                actual_sha,
                expected_sha
            );
        }
    }

    Some(BundleState {
        has_external_data: manifest.files.contains_key("model.onnx_data"),
    })
}

fn fetch_bundle(bundle_dir: &Path, model_id: &str, revision: &str) -> BundleState {
    fs::create_dir_all(bundle_dir).unwrap_or_else(|err| {
        panic!(
            "failed to create bundled embed cache directory {}: {err}",
            bundle_dir.display()
        )
    });

    let mut files = BTreeMap::new();
    fetch_file(
        bundle_dir,
        model_id,
        revision,
        ONNX_PATH,
        "model.onnx",
        &mut files,
    );

    let has_external_data = remote_exists(model_id, revision, ONNX_DATA_PATH);
    if has_external_data {
        fetch_file(
            bundle_dir,
            model_id,
            revision,
            ONNX_DATA_PATH,
            "model.onnx_data",
            &mut files,
        );
    }

    fetch_file(
        bundle_dir,
        model_id,
        revision,
        TOKENIZER_PATH,
        "tokenizer.json",
        &mut files,
    );
    fetch_file(
        bundle_dir,
        model_id,
        revision,
        CONFIG_PATH,
        "config.json",
        &mut files,
    );

    // Optional: sentence-transformers' canonical
    // `sentence_bert_config.json`. Present for sentence-bert-
    // class repos (bge-m3, paraphrase-multilingual-MiniLM-*,
    // etc.); absent for raw HF model repos. We HEAD-probe
    // before fetching to avoid 404 noise.
    if remote_exists(model_id, revision, SENTENCE_BERT_CONFIG_PATH) {
        fetch_file(
            bundle_dir,
            model_id,
            revision,
            SENTENCE_BERT_CONFIG_PATH,
            "sentence_bert_config.json",
            &mut files,
        );
    }

    let manifest = Manifest {
        model_id: model_id.to_owned(),
        revision: revision.to_owned(),
        fetched_at: "build-time".to_owned(),
        files,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .expect("serializing bundled embed manifest should not fail");
    fs::write(bundle_dir.join("manifest.json"), manifest_bytes).unwrap_or_else(|err| {
        panic!(
            "failed to write bundled embed manifest at {}: {err}",
            bundle_dir.join("manifest.json").display()
        )
    });

    BundleState { has_external_data }
}

fn fetch_file(
    bundle_dir: &Path,
    model_id: &str,
    revision: &str,
    remote_path: &str,
    local_name: &str,
    files: &mut BTreeMap<String, String>,
) {
    let url = hf_url(model_id, revision, remote_path);
    let bytes = fetch_bytes(&url, model_id, revision);
    let sha = sha256_hex(&bytes);
    let path = bundle_dir.join(local_name);
    fs::write(&path, bytes).unwrap_or_else(|err| {
        panic!(
            "failed to write bundled embed file {} fetched from {url}: {err}",
            path.display()
        )
    });
    files.insert(local_name.to_owned(), sha);
}

fn remote_exists(model_id: &str, revision: &str, remote_path: &str) -> bool {
    let url = hf_url(model_id, revision, remote_path);
    match ureq::head(&url).call() {
        Ok(response) => response.status().as_u16() == 200,
        Err(ureq::Error::StatusCode(404)) => false,
        Err(ureq::Error::StatusCode(code)) => panic!(
            "HTTP {code} probing bundled embed external-data URL {url} for {model_id}@{revision}; use --no-default-features to opt out"
        ),
        Err(err) => panic!(
            "failed to probe bundled embed external-data URL {url} for {model_id}@{revision}: {err}; use --no-default-features to opt out"
        ),
    }
}

fn fetch_bytes(url: &str, model_id: &str, revision: &str) -> Vec<u8> {
    let mut response = ureq::get(url).call().unwrap_or_else(|err| {
        panic!(
            "failed to fetch bundled embed model {model_id}@{revision} from {url}: {err}; use --no-default-features to opt out"
        )
    });
    let mut bytes = Vec::new();
    response
        .body_mut()
        .as_reader()
        .read_to_end(&mut bytes)
        .unwrap_or_else(|err| {
            panic!(
                "failed to read bundled embed model response {model_id}@{revision} from {url}: {err}; use --no-default-features to opt out"
            )
        });
    bytes
}

fn hf_url(model_id: &str, revision: &str, remote_path: &str) -> String {
    format!("{HF_BASE}/{model_id}/resolve/{revision}/{remote_path}")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String should not fail");
    }
    out
}

fn read_config(path: &Path) -> ModelConfig {
    let bytes = fs::read(path).unwrap_or_else(|err| {
        panic!(
            "failed to read bundled embed config {}: {err}",
            path.display()
        )
    });
    serde_json::from_slice(&bytes).unwrap_or_else(|err| {
        panic!(
            "failed to parse bundled embed config {}: {err}",
            path.display()
        )
    })
}

/// Returns the parsed sentence-bert config when the file
/// exists; `None` for raw HF repos that don't ship one.
/// Parse failures (malformed JSON, schema drift) panic loudly
/// — preferring a clear build break to a silently-wrong
/// `max_seq_length`.
fn read_sentence_bert_config(path: &Path) -> Option<SentenceBertConfig> {
    if !path.exists() {
        return None;
    }
    let bytes = fs::read(path).unwrap_or_else(|err| {
        panic!(
            "failed to read bundled embed sentence-bert config {}: {err}",
            path.display()
        )
    });
    let parsed: SentenceBertConfig = serde_json::from_slice(&bytes).unwrap_or_else(|err| {
        panic!(
            "failed to parse bundled embed sentence-bert config {}: {err}",
            path.display()
        )
    });
    Some(parsed)
}
