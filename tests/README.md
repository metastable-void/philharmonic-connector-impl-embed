# Integration Tests

This crate's live-inference integration tests are `#[ignore]` by default.
They require a local ONNX model file plus tokenizer files and are only run
when all of the following env vars are set:

- `EMBED_TEST_ONNX_PATH`
- `EMBED_TEST_TOKENIZER_DIR`
- `EMBED_TEST_MODEL_ID`
- `EMBED_TEST_DIMENSIONS`
- `EMBED_TEST_MAX_SEQ_LENGTH`

`EMBED_TEST_TOKENIZER_DIR` must contain:

- `tokenizer.json`
- `tokenizer_config.json`
- `config.json`
- `special_tokens_map.json`

Example:

```sh
EMBED_TEST_ONNX_PATH=/tmp/embed/model.onnx \
EMBED_TEST_TOKENIZER_DIR=/tmp/embed \
EMBED_TEST_MODEL_ID=paraphrase-multilingual-MiniLM-L12-v2 \
EMBED_TEST_DIMENSIONS=384 \
EMBED_TEST_MAX_SEQ_LENGTH=512 \
./scripts/rust-test.sh --ignored philharmonic-connector-impl-embed
```

The repository does not commit ONNX artifacts.
