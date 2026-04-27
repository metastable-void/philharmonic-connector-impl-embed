# philharmonic-connector-impl-embed

Part of the Philharmonic workspace: https://github.com/metastable-void/philharmonic-workspace

`philharmonic-connector-impl-embed` provides the Phase 7 Tier 1
`embed` connector implementation. It runs local CPU inference with
`tract-onnx`, tokenizes with `tokenizers` from a single
`tokenizer.json`, then mean-pools and L2-normalizes the output vectors.

The library performs no runtime network I/O, file I/O, or environment
variable lookup. Deployments can bundle their own ONNX and tokenizer
assets into a connector-service binary, for example with
`include_bytes!`, and construct `Embed` with `Embed::new_from_bytes(...)`.
For ONNX exports that use external tensor data, pass the external
`*.onnx_data` bytes in the constructor; pass `None` for single-file ONNX
exports.

By default, the crate also offers `Embed::new_default()` behind the
default-on `bundled-default-model` feature. Its build script fetches a
pinned HuggingFace bundle at compile time, caches it outside the repo,
and embeds the bytes into the compiled library. The default model is
`BAAI/bge-m3`; `PHILHARMONIC_EMBED_DEFAULT_MODEL`,
`PHILHARMONIC_EMBED_DEFAULT_REVISION`, and
`PHILHARMONIC_EMBED_CACHE_DIR` adjust the build-time bundle. Use
`--no-default-features` to opt out for offline or packaging builds.

## Contributing

This crate is developed as a submodule of the Philharmonic workspace.
Workspace-wide development conventions live in the workspace meta-repo at
[metastable-void/philharmonic-workspace](https://github.com/metastable-void/philharmonic-workspace),
authoritatively in its
[`CONTRIBUTING.md`](https://github.com/metastable-void/philharmonic-workspace/blob/main/CONTRIBUTING.md).

SPDX-License-Identifier: Apache-2.0 OR MPL-2.0
