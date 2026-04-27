# Changelog

All notable changes to this crate are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this crate adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-27

### Added

- Implemented the `embed` connector using pure-Rust `tract-onnx`,
  `tokenizers`, and `ndarray` inference plumbing.
- Added `Embed::new_from_bytes(...)` for eager loading from
  deployment-provided ONNX bytes, optional ONNX external-data bytes, and a
  single `tokenizer.json`.
- Added `Embed::new_default()` behind the default-on
  `bundled-default-model` feature, with build-time HuggingFace fetch,
  cache verification, docs.rs skip, and env-var model/revision/cache knobs.
- Added config/request/response wire types, batch and timeout validation,
  mean pooling, L2 normalization, and bundled-model integration tests.

### Notes

- The crate performs no runtime network I/O, file I/O, or environment
  variable lookup. Build-time bundling can be disabled with
  `--no-default-features`.
