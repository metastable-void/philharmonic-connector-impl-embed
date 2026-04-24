# Changelog

All notable changes to this crate are documented in this file.

The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this crate adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-24

### Added

- Implemented the `embed` connector using local fastembed CPU inference.
- Added `Embed::new_from_bytes(...)` for eager model loading from deployment-provided bytes.
- Added config/request/response wire types and `Implementation` execution flow with batch/timeout validation.
- Added env-gated `#[ignore]` integration tests for live inference shape, determinism, semantic sanity, and batch cap enforcement.
