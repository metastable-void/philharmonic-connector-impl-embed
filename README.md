# philharmonic-connector-impl-embed

Part of the Philharmonic workspace: https://github.com/metastable-void/philharmonic-workspace

`philharmonic-connector-impl-embed` provides the Phase 7 Tier 1 `embed`
connector implementation backed by local `fastembed` CPU inference.
Deployments bundle ONNX + tokenizer bytes into their connector-service binary
(for example with `include_bytes!`) and construct `Embed` via
`Embed::new_from_bytes(...)`; this crate does not fetch model files over the
network, perform disk I/O, or read environment variables at runtime.

## Contributing

This crate is developed as a submodule of the Philharmonic workspace.
Workspace-wide development conventions — git workflow, script wrappers,
Rust code rules, versioning, terminology — live in the workspace meta-repo at
[metastable-void/philharmonic-workspace](https://github.com/metastable-void/philharmonic-workspace),
authoritatively in its
[`CONTRIBUTING.md`](https://github.com/metastable-void/philharmonic-workspace/blob/main/CONTRIBUTING.md).

SPDX-License-Identifier: Apache-2.0 OR MPL-2.0
