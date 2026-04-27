# Embed Integration Tests

Integration tests are compiled only when the default bundle exists:
`bundled-default-model` must be enabled and build.rs must emit
`embed_default_bundle`.

For routine iteration, use the small multilingual override:

```sh
PHILHARMONIC_EMBED_DEFAULT_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
PHILHARMONIC_EMBED_DEFAULT_REVISION=e8f8c211226b894fcb81acc59f3b34ba3efd5f42 \
    ./scripts/pre-landing.sh philharmonic-connector-impl-embed
```

The default bge-m3 bundle is cached under
`$XDG_CACHE_HOME/philharmonic/embed-bundles/` or
`$HOME/.cache/philharmonic/embed-bundles/`, unless
`PHILHARMONIC_EMBED_CACHE_DIR` is set. Use
`--no-default-features` for offline unit-test-only iteration.
