//! tract + tokenizers model wrapper for `embed`.

use crate::error::{Error, Result};
use crate::pool::{l2_normalize_rows, mean_pool_with_mask};
use ndarray::{Array2, Ix2, Ix3};
use std::path::Path;
use std::sync::Arc;
use tokenizers::{
    PaddingParams, PaddingStrategy, Tokenizer, TruncationParams, tokenizer::TruncationStrategy,
};
use tract_onnx::data_resolver::ModelDataResolver;
use tract_onnx::prelude::*;
use tract_onnx::tract_hir::internal::{anyhow, bail};
use tract_onnx::tract_hir::tract_core::model::typed::{TypedModel, TypedSimplePlan};

pub(crate) struct Model {
    plan: TypedSimplePlan<TypedModel>,
    tokenizer: Tokenizer,
    input_names: Vec<String>,
    model_id: String,
    dimensions: usize,
    max_seq_length: usize,
}

impl Model {
    pub(crate) fn new_from_bytes(
        model_id: impl Into<String>,
        onnx_bytes: Vec<u8>,
        onnx_external_data: Option<&[u8]>,
        tokenizer_json_bytes: &[u8],
        dimensions: usize,
        max_seq_length: usize,
    ) -> Result<Self> {
        if dimensions == 0 {
            return Err(Error::InvalidConfig("dimensions must be >= 1".to_owned()));
        }
        if max_seq_length == 0 {
            return Err(Error::InvalidConfig(
                "max_seq_length must be >= 1".to_owned(),
            ));
        }

        let model_id = model_id.into();
        let inference_model = load_inference_model(&model_id, onnx_bytes, onnx_external_data)?;
        let input_names = input_names(&inference_model)?;
        validate_inputs(&input_names)?;

        let plan = inference_model
            .into_optimized()
            .and_then(|model| model.into_runnable())
            .map_err(|err| {
                Error::InvalidConfig(format!(
                    "failed to optimize runnable tract model '{model_id}': {err}"
                ))
            })?;

        let mut tokenizer = Tokenizer::from_bytes(tokenizer_json_bytes).map_err(|err| {
            Error::InvalidConfig(format!(
                "failed to load tokenizer.json for model '{model_id}': {err}"
            ))
        })?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_seq_length,
                strategy: TruncationStrategy::LongestFirst,
                stride: 0,
                ..TruncationParams::default()
            }))
            .map_err(|err| {
                Error::InvalidConfig(format!(
                    "failed to configure tokenizer truncation for model '{model_id}': {err}"
                ))
            })?;
        // BatchLongest pads to the longest encoding in the
        // batch (capped at `max_length` via the truncation
        // params above). Fixed(max_seq_length) would force
        // every batch to seq_len=max — for a model like
        // bge-m3 with max_seq_length=8192 that means a 5-token
        // query becomes an 8192-token forward pass, wasting
        // ~1500x the CPU. Tract handles dynamic seq lengths
        // for sentence-transformer ONNX exports natively
        // (their seq dim is symbolic).
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..PaddingParams::default()
        }));

        Ok(Self {
            plan,
            tokenizer,
            input_names,
            model_id,
            dimensions,
            max_seq_length,
        })
    }

    pub(crate) fn model_id(&self) -> &str {
        &self.model_id
    }

    pub(crate) fn dimensions(&self) -> usize {
        self.dimensions
    }

    pub(crate) fn forward(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|err| Error::Internal(format!("tokenizer encode_batch failed: {err}")))?;

        let batch = encodings.len();
        if batch != texts.len() {
            return Err(Error::Internal(format!(
                "tokenizer output count mismatch: expected {}, got {batch}",
                texts.len()
            )));
        }

        // Under `BatchLongest` padding every encoding shares
        // the same length — the longest in the batch, capped
        // by the truncation `max_length` we configured at
        // construction. Read it from the first encoding and
        // sanity-check the rest.
        let seq_len = encodings
            .first()
            .map(|e| e.get_ids().len())
            .ok_or_else(|| Error::Internal("tokenizer produced no encodings".to_owned()))?;
        if seq_len > self.max_seq_length {
            return Err(Error::Internal(format!(
                "tokenizer produced seq_len {seq_len} exceeding configured max_seq_length {}",
                self.max_seq_length
            )));
        }

        let mut input_ids = Array2::<i64>::zeros((batch, seq_len));
        let mut attention_mask = Array2::<i64>::zeros((batch, seq_len));
        let mut token_type_ids = Array2::<i64>::zeros((batch, seq_len));

        for (row, encoding) in encodings.iter().enumerate() {
            if encoding.get_ids().len() != seq_len
                || encoding.get_attention_mask().len() != seq_len
                || encoding.get_type_ids().len() != seq_len
            {
                return Err(Error::Internal(
                    "tokenizer produced inconsistent sequence lengths within batch".to_owned(),
                ));
            }

            for (col, id) in encoding.get_ids().iter().enumerate() {
                input_ids[(row, col)] = i64::from(*id);
            }
            for (col, mask) in encoding.get_attention_mask().iter().enumerate() {
                attention_mask[(row, col)] = i64::from(*mask);
            }
            for (col, type_id) in encoding.get_type_ids().iter().enumerate() {
                token_type_ids[(row, col)] = i64::from(*type_id);
            }
        }

        let mut inputs = TVec::new();
        for name in &self.input_names {
            if is_input_ids(name) {
                inputs.push(input_ids.clone().into_tensor().into_tvalue());
            } else if is_attention_mask(name) {
                inputs.push(attention_mask.clone().into_tensor().into_tvalue());
            } else if is_token_type_ids(name) {
                inputs.push(token_type_ids.clone().into_tensor().into_tvalue());
            } else {
                return Err(Error::Internal(format!(
                    "unsupported tract model input '{name}' survived constructor validation"
                )));
            }
        }

        let outputs = self
            .plan
            .run(inputs)
            .map_err(|err| Error::Internal(format!("tract inference failed: {err}")))?;

        pooled_embeddings(outputs, &attention_mask, batch, self.dimensions)
    }

    #[cfg(test)]
    pub(crate) fn for_tests(model_id: &str, dimensions: usize, max_seq_length: usize) -> Self {
        let mut model = TypedModel::default();
        let source = model
            .add_source("input_ids", i64::fact([1, max_seq_length]))
            .expect("test model source should build");
        model
            .set_output_outlets(&[source])
            .expect("test model output should build");
        let plan = model
            .into_runnable()
            .expect("test runnable model should build");
        Self {
            plan,
            tokenizer: Tokenizer::new(tokenizers::models::wordlevel::WordLevel::default()),
            input_names: vec!["input_ids".to_owned()],
            model_id: model_id.to_owned(),
            dimensions,
            max_seq_length,
        }
    }
}

struct ExternalDataBytes {
    data: Vec<u8>,
}

impl ModelDataResolver for ExternalDataBytes {
    fn read_bytes_from_path(
        &self,
        buf: &mut Vec<u8>,
        p: &Path,
        offset: usize,
        length: Option<usize>,
    ) -> TractResult<()> {
        let file_name = p.file_name().and_then(|name| name.to_str()).unwrap_or("");
        if file_name != "model.onnx_data" {
            bail!("unexpected ONNX external data file requested: {p:?}");
        }

        let available = self
            .data
            .len()
            .checked_sub(offset)
            .ok_or_else(|| anyhow!("external data offset {offset} exceeds supplied bytes"))?;
        let length = length.unwrap_or(available);
        let end = offset
            .checked_add(length)
            .ok_or_else(|| anyhow!("external data range overflows usize"))?;
        if end > self.data.len() {
            bail!(
                "external data range {offset}..{end} exceeds supplied {} bytes",
                self.data.len()
            );
        }

        buf.extend_from_slice(&self.data[offset..end]);
        Ok(())
    }
}

fn load_inference_model(
    model_id: &str,
    onnx_bytes: Vec<u8>,
    onnx_external_data: Option<&[u8]>,
) -> Result<InferenceModel> {
    match onnx_external_data {
        None => tract_onnx::onnx()
            .model_for_read(&mut &onnx_bytes[..])
            .map_err(|err| {
                Error::InvalidConfig(format!(
                    "failed to load ONNX model '{model_id}' from supplied bytes: {err}"
                ))
            }),
        Some(data) => {
            let mut onnx = tract_onnx::onnx();
            let proto = onnx
                .proto_model_for_read(&mut &onnx_bytes[..])
                .map_err(|err| {
                    Error::InvalidConfig(format!(
                        "failed to decode ONNX model '{model_id}' from supplied bytes: {err}"
                    ))
                })?;
            onnx.provider = Arc::new(ExternalDataBytes {
                data: data.to_vec(),
            });
            let parsed = onnx.parse(&proto, Some("")).map_err(|err| {
                Error::InvalidConfig(format!(
                    "failed to load ONNX model '{model_id}' with supplied external data: {err}"
                ))
            })?;
            if parsed.unresolved_inputs.is_empty() {
                Ok(parsed.model)
            } else {
                Err(Error::InvalidConfig(format!(
                    "ONNX model '{model_id}' has unresolved inputs: {:?}",
                    parsed.unresolved_inputs
                )))
            }
        }
    }
}

fn input_names(model: &InferenceModel) -> Result<Vec<String>> {
    let outlets = model
        .input_outlets()
        .map_err(|err| Error::InvalidConfig(format!("failed to inspect model inputs: {err}")))?;
    let mut names = Vec::with_capacity(outlets.len());
    for outlet in outlets {
        names.push(model.node(outlet.node).name.clone());
    }
    Ok(names)
}

fn validate_inputs(input_names: &[String]) -> Result<()> {
    let has_input_ids = input_names.iter().any(|name| is_input_ids(name));
    let has_attention_mask = input_names.iter().any(|name| is_attention_mask(name));
    if !has_input_ids || !has_attention_mask {
        return Err(Error::InvalidConfig(format!(
            "ONNX model must expose input_ids and attention_mask inputs, got {input_names:?}"
        )));
    }

    if let Some(name) = input_names
        .iter()
        .find(|name| !is_input_ids(name) && !is_attention_mask(name) && !is_token_type_ids(name))
    {
        return Err(Error::InvalidConfig(format!(
            "unsupported ONNX model input '{name}'"
        )));
    }

    Ok(())
}

fn is_input_ids(name: &str) -> bool {
    name == "input_ids"
}

fn is_attention_mask(name: &str) -> bool {
    name == "attention_mask"
}

fn is_token_type_ids(name: &str) -> bool {
    name == "token_type_ids"
}

fn pooled_embeddings(
    outputs: TVec<TValue>,
    attention_mask: &Array2<i64>,
    expected_batch: usize,
    dimensions: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut last_error = None;

    for output in outputs {
        let tensor = output.into_tensor();
        let view = tensor
            .to_array_view::<f32>()
            .map_err(|err| Error::Internal(format!("tract output is not f32 tensor: {err}")))?;

        match view.shape() {
            [batch, seq, hidden] if *batch == expected_batch && *hidden == dimensions => {
                if *seq != attention_mask.dim().1 {
                    last_error = Some(format!(
                        "last_hidden_state sequence length {seq} did not match attention mask sequence length {}",
                        attention_mask.dim().1
                    ));
                    continue;
                }
                let last_hidden = view
                    .into_dimensionality::<Ix3>()
                    .map_err(|err| Error::Internal(format!("failed to view rank-3 output: {err}")))?
                    .to_owned();
                let mut pooled = mean_pool_with_mask(&last_hidden, attention_mask);
                l2_normalize_rows(&mut pooled);
                return Ok(rows_to_vec(pooled));
            }
            [batch, hidden] if *batch == expected_batch && *hidden == dimensions => {
                let mut pooled = view
                    .into_dimensionality::<Ix2>()
                    .map_err(|err| Error::Internal(format!("failed to view rank-2 output: {err}")))?
                    .to_owned();
                l2_normalize_rows(&mut pooled);
                return Ok(rows_to_vec(pooled));
            }
            shape => {
                last_error = Some(format!(
                    "output shape {shape:?} did not match expected batch {expected_batch} and dimensions {dimensions}"
                ));
            }
        }
    }

    Err(Error::Internal(format!(
        "no usable embedding output found: {}",
        last_error.unwrap_or_else(|| "model returned no outputs".to_owned())
    )))
}

fn rows_to_vec(rows: Array2<f32>) -> Vec<Vec<f32>> {
    rows.outer_iter().map(|row| row.to_vec()).collect()
}
