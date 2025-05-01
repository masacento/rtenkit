use anyhow::{Error, Result};
use rten::{InputOrOutput, Model, NodeId, Output};
use rten_tensor::prelude::*;
use rten_tensor::Tensor;
use std::vec::Vec;

pub struct Embeddings {
    model: Model,
}

impl Embeddings {
    pub fn new(model_data: Vec<u8>) -> Result<Self, Error> {
        let model = Model::load(model_data)?;
        Ok(Embeddings { model })
    }

    pub fn embed(&self, input_ids: Vec<Vec<i32>>) -> Result<Vec<Vec<f32>>, Error> {
        if input_ids.is_empty() {
            return Ok(Vec::new());
        }

        let max_sequence_len = input_ids.iter().map(|enc| enc.len()).max().unwrap_or(0);
        if max_sequence_len == 0 {
            return Ok(vec![vec![]; input_ids.len()]);
        }

        let batch = input_ids.len();

        // Create input_ids using Vec and then convert to Tensor
        let mut input_ids_data = vec![0i32; batch * max_sequence_len];
        for (i, enc) in input_ids.iter().enumerate() {
            for (j, &token_id) in enc.iter().enumerate() {
                input_ids_data[i * max_sequence_len + j] = token_id;
            }
        }
        let input_ids_tensor =
            Tensor::from_vec(input_ids_data).into_shape([batch, max_sequence_len]);

        // Create attention_mask using Vec and then convert to Tensor
        let mut attention_mask_data = vec![0i32; batch * max_sequence_len];
        for (i, enc) in input_ids.iter().enumerate() {
            for j in 0..enc.len() {
                attention_mask_data[i * max_sequence_len + j] = 1;
            }
        }
        let attention_mask =
            Tensor::from_vec(attention_mask_data).into_shape([batch, max_sequence_len]);

        // Create token_type_ids (all zeros)
        let type_ids_data = vec![0i32; batch * max_sequence_len];
        let type_ids = Tensor::from_vec(type_ids_data).into_shape([batch, max_sequence_len]);

        let input_ids_id = self.model.node_id("input_ids")?;
        let attention_mask_id = self.model.node_id("attention_mask")?;

        let mut inputs: Vec<(NodeId, InputOrOutput)> = vec![
            (input_ids_id, input_ids_tensor.view().into()),
            (attention_mask_id, attention_mask.view().into()),
        ];

        // Add token_type_ids if needed
        if let Some(type_ids_id) = self.model.find_node("token_type_ids") {
            inputs.push((type_ids_id, type_ids.view().into()));
        }

        // Try to find sentence_embedding node first, fall back to last output node if not found
        let embedding_id = self
            .model
            .find_node("sentence_embedding")
            .unwrap_or_else(|| {
                *self
                    .model
                    .output_ids()
                    .last()
                    .expect("Model has no output nodes")
            });

        let [embeddings] = self.model.run_n(inputs, [embedding_id], None)?;

        // Convert output to tensor and extract embeddings
        match embeddings {
            Output::FloatTensor(tensor) => {
                let mut result = Vec::with_capacity(batch);
                let shape = tensor.shape();

                if shape.len() == 2 {
                    // Shape is [batch_size, embedding_dim]
                    for i in 0..batch {
                        let start = i * shape[1];
                        let embedding = tensor.iter().skip(start).take(shape[1]).copied().collect();
                        result.push(embedding);
                    }
                } else if shape.len() == 3 {
                    // Shape is [batch_size, sequence_length, embedding_dim]
                    for i in 0..batch {
                        let cls_embedding_start = i * shape[1] * shape[2]; // Offset for the i-th sequence
                        let embedding = tensor
                            .iter()
                            .skip(cls_embedding_start)
                            .take(shape[2]) // Take embedding_dim elements
                            .copied()
                            .collect();
                        result.push(embedding);
                    }
                } else {
                    return Err(anyhow::anyhow!(
                        "Unexpected embedding tensor shape: {:?}. Expected [batch, dim] or [batch, seq_len, dim].",
                        shape
                    ));
                }
                Ok(result)
            }
            _ => Err(anyhow::anyhow!("Model output is not a float tensor")),
        }
    }
}
