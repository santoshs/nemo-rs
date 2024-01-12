use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde::Serialize;
use std::fmt;

use crate::utils::post;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    pub content: Vec<String>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub embeddings: Box<Vec<Vec<f32>>>,
    pub model: EmbeddingModel,
}

// The supported models for embeddings
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EmbeddingModel {
    #[default]
    #[serde(rename = "e5-large-unsupervised")]
    E5LargeUnsupervised,
    #[serde(rename = "nre-002")]
    NRE002,
    #[serde(rename = "nre-001")]
    NRE001,
}

impl fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingModel::E5LargeUnsupervised => write!(f, "{}", "e5-large-unsupervised"),
            EmbeddingModel::NRE001 => write!(f, "{}", "nre-001"),
            EmbeddingModel::NRE002 => write!(f, "{}", "nre-002"),
        }
    }
}

pub struct Embeddings {
    model: EmbeddingModel,
    token: Option<String>,
}

impl Embeddings {
    pub fn new(model: EmbeddingModel, token: Option<String>) -> Result<Embeddings> {
        Ok(Embeddings { model, token })
    }

    pub async fn embeddings(self, content: Vec<String>) -> Result<Box<Vec<Vec<f32>>>> {
        if content.len() > 50 {
            return Err(anyhow!("Content list exceeds maximum limit of 50"));
        }
        let endpoint = format!("{}/embeddings/{}", crate::API_URL, self.model);
        let embeddings_request = EmbeddingsRequest { content };

        #[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
        struct NemoError {
            request_id: String,
            message: String,
        }
        let response = post(&embeddings_request, endpoint, self.token).await?;
        let status = response.status();
        if status.is_success() {
            Ok(response.json::<EmbeddingsResponse>().await?.embeddings)
        } else {
            let err = response.json::<NemoError>().await?;
            Err(anyhow!(format!("{}: {}", status, err.message)))
        }
    }
}
