use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde::Serialize;

use crate::utils::post;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsRequest {
    pub content: Vec<String>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingsResponse {
    pub embeddings: Vec<Vec<f64>>,
    pub model: String,
}

pub struct Embeddings {
    model: String,
    token: Option<String>,
}

impl Embeddings {
    pub fn new(model: String, token: Option<String>) -> Result<Embeddings> {
        Ok(Embeddings { model, token })
    }

    pub async fn embeddings(self, content: Vec<String>) -> Result<Vec<Vec<f64>>> {
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
