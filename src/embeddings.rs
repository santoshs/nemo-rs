use anyhow::{anyhow, Result};
use reqwest::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
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
    NVRetrievalQA,
}

impl fmt::Display for EmbeddingModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmbeddingModel::E5LargeUnsupervised => write!(f, "e5-large-unsupervised"),
            EmbeddingModel::NRE001 => write!(f, "nre-001"),
            EmbeddingModel::NRE002 => write!(f, "nre-002"),
            EmbeddingModel::NVRetrievalQA => write!(f, "NV RetrievalQA"),
        }
    }
}

pub struct Embeddings {
    model: EmbeddingModel,
    token: Option<String>,
}

#[derive(Debug, Serialize)]
struct RQARequest {
    input: Vec<String>,
    model: String,
    encoding_format: String,
}

#[derive(Debug, Deserialize)]
struct RQAResponse {
    data: Vec<RQAEmbedding>,
    usage: Option<Usage>,
}

#[derive(Debug, Deserialize, Clone)]
struct RQAEmbedding {
    index: u32,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct Usage {
    prompt_tokens: u32,
    total_tokens: u32,
}

impl Embeddings {
    pub fn new(model: EmbeddingModel, token: Option<String>) -> Result<Embeddings> {
        Ok(Embeddings { model, token })
    }

    pub async fn embeddings(&self, content: Vec<String>) -> Result<Box<Vec<Vec<f32>>>> {
        if content.len() > 50 {
            return Err(anyhow!("Content list exceeds maximum limit of 50"));
        }

        match self.model {
            EmbeddingModel::E5LargeUnsupervised
            | EmbeddingModel::NRE001
            | EmbeddingModel::NRE002 => {
                let endpoint = format!("{}/embeddings/{}", crate::API_URL, self.model);
                let embeddings_request = EmbeddingsRequest { content };

                let response = post(&embeddings_request, endpoint, self.token.clone()).await?;
                let status = response.status();
                if status.is_success() {
                    match response.json::<EmbeddingsResponse>().await {
                        Ok(er) => Ok(er.embeddings),
                        Err(e) => Err(e.into()),
                    }
                } else {
                    Err(anyhow!(format!(
                        "{}: {}",
                        status,
                        response.text().await.unwrap()
                    )))
                }
            }
            EmbeddingModel::NVRetrievalQA => Ok(Box::new(vec![
                nv_rqa(content[0].clone(), self.token.clone()).await?.data[0]
                    .to_owned()
                    .embedding,
            ])),
        }
    }
}

const INVOKE_URL: &str =
    "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/091a03bb-7364-4087-8090-bd71e9277520";
const FETCH_URL_FORMAT: &str = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/";

async fn nv_rqa(input: String, token: Option<String>) -> Result<RQAResponse> {
    let data = RQARequest {
        input: vec![input],
        model: "passage".to_string(),
        encoding_format: "float".to_string(),
    };
    let token = token.unwrap_or_else(|| {
        std::env::var("NV_FOUNDATIONAL_MODEL_TOKEN")
            .expect("NV_FOUNDATIONAL_MODEL_TOKEN must be set.")
    });

    let client = reqwest::Client::new();
    let res = client
        .post(INVOKE_URL)
        .header(AUTHORIZATION, format!("Bearer {}", token))
        .header(ACCEPT, "application/json")
        .header(CONTENT_TYPE, "application/json")
        .json(&data)
        .send()
        .await?;

    while res.status() == reqwest::StatusCode::ACCEPTED {
        let req_id = res.headers().get("NVCF-REQID").unwrap().to_str()?;
        let fetch_url = format!("{}{}", FETCH_URL_FORMAT, req_id);

        let result = client
            .get(&fetch_url)
            .header(AUTHORIZATION, format!("Bearer {}", token))
            .header(ACCEPT, "application/json")
            .send()
            .await?;

        if result.status() == reqwest::StatusCode::ACCEPTED {
            return Ok(result.json::<RQAResponse>().await?);
        }
    }

    if res.status() != reqwest::StatusCode::OK {
        let status = res.status();
        let body = res.text().await?;
        Err(anyhow!(
            "Invocation failed with status {}: {}",
            status,
            body
        ))
    } else {
        Ok(res.json::<RQAResponse>().await?)
    }
}
