use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde::Serialize;
use serde_json;

use crate::steerlm::SteerLM;
use crate::utils::post;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CompletionConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens_to_generate: Option<i64>,
    pub left_space_trim: bool,
    pub logprobs: bool,
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i64,
    pub stop: Vec<String>,
    pub random_seed: i64,
    pub repetition_penalty: f64,
    pub beam_search_diversity_rate: f64,
    pub beam_width: i64,
    pub length_penalty: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guardrail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steer_lm: Option<SteerLM>,
}

impl Default for CompletionConfig {
    fn default() -> CompletionConfig {
        CompletionConfig {
            stop: vec![],
            tokens_to_generate: None,
            temperature: 1.0,
            top_k: 1,
            top_p: 1.0,
            random_seed: 0,
            beam_search_diversity_rate: 0.0,
            beam_width: 1,
            repetition_penalty: 1.0,
            length_penalty: 0.0,
            guardrail: None,
            left_space_trim: true,
            logprobs: false,
            steer_lm: None,
        }
    }
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CompletionRequest {
    pub prompt: String,
    #[serde(flatten)]
    config: CompletionConfig,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub knowledge_base_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub citation_count: Option<i64>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct CompletionResponse {
    pub text: String,
    pub cumlogprobs: f64,
    pub logprobs: Option<Vec<f64>>,
    pub prompt_labels: Vec<ClassificationLabel>,
    pub completion_labels: Vec<ClassificationLabel>,
    pub completions: Option<Vec<_Completion>>,
    pub citations: Option<Vec<Citation>>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct ClassificationLabel {
    pub class_name: String,
    pub score: f64,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct _Completion {}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Citation {
    pub chunk_id: String,
    pub feedback: Feedback,
    pub id: String,
    pub text: String,
    pub document_id: String,
    pub document_name: String,
    pub knowledge_base_id: String,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub struct Feedback {
    pub category: String,
    pub comment: String,
    pub suggestion: String,
    pub completion_id: String,
    pub citation_id: String,
    pub rating: i64,
    #[serde(rename = "type")]
    pub type_field: String,
}

#[derive(Default, Debug, Clone, PartialEq)]
pub struct Completion {
    config: CompletionConfig,
    model: String,
    api_token: Option<String>,
}

impl Completion {
    pub fn new(
        config: CompletionConfig,
        model: String,
        token: Option<String>,
    ) -> Result<Completion> {
        return Ok(Completion {
            config,
            model,
            api_token: token,
        });
    }

    pub fn set_token(mut self, token: String) {
        self.api_token = Some(token);
    }

    pub async fn complete(self, prompt: String) -> Result<CompletionResponse> {
        let endpoint = format!("{}/models/{}/completions", crate::API_URL, self.model);

        let completion_request = CompletionRequest {
            prompt,
            config: self.config,
            ..Default::default()
        };

        log::debug!(
            "{}",
            serde_json::to_string_pretty(&completion_request).unwrap()
        );

        let response = post(&completion_request, endpoint, self.api_token).await?;

        if response.status().is_success() {
            Ok(response.json::<CompletionResponse>().await?)
        } else {
            Err(anyhow!(format!("{:?}", response)))
        }
    }
}
