use anyhow::Result;
use serde::Deserialize;
use serde::Serialize;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Models {
    pub models: Vec<Model>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub name: String,
    #[serde(rename = "type")]
    pub type_field: Option<String>,
    pub size: Option<i64>,
    pub context_size: i64,
    pub description: Option<String>,
    pub visibility: String,
    pub publisher: Option<String>,
    pub features: Features,
    pub deprecated: Option<Deprecated>,
    pub model_card: Option<bool>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Features {
    pub support_ptuning: Option<bool>,
    pub support_lora_tuning: Option<bool>,
    pub chat_compatible: Option<bool>,
    pub steer_lm: Option<bool>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Deprecated {
    pub suggested_model: String,
    pub date: String,
}

pub async fn list_models(token: Option<String>) -> Result<Vec<Model>> {
    let endpoint = format!("{}/models", crate::API_URL);
    let client = reqwest::Client::new();
    let token = if token.is_some() {
        token.unwrap()
    } else {
        std::env::var("NGC_API_KEY").expect("NGC_API_KEY must be set.")
    };

    let response = client
        .get(endpoint)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", token))
        .send()
        .await?;

    Ok(response.json::<Models>().await?.models)
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CountPromptTokens {
    pub prompt: String,
    pub knowledge_base_ids: Option<Vec<String>>,
    pub citation_count: Option<i64>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CountResponse {
    pub input_length: i32,
}

pub async fn count_prompt_tokens(
    prompt: String,
    model: String,
    knowledge_base_ids: Option<Vec<String>>,
    citation_count: Option<i64>,
    token: Option<String>,
) -> Result<i32> {
    let endpoint = format!("{}/models/{}/count_tokens", crate::API_URL, model);
    let response = crate::utils::post(
        &CountPromptTokens {
            prompt,
            knowledge_base_ids,
            citation_count,
        },
        endpoint,
        token,
    )
    .await?;

    Ok(response.json::<CountResponse>().await?.input_length)
}
