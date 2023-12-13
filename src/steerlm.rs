use serde::Deserialize;
use serde::Serialize;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SteerLM {
    pub quality: i64,
    pub toxicity: i64,
    pub humor: i64,
    pub creativity: i64,
    pub violence: i64,
    pub helpfulness: i64,
    #[serde(rename = "not_appropriate")]
    pub not_appropriate: i64,
    #[serde(rename = "hate_speech")]
    pub hate_speech: i64,
    #[serde(rename = "sexual_content")]
    pub sexual_content: i64,
    #[serde(rename = "fails_task")]
    pub fails_task: i64,
    #[serde(rename = "political_content")]
    pub political_content: i64,
    #[serde(rename = "moral_judgement")]
    pub moral_judgement: i64,
    pub lang: String,
}
