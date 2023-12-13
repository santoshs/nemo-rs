use reqwest::{Error, Response};
use serde::Serialize;

pub async fn post<T: Serialize + ?Sized>(
    json: &T,
    endpoint: String,
    token: Option<String>,
) -> Result<Response, Error> {
    let client = reqwest::Client::new();
    let token = if token.is_some() {
        token.unwrap()
    } else {
        std::env::var("NGC_API_KEY").expect("NGC_API_KEY must be set.")
    };

    client
        .post(endpoint)
        .header("Content-Type", "application/json")
        .header("Authorization", format!("Bearer {}", token))
        .header("x-stream", "false")
        .json(json)
        .send()
        .await
}
