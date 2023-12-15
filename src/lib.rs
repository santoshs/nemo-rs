pub mod completion;
pub mod embeddings;
pub mod models;
pub mod steerlm;
mod utils;

const API_URL: &str = "https://api.llm.ngc.nvidia.com/v1";

#[cfg(test)]
mod tests {
    use dotenv::dotenv;

    use crate::completion::*;
    use crate::embeddings::*;
    use crate::models::*;

    macro_rules! async_test {
        ($e:expr) => {
            tokio_test::block_on($e)
        };
    }

    #[test]
    fn completion_question() {
        dotenv().ok();
        let c = Completion::new(
            CompletionConfig {
                tokens_to_generate: Some(3),
                ..Default::default()
            },
            "llama-2-70b-hf".to_string(),
            None,
        )
        .unwrap();
        assert_eq!(async_test!(c.complete(
            "Q: What is the capital of Spain? \
             A: Madrid \
             \
             Q: What is synthetic biology? \
             A: Synthetic Biology is about designing biological systems at multiple levels from individual molecules up to whole cells and even multicellular assemblies like tissues and organs to perform specific functions. \
             \
             Q: How far is the Sun from the Earth? \
             A:  93 million miles \
             \
             Q: What is the deepest part of the ocean? \
             A: The Mariana Trench \
             \
             Q: What is the capital of India? \
             A:".to_string())).unwrap().text, "New Delhi");
    }

    #[test]
    fn embeddings_test() {
        dotenv().ok();
        let e = Embeddings::new(EmbeddingModel::E5LargeUnsupervised, None).unwrap();
        async_test!(e.embeddings(vec!["Rust embeddings test".to_string()])).unwrap();
    }

    #[test]
    fn get_models() {
        dotenv().ok();
        async_test!(list_models(None)).unwrap();
    }

    #[test]
    fn count_tokens() {
        dotenv().ok();
        let count = async_test!(count_prompt_tokens(
            "Token count test!".to_string(),
            "gpt-8b-000".to_string(),
            None,
            None,
            None
        ))
        .unwrap();

        assert_eq!(count, 4);
    }
}
