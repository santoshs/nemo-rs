pub mod completion;
pub mod steerlm;

#[cfg(test)]
mod tests {
    use dotenv::dotenv;

    use crate::completion::*;

    macro_rules! aw {
        ($e:expr) => {
            tokio_test::block_on($e)
        };
    }

    #[test]
    fn completion() {
        dotenv().ok();
        let c = Completion::new(
            CompletionConfig {
                tokens_to_generate: Some(3),
                ..Default::default()
            },
            "llama-2-70b-hf".to_string(),
        )
        .unwrap();
        assert_eq!(aw!(c.complete(
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
}
