![Image](https://github.com/janduplessis883/friends-and-family-test-analysis/blob/master/images/fftestabout.png?raw=true)

## Friends & Family Test Dashboard for Primary Care
Welcome to my new dashboard, aimed at enhancing how healthcare providers understand and use patient feedback. This tool focuses on the Friends and Family Test (FFT), which is essential for collecting patients' views on healthcare services. Our approach uses advanced text classification and sentiment analysis to organize and interpret this feedback more effectively.

Here's the core idea: Instead of just counting responses, the dashboard analyzes the sentiments behind them—whether positive, negative, or neutral. It assigns a detailed score to each piece of feedback, allowing for a more nuanced understanding of patient satisfaction. This method helps identify specific areas needing improvement and those that are performing well, based on real patient experiences.

For healthcare providers, this tool offers a more insightful way to look at patient feedback. It doesn’t just provide data; it offers a clearer picture of how patients feel about their care. This can help highlight excellence in services and pinpoint areas for potential improvements.

The data we use comes from a GP surgery in West London, showing how this tool can be applied in a real healthcare setting.

We employ several machine learning techniques for analysis:

1. **Sentiment Analysis:** Using Huggingface's 'cardiffnlp/twitter-roberta-base-sentiment-latest' model, we determine the emotional tone of the feedback.
2. **Text Classification** of Patient Feedback: To categorize feedback into different emotional themes, we use the 'SamLowe/roberta-base-go_emotions' model from Huggingface.
3. **Zero-shot Classification** of Patient Improvement Suggestions: The 'facebook/bart-large-mnli' model helps us identify and classify suggestions for improving patient care, even when the model hasn’t been specifically trained on healthcare data.
4. **Fine-tuned Zero-shot Calssification with FitSet** Classification of GP Reviews achieved with SetFit Algorithm. SetFit first fine-tunes a Sentence Transformer model on a small number of labeled examples (8 per class). This is followed by training a classifier head on the embeddings generated from the fine-tuned Sentence Transformer. https://huggingface.co/blog/setfit

Developed by [janduplessis883](https://github.com/janduplessis883/friends-and-family-test-analysis)

Collaboration welcomed!


# [View the Streamlit App](https://friends-and-family-test-analysis-pqev4j3c9katnrlv8kktnb.streamlit.app/)
