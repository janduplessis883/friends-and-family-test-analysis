![Image](images/fftest2.png)
# Friends & Family Test Analysis 
**Primary Care**

The NHS Friends and Family Test (FFT) is a quick and anonymous way for patients to give their views after receiving care or treatment across the NHS in England. Essentially, it's a single-question survey asking whether the patient would recommend the service they've received to friends and family who need similar treatment or care. 

The test aims to highlight both good and bad patient experiences, giving healthcare providers and commissioners a more clear understanding of the areas where improvements are needed. By analyzing the responses, NHS facilities can make informed decisions on how to enhance the quality of their services. Overall, the FFT serves as a valuable tool for continuous, patient-centered feedback in the healthcare system.

---
### Under the Hood
FFT Data is loaded from a Google Sheet, and the data is prepared by a Python script on a local machine and then pushed to GitHub ready for access via Streamlit.

**Machine Learning**<BR>
Various Machine Learning techniques are used to analyse the data:
- Sentiment Analysis - [Huggingface](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest) `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Text Classification of Patient Feedback - [Huggingface](https://huggingface.co/SamLowe/roberta-base-go_emotions) `SamLowe/roberta-base-go_emotions`
- Zero-shot Classification - [Huggingface](https://huggingface.co/facebook/bart-large-mnli) `facebook/bart-large-mnli`


## [View the Stremlit App](https://friends-and-family-test-analysis-pqev4j3c9katnrlv8kktnb.streamlit.app/)
