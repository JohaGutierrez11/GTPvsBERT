from sentiment_model import SentimentAnalyzer

bert_model = SentimentAnalyzer(
    "./results/bert-base-uncased/checkpoint-750",
    "bert-base-uncased"
)

gpt_model = SentimentAnalyzer(
    "./results/gpt2/checkpoint-750",
    "gpt2"
)

texto = "The movie was the best"
print(texto)
print("BERT:", bert_model.predict(texto))
print("GPT:", gpt_model.predict(texto))