from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SentimentAnalyzer:

    def __init__(self, model_path, tokenizer_name):

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # GPT needs special padding
        if tokenizer_name == "gpt2":
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def predict(self, text):

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)

        outputs = self.model(**inputs)

        pred = torch.argmax(outputs.logits).item()

        if pred == 1:
            return "Positive"
        else:
            return "Negative"