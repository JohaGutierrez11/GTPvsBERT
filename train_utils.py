from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score
import numpy as np

def load_data(sample_train=2000, sample_test=1000):
    dataset = load_dataset("imdb")
    train = dataset["train"].shuffle(seed=42).select(range(sample_train))
    test = dataset["test"].shuffle(seed=42).select(range(sample_test))
    return train, test

def tokenize_dataset(train, test, tokenizer):
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)

    train = train.map(tokenize, batched=True)
    test = test.map(tokenize, batched=True)

    train.set_format(type="torch", columns=["input_ids","attention_mask","label"])
    test.set_format(type="torch", columns=["input_ids","attention_mask","label"])

    return train, test

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

def train_model(model_name, train_dataset, test_dataset, tokenizer):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.pad_token_id = tokenizer.pad_token_id

    args = TrainingArguments(
        output_dir=f"./results/{model_name}",
        eval_strategy="epoch",
        logging_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()

    return trainer, results