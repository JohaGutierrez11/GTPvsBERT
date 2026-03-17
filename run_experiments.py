from transformers import AutoTokenizer
from train_utils import load_data, tokenize_dataset, train_model
from plots import plot_confusion_matrix, plot_training_curves

print("Loading dataset...")
train, test = load_data()

#Models
models = {
    "GPT2": "gpt2",
    "BERT": "bert-base-uncased"
}

results_summary = {}

for name, model_name in models.items():

    print("\n==========================")
    print("Training model:", name)
    print("==========================")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if model_name == "gpt2":
        tokenizer.pad_token = tokenizer.eos_token

    train_tok, test_tok = tokenize_dataset(train, test, tokenizer)

    trainer, results = train_model(model_name, train_tok, test_tok, tokenizer)

    print("Results:", results)

    results_summary[name] = results["eval_accuracy"]

    plot_confusion_matrix(trainer, test_tok, f"Confusion Matrix - {name}")
    plot_training_curves(trainer, f"Training Curves - {name}")

print("\n====== MODEL COMPARISON ======")

for model, acc in results_summary.items():
    print(model, "Accuracy:", acc)

best_model = max(results_summary, key=results_summary.get)

print("\nBest model:", best_model)