import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(trainer, test_dataset, title):

    predictions = trainer.predict(test_dataset)

    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    cm = confusion_matrix(labels, preds)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)

    plt.show()

def plot_training_curves(trainer, title):

    logs = trainer.state.log_history

    train_loss = [x["loss"] for x in logs if "loss" in x]
    eval_loss = [x["eval_loss"] for x in logs if "eval_loss" in x]

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(eval_loss, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)

    plt.legend()
    plt.show()