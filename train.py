"""Train a support-ticket text classifier (TF-IDF + Logistic Regression).

Reads data/tickets.csv, splits it into train/test sets, trains a
TF-IDF + Logistic Regression pipeline, prints a classification report,
saves the confusion matrix chart and the trained model.
"""
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

CATEGORIES = ["account", "billing", "returns", "shipping", "technical"]


def main() -> None:
    df = pd.read_csv("data/tickets.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.2,
        random_state=RANDOM_STATE, stratify=df["category"],
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=3)

    print(f"Test set size: {len(X_test)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(report)

    with open("outputs/metrics.txt", "w") as f:
        f.write(f"Test set size: {len(X_test)}\n")
        f.write(f"Accuracy: {accuracy:.3f}\n\n")
        f.write(report)

    cm = confusion_matrix(y_test, y_pred, labels=CATEGORIES)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CATEGORIES)))
    ax.set_yticks(range(len(CATEGORIES)))
    ax.set_xticklabels(CATEGORIES, rotation=30, ha="right")
    ax.set_yticklabels(CATEGORIES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix — Support Ticket Classifier")
    for i in range(len(CATEGORIES)):
        for j in range(len(CATEGORIES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("outputs/confusion_matrix.png", dpi=150)
    print("Saved outputs/confusion_matrix.png")

    joblib.dump(model, "outputs/model.joblib")
    print("Saved outputs/model.joblib")


if __name__ == "__main__":
    main()
