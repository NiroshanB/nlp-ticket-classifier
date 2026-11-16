"""Classify a new support ticket from the command line.

Usage:
    python predict.py "I was charged twice for my order"
    python predict.py --file ticket.txt
"""
import argparse

import joblib


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify a support ticket.")
    parser.add_argument("text", nargs="?", help="Ticket text to classify.")
    parser.add_argument("--file", help="Read ticket text from a file.")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide ticket text or --file.")

    model = joblib.load("outputs/model.joblib")
    predicted = model.predict([text])[0]
    proba = model.predict_proba([text])[0]
    classes = list(model.classes_)
    ranked = sorted(zip(classes, proba), key=lambda x: x[1], reverse=True)

    print(f"Ticket: {text}")
    print(f"Predicted category: {predicted}")
    print("Probabilities:")
    for label, p in ranked:
        print(f"  {label:<10} {p:.3f}")


if __name__ == "__main__":
    main()
