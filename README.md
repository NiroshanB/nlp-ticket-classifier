# Support Ticket Classifier

Text classification for customer support tickets using TF-IDF features and
Logistic Regression (scikit-learn). Given the text of a ticket, the model
predicts which team should handle it: billing, technical, account,
shipping, or returns.

## Skills demonstrated

- Python, Pandas, NumPy
- Natural language processing with TF-IDF (unigrams + bigrams)
- Logistic Regression classification with scikit-learn
- Train/test evaluation: classification report, confusion matrix
- Saving and loading a trained model (joblib)
- Command-line prediction tool

## Data

The dataset is **synthetic**: `generate_data.py` builds 2,200 realistic
tickets (400 per category from hand-written templates, plus 40 harder
cross-category tickets per category) from templates with randomised
slots (products, order numbers, amounts, dates). It is clearly labelled
synthetic because it is not real customer data. To keep the task
realistic rather than trivially separable, 10% of tickets have a random
word dropped and 3% carry a deliberately noisy label, simulating the
messiness of a real support queue. The generation seed is fixed, so the
dataset is reproducible.

## Setup

```bash
pip install -r requirements.txt
```

## How to run

```bash
# 1. Generate the dataset (creates data/tickets.csv)
python generate_data.py

# 2. Train the model (creates outputs/model.joblib,
#    outputs/metrics.txt, outputs/confusion_matrix.png)
python train.py

# 3. Classify a new ticket
python predict.py "I was charged twice for my order, please refund me"
```

## Results

Test set accuracy: **0.975** (440 held-out tickets).

```
              precision    recall  f1-score   support

     account      0.966     0.966     0.966        89
     billing      0.955     0.977     0.966        87
     returns      0.978     0.989     0.983        88
    shipping      0.989     0.989     0.989        89
   technical      0.988     0.954     0.971        87

    accuracy                          0.975       440
   macro avg      0.975     0.975     0.975       440
weighted avg      0.975     0.975     0.975       440
```

(The numbers above are from the run in this repo; re-running `train.py`
reproduces them. The remaining errors come mostly from the deliberately
noisy labels and the cross-category tickets, which is also a fair
limitation to note: real tickets are messier and would score lower.)

## Project structure

```
nlp-ticket-classifier/
├── generate_data.py   # builds the synthetic ticket dataset
├── train.py           # trains and evaluates the classifier
├── predict.py         # classifies a ticket from the command line
├── data/
│   └── tickets.csv    # generated dataset (2,200 rows)
├── outputs/
│   ├── metrics.txt          # classification report
│   ├── confusion_matrix.png # confusion matrix chart
│   └── model.joblib         # trained model
└── requirements.txt
```
