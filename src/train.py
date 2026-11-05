"""

Pipeline:   ticket description -> TF-IDF numbers -> Logistic Regression -> category

"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
CLEAN_PATH = ROOT / "data" / "tickets_clean.csv"
MODEL_PATH = ROOT / "model" / "ticket_classifier.joblib"
METRICS_PATH = ROOT / "model" / "metrics.txt"
CM_PATH = ROOT / "screenshots" / "confusion_matrix.png"

df = pd.read_csv(CLEAN_PATH)
X = df["clean_description"]   # input  (what the model reads)
y = df["category"]            # output (what the model must predict)


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,      
    random_state=42,    
    stratify=y,         
)
print(f"Training tickets: {len(X_train)} | Testing tickets: {len(X_test)}")


def make_pipeline(classifier):
    """TF-IDF + a classifier, bundled so they always work together."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2))),
        ("clf", classifier),
    ])



# 5-fold cross-validation on the TRAINING data only (the test set stays untouched)
candidates = {
    "Naive Bayes":         make_pipeline(MultinomialNB()),
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
}
print("\nModel comparison (5-fold cross-validation on training data):")
comparison_lines = []
for name, pipe in candidates.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    line = f"  {name:20s} mean accuracy = {scores.mean():.3f}"
    print(line)
    comparison_lines.append(line)

model = make_pipeline(LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="weighted", zero_division=0
)
report = classification_report(y_test, y_pred, zero_division=0)

print("\n" + "=" * 50)
print("TEST RESULTS")
print("=" * 50)
print(f"Accuracy : {accuracy:.3f}  ({accuracy * 100:.1f}%)")
print(f"Precision: {precision:.3f}  (weighted average)")
print(f"Recall   : {recall:.3f}  (weighted average)")
print(f"F1 score : {f1:.3f}  (weighted average)")
print("\nPer-category report:\n", report)

# Which test tickets did the model get wrong?
wrong = pd.DataFrame({"text": X_test.values, "actual": y_test.values, "predicted": y_pred})
wrong = wrong[wrong["actual"] != wrong["predicted"]]
print(f"Wrongly classified test tickets: {len(wrong)}")
if len(wrong):
    print(wrong.to_string(index=False))

# ------------------------------------------------------------------ 6. confusion matrix
labels = sorted(y.unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix (test data)")
plt.xlabel("Predicted category")
plt.ylabel("Actual category")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig(CM_PATH, dpi=150)
plt.close()


MODEL_PATH.parent.mkdir(exist_ok=True)
CM_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

with open(METRICS_PATH, "w") as f:
    f.write(f"Training tickets: {len(X_train)} | Testing tickets: {len(X_test)}\n\n")
    f.write("Model comparison (5-fold CV on training data):\n")
    f.write("\n".join(comparison_lines) + "\n\n")
    f.write(f"Accuracy : {accuracy:.3f}\n")
    f.write(f"Precision: {precision:.3f} (weighted)\n")
    f.write(f"Recall   : {recall:.3f} (weighted)\n")
    f.write(f"F1 score : {f1:.3f} (weighted)\n\n")
    f.write(report)
    f.write("\nConfusion matrix (rows = actual, columns = predicted):\n")
    f.write(f"Labels: {labels}\n{cm}\n")

print(f"\nModel saved to   {MODEL_PATH}")
print(f"Confusion matrix saved to {CM_PATH}")
