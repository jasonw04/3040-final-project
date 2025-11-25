
import kagglehub
import bz2
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

#3 data processing 

path = kagglehub.dataset_download("bittlingmayer/amazonreviews")

file_path = path + "/train.ft.txt.bz2"

with bz2.open(file_path, "rt", encoding="utf-8") as f:
    lines = [line.strip() for line in f]

# Raw DataFrame
df = pd.DataFrame(lines, columns=["raw"])

# Extract labels and review text
df["__label__"] = df["raw"].str.extract(r"(__label__\d)")
df["review_text"] = df["raw"].str.replace(r"__label__\d\s+", "", regex=True)

print("\nInitial data shape:", df.shape)
print(df[["__label__", "review_text"]].head())

# Numeric labels
df["label_num"] = df["__label__"].map({"__label__1": 0, "__label__2": 1})

# Keep only what we need
df = df[["review_text", "label_num"]].dropna()
print("\nAfter selecting columns & dropna, shape:", df.shape)

# 🔹 NEW: sample a subset for modeling
sample_size = 200_000
df = df.sample(sample_size, random_state=42)
print(f"\nAfter sampling for modelling (n={sample_size}), shape:", df.shape)

# Clean text
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["review_text"].apply(clean_text)

print("\nSample cleaned text:")
print(df["clean_text"].head())

# Train-test split
X = df["clean_text"]
y = df["label_num"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain/Test sizes (after sampling):")
print("  X_train:", X_train.shape)
print("  X_test :", X_test.shape)

#4 classification problem - tf-idf + baseline

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.95,
    min_df=5,
    ngram_range=(1, 2),
    max_features=20_000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF–IDF shapes:")
print("  X_train_tfidf:", X_train_tfidf.shape)
print("  X_test_tfidf :", X_test_tfidf.shape)

# Class weights
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))
print("\nClass weights:")
print(class_weight_dict)

# Baseline classifier
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train_tfidf, y_train)

print("\n=== Baseline (Most Frequent Class) ===")
print(f"Train accuracy = {baseline.score(X_train_tfidf, y_train):.4f}")
print(f"Test accuracy  = {baseline.score(X_test_tfidf, y_test):.4f}")

#5 methods + results - decision tree, nb, logreg


def evaluate_model(name, model, X_test, y_test, save_confusion=True):
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    auc = None
    if hasattr(model, "predict_proba"):
        auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    elif hasattr(model, "decision_function"):
        auc = roc_auc_score(y_test, model.decision_function(X_test))

    print(f"\n=== {name} ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"AUC      : {auc:.4f}" if auc is not None else "AUC      : N/A")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    if save_confusion:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap="Blues")
        plt.title(f"Confusion Matrix – {name}")
        plt.grid(False)
        fname = f"cm_{name.replace(' ', '_').lower()}.png"
        plt.savefig(fname, bbox_inches="tight")
        plt.close()


    return acc, prec, rec, f1, auc


results = {}

print("\nTraining KNN on sampled data...")

knn_sample_size = 20_000

train_size = min(knn_sample_size, X_train_tfidf.shape[0])

X_train_knn, _, y_train_knn, _ = train_test_split(
    X_train_tfidf,
    y_train,
    train_size=train_size,
    random_state=42,
    stratify=y_train
)

knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
knn.fit(X_train_knn, y_train_knn)
results["KNN"] = evaluate_model("KNN", knn, X_test_tfidf, y_test)

# decision tree
print("\nTraining Decision Tree on sampled data...")
tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=25,        # shallower tree to avoid crazy overfitting / runtime
    min_samples_leaf=20  # avoid tiny leaves
)
tree.fit(X_train_tfidf, y_train)
results["Decision Tree"] = evaluate_model("Decision Tree", tree, X_test_tfidf, y_test)

# naive bayes
print("\nTraining Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
results["Naive Bayes"] = evaluate_model("Naive Bayes", nb, X_test_tfidf, y_test)


# logistic regression
print("\nTraining Logistic Regression...")
logreg = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    n_jobs=-1
)
logreg.fit(X_train_tfidf, y_train)
results["Logistic Regression"] = evaluate_model("Logistic Regression", logreg, X_test_tfidf, y_test)


# summary
rows = []
for model_name, (acc, prec, rec, f1, auc) in results.items():
    rows.append({
        "Model": model_name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "AUC": auc
    })

results_df = pd.DataFrame(rows).sort_values(by="Accuracy", ascending=False)

print("\n\n=== SUMMARY OF ALL MODEL RESULTS (sampled 200k) ===")
print(results_df)

results_df.to_csv("model_results_summary_sample200k.csv", index=False)
