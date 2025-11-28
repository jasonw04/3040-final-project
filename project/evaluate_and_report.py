import argparse
import os
import bz2
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, classification_report
)
from scipy.sparse import hstack, csr_matrix

sns.set_style('whitegrid')


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate_model(name, model, X_test, y_test, save_confusion=True):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    auc_score = None
    y_score = None
    if hasattr(model, 'predict_proba'):
        y_score = model.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_score)
    elif hasattr(model, 'decision_function'):
        y_score = model.decision_function(X_test)
        auc_score = roc_auc_score(y_test, y_score)

    if save_confusion:
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(cmap='Blues')
        plt.title(f'Confusion Matrix – {name}')
        plt.grid(False)
        fname = f'cm_{name.replace(" ", "_").lower()}.png'
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'AUC': auc_score,
        'y_score': y_score
    }


def main(sample_size: int, quick: bool):
    print('Downloading dataset (kagglehub). This will reuse cached files if present...')
    path = kagglehub.dataset_download('bittlingmayer/amazonreviews')
    file_path = os.path.join(path, 'train.ft.txt.bz2')
    print('Loading:', file_path)

    with bz2.open(file_path, 'rt', encoding='utf-8') as f:
        lines = [line.strip() for line in f]

    df = pd.DataFrame(lines, columns=['raw'])
    df['__label__'] = df['raw'].str.extract(r'(__label__\d)')
    df['review_text'] = df['raw'].str.replace(r'__label__\d\s+', '', regex=True)
    df = df[['review_text', '__label__']].dropna()
    df['label_num'] = df['__label__'].map({'__label__1': 0, '__label__2': 1})

    if sample_size is not None and sample_size < len(df):
        print(f'Sampling {sample_size} rows for evaluation...')
        df = df.sample(sample_size, random_state=42)

    # Clean
    df['clean_text'] = df['review_text'].apply(clean_text)
    
    # Compute review length (word count) as an additional feature
    df['review_length'] = df['clean_text'].str.split().str.len()

    X = df['clean_text']
    X_len = df['review_length']
    y = df['label_num']

    X_train, X_test, X_train_len, X_test_len, y_train, y_test = train_test_split(
        X, X_len, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.95, min_df=5, ngram_range=(1, 2), max_features=20000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Convert review_length to sparse matrix and append to TF-IDF features
    X_train_len_arr = X_train_len.to_numpy().reshape(-1, 1)
    X_test_len_arr = X_test_len.to_numpy().reshape(-1, 1)
    X_train_len_sparse = csr_matrix(X_train_len_arr)
    X_test_len_sparse = csr_matrix(X_test_len_arr)
    
    # Combine TF-IDF and review_length
    X_train_final = hstack([X_train_tfidf, X_train_len_sparse])
    X_test_final = hstack([X_test_tfidf, X_test_len_sparse])
    
    print(f'Feature matrix shape (TF-IDF + review_length): {X_train_final.shape}')

    # Class weights (for info)
    classes = np.unique(y_train)
    class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    print('Class weights:', class_weight_dict)

    results = []

    # Baseline
    baseline = DummyClassifier(strategy='most_frequent')
    baseline.fit(X_train_final, y_train)
    results.append(evaluate_model('Baseline (Most Frequent)', baseline, X_test_final, y_test))

    # Decision Tree
    tree_max_depth = 25 if not quick else 10
    tree_min_samples_leaf = 20 if not quick else 50
    print('Training Decision Tree...')
    tree = DecisionTreeClassifier(random_state=42, max_depth=tree_max_depth, min_samples_leaf=tree_min_samples_leaf)
    tree.fit(X_train_final, y_train)
    results.append(evaluate_model('Decision Tree', tree, X_test_final, y_test))

    # Naive Bayes
    print('Training Multinomial Naive Bayes...')
    nb = MultinomialNB()
    nb.fit(X_train_final, y_train)
    results.append(evaluate_model('Naive Bayes', nb, X_test_final, y_test))

    # Logistic Regression
    print('Training Logistic Regression...')
    logreg_max_iter = 2000 if not quick else 500
    logreg = LogisticRegression(max_iter=logreg_max_iter, class_weight='balanced', n_jobs=-1)
    logreg.fit(X_train_final, y_train)
    results.append(evaluate_model('Logistic Regression', logreg, X_test_final, y_test))

    # K-Nearest Neighbors (convert sparse matrix to dense for KNN, as it doesn't support sparse input efficiently)
    print('Training K-Nearest Neighbors (k=5)...')
    X_train_dense = X_train_final.toarray()
    X_test_dense = X_test_final.toarray()
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train_dense, y_train)
    results.append(evaluate_model('KNN (k=5)', knn, X_test_dense, y_test))

    # Build results dataframe
    rows = []
    for r in results:
        rows.append({k: v for k, v in r.items() if k != 'y_score'})
    results_df = pd.DataFrame(rows).sort_values(by='Accuracy', ascending=False)
    results_df.to_csv('model_comparison.csv', index=False)
    print('\nSaved: model_comparison.csv')

    # ROC plot (combine models that have y_score)
    plt.figure(figsize=(8, 6))
    for r in results:
        if r.get('y_score') is not None:
            fpr, tpr, _ = roc_curve(y_test, r['y_score'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{r['Model']} (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('roc_curves.png', bbox_inches='tight', dpi=300)
    plt.close()
    print('Saved: roc_curves.png')

    # Write quick insights draft
    insight_lines = [
        '# Model Evaluation Insights (automated draft)\n',
        '\n',
        'This file is an automated draft generated by `evaluate_and_report.py`. Edit for clarity and context.\n',
        '\n',
        '## Summary table\n',
    ]
    insight_lines.append(results_df.to_markdown(index=False))
    insight_lines.append('\n\n## Quick observations\n')

    best = results_df.iloc[0]
    insight_lines.append(f"- Best model by accuracy: **{best['Model']}** (Accuracy = {best['Accuracy']:.4f}, AUC = {best['AUC']})\n")
    insight_lines.append('- Check the ROC plot (`roc_curves.png`) and confusion matrices (`cm_*.png`) for class-specific errors.\n')
    insight_lines.append('- Consider trade-offs: precision vs recall depending on business needs.\n')
    insight_lines.append('- Next steps: hyperparameter tuning, cross-validation, try regularized models or transformer-based classifiers for improved performance.\n')

    with open('insights.md', 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(insight_lines))
    print('Saved: insights.md')

    print('\nAll done. Artefacts saved: model_comparison.csv, roc_curves.png, cm_*.png, insights.md')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=200000, help='Number of rows to sample for modelling (default 200000)')
    parser.add_argument('--quick', action='store_true', help='Quick mode (smaller models/sample for testing)')
    args = parser.parse_args()

    sample_size = args.sample_size
    if args.quick:
        sample_size = min(10000, sample_size)

    main(sample_size=sample_size, quick=args.quick)
