"""
nlp_classification.py
---------------------
TF-IDF + Logistic Regression text classification of San Rafael 311
service request descriptions into service categories.

Model Design
------------
- Features    : TF-IDF vectors of free-text description field
                (unigrams + bigrams, max 10,000 features)
- Labels      : service request category (20 classes with >= 50 records)
- Algorithm   : Logistic Regression with L2 regularization
                (multi-class via softmax / multinomial)
- Split       : 80% train / 20% stratified test
- Classes     : 20 categories with >= 50 labeled records (98.1% of
                described records; 16 rare categories excluded)

Why TF-IDF + Logistic Regression over alternatives
---------------------------------------------------
- Neural networks (LSTM, BERT) require substantially more data per class
  and GPU infrastructure not available in this deployment environment.
  For n < 3,000 per class, LR typically matches or outperforms small nets.
- Naive Bayes assumes feature independence — problematic for multi-word
  phrases like "storm drain" or "road crack" where co-occurrence matters.
- TF-IDF bigrams capture these phrases naturally; LR coefficients are
  directly interpretable (top weighted terms per class visible).
- This approach is the practical standard for short-text classification
  in government/civic tech contexts (see:311 NYC, Boston City Score).

Business Value
--------------
1. Auto-categorization: New submissions without a selected category
   can be classified automatically, reducing dispatcher workload.
2. Quality control: Flags mismatches between submitted category and
   predicted category — potential data entry errors or category misuse.
3. Trend detection: Apply model to uncategorized or legacy records to
   expand the analyzable dataset.

Evaluation Metrics
------------------
- Accuracy    : Overall fraction correctly classified
- Precision   : Of predicted class X, how many were actually X
- Recall      : Of actual class X, how many did we catch
- F1-score    : Harmonic mean of precision/recall per class
- Macro F1    : Unweighted mean F1 across all classes (penalizes
                poor performance on small classes)
- Weighted F1 : Support-weighted F1 (overall practical performance)
- Confusion matrix highlights systematic misclassifications
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
CLEAN_311     = ROOT / "data" / "processed" / "san_rafael_311_clean.csv"
OUT_DIR       = ROOT / "data" / "processed"
OUT_METRICS   = OUT_DIR / "nlp_metrics.csv"
OUT_REPORT    = OUT_DIR / "nlp_classification_report.csv"
OUT_CONFUSION = OUT_DIR / "nlp_confusion_matrix.csv"
OUT_PREDS     = OUT_DIR / "nlp_predictions.csv"
OUT_MODEL     = ROOT / "models" / "nlp_tfidf_lr_pipeline.joblib"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_CLASS_SIZE  = 50       # exclude categories with fewer records
TEST_SIZE       = 0.20     # 80/20 train/test split
RANDOM_STATE    = 42
MAX_FEATURES    = 10_000   # TF-IDF vocabulary cap
NGRAM_RANGE     = (1, 2)   # unigrams + bigrams
MIN_DF          = 2        # ignore terms appearing in < 2 documents
MAX_DF          = 0.95     # ignore terms in > 95% of documents
CV_FOLDS        = 5        # cross-validation folds


# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------

def load_and_prepare(path: Path = CLEAN_311) -> pd.DataFrame:
    """
    Load cleaned 311 data, filter to records with usable descriptions,
    and restrict to categories with sufficient training data.
    """
    df = pd.read_csv(path, low_memory=False)

    # Keep only records with a meaningful description
    has_desc = (
        df["description"].notna() &
        (df["description"].str.strip().str.len() > 5)
    )
    df = df[has_desc].copy()
    print(f"[NLP] Records with description : {len(df):,}")

    # Use shortened category name
    if "category_short" not in df.columns:
        df["category_short"] = df["category"].apply(
            lambda x: x.split(" / ")[0].strip() if isinstance(x, str) else x
        )

    # Filter to categories with >= MIN_CLASS_SIZE records
    counts = df["category_short"].value_counts()
    viable = counts[counts >= MIN_CLASS_SIZE].index
    df = df[df["category_short"].isin(viable)].copy()

    excluded = counts[counts < MIN_CLASS_SIZE]
    print(f"[NLP] Viable categories        : {len(viable)} "
          f"(excluded {len(excluded)} with < {MIN_CLASS_SIZE} records)")
    print(f"[NLP] Records in viable set    : {len(df):,}")
    print(f"\n[NLP] Class distribution:")
    print(df["category_short"].value_counts().to_string())

    return df


# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------

def preprocess_text(series: pd.Series) -> pd.Series:
    """
    Light text cleaning before TF-IDF vectorization.

    We keep preprocessing minimal intentionally — TF-IDF with sublinear_tf
    and IDF weighting already handles most normalization. Over-cleaning
    (e.g., removing all punctuation) can lose signal like "storm drain" → 2
    separate tokens losing their bigram connection.
    """
    return (
        series
        .str.lower()
        .str.replace(r"http\S+", " ", regex=True)   # strip URLs
        .str.replace(r"[^\w\s]", " ", regex=True)   # punctuation → space
        .str.replace(r"\s+", " ", regex=True)        # collapse whitespace
        .str.strip()
    )


# ---------------------------------------------------------------------------
# 3. Build and train pipeline
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    TF-IDF vectorizer + Logistic Regression as a single sklearn Pipeline.

    TF-IDF parameters:
    - sublinear_tf=True : apply log(1 + tf) to dampen high-frequency terms
    - ngram_range=(1,2) : captures both single words and 2-word phrases
    - max_df=0.95       : removes near-universal terms (acts like stopwords)
    - min_df=2          : removes hapax legomena (appear once — pure noise)

    LR parameters:
    - max_iter=1000     : sufficient for convergence on sparse TF-IDF
    - C=1.0             : standard L2 regularization strength
    - solver='lbfgs'    : efficient for multinomial problems
                          (multi_class parameter removed in sklearn 1.5;
                           lbfgs defaults to multinomial automatically)
    """
    tfidf = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    lr = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    return Pipeline([("tfidf", tfidf), ("clf", lr)])


# ---------------------------------------------------------------------------
# 4. Train, evaluate, and cross-validate
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame) -> tuple:
    """
    Train the pipeline on 80% of data, evaluate on held-out 20%.
    Also runs 5-fold cross-validation on the training set.

    Returns (pipeline, metrics_dict, report_df, confusion_df, pred_df).
    """
    X = preprocess_text(df["description"])
    y = df["category_short"]

    # Stratified split preserves class proportions in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n[NLP] Train size : {len(X_train):,}")
    print(f"[NLP] Test size  : {len(X_test):,}")

    # -- Cross-validation on training set --------------------------------
    print(f"\n[NLP] Running {CV_FOLDS}-fold cross-validation...")
    pipeline = build_pipeline()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=CV_FOLDS, scoring="f1_weighted", n_jobs=-1
        )
    print(f"[NLP] CV Weighted F1 : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # -- Final fit on full training set ----------------------------------
    print(f"\n[NLP] Fitting final model on training set...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline.fit(X_train, y_train)

    # -- Evaluate on held-out test set -----------------------------------
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)

    accuracy  = accuracy_score(y_test, y_pred)
    f1_macro  = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\n[NLP] Test set evaluation:")
    print(f"[NLP]   Accuracy        : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"[NLP]   F1 (macro)      : {f1_macro:.4f}")
    print(f"[NLP]   F1 (weighted)   : {f1_weighted:.4f}")
    print(f"[NLP]   CV F1 (weighted): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Per-class report
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).T.reset_index()
    report_df.columns = ["class", "precision", "recall", "f1_score", "support"]
    report_df = report_df[report_df["class"].isin(
        df["category_short"].unique().tolist() + ["accuracy", "macro avg", "weighted avg"]
    )]

    print(f"\n[NLP] Per-class results (sorted by F1):")
    class_rows = report_df[~report_df["class"].isin(
        ["accuracy", "macro avg", "weighted avg"]
    )].sort_values("f1_score", ascending=False)
    pd.set_option("display.max_colwidth", 45)
    print(class_rows[["class","precision","recall","f1_score","support"]]
          .round(3).to_string(index=False))

    # Confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    confusion_df = pd.DataFrame(cm, index=labels, columns=labels)

    # Top misclassification pairs
    cm_copy = cm.copy()
    np.fill_diagonal(cm_copy, 0)
    flat_idx = np.argsort(cm_copy.flatten())[::-1][:10]
    print(f"\n[NLP] Top misclassification pairs:")
    for idx in flat_idx:
        if cm_copy.flatten()[idx] == 0:
            break
        true_cls  = labels[idx // len(labels)]
        pred_cls  = labels[idx  % len(labels)]
        count     = cm_copy.flatten()[idx]
        print(f"  True: {true_cls[:35]:35s} → Predicted: {pred_cls[:35]:35s}  ({count}x)")

    # Predictions on test set with confidence scores
    max_proba = y_prob.max(axis=1)
    pred_df = pd.DataFrame({
        "description":       X_test.values,
        "true_category":     y_test.values,
        "predicted_category": y_pred,
        "confidence":        max_proba.round(3),
        "correct":           (y_test.values == y_pred).astype(int),
    })

    metrics = {
        "model":             "TF-IDF + Logistic Regression",
        "n_classes":         len(labels),
        "n_train":           len(X_train),
        "n_test":            len(X_test),
        "accuracy":          round(accuracy, 4),
        "f1_macro":          round(f1_macro, 4),
        "f1_weighted":       round(f1_weighted, 4),
        "cv_f1_weighted_mean": round(cv_scores.mean(), 4),
        "cv_f1_weighted_std":  round(cv_scores.std(), 4),
        "tfidf_max_features":  MAX_FEATURES,
        "tfidf_ngram_range":   str(NGRAM_RANGE),
        "lr_C":                1.0,
        "min_class_size":      MIN_CLASS_SIZE,
    }

    return pipeline, metrics, report_df, confusion_df, pred_df


# ---------------------------------------------------------------------------
# 5. Top terms per category (interpretability)
# ---------------------------------------------------------------------------

def top_terms_per_class(pipeline: Pipeline,
                        n_terms: int = 10) -> pd.DataFrame:
    """
    Extract the top TF-IDF weighted terms per class from the LR coefficients.
    These are the terms that most strongly push predictions toward each class.
    Useful for slide explainability and validating the model makes sense.
    """
    tfidf   = pipeline.named_steps["tfidf"]
    clf     = pipeline.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    classes       = clf.classes_

    records = []
    for i, cls in enumerate(classes):
        top_idx = np.argsort(clf.coef_[i])[::-1][:n_terms]
        top_terms = [(feature_names[j], round(clf.coef_[i][j], 4))
                     for j in top_idx]
        records.append({
            "category":  cls,
            "top_terms": " | ".join(f"{t}({w:.3f})" for t, w in top_terms),
        })

    terms_df = pd.DataFrame(records)
    print(f"\n[NLP] Top {n_terms} terms per category:")
    for _, row in terms_df.iterrows():
        print(f"  {row['category'][:40]:40s}: {row['top_terms'][:100]}")

    return terms_df


# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(pipeline:     Pipeline,
                 metrics:      dict,
                 report_df:    pd.DataFrame,
                 confusion_df: pd.DataFrame,
                 pred_df:      pd.DataFrame,
                 terms_df:     pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(OUT_METRICS, index=False)
    print(f"\n[NLP] Metrics saved            → {OUT_METRICS}")

    report_df.to_csv(OUT_REPORT, index=False)
    print(f"[NLP] Classification report    → {OUT_REPORT}")

    confusion_df.to_csv(OUT_CONFUSION)
    print(f"[NLP] Confusion matrix saved   → {OUT_CONFUSION}")

    pred_df.to_csv(OUT_PREDS, index=False)
    print(f"[NLP] Test predictions saved   → {OUT_PREDS}")

    terms_path = OUT_DIR / "nlp_top_terms.csv"
    terms_df.to_csv(terms_path, index=False)
    print(f"[NLP] Top terms saved          → {terms_path}")

    joblib.dump(pipeline, OUT_MODEL)
    print(f"[NLP] Model saved              → {OUT_MODEL}")


# ---------------------------------------------------------------------------
# 7. Full pipeline runner
# ---------------------------------------------------------------------------

def run(data_path: Path = CLEAN_311) -> tuple:
    """
    Execute full NLP classification pipeline.
    Returns (pipeline, metrics, report_df).
    """
    print("=" * 60)
    print("TF-IDF + Logistic Regression — 311 Category Classification")
    print("=" * 60)

    df                                         = load_and_prepare(data_path)
    pipeline, metrics, report_df, confusion_df, pred_df = train_and_evaluate(df)
    terms_df                                   = top_terms_per_class(pipeline)
    save_outputs(pipeline, metrics, report_df, confusion_df, pred_df, terms_df)

    print("\n✓ NLP classification pipeline complete.")
    return pipeline, metrics, report_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Run TF-IDF + LR 311 category classification."
    )
    parser.add_argument("--input", type=Path, default=CLEAN_311)
    args = parser.parse_args()
    run(data_path=args.input)
