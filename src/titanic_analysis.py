import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

import matplotlib.pyplot as plt
import seaborn as sns


RANDOM_SEED = 42
TEST_SIZE = 0.2


@dataclass
class DatasetBundle:
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def extract_title(name: str) -> str:
    if not isinstance(name, str):
        return "Unknown"
    # Example: "Braund, Mr. Owen Harris" -> "Mr"
    parts = name.split(",")
    if len(parts) < 2:
        return "Unknown"
    after_comma = parts[1]
    if "." not in after_comma:
        return "Unknown"
    title = after_comma.split(".")[0].strip()
    return title


def normalize_titles(series: pd.Series) -> pd.Series:
    title_map = {
        "Mr": "Mr",
        "Miss": "Miss",
        "Mrs": "Mrs",
        "Master": "Master",
        "Dr": "Rare",
        "Rev": "Rare",
        "Col": "Rare",
        "Major": "Rare",
        "Mlle": "Miss",
        "Ms": "Miss",
        "Mme": "Mrs",
        "Sir": "Rare",
        "Lady": "Rare",
        "Don": "Rare",
        "Jonkheer": "Rare",
        "Capt": "Rare",
        "Countess": "Rare",
    }
    return series.map(lambda x: title_map.get(x, "Rare"))


def minimal_prep(df: pd.DataFrame) -> DatasetBundle:
    df_min = df.dropna().copy()

    # Basic encodings
    df_min["Sex"] = df_min["Sex"].map({"male": 0, "female": 1})
    df_min["Embarked"] = df_min["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df_min = df_min.drop(columns=[c for c in drop_cols if c in df_min.columns])

    y = df_min["Survived"].astype(int)
    X = df_min.drop(columns=["Survived"])

    meta = pd.DataFrame({
        "Sex": df_min["Sex"],
        "Pclass": df_min["Pclass"],
    })

    return DatasetBundle(X=X, y=y, meta=meta)


def full_prep(df: pd.DataFrame) -> DatasetBundle:
    df_full = df.copy()

    # Embarked: mode imputation
    if df_full["Embarked"].isna().any():
        df_full["Embarked"] = df_full["Embarked"].fillna(df_full["Embarked"].mode()[0])

    # Age: class-stratified median imputation
    df_full["Age"] = df_full.groupby("Pclass")["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    # Cabin features
    df_full["Deck"] = df_full["Cabin"].str[0].fillna("Unknown")
    df_full["Has_Cabin"] = df_full["Cabin"].notna().astype(int)

    # Title extraction
    df_full["Title"] = df_full["Name"].apply(extract_title)
    df_full["Title"] = normalize_titles(df_full["Title"])

    # Family features
    df_full["FamilySize"] = df_full["SibSp"] + df_full["Parch"] + 1
    df_full["IsAlone"] = (df_full["FamilySize"] == 1).astype(int)

    # Age groups (for interpretability)
    df_full["AgeGroup"] = pd.cut(
        df_full["Age"],
        bins=[0, 12, 18, 35, 60, 100],
        labels=["Child", "Teen", "Adult", "Middle", "Senior"],
    )

    # Fare log transform
    df_full["Fare_Log"] = np.log1p(df_full["Fare"])

    # Drop unused text columns
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    df_full = df_full.drop(columns=[c for c in drop_cols if c in df_full.columns])

    y = df_full["Survived"].astype(int)
    X = df_full.drop(columns=["Survived"])

    # One-hot encode non-ordinal categoricals
    X = pd.get_dummies(
        X,
        columns=["Sex", "Embarked", "Title", "Deck", "AgeGroup"],
        drop_first=False,
    )

    meta = pd.DataFrame({
        "Sex": df_full["Sex"],
        "Pclass": df_full["Pclass"],
    })

    return DatasetBundle(X=X, y=y, meta=meta)


def train_and_eval(
    bundle: DatasetBundle,
    model_name: str,
) -> Dict[str, float]:
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        bundle.X,
        bundle.y,
        bundle.meta,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=bundle.y,
    )

    if model_name == "logreg":
        model = LogisticRegression(max_iter=1000, solver="liblinear")
    elif model_name == "tree":
        model = DecisionTreeClassifier(
            random_state=RANDOM_SEED,
            max_depth=5,
            min_samples_leaf=10,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    # ROC-AUC if possible
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    except Exception:
        metrics["roc_auc"] = np.nan

    # Fairness metrics by sex
    sex_values = meta_test["Sex"].values
    male_mask = sex_values == "male" if sex_values.dtype == object else sex_values == 0
    female_mask = sex_values == "female" if sex_values.dtype == object else sex_values == 1

    if female_mask.any():
        metrics["acc_female"] = accuracy_score(y_test[female_mask], y_pred[female_mask])
    else:
        metrics["acc_female"] = np.nan

    if male_mask.any():
        metrics["acc_male"] = accuracy_score(y_test[male_mask], y_pred[male_mask])
    else:
        metrics["acc_male"] = np.nan

    # Accuracy by class
    for pclass in [1, 2, 3]:
        class_mask = meta_test["Pclass"] == pclass
        if class_mask.any():
            metrics[f"acc_class_{pclass}"] = accuracy_score(
                y_test[class_mask], y_pred[class_mask]
            )
        else:
            metrics[f"acc_class_{pclass}"] = np.nan

    # Disparate impact ratio (predicted positive rate)
    if female_mask.any() and male_mask.any():
        female_rate = y_pred[female_mask].mean()
        male_rate = y_pred[male_mask].mean()
        denom = max(female_rate, male_rate)
        metrics["disparate_impact"] = (
            min(female_rate, male_rate) / denom if denom > 0 else np.nan
        )
    else:
        metrics["disparate_impact"] = np.nan

    return metrics


def save_metrics(results: Dict[str, Dict[str, float]], out_path: str) -> None:
    df = pd.DataFrame(results).T
    df.to_csv(out_path, index=True)

def save_feature_impacts_logreg(model: LogisticRegression, columns: pd.Index, out_path: str) -> None:
    coefs = pd.Series(model.coef_[0], index=columns).sort_values(ascending=False)
    df = pd.DataFrame({
        "feature": coefs.index,
        "coef": coefs.values,
    })
    df.to_csv(out_path, index=False)


def save_feature_importance_tree(model: DecisionTreeClassifier, columns: pd.Index, out_path: str) -> None:
    importances = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    df = pd.DataFrame({
        "feature": importances.index,
        "importance": importances.values,
    })
    df.to_csv(out_path, index=False)


def plot_basics(df: pd.DataFrame, fig_dir: str) -> None:
    os.makedirs(fig_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "survival_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Sex", hue="Survived", data=df)
    plt.title("Survival by Sex")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "survival_by_sex.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(x="Pclass", hue="Survived", data=df)
    plt.title("Survival by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "survival_by_class.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df["Age"].dropna(), kde=True, bins=30)
    plt.title("Age Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "age_distribution.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.histplot(df["Fare"].dropna(), kde=True, bins=30)
    plt.title("Fare Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fare_distribution.png"), dpi=150)
    plt.close()

    # Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "Titanic-Dataset.csv")

    df = load_data(csv_path)

    # Basic plots from raw data
    plot_basics(df, os.path.join(base_dir, "figures"))

    results = {}

    minimal_bundle = minimal_prep(df)
    full_bundle = full_prep(df)

    results["minimal_logreg"] = train_and_eval(minimal_bundle, "logreg")
    results["full_logreg"] = train_and_eval(full_bundle, "logreg")
    results["minimal_tree"] = train_and_eval(minimal_bundle, "tree")
    results["full_tree"] = train_and_eval(full_bundle, "tree")

    out_dir = os.path.join(base_dir, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    save_metrics(results, os.path.join(out_dir, "model_metrics.csv"))

    # Feature impact outputs (full prep)
    # Train once on full data for interpretability exports
    X_train, X_test, y_train, y_test = train_test_split(
        full_bundle.X,
        full_bundle.y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=full_bundle.y,
    )

    logreg = LogisticRegression(max_iter=1000, solver="liblinear")
    logreg.fit(X_train, y_train)
    save_feature_impacts_logreg(
        logreg,
        full_bundle.X.columns,
        os.path.join(out_dir, "logreg_coefficients.csv"),
    )

    tree = DecisionTreeClassifier(
        random_state=RANDOM_SEED,
        max_depth=5,
        min_samples_leaf=10,
    )
    tree.fit(X_train, y_train)
    save_feature_importance_tree(
        tree,
        full_bundle.X.columns,
        os.path.join(out_dir, "tree_feature_importance.csv"),
    )

    print("Analysis complete.")
    print("Saved metrics to outputs/model_metrics.csv")
    print("Saved feature impacts to outputs/logreg_coefficients.csv and outputs/tree_feature_importance.csv")
    print("Saved figures to figures/")


if __name__ == "__main__":
    main()
