"""
Trains and compares 4 models for 30-day readmission risk: logistic
regression, random forest, gradient-boosted trees, and a small Keras MLP.

Readmitted is a minority class (~15-20%), so accuracy is a bad metric here
(always predicting "no" scores ~80%+ while being useless) — everything's
evaluated on precision/recall/PR-AUC with class-weighted training instead.
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.features import build_training_frame

warnings.filterwarnings("ignore", category=UserWarning)

NUMERIC_FEATURES = [
    "prior_encounter_count", "prior_inpatient_count", "prior_condition_count",
    "prior_procedure_count", "prior_medication_count", "chronic_condition_count",
    "bmi", "systolic_bp", "diastolic_bp", "heart_rate", "glucose", "hba1c",
    "age_at_encounter", "base_cost", "total_claim_cost",
]
CATEGORICAL_FEATURES = ["gender", "race", "ethnicity"]
TARGET = "readmitted_30d"


def make_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def split_by_patient(df: pd.DataFrame):
    splitter = GroupShuffleSplit(test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_id"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def evaluate(name: str, y_true, y_proba) -> dict:
    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "model": name,
        "precision": round(report["1"]["precision"], 3),
        "recall": round(report["1"]["recall"], 3),
        "f1": round(report["1"]["f1-score"], 3),
        "pr_auc": round(average_precision_score(y_true, y_proba), 3),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 3),
    }


def train_sklearn_models(preprocessor, X_train, y_train, X_test, y_test) -> list[dict]:
    results = []
    models = {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosted_trees": HistGradientBoostingClassifier(random_state=42),
    }

    fitted = {}
    for name, model in models.items():
        pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        if name == "gradient_boosted_trees":
            # no class_weight param on HistGradientBoostingClassifier, so
            # compute balanced sample weights by hand instead
            counts = y_train.value_counts()
            weight_map = {cls: len(y_train) / (2 * n) for cls, n in counts.items()}
            sample_weight = y_train.map(weight_map)
            pipeline.fit(X_train, y_train, model__sample_weight=sample_weight)
        else:
            pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        results.append(evaluate(name, y_test, proba))
        fitted[name] = pipeline

    return results, fitted


def train_keras_mlp(preprocessor, X_train, y_train, X_test, y_test):
    import tensorflow as tf

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    if hasattr(X_train_t, "toarray"):
        X_train_t, X_test_t = X_train_t.toarray(), X_test_t.toarray()

    counts = y_train.value_counts()
    class_weight = {cls: len(y_train) / (2 * n) for cls, n in counts.items()}

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_t.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.AUC(name="pr_auc", curve="PR")])
    model.fit(
        X_train_t, y_train, validation_split=0.15, epochs=30, batch_size=64,
        class_weight=class_weight, verbose=0,
    )
    proba = model.predict(X_test_t, verbose=0).ravel()
    result = evaluate("neural_network_mlp", y_test, proba)
    return result, model, preprocessor  # preprocessor is fitted in here, needed later for inference


def hypothesis_test(df: pd.DataFrame):
    # Mann-Whitney U, not a t-test — prior_inpatient_count is a small,
    # right-skewed integer count, not normally distributed.
    readmitted = df.loc[df[TARGET] == 1, "prior_inpatient_count"]
    not_readmitted = df.loc[df[TARGET] == 0, "prior_inpatient_count"]
    u_stat, p_value = stats.mannwhitneyu(readmitted, not_readmitted, alternative="greater")
    print("\nHypothesis test: do readmitted patients have more prior inpatient stays?")
    print(f"  H0: no difference in prior_inpatient_count distribution between groups")
    print(f"  Mann-Whitney U = {u_stat:.1f}, p = {p_value:.6f}")
    print(f"  {'Reject H0' if p_value < 0.05 else 'Fail to reject H0'} at alpha=0.05")
    print(f"  Median prior inpatient stays: readmitted={readmitted.median():.1f}, not readmitted={not_readmitted.median():.1f}")


def main():
    print("Building training frame from warehouse ...")
    df = build_training_frame()
    print(f"Cohort: {len(df):,} inpatient encounters, {df[TARGET].mean():.1%} readmitted within 30 days\n")

    hypothesis_test(df)

    train_df, test_df = split_by_patient(df)
    X_train, y_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train_df[TARGET]
    X_test, y_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test_df[TARGET]

    results, fitted = train_sklearn_models(make_preprocessor(), X_train, y_train, X_test, y_test)
    nn_result, nn_model, nn_preprocessor = train_keras_mlp(make_preprocessor(), X_train, y_train, X_test, y_test)
    results.append(nn_result)

    print("\nModel comparison (held-out test set, group-split by patient):")
    results_df = pd.DataFrame(results).sort_values("pr_auc", ascending=False)
    print(results_df.to_string(index=False))

    # save all 4, not just the PR-AUC winner — production model is a separate
    # call (see README): logistic regression, not the NN, for interpretability
    # and to avoid a TensorFlow runtime in the serving container
    for name, pipeline in fitted.items():
        joblib.dump(pipeline, f"models/{name}.joblib")
        print(f"Saved models/{name}.joblib")

    nn_model.save("models/neural_network_mlp.keras")
    joblib.dump(nn_preprocessor, "models/neural_network_mlp_preprocessor.joblib")
    print("Saved models/neural_network_mlp.keras + models/neural_network_mlp_preprocessor.joblib")

    results_df.to_csv("models/model_comparison.csv", index=False)
    print("Saved models/model_comparison.csv")


if __name__ == "__main__":
    main()
