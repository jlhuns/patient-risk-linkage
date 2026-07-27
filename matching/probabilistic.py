"""
Probabilistic matcher: logistic regression over similarity features, weights
learned from labeled pairs instead of hand-set — basically Fellegi-Sunter
but with the weights fit by regression instead of EM.

    python matching/probabilistic.py
"""

from pathlib import Path

import jellyfish
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from matching.deterministic import load_system_a, load_system_b


FEATURE_COLUMNS = ["first_sim", "last_sim", "zip_sim", "dob_year_match", "ssn_agree", "ssn_both_present"]


def zip3(z) -> str:
    z = str(z) if pd.notna(z) else ""
    return z[:3]


def birth_year(dob) -> str:
    return str(dob)[:4] if pd.notna(dob) else ""


def build_blocks(system_a: pd.DataFrame) -> dict[tuple, list[dict]]:
    blocks: dict[tuple, list[dict]] = {}
    for record in system_a.to_dict("records"):
        key = (birth_year(record["dob"]), zip3(record["zip"]))
        blocks.setdefault(key, []).append(record)
    return blocks


def extract_features(a: dict, b: dict) -> dict:
    a_ssn, b_ssn = a.get("ssn"), b.get("ssn")
    both_present = isinstance(a_ssn, str) and isinstance(b_ssn, str) and a_ssn and b_ssn
    return {
        "first_sim": jellyfish.jaro_winkler_similarity(str(a["first"]).lower(), str(b["first"]).lower()),
        "last_sim": jellyfish.jaro_winkler_similarity(str(a["last"]).lower(), str(b["last"]).lower()),
        "zip_sim": jellyfish.jaro_winkler_similarity(str(a["zip"]), str(b["zip"])),
        "dob_year_match": 1.0 if birth_year(a["dob"]) == birth_year(b["dob"]) else 0.0,
        "ssn_agree": 1.0 if (both_present and a_ssn == b_ssn) else 0.0,
        "ssn_both_present": 1.0 if both_present else 0.0,
    }


def build_candidate_pairs(system_a: pd.DataFrame, system_b: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    blocks = build_blocks(system_a)
    truth_map = dict(zip(ground_truth["system_b_id"], ground_truth["true_patient_id"]))

    rows = []
    for b in system_b.to_dict("records"):
        key = (birth_year(b["dob"]), zip3(b["zip"]))
        for a in blocks.get(key, []):
            features = extract_features(a, b)
            is_match = 1 if truth_map.get(b["system_b_id"]) == a["patient_id"] else 0
            rows.append({
                "system_b_id": b["system_b_id"],
                "patient_id": a["patient_id"],
                "is_match": is_match,
                **features,
            })
    return pd.DataFrame(rows)


def train_and_evaluate(pairs: pd.DataFrame):
    # Group split on system_b_id so all candidate pairs for one B-record
    # stay together — otherwise the same record's pairs could leak across
    # train/test and inflate the reported score.
    splitter = GroupShuffleSplit(test_size=0.3, random_state=42)
    train_idx, test_idx = next(splitter.split(pairs, groups=pairs["system_b_id"]))
    train, test = pairs.iloc[train_idx], pairs.iloc[test_idx]

    model = LogisticRegression(class_weight="balanced")
    model.fit(train[FEATURE_COLUMNS], train["is_match"])

    test_scores = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    test_pred = (test_scores >= 0.5).astype(int)

    metrics = {
        "candidate_pairs_total": len(pairs),
        "candidate_pairs_test": len(test),
        "test_precision": round(precision_score(test["is_match"], test_pred), 4),
        "test_recall": round(recall_score(test["is_match"], test_pred), 4),
        "test_f1": round(f1_score(test["is_match"], test_pred), 4),
        "learned_weights": dict(zip(FEATURE_COLUMNS, model.coef_[0].round(3))),
    }
    return model, metrics


def predict_best_match_per_record(model, pairs: pd.DataFrame) -> pd.DataFrame:
    scored = pairs.copy()
    scored["score"] = model.predict_proba(scored[FEATURE_COLUMNS])[:, 1]
    best = scored.sort_values("score", ascending=False).drop_duplicates("system_b_id")
    return best[best["score"] >= 0.5][["system_b_id", "patient_id", "score"]].rename(
        columns={"patient_id": "matched_patient_id"}
    )


def main():
    system_a = load_system_a()
    system_b = load_system_b()
    ground_truth = pd.read_csv(Path(__file__).parent / "ground_truth.csv", dtype=str)

    print("Building candidate pairs (blocked by birth year + zip3) ...")
    pairs = build_candidate_pairs(system_a, system_b, ground_truth)
    print(f"  {len(pairs):,} candidate pairs from {len(system_b):,} System B records")
    print(f"  {pairs['is_match'].sum():,} true matches present among candidates "
          f"({pairs['is_match'].sum() / len(system_b):.1%} of all System B records had their true match in-block)")

    model, metrics = train_and_evaluate(pairs)
    print("\nProbabilistic matcher (held-out test set):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    full_matches = predict_best_match_per_record(model, pairs)
    correct = full_matches.merge(ground_truth, on="system_b_id")
    correct = correct[correct["matched_patient_id"] == correct["true_patient_id"]]
    print(f"\nEnd-to-end (all {len(system_b):,} System B records, best-match-per-record):")
    print(f"  matched: {len(full_matches):,}")
    print(f"  correct: {len(correct):,}")
    print(f"  precision: {len(correct) / len(full_matches):.4f}")
    print(f"  recall:    {len(correct) / len(system_b):.4f}")

    joblib.dump(model, Path(__file__).parent / "probabilistic_model.joblib")
    print(f"\nSaved {Path(__file__).parent / 'probabilistic_model.joblib'}")


if __name__ == "__main__":
    main()
