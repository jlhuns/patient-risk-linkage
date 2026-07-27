"""
Deterministic matcher: block on DOB, then apply fixed rules within each block.

    python matching/deterministic.py
"""

from pathlib import Path

import pandas as pd


def load_system_a() -> pd.DataFrame:
    raw = pd.read_csv("output/csv/patients.csv", dtype=str, low_memory=False)
    raw = raw.rename(columns={
        "Id": "patient_id", "BIRTHDATE": "dob", "FIRST": "first", "LAST": "last",
        "SSN": "ssn", "ZIP": "zip",
    })
    return raw[["patient_id", "dob", "first", "last", "ssn", "zip"]]


def load_system_b() -> pd.DataFrame:
    return pd.read_csv(Path(__file__).parent / "system_b_patients.csv", dtype=str)


def build_blocks(df: pd.DataFrame, id_col: str) -> dict[str, list[dict]]:
    blocks: dict[str, list[dict]] = {}
    for record in df.to_dict("records"):
        blocks.setdefault(record["dob"], []).append(record)
    return blocks


def is_match(a: dict, b: dict) -> bool:
    # SSN match wins if both sides have one. If both present but disagree,
    # that's a hard no — don't fall through to the weaker rule below.
    if a["ssn"] and b["ssn"] and isinstance(a["ssn"], str) and isinstance(b["ssn"], str):
        return a["ssn"] == b["ssn"]

    # no usable SSN: last name exact + first-name prefix + zip
    same_last = str(a["last"]).lower() == str(b["last"]).lower()
    first_prefix = str(a["first"])[:3].lower() == str(b["first"])[:3].lower()
    same_zip = str(a["zip"])[:5] == str(b["zip"])[:5]
    return same_last and first_prefix and same_zip


def match(system_a: pd.DataFrame, system_b: pd.DataFrame) -> pd.DataFrame:
    blocks_a = build_blocks(system_a, "patient_id")
    matches = []
    for _, b in system_b.iterrows():
        b_dict = b.to_dict()
        candidates = blocks_a.get(b_dict["dob"], [])
        for a in candidates:
            if is_match(a, b_dict):
                matches.append({"system_b_id": b_dict["system_b_id"], "matched_patient_id": a["patient_id"]})
                break  # deterministic rules assume at most one true match per block
    return pd.DataFrame(matches)


def evaluate(predicted: pd.DataFrame, ground_truth: pd.DataFrame, total_system_b: int) -> dict:
    merged = predicted.merge(ground_truth, on="system_b_id", how="left")
    true_positives = (merged["matched_patient_id"] == merged["true_patient_id"]).sum()
    false_positives = len(merged) - true_positives
    false_negatives = total_system_b - len(predicted)  # records System B had, but we found no match for

    precision = true_positives / len(merged) if len(merged) else 0.0
    recall = true_positives / total_system_b if total_system_b else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "total_system_b_records": total_system_b,
        "predicted_matches": len(predicted),
        "true_positives": int(true_positives),
        "false_positives": int(false_positives),
        "false_negatives_unmatched": int(false_negatives),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def main():
    system_a = load_system_a()
    system_b = load_system_b()
    ground_truth = pd.read_csv(Path(__file__).parent / "ground_truth.csv", dtype=str)

    predicted = match(system_a, system_b)
    results = evaluate(predicted, ground_truth, total_system_b=len(system_b))

    print("Deterministic matching results:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
