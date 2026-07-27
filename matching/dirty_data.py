"""
Simulates a second hospital system ("System B") holding the same patients as
dim_patient ("System A"), entered independently with typos/missing fields —
the reason patient matching is a real problem in the first place.

Writes matching/system_b_patients.csv (the noisy records) and
matching/ground_truth.csv (system_b_id -> true patient_id), so the matcher
can actually be scored instead of eyeballed.
"""

import random
import uuid
from pathlib import Path

import pandas as pd

random.seed(42)

NICKNAMES = {
    "robert": ["rob", "bob", "bobby", "robby"],
    "william": ["will", "bill", "billy"],
    "richard": ["rick", "rich", "dick"],
    "james": ["jim", "jimmy"],
    "elizabeth": ["liz", "beth", "eliza"],
    "margaret": ["meg", "maggie", "peggy"],
    "katherine": ["kate", "katie", "kathy"],
    "michael": ["mike", "mikey"],
    "christopher": ["chris"],
    "jennifer": ["jen", "jenny"],
}


def strip_trailing_digits(name: str) -> str:
    # Synthea appends digits to names for uniqueness ("Mila257") — strip
    # before adding noise or every match becomes trivial on the suffix alone.
    return "".join(ch for ch in name if not ch.isdigit())


def noisy_name(name: str) -> str:
    name = strip_trailing_digits(name)
    key = name.lower()
    if key in NICKNAMES and random.random() < 0.5:
        return random.choice(NICKNAMES[key]).capitalize()
    if len(name) > 3 and random.random() < 0.4:
        # single adjacent-letter transposition typo
        i = random.randint(0, len(name) - 2)
        chars = list(name)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
        return "".join(chars)
    return name


def noisy_ssn(ssn: str) -> str | None:
    if random.random() < 0.25:
        return None  # missing SSN in System B — common in practice
    if random.random() < 0.15 and ssn:
        digits = list(ssn)
        idx_candidates = [i for i, c in enumerate(digits) if c.isdigit()]
        if idx_candidates:
            i = random.choice(idx_candidates)
            digits[i] = random.choice("0123456789")
        return "".join(digits)
    return ssn


def noisy_zip(zip_code: str) -> str:
    if pd.isna(zip_code):
        return zip_code
    zip_code = str(zip_code)
    if random.random() < 0.2 and len(zip_code) >= 5:
        return zip_code[:4]  # truncated zip, common CSV round-trip issue
    return zip_code


def noisy_dob(dob: str) -> str:
    if random.random() < 0.03:  # transposed month/day, rare but real
        try:
            year, month, day = dob.split("-")
            if month != day:
                return f"{year}-{day}-{month}"
        except ValueError:
            pass
    return dob


def build_system_b(dim_patient: pd.DataFrame, duplicate_fraction: float = 0.4) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = dim_patient.sample(frac=duplicate_fraction, random_state=42)

    rows = []
    ground_truth = []
    for _, p in sample.iterrows():
        system_b_id = f"B-{uuid.uuid4().hex[:12]}"
        rows.append({
            "system_b_id": system_b_id,
            "first": noisy_name(p["first"]),
            "last": noisy_name(p["last"]),
            "dob": noisy_dob(p["birth_date"]),
            "ssn": noisy_ssn(p["ssn"]),
            "zip": noisy_zip(p["zip"]),
        })
        ground_truth.append({"system_b_id": system_b_id, "true_patient_id": p["patient_id"]})

    return pd.DataFrame(rows), pd.DataFrame(ground_truth)


def main():
    # dim_patient doesn't keep SSN/first/last (identifiers, not demographics),
    # so pull those from the raw Synthea export instead.
    raw_patients = pd.read_csv("output/csv/patients.csv", dtype=str, low_memory=False)
    raw_patients = raw_patients.rename(columns={
        "Id": "patient_id", "BIRTHDATE": "birth_date", "FIRST": "first", "LAST": "last",
        "SSN": "ssn", "ZIP": "zip",
    })[["patient_id", "birth_date", "first", "last", "ssn", "zip"]]

    system_b, ground_truth = build_system_b(raw_patients)

    out_dir = Path(__file__).parent
    system_b.to_csv(out_dir / "system_b_patients.csv", index=False)
    ground_truth.to_csv(out_dir / "ground_truth.csv", index=False)

    print(f"System A (warehouse dim_patient): {len(raw_patients):,} patients")
    print(f"System B (noisy duplicates):      {len(system_b):,} records ({len(system_b)/len(raw_patients):.0%} of System A)")
    print(f"Wrote {out_dir / 'system_b_patients.csv'} and {out_dir / 'ground_truth.csv'}")


if __name__ == "__main__":
    main()
