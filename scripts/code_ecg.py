import os
import re
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

MIMIC_ROOT = "/home/asatsan2/Projects/ECG/mimic_ecg/p1000"

def extract_header_text(hea_path):
    with open(hea_path, "r") as f:
        raw = f.read()

    lines = []
    for line in raw.splitlines():
        if line.startswith("#"):
            lines.append(line.lstrip("#").strip())

    text = " ".join(lines).strip()
    if not text:
        text = raw.strip()
    return text

def map_mimic_label_from_text(text):
    t = str(text).lower()

    if any(x in t for x in ["normal ecg", "normal sinus rhythm", "sinus rhythm", "within normal limits"]):
        return "NORM"
    if any(x in t for x in ["myocardial infarction", "infarct", "stemi", "nstemi", "anterior infarct", "inferior infarct"]):
        return "MI"
    if any(x in t for x in ["atrial fibrillation", "afib", "a-fib", "atrial fib"]):
        return "AFIB"
    if any(x in t for x in ["sinus tachycardia", "tachycardia", "svt", "supraventricular tachycardia"]):
        return "TACHY"
    if any(x in t for x in ["sinus bradycardia", "bradycardia"]):
        return "BRADY"

    return "OTHER"

def build_mimic_metadata(root_dir, max_samples=200, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    hea_files = sorted(glob.glob(os.path.join(root_dir, "**", "*.hea"), recursive=True))
    print("Found", len(hea_files), ".hea files")

    rows = []
    for hea_path in tqdm(hea_files):
        try:
            record_path = hea_path[:-4]
            report = extract_header_text(hea_path)

            patient_match = re.search(r"/(p\d+)/", hea_path)
            patient_id = patient_match.group(1) if patient_match else "unknown"

            study_match = re.search(r"/(s\d+)/", hea_path)
            study_id = study_match.group(1) if study_match else os.path.basename(record_path)

            label = map_mimic_label_from_text(report)

            rows.append({
                "patient_id": patient_id,
                "study_id": study_id,
                "filename_lr": record_path,
                "report": report,
                "chosen_label": label
            })
        except Exception:
            continue

    meta = pd.DataFrame(rows).drop_duplicates(subset=["filename_lr"]).reset_index(drop=True)

    print("\nRaw label counts:")
    print(meta["chosen_label"].value_counts())

    # optionally remove OTHER if too noisy
    vc = meta["chosen_label"].value_counts()
    keep_labels = vc[vc >= 5].index.tolist()
    meta = meta[meta["chosen_label"].isin(keep_labels)].reset_index(drop=True)

    if len(meta) > max_samples:
        meta = meta.groupby("chosen_label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples // max(1, meta["chosen_label"].nunique())), random_state=seed)
        ).reset_index(drop=True)

        if len(meta) > max_samples:
            meta = meta.sample(max_samples, random_state=seed).reset_index(drop=True)

    # study-level split is safer for now
    idx = np.arange(len(meta))
    np.random.shuffle(idx)

    n = len(idx)
    train_idx = set(idx[:int(0.7 * n)])
    val_idx = set(idx[int(0.7 * n):int(0.85 * n)])
    test_idx = set(idx[int(0.85 * n):])

    meta["split"] = [
        "train" if i in train_idx else ("val" if i in val_idx else "test")
        for i in range(len(meta))
    ]

    label_names = sorted(meta["chosen_label"].unique())
    label_map = {lab: i for i, lab in enumerate(label_names)}

    print("\nFinal size:", len(meta))
    print("\nFinal label counts:")
    print(meta["chosen_label"].value_counts())
    print("\nSplit counts:")
    print(meta["split"].value_counts())
    print("\nLabel map:", label_map)

    return meta, label_map

mimic_meta, mimic_label_map = build_mimic_metadata(MIMIC_ROOT, max_samples=200)
print(mimic_meta.head())