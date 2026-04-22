
import os
import re
import gc
import ast
import glob
import json
import math
import random
import warnings
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import wfdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================================
# UTIL
# ============================================================================
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # dataset
    DATASET: str = "mimic"  # mimic | ptbxl
    DATA_DIR: str = "/home/asatsan2/Projects/ECG/mimic_ecg"
    CSV_META: Optional[str] = None

    # model
    TOPK: int = 5
    ECG_CHANNELS: int = 12
    ECG_SEQ_LEN: int = 2500
    TEXT_MAX_LEN: int = 64
    ECG_EMBED_DIM: int = 128
    FUSION_DIM: int = 128

    # train
    BATCH_SIZE: int = 16
    GRADIENT_ACCUM: int = 2
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 5
    WEIGHT_DECAY: float = 1e-5
    MAX_GRAD_NORM: float = 1.0

    # runtime
    USE_MIXED_PRECISION: bool = True
    CLEAR_CACHE_EVERY_N_BATCHES: int = 20
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42

    # mimic only
    MIMIC_MAX_SAMPLES: int = 500
    MIMIC_KEEP_LABELS: Optional[List[str]] = None

    def __post_init__(self):
        if self.CSV_META is None and self.DATASET == "ptbxl":
            self.CSV_META = os.path.join(self.DATA_DIR, "ptbxl_database.csv")
        if self.MIMIC_KEEP_LABELS is None:
            self.MIMIC_KEEP_LABELS = ["NORM", "TACHY", "BRADY", "OTHER", "AFIB"]


# ============================================================================
# PTB-XL DATA
# ============================================================================
def load_ptbxl_metadata(config: Config):
    print("\nLoading PTB-XL metadata...")
    meta = pd.read_csv(config.CSV_META)

    keep_cols = ["patient_id", "filename_lr", "scp_codes"]
    if "report" in meta.columns:
        keep_cols.append("report")
    meta = meta[keep_cols].copy()

    if "report" not in meta.columns:
        meta["report"] = meta["scp_codes"].apply(
            lambda x: f"ECG diagnosis: {x}" if pd.notna(x) else "Normal ECG"
        )

    all_codes = {}
    for scp_str in meta["scp_codes"]:
        if pd.isna(scp_str):
            continue
        try:
            scp_dict = ast.literal_eval(scp_str)
            for code in scp_dict.keys():
                all_codes[code] = all_codes.get(code, 0) + 1
        except Exception:
            continue

    top_codes = sorted(all_codes.items(), key=lambda x: -x[1])[: config.TOPK]
    label_map = {k: i for i, (k, _) in enumerate(top_codes)}

    def get_label(s):
        if pd.isna(s):
            return None
        try:
            d = ast.literal_eval(s)
            for k in label_map:
                if k in d:
                    return k
        except Exception:
            pass
        return None

    meta["chosen_label"] = meta["scp_codes"].apply(get_label)
    meta = meta[meta["chosen_label"].notnull()].reset_index(drop=True)

    unique_pat = meta["patient_id"].unique()
    np.random.shuffle(unique_pat)
    n = len(unique_pat)

    train_p = set(unique_pat[: int(0.7 * n)])
    val_p = set(unique_pat[int(0.7 * n) : int(0.85 * n)])
    test_p = set(unique_pat[int(0.85 * n) :])

    meta["split"] = meta["patient_id"].apply(
        lambda x: "train" if x in train_p else ("val" if x in val_p else "test")
    )

    print(f"PTB-XL records: {len(meta)}")
    print(meta["chosen_label"].value_counts())
    print(meta["split"].value_counts())
    print(label_map)
    return meta, label_map


# ============================================================================
# MIMIC DATA
# ============================================================================
def normalize_record_path(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"^\./", "", x)
    x = re.sub(r"^/", "", x)
    x = re.sub(r"^files/", "", x)
    x = re.sub(r"\.hea$|\.dat$", "", x)
    return x


def build_mimic_metadata_from_csv(config: Config):
    mimic_root = config.DATA_DIR
    record_csv = os.path.join(mimic_root, "record_list.csv")
    mm_csv = os.path.join(mimic_root, "machine_measurements.csv")

    record_df = pd.read_csv(record_csv)
    mm_df = pd.read_csv(mm_csv, low_memory=False)

    print("record_list shape:", record_df.shape)
    print("machine_measurements shape:", mm_df.shape)

    hea_files = sorted(glob.glob(os.path.join(mimic_root, "p*", "**", "*.hea"), recursive=True))
    local_record_paths = [hp[:-4] for hp in hea_files]
    local_rel_paths = [os.path.relpath(p, mimic_root) for p in local_record_paths]
    local_rel_paths = [normalize_record_path(x) for x in local_rel_paths]
    local_set = set(local_rel_paths)

    print("Local downloaded .hea files:", len(hea_files))
    if len(hea_files) == 0:
        raise ValueError(f"No .hea files found under {mimic_root}")

    candidate_cols = ["path", "record_path", "file_name", "wfdb_path"]
    path_col = None
    for c in candidate_cols:
        if c in record_df.columns:
            path_col = c
            break
    if path_col is None:
        raise ValueError(f"Could not find path column in record_list.csv. Columns: {record_df.columns.tolist()}")

    record_df[path_col] = record_df[path_col].astype(str).apply(normalize_record_path)
    record_df = record_df[record_df[path_col].isin(local_set)].copy()

    print("Matched locally available records:", len(record_df))
    if len(record_df) == 0:
        raise ValueError("Still 0 matched records after normalization.")

    meta = record_df.merge(mm_df, on=["subject_id", "study_id"], how="left")
    print("Merged shape:", meta.shape)

    report_cols = [c for c in meta.columns if re.fullmatch(r"report_\d+", str(c))]
    report_cols = sorted(report_cols, key=lambda x: int(x.split("_")[1]))
    if len(report_cols) == 0:
        raise ValueError("No report_# columns found in machine_measurements.csv")

    def build_machine_text(row):
        parts = []
        for c in report_cols:
            val = row.get(c, None)
            if pd.notna(val):
                sval = str(val).strip()
                if sval and sval.lower() != "nan":
                    parts.append(sval)
        return " ".join(parts).strip()

    meta["report"] = meta.apply(build_machine_text, axis=1)
    meta["report"] = meta["report"].fillna("").astype(str).str.strip()
    meta = meta[meta["report"] != ""].copy()

    print("Rows with usable machine text:", len(meta))
    print("Sample reports:", meta["report"].head(3).tolist())

    def map_label(text):
        t = str(text).lower()
        if any(x in t for x in ["normal sinus rhythm", "normal ecg", "sinus rhythm", "within normal limits"]):
            return "NORM"
        if any(x in t for x in ["myocardial infarction", "infarct", "stemi", "nstemi", "anterior infarct", "inferior infarct"]):
            return "MI"
        if any(x in t for x in ["atrial fibrillation", "afib", "a-fib", "atrial fib"]):
            return "AFIB"
        if any(x in t for x in ["sinus tachycardia", "tachycardia", "supraventricular tachycardia", "svt"]):
            return "TACHY"
        if any(x in t for x in ["sinus bradycardia", "bradycardia"]):
            return "BRADY"
        return "OTHER"

    meta["chosen_label"] = meta["report"].apply(map_label)
    print("\nRaw label counts:")
    print(meta["chosen_label"].value_counts())

    keep = config.MIMIC_KEEP_LABELS
    meta = meta[meta["chosen_label"].isin(keep)].copy()
    meta["filename_lr"] = meta[path_col].apply(lambda x: os.path.join(mimic_root, x))

    # balanced subset
    if len(meta) > config.MIMIC_MAX_SAMPLES:
        n_classes = meta["chosen_label"].nunique()
        per_class = max(1, config.MIMIC_MAX_SAMPLES // n_classes)
        chunks = []
        for lab in sorted(meta["chosen_label"].unique()):
            part = meta[meta["chosen_label"] == lab]
            chunks.append(part.sample(min(len(part), per_class), random_state=config.SEED))
        meta = pd.concat(chunks).sample(frac=1, random_state=config.SEED).reset_index(drop=True)
        if len(meta) > config.MIMIC_MAX_SAMPLES:
            meta = meta.sample(config.MIMIC_MAX_SAMPLES, random_state=config.SEED).reset_index(drop=True)

    idx = np.arange(len(meta))
    np.random.shuffle(idx)
    n = len(idx)
    train_idx = set(idx[: int(0.7 * n)])
    val_idx = set(idx[int(0.7 * n) : int(0.85 * n)])
    test_idx = set(idx[int(0.85 * n) :])

    meta["split"] = [
        "train" if i in train_idx else ("val" if i in val_idx else "test")
        for i in range(len(meta))
    ]

    label_names = sorted(meta["chosen_label"].unique())
    label_map = {lab: i for i, lab in enumerate(label_names)}

    out_cols = ["subject_id", "study_id", "filename_lr", "report", "chosen_label", "split"]
    meta = meta[out_cols].reset_index(drop=True)

    print("\nFinal size:", len(meta))
    print(meta["chosen_label"].value_counts())
    print(meta["split"].value_counts())
    print(label_map)
    return meta, label_map


# ============================================================================
# DATASETS
# ============================================================================
class ECGDataset(Dataset):
    def __init__(self, meta_df, data_dir, label_map, tokenizer=None, max_len=64, seq_len=2500, use_text=True):
        self.meta = meta_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seq_len = seq_len
        self.use_text = use_text

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        try:
            record_path = os.path.join(self.data_dir, row["filename_lr"])
            record = wfdb.rdrecord(record_path)
            ecg = record.p_signal.T
            if ecg.shape[1] > self.seq_len * 2:
                ecg = ecg[:, ::2]
            if ecg.shape[1] < self.seq_len:
                pad_width = self.seq_len - ecg.shape[1]
                ecg = np.pad(ecg, ((0, 0), (0, pad_width)), mode="constant")
            else:
                ecg = ecg[:, :self.seq_len]
            ecg = (ecg - ecg.mean(axis=1, keepdims=True)) / (ecg.std(axis=1, keepdims=True) + 1e-6)
            ecg = torch.tensor(ecg, dtype=torch.float32)
        except Exception:
            ecg = torch.zeros((12, self.seq_len), dtype=torch.float32)

        if self.use_text and self.tokenizer is not None:
            text = str(row.get("report", ""))[:1000]
            text_enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            text_enc = {k: v.squeeze(0) for k, v in text_enc.items()}
        else:
            text_enc = None

        label_idx = self.label_map[row["chosen_label"]]
        label = torch.zeros(len(self.label_map), dtype=torch.float32)
        label[label_idx] = 1.0
        return ecg, text_enc, label


class MIMICECGDataset(Dataset):
    def __init__(self, meta_df, label_map, tokenizer=None, max_len=64, seq_len=2500, use_text=True):
        self.meta = meta_df.reset_index(drop=True)
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.seq_len = seq_len
        self.use_text = use_text

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]

        try:
            record = wfdb.rdrecord(row["filename_lr"])
            ecg = record.p_signal.T.astype(np.float32)

            # replace bad values first
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

            # optional downsample if too long
            if ecg.shape[1] > self.seq_len * 2:
                ecg = ecg[:, ::2]

            # pad / trim
            if ecg.shape[1] < self.seq_len:
                pad = self.seq_len - ecg.shape[1]
                ecg = np.pad(ecg, ((0, 0), (0, pad)), mode="constant")
            else:
                ecg = ecg[:, :self.seq_len]

            # recompute after pad/trim and sanitize again
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

            mean = ecg.mean(axis=1, keepdims=True)
            std = ecg.std(axis=1, keepdims=True)

            # protect against zero or tiny std
            std = np.where(std < 1e-6, 1.0, std)

            ecg = (ecg - mean) / std

            # final sanitize
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)

            ecg = torch.tensor(ecg, dtype=torch.float32)

        except Exception as e:
            print(f"ECG load failed for {row['filename_lr']}: {e}")
            ecg = torch.zeros((12, self.seq_len), dtype=torch.float32)

        """
        try:
            record = wfdb.rdrecord(row["filename_lr"])
            ecg = record.p_signal.T
            if ecg.shape[1] > self.seq_len * 2:
                ecg = ecg[:, ::2]
            if ecg.shape[1] < self.seq_len:
                pad = self.seq_len - ecg.shape[1]
                ecg = np.pad(ecg, ((0, 0), (0, pad)), mode="constant")
            else:
                ecg = ecg[:, :self.seq_len]
            ecg = (ecg - ecg.mean(axis=1, keepdims=True)) / (ecg.std(axis=1, keepdims=True) + 1e-6)
            ecg = torch.tensor(ecg, dtype=torch.float32)
        except Exception:
            ecg = torch.zeros((12, self.seq_len), dtype=torch.float32)
        """

        if self.use_text and self.tokenizer is not None:
            text = str(row.get("report", ""))[:1000]
            text_enc = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            text_enc = {k: v.squeeze(0) for k, v in text_enc.items()}
        else:
            text_enc = None

        label_idx = self.label_map[row["chosen_label"]]
        label = torch.zeros(len(self.label_map), dtype=torch.float32)
        label[label_idx] = 1.0
        return ecg, text_enc, label


def custom_collate_fn(batch):
    ecgs, text_encs, labels = zip(*batch)
    ecgs = torch.stack(ecgs)
    labels = torch.stack(labels)

    if text_encs[0] is not None:
        text_batch = {
            "input_ids": torch.stack([t["input_ids"] for t in text_encs]),
            "attention_mask": torch.stack([t["attention_mask"] for t in text_encs]),
        }
    else:
        text_batch = None

    return ecgs, text_batch, labels


# ============================================================================
# MODELS
# ============================================================================
class LightECGEncoder(nn.Module):
    def __init__(self, in_channels=12, embed_dim=128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(), nn.AdaptiveAvgPool1d(32),
        )
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        return x


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ecg_embeds, text_embeds, text_mask=None):
        attn_out, attn_weights = self.cross_attn(
            query=ecg_embeds,
            key=text_embeds,
            value=text_embeds,
            key_padding_mask=~text_mask.bool() if text_mask is not None else None,
        )
        fused = self.norm(attn_out + ecg_embeds)
        return fused, attn_weights


class MultimodalModel(nn.Module):
    def __init__(self, num_classes, text_model_name, fusion_type, config):
        super().__init__()
        self.fusion_type = fusion_type

        self.ecg_encoder = LightECGEncoder(config.ECG_CHANNELS, config.ECG_EMBED_DIM)

        self.text_model = AutoModel.from_pretrained(text_model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.FUSION_DIM)

        self.fusion = CrossAttentionFusion(config.FUSION_DIM, num_heads=4)

        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def encode_text(self, text_enc):
        outputs = self.text_model(input_ids=text_enc["input_ids"], attention_mask=text_enc["attention_mask"])
        hidden = outputs.last_hidden_state
        return self.text_proj(hidden)

    def forward(self, ecg, text_enc):
        ecg_embeds = self.ecg_encoder(ecg)
        text_embeds = self.encode_text(text_enc)
        fused, _ = self.fusion(ecg_embeds, text_embeds, text_enc["attention_mask"])
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


class ECGOnlyModel(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()
        self.ecg_encoder = LightECGEncoder(config.ECG_CHANNELS, config.ECG_EMBED_DIM)
        self.classifier = nn.Sequential(
            nn.Linear(config.ECG_EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, ecg, text_enc=None):
        ecg_embeds = self.ecg_encoder(ecg)
        pooled = ecg_embeds.mean(dim=1)
        return self.classifier(pooled)


class TextOnlyModel(nn.Module):
    def __init__(self, num_classes, text_model_name, config):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(text_model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        hidden_size = self.text_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, text_enc):
        outputs = self.text_model(input_ids=text_enc["input_ids"], attention_mask=text_enc["attention_mask"])
        cls_token = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_token)


# ============================================================================
# BASE TRAIN / EVAL
# ============================================================================
def safe_metrics(all_labels, all_preds):
    try:
        auroc = roc_auc_score(all_labels, all_preds, average="macro")
    except Exception:
        auroc = float("nan")
    preds_binary = (all_preds > 0.5).astype(int)
    f1_macro = f1_score(all_labels, preds_binary, average="macro", zero_division=0)
    try:
        acc = accuracy_score(all_labels.argmax(axis=1), preds_binary.argmax(axis=1))
    except Exception:
        acc = float("nan")
    return auroc, f1_macro, acc


def train_epoch(model, loader, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()


    for i, (ecg, text_enc, labels) in enumerate(tqdm(loader, desc="Training")):
        ecg = ecg.to(device)
        labels = labels.to(device)

        if torch.isnan(ecg).any() or torch.isinf(ecg).any():
            print("Bad ECG batch detected")
            print("NaN count:", torch.isnan(ecg).sum().item())
            print("Inf count:", torch.isinf(ecg).sum().item())
            ecg = torch.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
            
        if text_enc is not None:
            text_enc = {k: v.to(device) for k, v in text_enc.items()}

        with torch.cuda.amp.autocast(enabled=config.USE_MIXED_PRECISION):
            logits = model(ecg, text_enc)
            loss = criterion(logits, labels) / config.GRADIENT_ACCUM

        scaler.scale(loss).backward()

        if (i + 1) % config.GRADIENT_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUM

        if (i + 1) % config.CLEAR_CACHE_EVERY_N_BATCHES == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for ecg, text_enc, labels in tqdm(loader, desc="Evaluating"):
        ecg = ecg.to(device)
        labels = labels.to(device)
        if text_enc is not None:
            text_enc = {k: v.to(device) for k, v in text_enc.items()}

        logits = model(ecg, text_enc)
        loss = criterion(logits, labels)

        preds = torch.sigmoid(logits)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_loss = total_loss / max(1, len(loader))
    auroc, f1_macro, acc = safe_metrics(all_labels, all_preds)
    return avg_loss, auroc, f1_macro, acc


def train_epoch_text_only(model, loader, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (_, text_enc, labels) in enumerate(tqdm(loader, desc="Training Text-Only")):
        labels = labels.to(device)
        text_enc = {k: v.to(device) for k, v in text_enc.items()}

        with torch.cuda.amp.autocast(enabled=config.USE_MIXED_PRECISION):
            logits = model(text_enc)
            loss = criterion(logits, labels) / config.GRADIENT_ACCUM

        scaler.scale(loss).backward()

        if (i + 1) % config.GRADIENT_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUM

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_text_only(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for _, text_enc, labels in tqdm(loader, desc="Evaluating Text-Only"):
        labels = labels.to(device)
        text_enc = {k: v.to(device) for k, v in text_enc.items()}

        logits = model(text_enc)
        loss = criterion(logits, labels)

        preds = torch.sigmoid(logits)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_loss = total_loss / max(1, len(loader))
    auroc, f1_macro, acc = safe_metrics(all_labels, all_preds)
    return avg_loss, auroc, f1_macro, acc


# ============================================================================
# RAG
# ============================================================================
class SimpleRetrievalDatabase:
    def __init__(self, embed_dim=128):
        self.embed_dim = embed_dim
        self.embeddings = None
        self.reports = []
        self.labels = []
        self.ecg_ids = []

    def add_samples(self, ecg_embeddings, reports, labels=None, ecg_ids=None):
        ecg_embeddings = ecg_embeddings / (ecg_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        if self.embeddings is None:
            self.embeddings = ecg_embeddings.cpu()
        else:
            self.embeddings = torch.cat([self.embeddings, ecg_embeddings.cpu()], dim=0)

        self.reports.extend(reports)
        if labels is not None:
            self.labels.extend(labels)
        if ecg_ids is not None:
            self.ecg_ids.extend(ecg_ids)

    def retrieve(self, query_embeddings, k=3):
        query_embeddings = query_embeddings / (query_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        similarities = torch.mm(query_embeddings.cpu(), self.embeddings.t())
        topk_sims, topk_indices = torch.topk(similarities, k=min(k, len(self.reports)), dim=1)

        retrieved_reports, retrieved_labels, retrieved_sims = [], [], []
        for i in range(len(query_embeddings)):
            batch_reports, batch_labels, batch_sims = [], [], []
            for idx, sim in zip(topk_indices[i], topk_sims[i]):
                idx = idx.item()
                batch_reports.append(str(self.reports[idx]))
                if len(self.labels) > 0:
                    batch_labels.append(self.labels[idx])
                batch_sims.append(sim.item())
            retrieved_reports.append(batch_reports)
            retrieved_labels.append(batch_labels if len(batch_labels) > 0 else None)
            retrieved_sims.append(batch_sims)

        return retrieved_reports, retrieved_labels, np.array(retrieved_sims)


class RAGFusionLayer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.gate = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

    def forward(self, ecg_embeds, text_embeds, retrieved_embeds_list=None, text_mask=None, retrieved_masks=None):
        if retrieved_embeds_list is not None and len(retrieved_embeds_list) > 0:
            all_retrieved = torch.cat(retrieved_embeds_list, dim=1)
            combined_text = torch.cat([text_embeds, all_retrieved], dim=1)

            if retrieved_masks is not None and len(retrieved_masks) > 0:
                all_ret_masks = torch.cat(retrieved_masks, dim=1)
                combined_mask = torch.cat([text_mask, all_ret_masks], dim=1)
            else:
                combined_mask = text_mask
        else:
            combined_text = text_embeds
            combined_mask = text_mask

        attn_out, attn_weights = self.cross_attn(
            query=ecg_embeds,
            key=combined_text,
            value=combined_text,
            key_padding_mask=~combined_mask.bool() if combined_mask is not None else None,
        )

        gate_values = self.gate(ecg_embeds)
        fused = gate_values * attn_out + (1 - gate_values) * ecg_embeds
        fused = self.norm(fused)
        return fused, attn_weights


class RAGFusionModel(nn.Module):
    def __init__(self, num_classes, text_model_name, config, k_retrieve=3):
        super().__init__()
        self.k_retrieve = k_retrieve

        self.ecg_encoder = LightECGEncoder(config.ECG_CHANNELS, config.ECG_EMBED_DIM)

        self.text_model = AutoModel.from_pretrained(text_model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.FUSION_DIM)

        self.rag_fusion = RAGFusionLayer(config.FUSION_DIM, num_heads=4)
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def encode_ecg(self, ecg):
        return self.ecg_encoder(ecg)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        return self.text_proj(hidden)

    def forward(self, ecg, text_enc, retrieved_text_encs=None, retrieved_masks=None):
        ecg_embeds = self.encode_ecg(ecg)
        text_embeds = self.encode_text(text_enc["input_ids"], text_enc["attention_mask"])

        retrieved_embeds = []
        if retrieved_text_encs is not None and len(retrieved_text_encs) > 0:
            for ret_enc in retrieved_text_encs:
                ret_embeds = self.encode_text(ret_enc["input_ids"], ret_enc["attention_mask"])
                retrieved_embeds.append(ret_embeds)

        fused, _ = self.rag_fusion(
            ecg_embeds=ecg_embeds,
            text_embeds=text_embeds,
            retrieved_embeds_list=retrieved_embeds if len(retrieved_embeds) > 0 else None,
            text_mask=text_enc["attention_mask"],
            retrieved_masks=retrieved_masks,
        )
        pooled = fused.mean(dim=1)
        return self.classifier(pooled)


def build_retrieval_database(model, dataset, loader, device, config):
    print("\nBuilding retrieval database with REAL reports...")
    db = SimpleRetrievalDatabase(embed_dim=config.ECG_EMBED_DIM)
    model.eval()

    all_embeddings = []
    all_reports = []
    all_labels = []
    offset = 0

    with torch.no_grad():
        for ecg, text_enc, labels in loader:
            bs = ecg.size(0)
            ecg = ecg.to(device)

            ecg_embeds = model.encode_ecg(ecg).mean(dim=1)
            all_embeddings.append(ecg_embeds.cpu())

            batch_meta = dataset.meta.iloc[offset : offset + bs]
            all_reports.extend(batch_meta["report"].astype(str).tolist())
            all_labels.extend(batch_meta["chosen_label"].astype(str).tolist())
            offset += bs

    all_embeddings = torch.cat(all_embeddings, dim=0)
    db.add_samples(all_embeddings, all_reports, all_labels)
    print(f"Indexed {len(all_reports)} real-text samples into retrieval DB")
    return db


def prepare_retrieved_text_batch(retrieved_reports, tokenizer, device, max_len):
    batch_size = len(retrieved_reports)
    k = max(len(x) for x in retrieved_reports) if batch_size > 0 else 0

    padded = []
    for reps in retrieved_reports:
        rr = list(reps)
        while len(rr) < k:
            rr.append("")
        padded.append(rr)

    retrieved_text_encs = []
    retrieved_masks = []

    for j in range(k):
        texts_j = [padded[i][j] for i in range(batch_size)]
        enc = tokenizer(
            texts_j,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        enc = {kk: vv.to(device) for kk, vv in enc.items()}
        retrieved_text_encs.append(enc)
        retrieved_masks.append(enc["attention_mask"])

    return retrieved_text_encs, retrieved_masks


def train_epoch_rag(model, loader, retrieval_db, tokenizer, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (ecg, text_enc, labels) in enumerate(tqdm(loader, desc="Training RAG")):
        ecg = ecg.to(device)
        labels = labels.to(device)
        text_enc = {k: v.to(device) for k, v in text_enc.items()}

        with torch.no_grad():
            query_embeds = model.encode_ecg(ecg).mean(dim=1)
            retrieved_reports, _, _ = retrieval_db.retrieve(query_embeds, k=model.k_retrieve)

        retrieved_text_encs, retrieved_masks = prepare_retrieved_text_batch(
            retrieved_reports, tokenizer, device, config.TEXT_MAX_LEN
        )

        with torch.cuda.amp.autocast(enabled=config.USE_MIXED_PRECISION):
            logits = model(ecg, text_enc, retrieved_text_encs=retrieved_text_encs, retrieved_masks=retrieved_masks)
            loss = criterion(logits, labels) / config.GRADIENT_ACCUM

        scaler.scale(loss).backward()

        if (i + 1) % config.GRADIENT_ACCUM == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * config.GRADIENT_ACCUM

        if (i + 1) % config.CLEAR_CACHE_EVERY_N_BATCHES == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate_rag(model, loader, retrieval_db, tokenizer, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for ecg, text_enc, labels in tqdm(loader, desc="Evaluating RAG"):
        ecg = ecg.to(device)
        labels = labels.to(device)
        text_enc = {k: v.to(device) for k, v in text_enc.items()}

        query_embeds = model.encode_ecg(ecg).mean(dim=1)
        retrieved_reports, _, _ = retrieval_db.retrieve(query_embeds, k=model.k_retrieve)

        retrieved_text_encs, retrieved_masks = prepare_retrieved_text_batch(
            retrieved_reports, tokenizer, device, config.TEXT_MAX_LEN
        )

        logits = model(ecg, text_enc, retrieved_text_encs=retrieved_text_encs, retrieved_masks=retrieved_masks)
        loss = criterion(logits, labels)

        preds = torch.sigmoid(logits)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    avg_loss = total_loss / max(1, len(loader))
    auroc, f1_macro, acc = safe_metrics(all_labels, all_preds)
    return avg_loss, auroc, f1_macro, acc


# ============================================================================
# EXPERIMENT RUNNERS
# ============================================================================
def build_dataset_and_loaders(meta, label_map, config, tokenizer=None, use_text=True):
    train_meta = meta[meta["split"] == "train"].reset_index(drop=True)
    val_meta = meta[meta["split"] == "val"].reset_index(drop=True)
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)

    dataset_cls = MIMICECGDataset if config.DATASET == "mimic" else ECGDataset

    if config.DATASET == "mimic":
        train_ds = dataset_cls(train_meta, label_map, tokenizer=tokenizer, max_len=config.TEXT_MAX_LEN,
                               seq_len=config.ECG_SEQ_LEN, use_text=use_text)
        val_ds = dataset_cls(val_meta, label_map, tokenizer=tokenizer, max_len=config.TEXT_MAX_LEN,
                             seq_len=config.ECG_SEQ_LEN, use_text=use_text)
        test_ds = dataset_cls(test_meta, label_map, tokenizer=tokenizer, max_len=config.TEXT_MAX_LEN,
                              seq_len=config.ECG_SEQ_LEN, use_text=use_text)
    else:
        train_ds = dataset_cls(train_meta, config.DATA_DIR, label_map, tokenizer, config.TEXT_MAX_LEN,
                               config.ECG_SEQ_LEN, use_text)
        val_ds = dataset_cls(val_meta, config.DATA_DIR, label_map, tokenizer, config.TEXT_MAX_LEN,
                             config.ECG_SEQ_LEN, use_text)
        test_ds = dataset_cls(test_meta, config.DATA_DIR, label_map, tokenizer, config.TEXT_MAX_LEN,
                              config.ECG_SEQ_LEN, use_text)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0,
                              collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
                            collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
                             collate_fn=custom_collate_fn)

    return (train_meta, val_meta, test_meta), (train_ds, val_ds, test_ds), (train_loader, val_loader, test_loader)


def run_experiment(model_name, text_model, fusion_type, meta, label_map, config):
    print(f"\n{'='*70}\n{model_name}\n{'='*70}")
    tokenizer = AutoTokenizer.from_pretrained(text_model) if text_model else None
    (_, _, _), (_, _, _), (train_loader, val_loader, test_loader) = build_dataset_and_loaders(
        meta, label_map, config, tokenizer=tokenizer, use_text=(text_model is not None)
    )

    if fusion_type is None:
        model = ECGOnlyModel(num_classes=len(label_map), config=config).to(config.DEVICE)
    else:
        model = MultimodalModel(num_classes=len(label_map), text_model_name=text_model,
                                fusion_type=fusion_type, config=config).to(config.DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_MIXED_PRECISION)

    best_val_f1 = -1
    best_path = f"{model_name.replace(' ', '_')}_best.pt"

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE, config)
        val_loss, val_auroc, val_f1, val_acc = evaluate(model, val_loader, criterion, config.DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val: AUROC={val_auroc:.4f} | F1={val_f1:.4f} | Acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
    test_loss, test_auroc, test_f1, test_acc = evaluate(model, test_loader, criterion, config.DEVICE)
    print(f"TEST: AUROC={test_auroc:.4f} | F1={test_f1:.4f} | Acc={test_acc:.4f}")

    return {
        "model_name": model_name,
        "test_loss": test_loss,
        "test_auroc": test_auroc,
        "test_f1": test_f1,
        "test_acc": test_acc,
    }


def run_text_only_experiment(model_name, text_model, meta, label_map, config):
    print(f"\n{'='*70}\n{model_name}\n{'='*70}")
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    (_, _, _), (_, _, _), (train_loader, val_loader, test_loader) = build_dataset_and_loaders(
        meta, label_map, config, tokenizer=tokenizer, use_text=True
    )

    model = TextOnlyModel(num_classes=len(label_map), text_model_name=text_model, config=config).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_MIXED_PRECISION)

    best_val_f1 = -1
    best_path = f"{model_name.replace(' ', '_')}_best.pt"

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_loss = train_epoch_text_only(model, train_loader, criterion, optimizer, scaler, config.DEVICE, config)
        val_loss, val_auroc, val_f1, val_acc = evaluate_text_only(model, val_loader, criterion, config.DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val: AUROC={val_auroc:.4f} | F1={val_f1:.4f} | Acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
    test_loss, test_auroc, test_f1, test_acc = evaluate_text_only(model, test_loader, criterion, config.DEVICE)
    print(f"TEST: AUROC={test_auroc:.4f} | F1={test_f1:.4f} | Acc={test_acc:.4f}")

    return {
        "model_name": model_name,
        "test_loss": test_loss,
        "test_auroc": test_auroc,
        "test_f1": test_f1,
        "test_acc": test_acc,
    }


def run_rag_experiment(model_name, text_model, k_retrieve, meta, label_map, config):
    print(f"\n{'='*70}\n{model_name} (K={k_retrieve})\n{'='*70}")
    tokenizer = AutoTokenizer.from_pretrained(text_model)
    (_, _, _), (train_ds, val_ds, test_ds), (train_loader, val_loader, test_loader) = build_dataset_and_loaders(
        meta, label_map, config, tokenizer=tokenizer, use_text=True
    )

    train_loader_no_shuffle = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0,
                                         collate_fn=custom_collate_fn)

    model = RAGFusionModel(num_classes=len(label_map), text_model_name=text_model,
                           config=config, k_retrieve=k_retrieve).to(config.DEVICE)

    retrieval_db = build_retrieval_database(model, train_ds, train_loader_no_shuffle, config.DEVICE, config)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_MIXED_PRECISION)

    best_val_f1 = -1
    best_path = f"{model_name.replace(' ', '_')}_best.pt"

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        train_loss = train_epoch_rag(model, train_loader, retrieval_db, tokenizer, criterion,
                                     optimizer, scaler, config.DEVICE, config)
        val_loss, val_auroc, val_f1, val_acc = evaluate_rag(
            model, val_loader, retrieval_db, tokenizer, criterion, config.DEVICE, config
        )
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val: AUROC={val_auroc:.4f} | F1={val_f1:.4f} | Acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
    test_loss, test_auroc, test_f1, test_acc = evaluate_rag(
        model, test_loader, retrieval_db, tokenizer, criterion, config.DEVICE, config
    )
    print(f"TEST: AUROC={test_auroc:.4f} | F1={test_f1:.4f} | Acc={test_acc:.4f}")

    return {
        "model_name": model_name,
        "test_loss": test_loss,
        "test_auroc": test_auroc,
        "test_f1": test_f1,
        "test_acc": test_acc,
    }


def run_all_experiments(meta, label_map, config):
    experiments = [
        ("ECG Only", None, None, None),
        ("Text Only DistilBERT", "distilbert-base-uncased", "text_only", None),
        ("DistilBERT + Cross-Attn", "distilbert-base-uncased", "cross_attention", None),
        ("BioBERT + Cross-Attn", "dmis-lab/biobert-base-cased-v1.1", "cross_attention", None),
        ("PubMedBERT + Cross-Attn", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "cross_attention", None),
        ("RAG-Fusion K=1", "distilbert-base-uncased", "rag", 1),
        ("RAG-Fusion K=3", "distilbert-base-uncased", "rag", 3),
        ("RAG-Fusion K=5", "distilbert-base-uncased", "rag", 5),
        ("BioBERT + RAG K=3", "dmis-lab/biobert-base-cased-v1.1", "rag", 3),
        ("PubMedBERT + RAG K=3", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "rag", 3),
    ]

    all_results = []
    for model_name, text_model, fusion_type, k_retrieve in experiments:
        try:
            if fusion_type == "text_only":
                result = run_text_only_experiment(model_name, text_model, meta, label_map, config)
            elif fusion_type == "rag":
                result = run_rag_experiment(model_name, text_model, k_retrieve, meta, label_map, config)
            else:
                result = run_experiment(model_name, text_model, fusion_type, meta, label_map, config)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR in {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    out_csv = f"{config.DATASET}_results.csv"
    results_df.to_csv(out_csv, index=False)

    print(f"\nSaved results to {out_csv}")
    print(results_df)
    return results_df


# ============================================================================
# ENTRY
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ptbxl", "mimic"], default="mimic")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mimic_max_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        DATASET=args.dataset,
        DATA_DIR=args.data_dir,
        NUM_EPOCHS=args.epochs,
        BATCH_SIZE=args.batch_size,
        MIMIC_MAX_SAMPLES=args.mimic_max_samples,
        SEED=args.seed,
    )
    set_all_seeds(cfg.SEED)

    print("Configuration:")
    print(cfg)

    if cfg.DATASET == "ptbxl":
        meta, label_map = load_ptbxl_metadata(cfg)
    else:
        meta, label_map = build_mimic_metadata_from_csv(cfg)

    run_all_experiments(meta, label_map, cfg)


if __name__ == "__main__":
    main()
"""
/home/asatsan2/Projects/ECG

/home/asatsan2/Projects/ECG

python /home/asatsan2/Projects/ECG/pipeline.py  \
  --dataset mimic \
  --data_dir /home/asatsan2/Projects/ECG/mimic_ecg \ 
  --epochs 5 \
  --batch_size 16 \
  --mimic_max_samples 500


"""