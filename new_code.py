import os
import re
import gc
import ast
import glob
import random
import warnings
import argparse
from dataclasses import dataclass
from typing import Optional, List, Dict
# final
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


def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    DATASET: str = "mimic"
    DATA_DIR: str = "/home/asatsan2/Projects/ECG/mimic_ecg"
    CSV_META: Optional[str] = None

    TOPK: int = 5
    ECG_CHANNELS: int = 12
    ECG_SEQ_LEN: int = 2500
    TEXT_MAX_LEN: int = 64
    ECG_EMBED_DIM: int = 128
    FUSION_DIM: int = 128

    BATCH_SIZE: int = 16
    GRADIENT_ACCUM: int = 2
    LEARNING_RATE: float = 2e-5
    RAG_LEARNING_RATE: float = 1e-5
    NUM_EPOCHS: int = 8
    WEIGHT_DECAY: float = 1e-5
    MAX_GRAD_NORM: float = 1.0

    USE_MIXED_PRECISION: bool = False
    CLEAR_CACHE_EVERY_N_BATCHES: int = 20
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42

    MIMIC_MAX_SAMPLES: int = 1000
    MIMIC_KEEP_LABELS: Optional[List[str]] = None
    RAG_K: int = 1

    def __post_init__(self):
        if self.CSV_META is None and self.DATASET == "ptbxl":
            self.CSV_META = os.path.join(self.DATA_DIR, "ptbxl_database.csv")
        if self.MIMIC_KEEP_LABELS is None:
            self.MIMIC_KEEP_LABELS = ["NORM", "TACHY", "BRADY", "AFIB"]


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
    val_p = set(unique_pat[int(0.7 * n): int(0.85 * n)])
    test_p = set(unique_pat[int(0.85 * n):])

    meta["split"] = meta["patient_id"].apply(
        lambda x: "train" if x in train_p else ("val" if x in val_p else "test")
    )

    print(f"PTB-XL records: {len(meta)}")
    print(meta["chosen_label"].value_counts())
    print(meta["split"].value_counts())
    print(label_map)
    return meta, label_map


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

    meta = meta[meta["chosen_label"].isin(config.MIMIC_KEEP_LABELS)].copy()
    meta["filename_lr"] = meta[path_col].apply(lambda x: os.path.join(mimic_root, x))

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
    val_idx = set(idx[int(0.7 * n): int(0.85 * n)])
    test_idx = set(idx[int(0.85 * n):])

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
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
            if ecg.shape[1] > self.seq_len * 2:
                ecg = ecg[:, ::2]
            if ecg.shape[1] < self.seq_len:
                pad = self.seq_len - ecg.shape[1]
                ecg = np.pad(ecg, ((0, 0), (0, pad)), mode="constant")
            else:
                ecg = ecg[:, :self.seq_len]
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
            mean = ecg.mean(axis=1, keepdims=True)
            std = ecg.std(axis=1, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            ecg = (ecg - mean) / std
            ecg = np.nan_to_num(ecg, nan=0.0, posinf=0.0, neginf=0.0)
            ecg = torch.tensor(ecg, dtype=torch.float32)
        except Exception as e:
            print(f"ECG load failed for {row['filename_lr']}: {e}")
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

        label = torch.tensor(self.label_map[row["chosen_label"]], dtype=torch.long)
        return ecg, text_enc, label


def custom_collate_fn(batch):
    ecgs, text_encs, labels = zip(*batch)
    ecgs = torch.stack(ecgs)
    labels = torch.stack(labels).long()

    if text_encs[0] is not None:
        text_batch = {
            "input_ids": torch.stack([t["input_ids"] for t in text_encs]),
            "attention_mask": torch.stack([t["attention_mask"] for t in text_encs]),
        }
    else:
        text_batch = None

    return ecgs, text_batch, labels


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

    def forward(self, query_embeds, key_value_embeds, mask=None):
        attn_out, _ = self.cross_attn(
            query=query_embeds,
            key=key_value_embeds,
            value=key_value_embeds,
            key_padding_mask=~mask.bool() if mask is not None else None,
        )
        return self.norm(attn_out + query_embeds)


class BioBERTCrossAttn(nn.Module):
    def __init__(self, num_classes, text_model_name, config):
        super().__init__()
        self.ecg_encoder = LightECGEncoder(config.ECG_CHANNELS, config.ECG_EMBED_DIM)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.FUSION_DIM)
        self.fusion = CrossAttentionFusion(config.FUSION_DIM, 4)
        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def encode_ecg(self, ecg):
        return self.ecg_encoder(ecg)

    def encode_text_tokens(self, text_enc):
        outputs = self.text_model(input_ids=text_enc["input_ids"], attention_mask=text_enc["attention_mask"])
        return self.text_proj(outputs.last_hidden_state)

    def encode_text_vector(self, text_enc):
        return self.encode_text_tokens(text_enc)[:, 0, :]

    def forward(self, ecg, text_enc):
        ecg_embeds = self.encode_ecg(ecg)
        text_tokens = self.encode_text_tokens(text_enc)
        fused = self.fusion(ecg_embeds, text_tokens, text_enc["attention_mask"])
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
        pooled = self.ecg_encoder(ecg).mean(dim=1)
        return self.classifier(pooled)


class TextOnlyModel(nn.Module):
    def __init__(self, num_classes, text_model_name):
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
        return self.classifier(outputs.last_hidden_state[:, 0, :])


class RetrievalVectorProjector(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class TwoStageRAGModel(nn.Module):
    def __init__(self, num_classes, text_model_name, config, k_retrieve=1):
        super().__init__()
        self.k_retrieve = k_retrieve
        self.ecg_encoder = LightECGEncoder(config.ECG_CHANNELS, config.ECG_EMBED_DIM)

        self.text_model = AutoModel.from_pretrained(text_model_name)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, config.FUSION_DIM)
        self.ret_vec_proj = RetrievalVectorProjector(config.FUSION_DIM, config.FUSION_DIM)

        self.base_fusion = CrossAttentionFusion(config.FUSION_DIM, 4)
        self.ret_fusion = CrossAttentionFusion(config.FUSION_DIM, 4)

        self.gate = nn.Sequential(
            nn.Linear(config.FUSION_DIM * 2, config.FUSION_DIM),
            nn.ReLU(),
            nn.Linear(config.FUSION_DIM, config.FUSION_DIM),
            nn.Sigmoid(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(config.FUSION_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def encode_ecg(self, ecg):
        return self.ecg_encoder(ecg)

    def encode_text_tokens(self, text_enc):
        outputs = self.text_model(input_ids=text_enc["input_ids"], attention_mask=text_enc["attention_mask"])
        return self.text_proj(outputs.last_hidden_state)

    def encode_text_vector(self, text_enc):
        return self.ret_vec_proj(self.encode_text_tokens(text_enc)[:, 0, :])

    def forward(self, ecg, text_enc, retrieved_text_encs=None):
        ecg_embeds = self.encode_ecg(ecg)
        own_text = self.encode_text_tokens(text_enc)
        base = self.base_fusion(ecg_embeds, own_text, text_enc["attention_mask"])

        if retrieved_text_encs:
            retrieved_vecs = [self.encode_text_vector(ret).unsqueeze(1) for ret in retrieved_text_encs]
            retrieved_stack = torch.cat(retrieved_vecs, dim=1)
            mask = torch.ones(retrieved_stack.shape[:2], dtype=torch.bool, device=retrieved_stack.device)
            ret = self.ret_fusion(base, retrieved_stack, mask)

            base_pool = base.mean(dim=1)
            ret_pool = ret.mean(dim=1)
            gate = self.gate(torch.cat([base_pool, ret_pool], dim=1))
            pooled = gate * ret_pool + (1.0 - gate) * base_pool
        else:
            pooled = base.mean(dim=1)

        return self.classifier(pooled)


def make_class_weights(meta, label_map, device):
    counts = meta["chosen_label"].value_counts().to_dict()
    weights = []
    for lab, idx in sorted(label_map.items(), key=lambda x: x[1]):
        weights.append(1.0 / counts.get(lab, 1))
    weights = np.array(weights, dtype=np.float32)
    weights = weights / weights.sum() * len(weights)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def safe_metrics(all_labels, all_probs):
    pred_classes = all_probs.argmax(axis=1)
    f1_macro = f1_score(all_labels, pred_classes, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, pred_classes)
    try:
        n_classes = all_probs.shape[1]
        auroc = roc_auc_score(np.eye(n_classes)[all_labels], all_probs, multi_class="ovr", average="macro")
    except Exception:
        auroc = float("nan")
    return auroc, f1_macro, acc


def build_dataset_and_loaders(meta, label_map, config, tokenizer=None, use_text=True):
    train_meta = meta[meta["split"] == "train"].reset_index(drop=True)
    val_meta = meta[meta["split"] == "val"].reset_index(drop=True)
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)

    train_ds = MIMICECGDataset(train_meta, label_map, tokenizer=tokenizer, max_len=config.TEXT_MAX_LEN,
                               seq_len=config.ECG_SEQ_LEN, use_text=use_text)
    val_ds = MIMICECGDataset(val_meta, label_map, tokenizer=tokenizer, max_len=config.TEXT_MAX_LEN,
                             seq_len=config.ECG_SEQ_LEN, use_text=use_text)
    test_ds = MIMICECGDataset(test_meta, label_map, tokenizer=tokenizer, max_len=config.TEXT_MAX_LEN,
                              seq_len=config.ECG_SEQ_LEN, use_text=use_text)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    return train_meta, train_ds, train_loader, val_loader, test_loader


def train_epoch(model, loader, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (ecg, text_enc, labels) in enumerate(tqdm(loader, desc="Training")):
        ecg = ecg.to(device)
        labels = labels.to(device)
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
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for ecg, text_enc, labels in tqdm(loader, desc="Evaluating"):
        ecg = ecg.to(device)
        labels = labels.to(device)
        if text_enc is not None:
            text_enc = {k: v.to(device) for k, v in text_enc.items()}

        logits = model(ecg, text_enc)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    auroc, f1, acc = safe_metrics(all_labels, all_probs)
    return total_loss / max(1, len(loader)), auroc, f1, acc


def train_epoch_text(model, loader, criterion, optimizer, scaler, device, config):
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
def evaluate_text(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for _, text_enc, labels in tqdm(loader, desc="Evaluating Text-Only"):
        labels = labels.to(device)
        text_enc = {k: v.to(device) for k, v in text_enc.items()}
        logits = model(text_enc)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    auroc, f1, acc = safe_metrics(all_labels, all_probs)
    return total_loss / max(1, len(loader)), auroc, f1, acc


class SimpleRetrievalDatabase:
    def __init__(self):
        self.ecg_embeddings = None
        self.text_embeddings = None
        self.reports = []
        self.labels = []

    def add_samples(self, ecg_embeddings, text_embeddings, reports, labels):
        ecg_embeddings = ecg_embeddings / (ecg_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        text_embeddings = text_embeddings / (text_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        if self.ecg_embeddings is None:
            self.ecg_embeddings = ecg_embeddings.cpu()
            self.text_embeddings = text_embeddings.cpu()
        else:
            self.ecg_embeddings = torch.cat([self.ecg_embeddings, ecg_embeddings.cpu()], dim=0)
            self.text_embeddings = torch.cat([self.text_embeddings, text_embeddings.cpu()], dim=0)
        self.reports.extend(reports)
        self.labels.extend(labels)

    def retrieve(self, query_ecg, query_text, k=1, alpha=0.6):
        q_ecg = query_ecg / (query_ecg.norm(dim=1, keepdim=True) + 1e-8)
        q_txt = query_text / (query_text.norm(dim=1, keepdim=True) + 1e-8)
        sim = alpha * torch.mm(q_ecg.cpu(), self.ecg_embeddings.t()) + (1 - alpha) * torch.mm(q_txt.cpu(), self.text_embeddings.t())
        vals, inds = torch.topk(sim, k=min(k, len(self.reports)), dim=1)

        outs = []
        for i in range(len(query_ecg)):
            rs = []
            seen = set()
            for idx in inds[i]:
                txt = str(self.reports[idx.item()]).strip()
                if txt and txt not in seen:
                    rs.append(txt)
                    seen.add(txt)
            outs.append(rs)
        return outs


def build_retrieval_database(model, dataset, loader, device):
    db = SimpleRetrievalDatabase()
    model.eval()
    all_e, all_t, all_r, all_l = [], [], [], []
    offset = 0

    with torch.no_grad():
        for ecg, text_enc, labels in loader:
            bs = ecg.size(0)
            ecg = ecg.to(device)
            text_enc = {k: v.to(device) for k, v in text_enc.items()}
            all_e.append(model.encode_ecg(ecg).mean(dim=1).cpu())
            all_t.append(model.encode_text_vector(text_enc).cpu())
            batch_meta = dataset.meta.iloc[offset:offset+bs]
            all_r.extend(batch_meta["report"].astype(str).tolist())
            all_l.extend(batch_meta["chosen_label"].astype(str).tolist())
            offset += bs

    db.add_samples(torch.cat(all_e, 0), torch.cat(all_t, 0), all_r, all_l)
    print(f"Indexed {len(all_r)} real-text samples into retrieval DB")
    return db


def prepare_retrieved_text_batch(retrieved_reports, tokenizer, device, max_len):
    k = max(len(x) for x in retrieved_reports) if len(retrieved_reports) > 0 else 0
    padded = []
    for reps in retrieved_reports:
        rr = list(reps)
        while len(rr) < k:
            rr.append("")
        padded.append(rr)

    encs = []
    for j in range(k):
        texts = [padded[i][j] for i in range(len(padded))]
        enc = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        enc = {kk: vv.to(device) for kk, vv in enc.items()}
        encs.append(enc)
    return encs

# class ECGGradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer
#         self.activations = None
#         self.gradients = None

#         self.fwd_hook = target_layer.register_forward_hook(self.save_activation)
#         self.bwd_hook = target_layer.register_full_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         self.activations = output.detach()

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0].detach()

#     def remove(self):
#         self.fwd_hook.remove()
#         self.bwd_hook.remove()

#     def generate(self, ecg, text_enc=None, retrieved_text_encs=None, class_idx=None):
#         self.model.zero_grad()

#         if retrieved_text_encs is None:
#             logits = self.model(ecg, text_enc)
#         else:
#             logits = self.model(ecg, text_enc, retrieved_text_encs)

#         if class_idx is None:
#             class_idx = logits.argmax(dim=1).item()

#         score = logits[:, class_idx].sum()
#         score.backward(retain_graph=True)

#         # activations: [B, C, T]
#         # gradients: [B, C, T]
#         grads = self.gradients.mean(dim=2, keepdim=True)   # [B, C, 1]
#         cam = (grads * self.activations).sum(dim=1)        # [B, T]
#         cam = torch.relu(cam)

#         cam = cam[0].cpu().numpy()
#         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#         return cam, class_idx

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

class ECGGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self.bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()   # [B,C,T]

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()   # [B,C,T]

    def remove(self):
        self.fwd_hook.remove()
        self.bwd_hook.remove()

    def generate(self, ecg, text_enc=None, retrieved_text_encs=None, class_idx=None):
        self.model.zero_grad()

        if retrieved_text_encs is None:
            logits = self.model(ecg, text_enc)
        else:
            logits = self.model(ecg, text_enc, retrieved_text_encs)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=2, keepdim=True)   # [B,C,1]
        cam = (weights * self.activations).sum(dim=1)        # [B,T]
        cam = F.relu(cam)

        cam = cam[0].cpu().numpy()
        # cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam = np.maximum(cam, 0)
        cam = cam ** 2   # 🔥 enhances peaks
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


class ECGScoreCAM:
    def __init__(self, model, target_layer, max_channels=32):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.max_channels = max_channels
        self.fwd_hook = target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def remove(self):
        self.fwd_hook.remove()

    @torch.no_grad()
    def generate(self, ecg, text_enc=None, retrieved_text_encs=None, class_idx=None):
        if retrieved_text_encs is None:
            logits = self.model(ecg, text_enc)
        else:
            logits = self.model(ecg, text_enc, retrieved_text_encs)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        acts = self.activations[0]   # [C,T]
        C, T = acts.shape
        chosen = torch.linspace(0, C - 1, steps=min(C, self.max_channels)).long()

        score_cam = None
        ecg_len = ecg.shape[-1]

        for c in chosen:
            act = acts[c].cpu().numpy()
            act = np.maximum(act, 0)
            if act.max() > 0:
                act = act / (act.max() + 1e-8)

            mask = np.interp(
                np.arange(ecg_len),
                np.linspace(0, ecg_len - 1, len(act)),
                act
            )

            mask_t = torch.tensor(mask, dtype=ecg.dtype, device=ecg.device).unsqueeze(0).unsqueeze(0)
            masked_ecg = ecg * mask_t

            if retrieved_text_encs is None:
                out = self.model(masked_ecg, text_enc)
            else:
                out = self.model(masked_ecg, text_enc, retrieved_text_encs)

            weight = F.softmax(out, dim=1)[0, class_idx].item()

            if score_cam is None:
                score_cam = weight * mask
            else:
                score_cam += weight * mask

        score_cam = np.maximum(score_cam, 0)
        score_cam = (score_cam - score_cam.min()) / (score_cam.max() - score_cam.min() + 1e-8)
        return score_cam, class_idx


def plot_plain_ecg(ax, ecg_1d, title, true_label):
    x = np.arange(len(ecg_1d))
    # ax.plot(x, ecg_1d, linewidth=1.0)
    ax.plot(x, ecg_1d, linewidth=1.5, color="black")
    ax.set_title(f"{title}\nTrue: {true_label}", fontsize=10)
    # 
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")


def overlay_cam_on_ecg(ax, ecg_1d, cam_1d, title, pred_label, true_label):
    x = np.arange(len(ecg_1d))
    cam_resized = np.interp(
        x,
        np.linspace(0, len(ecg_1d) - 1, len(cam_1d)),
        cam_1d
    )

    #ax.plot(x, ecg_1d, linewidth=1.0)
    ax.plot(x, ecg_1d, linewidth=1.5, color="black")
    
    # ax.imshow(
    #     cam_resized[np.newaxis, :],
    #     aspect="auto",
    #     extent=[0, len(ecg_1d), ecg_1d.min(), ecg_1d.max()],
    #     alpha=0.40,
    # )
    ax.imshow(
    cam_resized[np.newaxis, :],
    aspect="auto",
    extent=[0, len(ecg_1d), ecg_1d.min(), ecg_1d.max()],
    cmap="jet",   # 🔥 or "hot"
    alpha=0.35,
)
    # ax.set_title(f"{title}\nPred: {pred_label} | True: {true_label}", fontsize=10)
    ax.set_title(f"{title}  |  Pred: {pred_label}  |  True: {true_label}", fontsize=10)
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Amplitude")


def load_matching_weights(target_model, checkpoint_path):
    if checkpoint_path is None or not os.path.exists(checkpoint_path):
        return
    state = torch.load(checkpoint_path, map_location="cpu")
    target_state = target_model.state_dict()
    matched = {}
    for k, v in state.items():
        if k in target_state and target_state[k].shape == v.shape:
            matched[k] = v
    target_state.update(matched)
    target_model.load_state_dict(target_state)
    print(f"Loaded {len(matched)} tensors from {checkpoint_path}")


def train_epoch_rag(model, loader, retrieval_db, tokenizer, criterion, optimizer, scaler, device, config):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (ecg, text_enc, labels) in enumerate(tqdm(loader, desc="Training RAG")):
        ecg = ecg.to(device)
        labels = labels.to(device)
        text_enc = {k: v.to(device) for k, v in text_enc.items()}

        with torch.no_grad():
            q_ecg = model.encode_ecg(ecg).mean(dim=1)
            q_txt = model.encode_text_vector(text_enc)
            retrieved_reports = retrieval_db.retrieve(q_ecg, q_txt, k=model.k_retrieve, alpha=0.6)

        retrieved_text_encs = prepare_retrieved_text_batch(retrieved_reports, tokenizer, device, config.TEXT_MAX_LEN)

        with torch.cuda.amp.autocast(enabled=config.USE_MIXED_PRECISION):
            logits = model(ecg, text_enc, retrieved_text_encs)
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
def evaluate_rag(model, loader, retrieval_db, tokenizer, criterion, device, config):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for ecg, text_enc, labels in tqdm(loader, desc="Evaluating RAG"):
        ecg = ecg.to(device)
        labels = labels.to(device)
        text_enc = {k: v.to(device) for k, v in text_enc.items()}

        q_ecg = model.encode_ecg(ecg).mean(dim=1)
        q_txt = model.encode_text_vector(text_enc)
        retrieved_reports = retrieval_db.retrieve(q_ecg, q_txt, k=model.k_retrieve, alpha=0.6)
        retrieved_text_encs = prepare_retrieved_text_batch(retrieved_reports, tokenizer, device, config.TEXT_MAX_LEN)

        logits = model(ecg, text_enc, retrieved_text_encs)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        total_loss += loss.item()

    all_probs = np.vstack(all_probs)
    all_labels = np.concatenate(all_labels)
    auroc, f1, acc = safe_metrics(all_labels, all_probs)
    return total_loss / max(1, len(loader)), auroc, f1, acc


def run_ecg(meta, label_map, config):
    train_meta, _, train_loader, val_loader, test_loader = build_dataset_and_loaders(meta, label_map, config, tokenizer=None, use_text=False)
    model = ECGOnlyModel(len(label_map), config).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_meta, label_map, config.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_MIXED_PRECISION)

    best_f1, best_path = -1, "ECG_Only_best.pt"
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        tr = train_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE, config)
        vl, va, vf, vacc = evaluate(model, val_loader, criterion, config.DEVICE)
        print(f"Train Loss: {tr:.4f}")
        print(f"Val: AUROC={va:.4f} | F1={vf:.4f} | Acc={vacc:.4f}")
        if vf > best_f1:
            best_f1 = vf
            torch.save(model.state_dict(), best_path)
    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
    tl, ta, tf, tacc = evaluate(model, test_loader, criterion, config.DEVICE)
    print(f"TEST: AUROC={ta:.4f} | F1={tf:.4f} | Acc={tacc:.4f}")
    return {"model_name": "ECG Only", "checkpoint": best_path, "test_loss": tl, "test_auroc": ta, "test_f1": tf, "test_acc": tacc}



def run_text(meta, label_map, config, text_model_name, model_label):
    tok = AutoTokenizer.from_pretrained(text_model_name)
    train_meta, _, train_loader, val_loader, test_loader = build_dataset_and_loaders(
        meta, label_map, config, tokenizer=tok, use_text=True
    )

    model = TextOnlyModel(len(label_map), text_model_name).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_meta, label_map, config.DEVICE))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_MIXED_PRECISION)

    best_f1 = -1
    best_path = f"{model_label.replace(' ', '_')}_best.pt"

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        tr = train_epoch_text(model, train_loader, criterion, optimizer, scaler, config.DEVICE, config)
        vl, va, vf, vacc = evaluate_text(model, val_loader, criterion, config.DEVICE)
        print(f"Train Loss: {tr:.4f}")
        print(f"Val: AUROC={va:.4f} | F1={vf:.4f} | Acc={vacc:.4f}")

        if vf > best_f1:
            best_f1 = vf
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
    tl, ta, tf, tacc = evaluate_text(model, test_loader, criterion, config.DEVICE)
    print(f"TEST: AUROC={ta:.4f} | F1={tf:.4f} | Acc={tacc:.4f}")

    return {
        "model_name": model_label,
        "checkpoint": best_path,
        "test_loss": tl,
        "test_auroc": ta,
        "test_f1": tf,
        "test_acc": tacc,
    }

def run_cross(meta, label_map, config, text_model_name, model_label):
    tok = AutoTokenizer.from_pretrained(text_model_name)
    train_meta, _, train_loader, val_loader, test_loader = build_dataset_and_loaders(
        meta, label_map, config, tokenizer=tok, use_text=True
    )

    model = BioBERTCrossAttn(len(label_map), text_model_name, config).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_meta, label_map, config.DEVICE))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_MIXED_PRECISION)

    best_f1 = -1
    best_path = f"{model_label.replace(' ', '_')}_best.pt"

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        tr = train_epoch(model, train_loader, criterion, optimizer, scaler, config.DEVICE, config)
        vl, va, vf, vacc = evaluate(model, val_loader, criterion, config.DEVICE)
        print(f"Train Loss: {tr:.4f}")
        print(f"Val: AUROC={va:.4f} | F1={vf:.4f} | Acc={vacc:.4f}")

        if vf > best_f1:
            best_f1 = vf
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
    tl, ta, tf, tacc = evaluate(model, test_loader, criterion, config.DEVICE)
    print(f"TEST: AUROC={ta:.4f} | F1={tf:.4f} | Acc={tacc:.4f}")

    return {
        "model_name": model_label,
        "checkpoint": best_path,
        "test_loss": tl,
        "test_auroc": ta,
        "test_f1": tf,
        "test_acc": tacc,
    }

def run_rag(meta, label_map, config, text_model_name, model_label, k_retrieve, init_ckpt=None):
    tok = AutoTokenizer.from_pretrained(text_model_name)
    train_meta, train_ds, train_loader, val_loader, test_loader = build_dataset_and_loaders(
        meta, label_map, config, tokenizer=tok, use_text=True
    )
    index_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    model = TwoStageRAGModel(
        len(label_map),
        text_model_name,
        config,
        k_retrieve=k_retrieve
    ).to(config.DEVICE)

    load_matching_weights(model, init_ckpt)

    retrieval_db = build_retrieval_database(model, train_ds, index_loader, config.DEVICE)
    criterion = nn.CrossEntropyLoss(weight=make_class_weights(train_meta, label_map, config.DEVICE))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.RAG_LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    scaler = torch.cuda.amp.GradScaler(enabled=config.USE_MIXED_PRECISION)

    best_f1 = -1
    best_path = f"{model_label.replace(' ', '_')}_best.pt"

    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
        tr = train_epoch_rag(model, train_loader, retrieval_db, tok, criterion, optimizer, scaler, config.DEVICE, config)
        vl, va, vf, vacc = evaluate_rag(model, val_loader, retrieval_db, tok, criterion, config.DEVICE, config)
        print(f"Train Loss: {tr:.4f}")
        print(f"Val: AUROC={va:.4f} | F1={vf:.4f} | Acc={vacc:.4f}")

        if vf > best_f1:
            best_f1 = vf
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=config.DEVICE))
    tl, ta, tf, tacc = evaluate_rag(model, test_loader, retrieval_db, tok, criterion, config.DEVICE, config)
    print(f"TEST: AUROC={ta:.4f} | F1={tf:.4f} | Acc={tacc:.4f}")

    return {
        "model_name": model_label,
        "checkpoint": best_path,
        "test_loss": tl,
        "test_auroc": ta,
        "test_f1": tf,
        "test_acc": tacc,
    }

def run_all(meta, label_map, config):
    results = []
    ckpts = {}

    # ECG only
    r1 = run_ecg(meta, label_map, config)
    results.append(r1)

    # Text-only family
    text_models = [
        ("distilbert-base-uncased", "Text Only DistilBERT"),
        ("dmis-lab/biobert-base-cased-v1.1", "Text Only BioBERT"),
        ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "Text Only PubMedBERT"),
    ]

    for text_model_name, model_label in text_models:
        r = run_text(meta, label_map, config, text_model_name, model_label)
        results.append(r)

    # Cross-attn family
    cross_models = [
        ("distilbert-base-uncased", "DistilBERT + Cross-Attn"),
        ("dmis-lab/biobert-base-cased-v1.1", "BioBERT + Cross-Attn"),
        ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "PubMedBERT + Cross-Attn"),
    ]

    for text_model_name, model_label in cross_models:
        r = run_cross(meta, label_map, config, text_model_name, model_label)
        results.append(r)
        ckpts[model_label] = r["checkpoint"]

    # RAG family
    rag_models = [
        ("distilbert-base-uncased", "DistilBERT", "DistilBERT + Cross-Attn"),
        ("dmis-lab/biobert-base-cased-v1.1", "BioBERT", "BioBERT + Cross-Attn"),
        ("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", "PubMedBERT", "PubMedBERT + Cross-Attn"),
    ]

    for text_model_name, short_name, init_name in rag_models:
        for k in [1, 3, 5]:
            model_label = f"{short_name} + RAG K={k}"
            init_ckpt = ckpts.get(init_name)
            r = run_rag(meta, label_map, config, text_model_name, model_label, k, init_ckpt=init_ckpt)
            results.append(r)

    df = pd.DataFrame(results)
    out_csv = f"{config.DATASET}_results_updated_rag.csv"
    df.to_csv(out_csv, index=False)

    print(f"\nSaved results to {out_csv}")
    print(df[["model_name", "test_loss", "test_auroc", "test_f1", "test_acc"]])
    return df

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ptbxl", "mimic"], default="mimic")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mimic_max_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

# @torch.no_grad()
# def export_mimic_retrieval_examples(
#     model,
#     retrieval_db,
#     dataset,
#     tokenizer,
#     device,
#     config,
#     out_csv="mimic_retrieval_examples.csv",
#     num_examples=3,
# ):
#     model.eval()

#     loader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=custom_collate_fn,
#     )

#     rows = []
#     inv_label_map = {v: k for k, v in dataset.label_map.items()}

#     for i, (ecg, text_enc, labels) in enumerate(loader):
#         ecg = ecg.to(device)
#         text_enc = {k: v.to(device) for k, v in text_enc.items()}
#         labels = labels.to(device)

#         q_ecg = model.encode_ecg(ecg).mean(dim=1)
#         q_txt = model.encode_text_vector(text_enc)

#         retrieved_reports = retrieval_db.retrieve(q_ecg, q_txt, k=3, alpha=0.6)[0]
#         retrieved_text_encs = prepare_retrieved_text_batch(
#             [retrieved_reports], tokenizer, device, config.TEXT_MAX_LEN
#         )

#         logits = model(ecg, text_enc, retrieved_text_encs)
#         probs = torch.softmax(logits, dim=1)

#         pred_idx = probs.argmax(dim=1).item()
#         true_idx = labels.item()

#         meta_row = dataset.meta.iloc[i]

#         rows.append({
#             "sample_idx": i,
#             "filename_lr": meta_row["filename_lr"],
#             "true_label": inv_label_map[true_idx],
#             "pred_label": inv_label_map[pred_idx],
#             "orig_report": meta_row["report"],
#             "retrieved_1": retrieved_reports[0] if len(retrieved_reports) > 0 else "",
#             "retrieved_2": retrieved_reports[1] if len(retrieved_reports) > 1 else "",
#             "retrieved_3": retrieved_reports[2] if len(retrieved_reports) > 2 else "",
#         })

#         if len(rows) >= num_examples:
#             break

#     df = pd.DataFrame(rows)
#     df.to_csv(out_csv, index=False)
#     print(f"Saved retrieval examples to {out_csv}")
#     print(df)
#     return df

@torch.no_grad()
def export_mimic_retrieval_examples(
    model,
    retrieval_db,
    dataset,
    tokenizer,
    device,
    config,
    out_csv="mimic_retrieval_examples.csv",
    num_examples=3,
):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn,
    )

    rows = []
    inv_label_map = {v: k for k, v in dataset.label_map.items()}

    for i, (ecg, text_enc, labels) in enumerate(loader):
        ecg = ecg.to(device)
        text_enc = {k: v.to(device) for k, v in text_enc.items()}
        labels = labels.to(device)

        q_ecg = model.encode_ecg(ecg).mean(dim=1)
        q_txt = model.encode_text_vector(text_enc)

        retrieved_reports = retrieval_db.retrieve(q_ecg, q_txt, k=3, alpha=0.6)[0]
        retrieved_text_encs = prepare_retrieved_text_batch(
            [retrieved_reports], tokenizer, device, config.TEXT_MAX_LEN
        )

        logits = model(ecg, text_enc, retrieved_text_encs)
        probs = torch.softmax(logits, dim=1)

        pred_idx = probs.argmax(dim=1).item()
        true_idx = labels.item()
        meta_row = dataset.meta.iloc[i]

        rows.append({
            "sample_idx": i,
            "filename_lr": meta_row["filename_lr"],
            "true_label": inv_label_map[true_idx],
            "pred_label": inv_label_map[pred_idx],
            "orig_report": meta_row["report"],
            "retrieved_1": retrieved_reports[0] if len(retrieved_reports) > 0 else "",
            "retrieved_2": retrieved_reports[1] if len(retrieved_reports) > 1 else "",
            "retrieved_3": retrieved_reports[2] if len(retrieved_reports) > 2 else "",
        })

        if len(rows) >= num_examples:
            break

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved retrieval examples to {out_csv}")
    print(df)
    return df

import matplotlib.pyplot as plt
import numpy as np

def plot_ecg_cam(ecg_tensor, cam, lead_idx=0, title="Grad-CAM", out_path=None):
    ecg = ecg_tensor[lead_idx].cpu().numpy()
    x = np.arange(len(ecg))

    cam_resized = np.interp(x, np.linspace(0, len(ecg)-1, len(cam)), cam)

    plt.figure(figsize=(12, 3))
    plt.plot(x, ecg, linewidth=1)
    plt.imshow(
        cam_resized[np.newaxis, :],
        aspect="auto",
        extent=[0, len(ecg), ecg.min(), ecg.max()],
        alpha=0.3
    )
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(f"Lead {lead_idx}")
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.show()

# def main():
#     args = parse_args()
#     cfg = Config(
#         DATASET=args.dataset,
#         DATA_DIR=args.data_dir,
#         NUM_EPOCHS=args.epochs,
#         BATCH_SIZE=args.batch_size,
#         MIMIC_MAX_SAMPLES=args.mimic_max_samples,
#         SEED=args.seed,
#     )
#     set_all_seeds(cfg.SEED)

#     print("Configuration:")
#     print(cfg)

#     if cfg.DATASET == "ptbxl":
#         meta, label_map = load_ptbxl_metadata(cfg)
#     else:
#         meta, label_map = build_mimic_metadata_from_csv(cfg)
    
#     run_all(meta, label_map, cfg)

# def main():
#     args = parse_args()
#     cfg = Config(
#         DATASET=args.dataset,
#         DATA_DIR=args.data_dir,
#         NUM_EPOCHS=args.epochs,
#         BATCH_SIZE=args.batch_size,
#         MIMIC_MAX_SAMPLES=args.mimic_max_samples,
#         SEED=args.seed,
#     )
#     set_all_seeds(cfg.SEED)

#     print("Configuration:")
#     print(cfg)

#     if cfg.DATASET == "ptbxl":
#         meta, label_map = load_ptbxl_metadata(cfg)
#     else:
#         meta, label_map = build_mimic_metadata_from_csv(cfg)

#     # tokenizer
#     tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

#     # datasets / loaders
#     train_meta, train_ds, train_loader, val_loader, test_loader = build_dataset_and_loaders(
#         meta, label_map, cfg, tokenizer=tok, use_text=True
#     )

#     test_ds = test_loader.dataset

#     # retrieval index loader
#     index_loader = DataLoader(
#         train_ds,
#         batch_size=cfg.BATCH_SIZE,
#         shuffle=False,
#         num_workers=0,
#         collate_fn=custom_collate_fn
#     )

#     # -------------------------
#     # load best RAG model
#     # -------------------------
#     rag_model = TwoStageRAGModel(
#         num_classes=len(label_map),
#         text_model_name="dmis-lab/biobert-base-cased-v1.1",
#         config=cfg,
#         k_retrieve=3,   # because you are loading K=3 checkpoint
#     ).to(cfg.DEVICE)

#     rag_ckpt = "/home/asatsan2/Projects/ECG/BioBERT_+_RAG_K=3_best.pt"
#     rag_model.load_state_dict(torch.load(rag_ckpt, map_location=cfg.DEVICE))
#     rag_model.eval()
#     print("Loaded RAG model from:", rag_ckpt)

#     # retrieval DB
#     retrieval_db = build_retrieval_database(
#         rag_model,
#         train_ds,
#         index_loader,
#         cfg.DEVICE
#     )
#     print("Retrieval DB ready")

#     # -------------------------
#     # export retrieval examples
#     # -------------------------
#     retrieval_df = export_mimic_retrieval_examples(
#         model=rag_model,
#         retrieval_db=retrieval_db,
#         dataset=test_ds,
#         tokenizer=tok,
#         device=cfg.DEVICE,
#         config=cfg,
#         out_csv="mimic_retrieval_examples.csv",
#         num_examples=3,
#     )

#     print(retrieval_df)

#     # -------------------------
#     # load best Cross-Attn model
#     # -------------------------
#     cross_model = BioBERTCrossAttn(
#         num_classes=len(label_map),
#         text_model_name="dmis-lab/biobert-base-cased-v1.1",
#         config=cfg,
#     ).to(cfg.DEVICE)

#     cross_ckpt = "/home/asatsan2/Projects/ECG/ECG_Only_best.pt"
#     cross_model.load_state_dict(torch.load(cross_ckpt, map_location=cfg.DEVICE))
#     cross_model.eval()
#     print("Loaded Cross-Attn model from:", cross_ckpt)

#     # -------------------------
#     # one Grad-CAM example
#     # -------------------------
#     sample_ecg, sample_text, sample_label = test_ds[0]
#     sample_ecg = sample_ecg.unsqueeze(0).to(cfg.DEVICE)
#     sample_text = {k: v.unsqueeze(0).to(cfg.DEVICE) for k, v in sample_text.items()}

#     print(cross_model.ecg_encoder.conv_layers)
#     cam_helper = ECGGradCAM(cross_model, cross_model.ecg_encoder.conv_layers[8])
#     cam, pred_class = cam_helper.generate(sample_ecg, sample_text)
#     cam_helper.remove()

#     print("Pred class:", pred_class)
#     print("CAM shape:", cam.shape)

#     plot_ecg_cam(
#         sample_ecg[0],
#         cam,
#         lead_idx=0,
#         title="ECG Only Grad-CAM",
#         out_path="ecg_gradcam.png"
#     )
    
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

    tok = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

    train_meta, train_ds, train_loader, val_loader, test_loader = build_dataset_and_loaders(
        meta, label_map, cfg, tokenizer=tok, use_text=True
    )
    test_ds = test_loader.dataset

    index_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=custom_collate_fn
    )

    # -------------------------
    # Load ECG-only model
    # -------------------------
    ecg_model = ECGOnlyModel(
        num_classes=len(label_map),
        config=cfg
    ).to(cfg.DEVICE)

    ecg_ckpt = "/home/asatsan2/Projects/ECG/ECG_Only_best.pt"
    ecg_model.load_state_dict(torch.load(ecg_ckpt, map_location=cfg.DEVICE))
    ecg_model.eval()
    print("Loaded ECG-only model from:", ecg_ckpt)

    # -------------------------
    # Load RAG K=3 model
    # -------------------------
    rag_model = TwoStageRAGModel(
        num_classes=len(label_map),
        text_model_name="dmis-lab/biobert-base-cased-v1.1",
        config=cfg,
        k_retrieve=3,
    ).to(cfg.DEVICE)

    rag_ckpt = "/home/asatsan2/Projects/ECG/BioBERT_+_RAG_K=3_best.pt"
    rag_model.load_state_dict(torch.load(rag_ckpt, map_location=cfg.DEVICE))
    rag_model.eval()
    print("Loaded RAG model from:", rag_ckpt)

    # Retrieval DB
    retrieval_db = build_retrieval_database(
        rag_model,
        train_ds,
        index_loader,
        cfg.DEVICE
    )
    print("Retrieval DB ready")

    # Optional retrieval export for paper qualitative table
    retrieval_df = export_mimic_retrieval_examples(
        model=rag_model,
        retrieval_db=retrieval_db,
        dataset=test_ds,
        tokenizer=tok,
        device=cfg.DEVICE,
        config=cfg,
        out_csv="mimic_retrieval_examples.csv",
        num_examples=3,
    )


    # ===========================================
    inv_label_map = {v: k for k, v in label_map.items()}

    sample_idx = 0   # change if you want a better-looking sample
    lead_idx = 0

    sample_ecg, sample_text, sample_label = test_ds[sample_idx]
    true_label = inv_label_map[sample_label.item()]

    sample_ecg_b = sample_ecg.unsqueeze(0).to(cfg.DEVICE)
    sample_text_b = {k: v.unsqueeze(0).to(cfg.DEVICE) for k, v in sample_text.items()}

    with torch.no_grad():
        q_ecg = rag_model.encode_ecg(sample_ecg_b).mean(dim=1)
        q_txt = rag_model.encode_text_vector(sample_text_b)
        retrieved_reports = retrieval_db.retrieve(q_ecg, q_txt, k=3, alpha=0.6)[0]
        retrieved_text_encs = prepare_retrieved_text_batch(
            [retrieved_reports], tok, cfg.DEVICE, cfg.TEXT_MAX_LEN
        )

    with torch.no_grad():
        ecg_logits = ecg_model(sample_ecg_b, None)
        ecg_pred = inv_label_map[ecg_logits.argmax(dim=1).item()]

        rag_logits = rag_model(sample_ecg_b, sample_text_b, retrieved_text_encs)
        rag_pred = inv_label_map[rag_logits.argmax(dim=1).item()]

    # target layer = last Conv1d
    ecg_target_layer = ecg_model.ecg_encoder.conv_layers[8]
    rag_target_layer = rag_model.ecg_encoder.conv_layers[8]

    # # ECG-only Grad-CAM
    # ecg_gc = ECGGradCAM(ecg_model, ecg_target_layer)
    # ecg_cam, _ = ecg_gc.generate(sample_ecg_b, None)
    # ecg_gc.remove()

    # ECG-only Score-CAM
    ecg_sc = ECGScoreCAM(ecg_model, ecg_target_layer, max_channels=32)
    with torch.no_grad():
        _ = ecg_model(sample_ecg_b, None)
    ecg_scorecam, _ = ecg_sc.generate(sample_ecg_b, None)
    ecg_sc.remove()

    # RAG Grad-CAM
    rag_gc = ECGGradCAM(rag_model, rag_target_layer)
    rag_gradcam, _ = rag_gc.generate(sample_ecg_b, sample_text_b, retrieved_text_encs)
    rag_gc.remove()

    # RAG Score-CAM
    rag_sc = ECGScoreCAM(rag_model, rag_target_layer, max_channels=32)
    with torch.no_grad():
        _ = rag_model(sample_ecg_b, sample_text_b, retrieved_text_encs)
    rag_scorecam, _ = rag_sc.generate(sample_ecg_b, sample_text_b, retrieved_text_encs)
    rag_sc.remove()

    ecg_1d = sample_ecg[lead_idx].cpu().numpy()

    # fig, axes = plt.subplots(4, 1, figsize=(14, 12), constrained_layout=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)

    plot_plain_ecg(
        axes[0],
        ecg_1d,
        title="Original ECG",
        true_label=true_label
    )

    overlay_cam_on_ecg(
        axes[1],
        ecg_1d,
        ecg_scorecam,
        title="ECG-only Score-CAM",
        pred_label=ecg_pred,
        true_label=true_label
    )

    overlay_cam_on_ecg(
        axes[2],
        ecg_1d,
        rag_scorecam,
        title="BioBERT + RAG K=3 Score-CAM",
        pred_label=rag_pred,
        true_label=true_label
    )

    plt.savefig("mimic_explainability_panel_scorecam.png", dpi=300, bbox_inches="tight")
    plt.show()

    # plt.savefig("mimic_explainability_panel_new.png", dpi=300, bbox_inches="tight")
    # plt.show()

    print("Saved final figure to mimic_explainability_panelnew.png")
    print("Retrieved reports for this sample:")
    for i, rr in enumerate(retrieved_reports, 1):
        print(f"[{i}] {rr}")


if __name__ == "__main__":
    main()


"""

python /home/asatsan2/Projects/ECG/updatd_pipe.py --dataset mimic --data_dir /home/asatsan2/Projects/ECG/mimic_ecg --epochs 30 --batch_size 16 --mimic_max_samples 1000

mkdir -p /data/asatsan2/ECG_RAG_WORK/ptbxl 
cd /data/asatsan2/ECG_RAG_WORK/ptbxl 

wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.3/

"""