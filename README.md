# RAG-Fusion for Multimodal ECG Classification

This repository contains the code for a retrieval-augmented multimodal ECG classification framework that combines ECG waveforms with biomedical text using gated cross-attention and retrieval-based fusion.

The proposed framework integrates:

* ECG signal representations from 12-lead recordings
* Transformer-based biomedical language models
* Retrieval-augmented generation (RAG)
* Cross-attention fusion
* Activation-based interpretability analysis

The framework is evaluated on both PTB-XL and MIMIC-IV-ECG datasets.

## Overview

The proposed pipeline contains four main stages:

1. ECG encoding using a 1D CNN encoder
2. Text encoding using DistilBERT, BioBERT, or PubMedBERT
3. Retrieval of top-K relevant reports using FAISS
4. Gated cross-attention fusion for final classification

[Add architecture figure here]

## Datasets

### PTB-XL

Publicly available 12-lead ECG dataset containing diagnostic labels.

### MIMIC-IV-ECG

Large-scale clinical ECG dataset paired with free-text reports.

## Models Evaluated

* ECG Only
* Text Only DistilBERT
* Text Only BioBERT
* Text Only PubMedBERT
* DistilBERT + Cross-Attn
* BioBERT + Cross-Attn
* PubMedBERT + Cross-Attn
* DistilBERT + RAG
* BioBERT + RAG
* PubMedBERT + RAG

## Key Results

### MIMIC-IV-ECG

Best performance was achieved by BioBERT + RAG (K=3):

* AUROC: 0.9906
* Macro F1: 0.9179
* Accuracy: 0.9211

### PTB-XL

BioBERT + RAG and PubMedBERT + RAG variants achieved the strongest overall performance.

## Installation

```bash
git clone https://github.com/anonymnett/ECG.git
cd ECG

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Running

Example:

```bash
python pipeline.py
```

Or run the notebook:

```bash
jupyter notebook Mimic.ipynb
```

## Interpretability

The repository includes Grad-CAM visualizations for ECG signals to highlight diagnostically relevant waveform regions.

## Citation

If you use this repository, please cite our paper once published.

## License

This project is released under the MIT License.
