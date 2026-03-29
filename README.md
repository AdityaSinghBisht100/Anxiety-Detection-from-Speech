# Speech Anxiety Detection

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-F9AB00.svg)

A production-ready pipeline for detecting clinical anxiety and depression from raw speech using a **Dual-Branch Fusion Architecture**. 

This system leverages deep contextual embeddings from **Wav2Vec 2.0 (with LoRA)** alongside clinically validated acoustic markers from **eGeMAPS (openSMILE)**, optimized specifically for small clinical datasets like DAIC-WOZ and DEPAC.

---

##  Architecture Overview

The system uses a late-fusion approach to combine the representational power of large self-supervised models with the explainability of traditional speech features.

```text
Raw Audio (16kHz) ──┬──→ [Wav2Vec 2.0 + LoRA] → [Weighted Layer Aggregation] → [Attention Pooling] → [768-d]
                    │                                                                                    ↓
                    └──→ [eGeMAPS 88 features] → [MLP Projection] ─────────────────────────────────→ [128-d]
                                                                                                         ↓
                                                                                               [Fusion Layer (LayerNorm)]
                                                                                                         ↓
                                                                                               [Classification Head]
                                                                                                         ↓
                                                                                                Anxiety Score (0-1)
```

### Key Innovations (v2 Architecture)
1. **LoRA Fine-Huning**: Wav2Vec 2.0 is trained using Low-Rank Adaptation (r=8), injecting only ~150K trainable parameters to prevent catastrophic forgetting on tiny datasets.
2. **Weighted Layer Aggregation**: Instead of just using the final transformer layer, the model learns a 12-parameter scalar distribution to focus on middle layers (often richer in prosody/emotion signals).
3. **Attention Pooling**: Replaces simple mean pooling. A learned attention mechanism focuses on anxiety-salient sparse frames (e.g., specific pitch spikes or disfluencies).
4. **eGeMAPS Complement**: A lightweight 88-dimensional feature set (jitter, shimmer, formants, HNR) injected during late fusion to ground the deep representations with concrete physiological markers.

---

## 📂 Project Structure

```text
speech_anxiety_detection/
├── configs/
│   ├── model_config.yaml          # Architecture hyperparameters
│   └── training_config.yaml       # Phase 1 & 2 training schedules
├── src/
│   ├── preprocessing/             # audio_utils, VAD, segmentation, eGeMAPS
│   ├── data/                      # PyTorch Datasets (DAIC-WOZ loader), augmentation
│   ├── models/                    # Network definitions (LoRA backbone, pooling, fusion)
│   ├── training/                  # Custom loss (Focal/WeightedBCE), metrics, trainer
│   └── inference/                 # End-to-end predictor, FastAPI server
├── train.py                       # CLI entry point for training
├── predict.py                     # CLI entry point for local inference
├── verify.py                      # Automated test script for CI/CD
└── requirements.txt               # Pinned dependencies
```

---

## Setup & Installation

### Local Setup (Inference & Prototyping)

```bash
git clone <repo-url>
cd speech_anxiety_detection

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Training Workflow (Kaggle / Cloud GPU)

Since clinical datasets are small, full end-to-end training runs smoothly on a single T4 or P100 GPU on platforms like Kaggle.

### 1. Data Preparation
Our dataloader natively supports the **DAIC-WOZ** format. It automatically parses `*_TRANSCRIPT.csv` to strip out the virtual interviewer (Ellie) and only runs inference on participant speech. 

### 2. Multi-Phase Training Strategy
To maximize performance, training is broken into two phases:
*   **Phase 1 (WarmUp):** Train the LoRA adapters on standard Speech Emotion Recognition (SER) datasets (e.g., IEMOCAP) to teach the model to extract paralinguistic features.
*   **Phase 2 (Fine-Tuning):** Transfer the weights and train on the target anxiety dataset (DAIC-WOZ).

Upload the `speech_anxiety_detection/` repository to Kaggle, mount your datasets, and execute:

```bash
# Phase 1: SER warm-up (Optional but recommended)
python train.py --phase phase1 --train_csv /kaggle/input/ser_labels.csv --epochs 15 --device cuda

# Phase 2: Anxiety fine-tuning
python train.py --phase phase2 --data_dir /kaggle/input/daic-woz-dataset --epochs 35 --device cuda
```

### 3. Downloading Checkpoints
The trainer intentionally saves **only** the LoRA weights and the custom classification heads (1–3 MB). You do not need to download the 360MB Wav2Vec 2.0 backbone. 

Download `checkpoints/best_phase2.pt` to your local machine for inference.

---

## Local Inference & API

Once you have a downloaded Kaggle checkpoint, you can run rapid inference locally on a CPU. The model handles all internal slicing for long audio (>10s) and aggregates predictions automatically.

### CLI Usage

```bash
# Predict a single file
python predict.py --audio test_recording.wav --checkpoint checkpoints/best_phase2.pt

# Predict a batch of files in a directory
python predict.py --audio_dir path/to/audios/ --checkpoint checkpoints/best_phase2.pt --output results.json
```

### FastAPI REST Server

The repository comes with a production-ready FastAPI endpoint that exposes the model.

```bash
# Start the server on port 8000
python predict.py --serve --checkpoint checkpoints/best_phase2.pt
```

**Testing the API:**
```bash
curl -X POST http://localhost:8000/predict -F "audio=@test_recording.wav"
```

**Example JSON Response:**
```json
{
    "anxiety_score": 0.8123,
    "label": "anxious",
    "confidence": "high",
    "threshold": 0.5,
    "num_segments": 4,
    "audio_duration_sec": 38.5,
    "processing_time_ms": 312.4,
    "top_acoustic_markers": {
        "pitch_variability": "elevated",
        "jitter": "elevated",
        "hnr": "reduced"
    }
}
```
*(Notice how the API surfaces **acoustic markers** from the eGeMAPS branch to act as clinical evidence for the deep learning score).*

---

## Verification & Testing

To ensure your environment is set up correctly and the tensor shapes match across all model branches, run the verification harness:

```bash
# Tests imports, checks component shapes, and verifies loss computation
python verify.py

# Full test: Downloads Wav2Vec2-base to run a complete forward/backward pass
python verify.py --full
```

## Performance Targets

Based on the implemented architecture, the expected baseline metrics on the DAIC-WOZ dataset are:

| Metric | Target | Literature SOTA (DAIC-WOZ Baseline) |
|--------|--------|-------------------------------------|
| **AUC-ROC** | ≥ 0.70 | 0.68–0.69 |
| **UAR** | ≥ 0.60 | 0.60 |

---
*Developed based on modern computational paralinguistics and parameter-efficient fine-tuning principles.*
