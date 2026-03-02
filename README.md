# 🧠 NeuroFusion — Brain Tumor Detection & Segmentation

A **Unified Generative AI Model** that performs:
1. **Missing Modality Synthesis** — GAN-based generation of missing MRI scans (e.g., T1ce)
2. **Tumor Segmentation** — U-Net that precisely identifies and masks brain tumors

> **Dice Score: 0.83** | Inference: Real-time on GPU

---

## 📋 Table of Contents
- [Requirements](#requirements)
- [Step 1 — Clone the Repository](#step-1--clone-the-repository)
- [Step 2 — Create a Virtual Environment](#step-2--create-a-virtual-environment)
- [Step 3 — Install Dependencies](#step-3--install-dependencies)
- [Step 4 — Set Up the Logs Folder (Pre-trained Model)](#step-4--set-up-the-logs-folder-pre-trained-model)
- [Step 5 — Run the Web Interface](#step-5--run-the-web-interface)
- [Step 6 — Training from Scratch (Optional)](#step-6--training-from-scratch-optional)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## ✅ Requirements

Make sure the target system has the following installed before starting:

| Tool | Version | Download |
|---|---|---|
| Python | 3.9 or later | [python.org](https://www.python.org/downloads/) |
| Git | Any | [git-scm.com](https://git-scm.com/downloads) |
| NVIDIA GPU | Recommended (~4GB VRAM) | For fast inference & training |
| CUDA Toolkit | Match your GPU driver | [nvidia.com](https://developer.nvidia.com/cuda-downloads) |

> ℹ️ A GPU is **strongly recommended**. CPU-only inference is possible but will be very slow.

---

## Step 1 — Clone the Repository

Open a terminal (Command Prompt, PowerShell, or Bash) and run:

```bash
git clone https://github.com/vijayhub26/NeuroFusion.git
cd NeuroFusion
```

---

## Step 2 — Create a Virtual Environment

A virtual environment keeps all dependencies isolated from your system Python.

```bash
python -m venv venv
```

**Activate it:**

- **Windows (Command Prompt / PowerShell):**
  ```bash
  venv\Scripts\activate
  ```
- **Mac / Linux:**
  ```bash
  source venv/bin/activate
  ```

You will see `(venv)` appear at the beginning of your terminal prompt — this means it is active.

---

## Step 3 — Install Dependencies

With the virtual environment active, install all required packages:

```bash
pip install -r requirements.txt
```

### 🚀 GPU Users — Install PyTorch with CUDA First

If you have an NVIDIA GPU, install the GPU-accelerated version of PyTorch **before** running the above command. Go to [pytorch.org/get-started](https://pytorch.org/get-started/locally/) and select your OS and CUDA version. Example for CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

To verify GPU is available after installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Should print `True` if GPU is set up correctly.

---

## Step 4 — Set Up the Logs Folder (Pre-trained Model)

The trained model checkpoints are included in the repository inside the `logs/` folder. After cloning, they will already be in the correct location. Verify the structure looks like this:

```
NeuroFusion/
└── logs/
    ├── unified_segmentation/
    │   └── version_0/
    │       └── checkpoints/
    │           ├── unified-epoch=epoch=45-dice=val_dice_score=0.82.ckpt
    │           ├── unified-epoch=epoch=48-dice=val_dice_score=0.82.ckpt
    │           └── unified-epoch=epoch=49-dice=val_dice_score=0.83.ckpt  ← BEST MODEL
    └── simple_tumor_detection/
        └── version_2/
            └── checkpoints/
                └── simple-tumor-epoch=49-val_acc=0.85.ckpt
```

### ⚠️ If the `logs/` folder is missing or empty after cloning

This can happen if git did not download the checkpoint files correctly. Fix it by running:

```bash
git lfs pull
```

> If `git lfs` is not installed: [git-lfs.com](https://git-lfs.github.com/)

If that does not work, **manually create the folder structure** and place the `.ckpt` file you downloaded:

**Windows:**
```bash
mkdir logs\unified_segmentation\version_0\checkpoints
```

**Mac / Linux:**
```bash
mkdir -p logs/unified_segmentation/version_0/checkpoints
```

Then copy the `.ckpt` file into that `checkpoints/` folder.

---

## Step 5 — Run the Web Interface

With the checkpoints in place, start the Gradio web UI:

```bash
python app.py
```

Then open your browser and go to:

```
http://localhost:7860
```

### What you can do in the UI:
- 📂 Upload your own MRI scan files
- 🔀 Randomly load a patient from the dataset
- 🧪 Simulate missing MRI modalities to test the synthesis model
- 🧠 View the tumor segmentation mask overlaid on the scan

---

## Step 6 — Training from Scratch (Optional)

> ⚠️ Only do this if you want to retrain the model. You already have a trained model from Step 4.

### 6a. Download the BraTS 2020 Dataset

1. Go to [Kaggle BraTS 2020](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)
2. Download and extract the dataset (~10GB)
3. Note the path to the extracted folder

### 6b. Configure the Dataset Path

Open `config.py` and update `DATA_PATH` to point to your dataset:

```python
# config.py
DATA_PATH = "C:/path/to/BraTS2020_TrainingData"  # Windows
# DATA_PATH = "/home/user/BraTS2020_TrainingData"  # Mac/Linux
```

### 6c. Run Training

```bash
python train_full.py
```

- **Hardware required:** ~4GB VRAM GPU
- **Time:** Several hours depending on hardware
- **Logs:** Saved to `logs/unified_segmentation/`

Monitor training progress in real-time with TensorBoard:
```bash
tensorboard --logdir logs/unified_segmentation
```
Then open `http://localhost:6006` in your browser.

---

## 📂 Project Structure

```
NeuroFusion/
├── app.py                    # 🖥️  Main Gradio web interface
├── train_full.py             # 🏋️  Full training script (Segmentation + Synthesis)
├── train_simple.py           # 🏋️  Simple detector training script
├── config.py                 # ⚙️  All configuration (paths, hyperparams, epochs)
├── requirements.txt          # 📦  Python dependencies
│
├── models/
│   ├── unified_detector.py   # 🧩  Unified Model architecture
│   └── synthesis_2d.py       # 🧩  2D GAN components for synthesis
│
├── losses/
│   └── unified_loss.py       # 📉  Combined Dice + L1 Loss
│
├── data/
│   └── dataset_2d.py         # 📊  BraTS 2020 data loading logic
│
├── utils/                    # 🔧  Helper utilities
│
└── logs/                     # 💾  Saved model checkpoints (included in repo)
    ├── unified_segmentation/ #     Main model — Dice Score: 0.83
    └── simple_tumor_detection/ #  Simple detector — Accuracy: 0.85
```

---

## 📊 Performance

| Model | Metric | Score |
|---|---|---|
| Unified Segmentation (U-Net + GAN) | Dice Score | **0.83** |
| Simple Tumor Detector | Accuracy | **0.85** |

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Ensure venv is activated and run `pip install -r requirements.txt` again |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in `config.py` (try `2` or `1`) |
| `No module named torch` | Reinstall PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) |
| `FileNotFoundError` for checkpoint | Check `logs/` folder structure matches Step 4 exactly |
| Port 7860 already in use | Edit `app.py` and change `server_port=7860` to `7861` |
| `True` not printed for CUDA check | Your CUDA / GPU driver is not set up — revisit Step 3 |
| Dataset not found during training | Double-check `DATA_PATH` in `config.py` points to correct folder |

---

*Built with PyTorch Lightning · Gradio · BraTS 2020*
*Generated by Antigravity*
