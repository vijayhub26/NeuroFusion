# 📦 Deployment Guide: Running NeuroFusion on a New PC

Yes! You can run this on any computer without retraining. You just need to move the "Brains" (Model Weights) along with the code.

## 1. Transfer the Code (Easy)
On the new computer:
```bash
git clone https://github.com/vijayhub26/NeuroFusion.git
cd NeuroFusion
```

## 2. Transfer the Brains (Critical) 🧠
The trained model weights (`.ckpt` files) are **NOT on GitHub** (they are too big). You must copy them manually.

### Option A: USB Drive / Google Drive
1.  Go to `c:\antigravity works\logs\unified_segmentation\version_0\checkpoints\` on your current PC.
2.  Copy the `.ckpt` file (e.g., `unified-epoch=49...ckpt`).
3.  On the **New PC**, create this folder structure: `logs/unified_segmentation/version_0/checkpoints/` inside the project folder.
4.  Paste the `.ckpt` file there.

### Option B: Hugging Face (Recommended)
Upload the model to Hugging Face Hub so you can download it anywhere.

## 3. Install Dependencies (Fix Import Issues)
To avoid "Import Errors", run this on the new PC:

```bash
# 1. Install Python (3.10 recommended)

# 2. Install Libraries
pip install -r requirements.txt
```

## 4. Run It
```bash
python app.py
```

---
**Summary**:
-   **Code**: Syncs via GitHub.
-   **Model**: Must be copied manually (USB/Cloud).
-   **Dependencies**: Fixed by `requirements.txt`.
