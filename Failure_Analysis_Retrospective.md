# 📉 Failure Analysis & Retrospective

Building NeuroFusion was not a straight path. We faced significant hardware constraints that forced us to pivot multiple times. This document records what failed and why.

## 1. The "Full 3D" Dream (FAIL)
### What we tried
We initially attempted to feed full 3D MRI volumes (240x240x155) into a 3D U-Net.
### Why it failed
*   **VRAM Explosion**: A single 3D volume with gradients requires ~12GB+ VRAM. Your GTX 1650 has 4GB.
*   **Result**: Immediate `CUDA Out of Memory` errors.

## 2. 3D Patch-Based Training (PARTIAL FAIL)
### What we tried
Instead of the whole brain, we chopped it into small cubes (64x64x64) to train on patches.
### Why it failed
*   **Complexity**: Reassembling patches during inference is math-heavy and prone to "checkerboard" artifacts.
*   **Inefficiency**: It was still very slow to train on CPU (which we fell back to), taking ~45 minutes per epoch.

## 3. CPU Training (FAIL)
### What we tried
We tried ensuring the code ran on CPU to bypass VRAM limits.
### Why it failed
*   **Time**: Estimated training time was ~40 hours for 50 epochs. This feedback loop is too slow for development.

## 4. The "Binary Only" Simplified Model (PIVOT)
### What we tried
We removed the segmentation head and just asked the AI: "Is there a tumor? Yes/No."
### Why it was temporary
*   **Success**: It trained fast on GPU! (1.5 mins/epoch).
*   **Limitation**: "Yes/No" isn't clinically useful. Doctors need to see *where* the tumor is.
*   **Lesson**: It proved our *data pipeline* was fast enough, giving us confidence to try adding segmentation back.

## 5. The Solution: 2D Slice Training (SUCCESS) 🏆
### What worked
We realized we don't need 3D convolution for every task. By looking at the brain one "slice" at a time (like pages in a book):
1.  **Memory**: VRAM usage dropped to <2GB.
2.  **Speed**: Training hit real-time speeds.
3.  **Result**: We restored full segmentation capabilities without melting the GPU.
