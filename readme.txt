The Short Answer:
1)Code: Just pull from GitHub! (git clone ...)
2)Brains (Model Checkpoints): These are TOO BIG for GitHub. You must copy them manually (USB stick or Google Drive).
File to copy: logs/unified_segmentation/version_0/checkpoints/*.ckpt
3)Dependencies: I created a requirements.txt file. On the new PC, just run:
bash
pip install -r requirements.txt