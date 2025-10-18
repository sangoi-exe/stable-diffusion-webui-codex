Task: Upgrade PyTorch to 2.9.0 (+cu128) using plain `pip install torch torchvision` in ~/.venv

When: 2025-10-17

Steps
1) `pip install --upgrade --upgrade-strategy eager torch torchvision` (no index-url)
2) Validate versions: `python -c 'import torch; print(torch.__version__, torch.version.cuda)'`

Results
- torch==2.9.0+cu128, torchvision==0.24.0 installed
- CUDA runtime reported by torch: 12.8
- In sandbox: `torch.cuda.is_available() == False` (expected; GPU not exposed)

Notes
- Pillow auto-upgraded to 12.0.0; gradio 5.49.1 pins `<12.0`. On host, if UI errors appear, pin pillow `==11.3.0` again.
