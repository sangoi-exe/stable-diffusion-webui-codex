Task: Update pip, install WebUI requirements, install PyTorch 2.9 (cu128) via pip in ~/.venv

When: 2025-10-17

Context
- Python: ~/.venv (Python 3.12.3; real venv validated)
- OS: WSL Linux (GPU access not validated inside sandbox)

Commands Run
- Verify venv:
  - `~/.venv/bin/python - <<PY ...` (printed prefix, base_prefix, is_venv True)
- Upgrade tooling:
  - `~/.venv/bin/python -m pip install -U pip setuptools wheel`
- Install repo requirements (filtered to avoid direct torch lines; deps allowed):
  - `mkdir -p .pip-cache .pip-tmp`
  - `TMPDIR="$PWD/.pip-tmp" PIP_CACHE_DIR="$PWD/.pip-cache" ~/.venv/bin/python -m pip install --no-cache-dir --prefer-binary -r .pip-tmp/req-filtered.txt`
- Install torch/vision (as requested, no index-url):
  - `TMPDIR="$PWD/.pip-tmp" PIP_CACHE_DIR="$PWD/.pip-cache" ~/.venv/bin/pip install --no-cache-dir --prefer-binary torch torchvision`
- Validate torch/CUDA:
  - `~/.venv/bin/python - <<PY print(torch.__version__, torch.version.cuda, torch.cuda.is_available())`

Key Results
- pip upgraded to 25.2; setuptools 80.9.0; wheel 0.45.1
- requirements installed successfully from `requirements_versions.txt`
- torch==2.9.0+cu128, torchvision==0.24.0 installed from PyPI
- nvidia CUDA runtime packages resolved: cu12.8 (cublas 12.8.4.1, cudnn 9.10.2.21, nvrtc 12.8.93, etc.)
- Validation in sandbox: `torch.version.cuda == '12.8'`; `torch.cuda.is_available() == False` (expected here due to sandbox/WSL GPU not exposed)

Notes/Risks
- In this environment, CUDA devices are not visible; on host WSL2 with recent NVIDIA driver, `torch.cuda.is_available()` should be True.
- Requirements include `transformers==4.57.0` (yanked upstream). Installation succeeded with wheels; monitor for future breakage.

Next Steps
- On host, verify GPU path: `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name() if torch.cuda.is_available() else '-')"`
- If CUDA not available on host, update Windows NVIDIA driver (WSL) and reboot WSL.
