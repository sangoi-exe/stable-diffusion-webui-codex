Task: Windows launcher script `run-webui.bat`
Date: 2025-10-18

What it does
- Checks for Python in PATH and enforces 3.10.x on Windows.
- Creates/activates `.venv` at repo root if missing.
- Upgrades `pip/setuptools/wheel`.
- Installs `requirements_versions.txt`.
- Verifies core libs via a temporary Python script (`torch`, `torchvision`, `diffusers`, `gradio`, `fastapi`, `huggingface_hub`, `numpy`, `pydantic`).
- Warns if `ffmpeg` is not found.
- Launches `python webui.py` forwarding any CLI args.

Notes
- Does not write global state; uses repo-local venv.
- Exits with non‑zero on failures to prevent half‑configured runs.

