Task: Boot WebUI and verify it loads under sandbox

When: 2025-10-17

Env
- Python: ~/.venv -> 3.12.10 (validated)
- Torch in sandbox: 2.6.0+cu124 (CPU mode for boot test)
- Note: User host shows Torch 2.9.0+cu128 and CUDA OK

Steps
1) Reinstall project dependencies (fresh venv):
   - `python -m pip install -U pip setuptools wheel`
   - `pip install -r requirements_versions.txt` (torch/vision/audio excluded)
   - `pip install sentencepiece`
   - `pip install torch torchvision`
2) Launch via launcher with flags to avoid GPU calls and extension installers:
   - `python launch.py --skip-prepare-environment --skip-install --skip-python-version-check --always-cpu --skip-torch-cuda-test --listen --port 7860`

Observed Output (highlights)
- "Launching Web UI with arguments: ..."
- "Total VRAM 32046 MB, total RAM 32046 MB"
- "Device: cpu"
- "* Running on local URL:  http://0.0.0.0:7860"
- "Startup time: 14.3s (... gradio launch: 1.5s)"

Outcome
- WebUI booted successfully (CPU mode, sandbox). Gradio served on :7860.
- GPU blocked in sandbox (expected). On host WSL2, remove `--always-cpu --skip-torch-cuda-test` to use CUDA.

Notes
- Launcher’s `--skip-prepare-environment` triggers prepare step in this fork (inverted logic). Kept intentionally to ensure repo clones.
