Dependencies Policy
===================

Pins & Upgrades
- Use `tools/update_requirements.py` to refresh pins while excluding conflict-prone libs.
- Default exclusions (kept or annotated): torch, torchvision, torchaudio, fastapi, gradio, pydantic, protobuf, httpx, httpcore, open-clip-torch, onnxruntime-gpu.
- `--drop-excluded` to remove excluded pins (pip won’t touch local builds).

Manual Installer (TODO)
- Add a no-deps installer runner that installs each pinned package with `pip install --no-deps`, prints a summary, supports `--subset`, and never installs torch/vision/audio.

Known Good Combos
- pydantic 2.11.x with gradio 5.49.x & fastapi 0.118.x.
- protobuf < 5 with onnxruntime-gpu 1.23.x.

