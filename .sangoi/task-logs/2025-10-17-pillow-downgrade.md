Task: Downgrade Pillow to satisfy Gradio pin (<12)

When: 2025-10-17

Actions
- `pip install pillow==11.3.0` in `~/.venv`
- Verified: Pillow 11.3.0; Gradio 5.49.1

Note
- PyTorch remains 2.9.0+cu128, torchvision 0.24.0.
