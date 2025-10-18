Date: 2025-10-18
Task: Audit and fix memory management module

Findings
- Misuse of `torch.cuda.get_device_properties("cuda")` (string) in `should_use_fp16` could mis-query device; fixed to use actual `torch.device`/current device.
- Negative torch-free calculations possible (`reserved - active`) → clamped to >= 0 for CUDA/XPU.
- `unload_all_models()` printed confusing message even when no models were tracked; now logs and clears caches early without noisy output.
- Added structured logs under `backend.memory` for visibility (enter/exit in `free_memory`, initial device/VRAM state).

Changes
- backend/memory_management.py
  - Add logger `backend.memory`; init state logs.
  - `free_memory`: better logging, early return if no tracked models on free_all; debug loop stats.
  - Clamp negative values in `get_total_memory`/`get_free_memory`.
  - Fix device props query in `should_use_fp16` and robust default in `should_use_bf16`.
  - `unload_all_models` guards device resolution and logs tracked count.

Expected Impact
- Clearer unload behavior; fewer confusing messages; robust dtype/device checks; avoids potential errors when querying device properties; more actionable logs for VRAM.

