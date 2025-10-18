Date: 2025-10-18
Task: Implement GPU-first policy with explicit swap policy/method flags

Flags
- --swap-policy {never,cpu,shared} (env: CODEX_SWAP_POLICY)
- --swap-method {blocked,async} (env: CODEX_SWAP_METHOD)
- --gpu-prefer-construct (env: CODEX_GPU_PREFER_CONSTRUCT)

Changes
- backend/args.py: add flags + env overrides.
- backend/stream.py: streams controlled by swap-method; async attempts streams even on Windows with safe fallback.
- backend/memory_management.py: offload devices honor swap-policy (never => GPU; cpu/shared => CPU). Log policy summary at init.
- backend/loader.py: construct UNet on GPU when preferred; catch OOM and fallback only if policy allows; keep no-bf16/fp16 on CPU guard.
- run-webui.bat: inline CODEX_* vars + echo at startup.

Notes
- Default launcher sets CODEX_SWAP_POLICY=never and prefer_construct=1 for GPU-first behavior; users can change.
