Date: 2025-10-18
Task: Add Torch trace and guard against bf16 on CPU during construction

Changes
- backend/torch_trace.py: optional tracing (env: CODEX_TRACE_TORCH=1, CODEX_TRACE_LIMIT). Patches Module.to during trace section; logs compact move events and stages.
- modules/sd_models.py: wraps `forge_loader()` call with `trace_section("forge_loader")` and emits split_state_dict event.
- backend/loader.py: emits trace events for component construction; enforces no bf16/fp16 on CPU at construction time (overrides to fp32 or builds on GPU); logs load_state_dict start.
- backend/state_dict.py: emits `load_state_dict_done` events and DEBUG counts.
- backend/patcher/base.py: trace events for patch `.to()` moves.
- backend/memory_management.py: text_encoder_dtype returns fp32 on CPU (previously fp16), matching the “no bf16/fp16 on CPU” policy.
- run-webui.bat: echoes TRACE env when set.

Usage
- To enable trace: set `CODEX_TRACE_TORCH=1` (optional `CODEX_TRACE_LIMIT=1000`). Check logs under `backend.trace` and component logs.

