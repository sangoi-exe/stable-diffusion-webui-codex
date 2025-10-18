Title: Memory management audit and fixes
Date: 2025-10-18

Summary
- Audited `backend/memory_management.py` for issues after backend rewrite. Fixed confusing unload message and device/dtype edge cases; added structured logging.

Files touched
- backend/memory_management.py
- .sangoi/task-logs/2025-10-18-memory-management-audit-fixes.md

Key changes
- Logger `backend.memory`; boot logs for VRAM/CPU modes.
- `free_memory`: early return on free_all when no models tracked; debug loop metrics; clearer output.
- Clamp negative torch-free calculations; safer device property queries.
- `unload_all_models`: safe device resolution; logs tracked count.

Commands/validation
- Static checks; no destructive ops. Run normally; observe new DEBUG lines under backend.memory.

Open risks / next steps
- Consider periodic reconciliation of `current_loaded_models` with actual live objects to avoid stale refs in long sessions.
- Optional: add per-model IDs and timings in unload/load for deeper traces.
