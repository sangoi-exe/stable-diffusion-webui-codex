Date: 2025-10-18
Task: Ensure DEBUG logs appear even when third-party handlers set INFO

Problem
- User reported no DEBUG output despite default DEBUG. Likely due to later logging setup (uvicorn/gradio) attaching handlers at INFO and filtering root.

Fix
- backend/logging_utils.py: add dedicated StreamHandler for 'backend' logger hierarchy, set to env level (default DEBUG), and set propagate=False to avoid filtering and duplicates.
- backend/engines/base.py: set engine logger to module-qualified name under 'backend.*' so it is captured by the dedicated handler (e.g., backend.engines.sdxl.engine.SDXLEngine).

Expected Result
- DEBUG messages from SDXL engine and adapters should appear in console even if root handlers are INFO.

