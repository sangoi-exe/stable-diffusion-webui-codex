Date: 2025-10-18
Task: Summarize state_dict 'Missing/Unexpected' output to avoid massive dumps

Changes
- backend/state_dict.py: replace printing full key lists with counts only; include sample (first 10) at DEBUG level.
- Applies to both default load_state_dict and safe_load_state_dict.

Reason
- Avoid spamming console with hundreds/thousands of keys while preserving actionable diagnostics via DEBUG and trace events.
