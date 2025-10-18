Date: 2025-10-18
Task: Fix trace event on split_state_dict (avoid .keys() on path)

Fix
- modules/sd_models.py: replace `split_state_dict_done` (incorrect on str) with `split_state_dict_start` logging only the path and add count.
- backend/loader.py: emit `split_state_dict_done` after actual split with parts and sizes.

Reason
- Prevent AttributeError ('str' has no attribute 'keys') during trace when `state_dict` is still the checkpoint path.

