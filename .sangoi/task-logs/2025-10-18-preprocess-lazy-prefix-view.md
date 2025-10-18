Date: 2025-10-18
Task: Avoid materializing full state_dict on CPU by using a lazy key-prefix view

Problem
- preprocess_state_dict rebuilt the entire mapping with a new prefix, forcing all tensors to materialize on CPU (risk on Windows, high RAM). Crash occurred around build_unet_vae_start.

Change
- Added `KeyPrefixView` (backend/utils.py): Mapping view that exposes base keys with a prefix without loading values.
- preprocess_state_dict now returns `KeyPrefixView` instead of rebuilding a dict when prefix is missing.

Impact
- Keeps safetensors lazy; filtering/loading happens later on demand. Reduces CPU churn and potential native crashes in torch_cpu.dll during preprocessing.
