Task: Autonomous smoke inference test via API-only server

When: 2025-10-17

Server
- Command: `python launch.py --skip-prepare-environment --skip-install --skip-python-version-check --nowebui --api --api-server-stop --listen --port 7861 --ckpt-dir /mnt/f/stable-diffusion-webui-forge/models/Stable-diffusion --always-cpu --skip-torch-cuda-test`
- Result: API ready on :7861; models listed; using first checkpoint `cyberrealisticPony_v7.safetensors`

Smoke Request
- Endpoint: `POST /sdapi/v1/txt2img`
- Payload: `{prompt:"smoke test", steps:1, width:64, height:64, send_images:false, save_images:false}`
- Response: 200 OK; info includes: `Model: cyberrealisticPony_v7`, `Steps: 1`, `Sampler: Euler`, `Version: f2.0.1v1.10.1-previous-881-ga3c0490d`

Shutdown
- Endpoint: `POST /sdapi/v1/server-stop`; cleaned background process.

Artifacts
- Script added: `scripts/smoke_infer.sh` (parametrizable via env: CKPT_DIR, PORT, PROMPT, STEPS, SIZE, MODEL_TITLE)
- Boot log: `.webui-api.log`
