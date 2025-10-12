Backend Services Overview
=========================

Project Identity
- This repository is `stable-diffusion-webui-codex`, a fork built on top of Forge (itself a fork of AUTOMATIC1111’s WebUI). The goal is to preserve the A1111 legacy while modernising the backend.
- Maintenance uses OpenAI “codex” as a coding assistant in the development workflow; there is no LLM integrated into the runtime.
- The `legacy/` folder contains a snapshot of Forge’s `main` to serve as a functional pipeline reference during refactors.

Purpose
- Reduce duplication and decouple FastAPI routes from generation internals.
- Create a thin service layer that centralizes common flows and helpers.

Services
- ImageService (backend/services/image_service.py)
  - execute_generation(p_factory, ...): wraps queue lock, state begin/finish, script vs process_images, process_extra_images.
  - Used by txt2img/img2img API to run jobs with identical semantics.

- MediaService (backend/services/media_service.py)
  - decode_image(str): base64 or URL fetch with opts guards.
  - encode_image(PIL.Image): respects current output format and metadata.
  - Used across API endpoints: extras, png-info, progress, interrogate.

- OptionsService (backend/services/options_service.py)
  - get_config/set_config: proxies to modules.sysinfo.
  - get_cmd_flags: returns shared.cmd_opts.

- SamplerService (backend/services/sampler_service.py)
  - resolve(name_or_index, scheduler): maps aliases/indices.
  - ensure_valid_sampler(name): raises HTTPException if unknown.

API Changes (modules/api/api.py)
- API constructs p_factory/prepare_p and delegates to ImageService.
- All image encode/decode moved to MediaService.
- Options and flags routes delegate to OptionsService.
- Sampler resolution/validation uses SamplerService.

Behaviour
- Public responses, fields, and error semantics preserved.
- No change to endpoints or request/response models.

Next
- Optional: ProgressService to centralize task/time metrics and future telemetry.
- Optional: move sampler/scheduler listing to SamplerService.

Tokenizer Assets (Strict + Online Cache)
-------------------------

- Tokenizers are not embedded in `.safetensors`. They must exist in the model directory (or be cached locally from the Hub) as standard files:
  - Preferred: `tokenizer.json` (+ `tokenizer_config.json`)
  - BPE: `vocab.json` + `merges.txt`
  - SentencePiece: `tokenizer.model`
- SDXL requires two tokenizer folders: `tokenizer/` and `tokenizer_2/`.
- Loader behavior (backend/loader.py):
  - Primeiro tenta resolver no disco (pasta do componente e raiz do repositório local).
  - Se faltar e `--disable-online-tokenizer` NÃO estiver habilitado e `huggingface_repo` for conhecido, baixa do Hub apenas os artefatos do tokenizer e copia para o diretório do modelo (cache explícito, sem “magia”).
  - Se ainda assim não existir ou falhar, erro explícito (sem mascarar). Não há fallback silencioso.
- Recommended workflow when you only have a checkpoint `.safetensors`:
  - Download tokenizers (and `config.json`) from the original repo (e.g., via `huggingface_hub.snapshot_download(allow_patterns=["tokenizer.*", "vocab.json", "merges.txt", "tokenizer.model", "tokenizer_config.json", "config.json"])`).
  - Place them under the model directory alongside your weights.
