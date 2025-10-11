Backend Services Overview
=========================

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

