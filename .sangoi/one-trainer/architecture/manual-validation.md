# Manual Validation Routines (Phase 2 — Engine Stubs)

Goal: run hands-on checks for the new modular engine scaffolding without automated suites.

## Registry wiring
- Command:
  ```bash
  python - <<'PY'
  from backend.core.registry import registry
  from backend.engines import register_default_engines

  register_default_engines(replace=True)
  print("engines:", registry.list())
  for key in registry.list():
      descriptor = registry.get_descriptor(key)
      print(key, descriptor.metadata)
  PY
  ```
- Expectation: keys `sd15`, `sdxl`, `flux` listed with metadata (`family`, `version`).

### Model reload semantics
- Calling the orchestrator again with the same `engine_key` but a different `model_ref` triggers `engine.load()` and reload via Forge parameters before processing.

## Stub behavior
- Command:
  ```bash
  python - <<'PY'
  from backend.core.registry import registry
  from backend.engines import register_default_engines
  from backend.core.engine_interface import TaskType
  from backend.core.requests import Txt2ImgRequest

  register_default_engines(replace=True)
  orchestrator_engine = registry.create('sd15')
  try:
      list(orchestrator_engine.txt2img(Txt2ImgRequest(task=TaskType.TXT2IMG, prompt='hello')))
  except Exception as exc:
      print(type(exc).__name__, exc)
  PY
  ```
- Expectation: explicit `UnsupportedTaskError` referencing engine `sd15`, task `txt2img`, model/options context.

## Logging check
- Ensure application logger level set to INFO captures messages from `DiffusionEngine.load/unload`. For manual runs: watch `.webui-api.log` or console output for entries like `Loading engine sd15 ...` / `Unloading engine sd15 ...`.

## Functional SD15 (txt2img/img2img/inpaint/upscale)
- Command (txt2img):
  ```bash
  python - <<'PY'
  from backend.core.orchestrator import InferenceOrchestrator
  from backend.core.engine_interface import TaskType
  from backend.core.requests import Txt2ImgRequest, Img2ImgRequest
  from backend.engines import register_default_engines

  # Register engines and prepare orchestrator
  register_default_engines(replace=True)
  orch = InferenceOrchestrator()

  # Load SD15 by checkpoint title/alias (as shown in UI dropdown)
  engine_key = 'sd15'
  model_ref = 'Any SD15 checkpoint name or [hash] shown in UI'

  # Prepare request
  req = Txt2ImgRequest(
      task=TaskType.TXT2IMG,
      prompt='masterpiece, ultra-detailed',
      negative_prompt='lowres, blurry',
      width=512, height=512,
      steps=12, guidance_scale=7.0,
      sampler='Euler a', scheduler='Automatic', seed=42,
  )

  # Run and consume events
  gen = orch.run(TaskType.TXT2IMG, engine_key, req, model_ref=model_ref)
  for ev in gen:
      if getattr(ev, 'stage', None):
          print('EVT', ev.stage, ev.percent, ev.message)
      else:
          print('RES', type(ev.payload), list(ev.metadata.items()))
  PY
  ```
- Expectation: events include `start`, `prepare`, `end`; result includes `images` and `info` JSON. Errors must be explícitos (engine, task, model, stack). A rota `txt2img` do UI já usa o Orchestrator (sem caminho legacy).

## Functional SDXL (txt2img/img2img/inpaint/upscale)
- Commands: repeat the SD15 flow using `engine_key='sdxl'` and an SDXL checkpoint name/hash.
- Expectation: same events and result shape; hires-based upscale path should produce results; errors must carry full context.

## Functional Flux (txt2img)
- Command: repeat with `engine_key='flux'` and a Flux checkpoint (e.g., flux-dev).
- Note: only `txt2img` is supported for Flux at this stage; other tasks should raise `UnsupportedTaskError` with full details.

## UI handlers cutover
- txt2img: handler usa Orchestrator diretamente (sem caminho legacy).
- img2img: handler também usa Orchestrator (fonte/máscara detectados pelos campos ativos do UI).

## Presets (Engine×Mode×Task)
- Arquivos em `configs/presets/{engine}/{mode}/{task}.yaml` com `schema_version` e `defaults`.
- Aplicação: engines preenchem apenas campos ausentes/"Automatic" do request; log `applied_preset_patch` lista chaves aplicadas; ausência de arquivo gera log "Preset not found".
- Validação: executar geração com campos omitidos (ex.: steps) e verificar no log que o preset preencheu e que o parser não reclamou de faltas.

Notes
- No automated test runners should be introduced; capture console outputs and attach to PR descriptions.
