 # UI — Inference Type Dropdown (replacing radio buttons)

 Objective
 - Replace radio buttons for inference type selection with a dropdown to handle many options.

 Placement
- Quicksettings row: place `Engine` dropdown immediately to the left of the `Checkpoint` selector (replacing the previous radio preset). This applies to the whole UI layout (tabs mantidos).
- Tabs mantidos: `txt2img`, `img2img`; novos: `txt2vid`, `img2vid` (sem alterações no estilo dos tabs).

 Behavior
 - Changing the dropdown toggles visibility of only the controls relevant to that type.
 - The handler constructs the corresponding Request dataclass and calls the Orchestrator.
 - Persist last selection per tab (`config_states/`).

 Engine Selection
- Dropdown `Engine` (populado por `backend.core.registry.list()` após `register_default_engines()`); fallback para `[sd15, sdxl, flux]` se vazio.
- Ao mudar o `Engine`, atualiza presets de largura/altura, CFG e sampler/scheduler (lógicas específicas para sd15/sdxl/flux inicialmente). Valores avançados do Flux (ex.: Distilled CFG) ficam interativos conforme necessário.

 LoRA / Extra Networks
 - Follow A1111/Forge convention: tokens in prompt (e.g., `<lora:name:weight>`). UI keeps current controls for discovery/help but does not inject outside prompt.
 - If an engine does not support a given LoRA or extra, show a non-blocking warning with details.

 Telemetry & Logs
 - Log `inference_type`, `engine_id`, schema version, and any ignored fields with reasons.
