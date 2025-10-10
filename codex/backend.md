# Backend Systems

Forge maintains two cooperating back ends: the legacy AUTOMATIC1111 pipeline (`modules/processing.py`) and the modern Forge backend (`backend/`). The modern backend is modular so we can progressively migrate workloads while preserving existing behaviour.

## Package Layout
- `backend/args.py`: command-line arguments and configuration hydration for backend toggles.
- `backend/diffusion_engine/`: pluggable pipelines for txt2img, img2img, Flux, and other schedulers.
- `backend/memory_management.py`: VRAM accounting, offloading policies, and buffer pooling.
- `backend/attention.py` & `backend/nn/`: optimized attention kernels and neural network utilities.
- `backend/operations*.py`: operator registries for GGUF, bitsandbytes (BNB), and default Torch operations.
- `backend/text_processing/`: prompt parsing, tokenizer helpers, and conditioning assembly.
- `backend/modules/`: compatibility shims that bridge legacy module interfaces with the new backend.
- `backend/utils.py` & `backend/shared.py`: shared helpers, configuration, and runtime state objects.

## Key Execution Concepts
1. **Configuration intake**: CLI flags (`launch.py` + `backend/args.py`) populate a backend settings object stored in `backend/shared.py`.
2. **Engine selection**: Based on the active pipeline (e.g., txt2img vs img2img) and model family, `backend/diffusion_engine` chooses an engine class that orchestrates sampling, guidance, and refiners.
3. **Scheduler orchestration**: Each engine coordinates sampler implementations from `backend/sampling/` and can fall back to legacy samplers exposed through `modules/sd_samplers.py`.
4. **Memory strategy**: `backend/memory_management.py` tracks buffers, enabling VRAM offload/rewind strategies. Integrations with `modules/shared.py` ensure UI sliders (e.g., "GPU Weight") map to backend policies.
5. **Operations registry**: Modular operator registries (`operations.py`, `operations_bnb.py`, `operations_gguf.py`) abstract low-level tensor ops so multiple acceleration paths can coexist without scattering conditional logic.
6. **Streaming & async**: `backend/stream.py` handles generator-based outputs for streaming inference, coordinating with Gradio endpoints and WebSocket clients.

## Migration Notes
- When porting a feature, first mirror the legacy behaviour inside `backend/modules/` before replacing the call site in `modules/`.
- Avoid rewriting everything at once. Incrementally register backend features and guard them behind configuration flags until they are stable.
- Document deviations from the legacy pipeline in `codex/refactor-roadmap.md` so QA can focus on impacted features.

## Flux Configuration Schema
- `backend/diffusion_engine/flux_config.py` enumerates every Flux-specific UI toggle (width/height defaults, CFG sliders, GPU weight budgets, and sampler dropdowns) with a single data contract shared by the frontend and backend.
- `build_flux_option_info` converts the schema into `shared.OptionInfo` objects so presets defined in `modules_forge/main_entry.py` and `modules/processing_scripts/sampler.py` derive from the same metadata.
- Dynamic values such as GPU VRAM headroom and sampler lists are supplied at call time through the helper's `context` and `dynamic_choices` parameters, keeping runtime-sensitive logic near the consumers while avoiding duplicated defaults.
