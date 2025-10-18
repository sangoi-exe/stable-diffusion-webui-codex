Task: Filter sampler/scheduler per engine (UI) and add modular JS components
Date: 2025-10-18

Changes
- Backend policy: `backend/core/sampler_policy.py` with `allowed_samplers(engine, task)` and `allowed_schedulers(engine, task)`.
- Image Sampler UI: `modules/processing_scripts/sampler.py` now uses backend policy for initial choices and binds engine change to update dropdowns.
- Video Tabs: `modules/ui.py` now builds Txt2Vid/Img2Vid sampler/scheduler dropdowns from policy and updates on engine change.
- JS components: `javascript/codex.components.core.js` (bus bootstrap) and `javascript/codex.components.sampler.js` (helper to set choices; optional for future use).
- Injection allowlist: `modules/ui_gradio_extensions.py` includes new JS files in curated allowlist.

Rationale
- "Só mostre samplers compatíveis" para qualquer engine, com contratos claros front↔back. Política central evita divergências e garante logs/erros explícitos no back.

Notes
- Schedulers para SD/Flux ficam no conjunto comum [Automatic, Karras, Simple, Exponential]; WAN/Hunyuan/SVD usam o mesmo conjunto, coerente com o mapeamento do Diffusers.

