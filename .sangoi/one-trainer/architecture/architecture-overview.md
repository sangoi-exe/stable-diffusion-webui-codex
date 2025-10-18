**Objective**
- Reescrever a pipeline de inferência do zero mantendo o layout do Gradio, com engines modulares por família de modelo (SD15, SDXL, FLUX, etc.), e adicionando duas novas abas: `Txt2Vid` e `Img2Vid`.

**Design Principles**
- Separação nítida: UI (Gradio) fina → Orquestrador → Engines por modelo.
- Engines plug‑in: cada família implementa a mesma interface, expõe capacidades e tarefas suportadas.
- Config por preset: requests versionados (YAML/JSON) e adapters para UI.
- Fail‑fast com logs verbosos; nada de fallback silencioso.
- Progressos em tempo real: eventos com porcentagem/etapa/timer.
- Parametrização tipada: enums e dataclasses para Engines/Tasks/Mode/Params; validação precoce.

**High‑Level Components**
- `EngineInterface` (contrato): load/unload/configure; inferências `txt2img`, `img2img`, `txt2vid`, `img2vid` (métodos opcionais por engine); introspecção de capacidades.
- `EngineRegistry` (descoberta): resolve `engine_key` → classe; criação por `model_type`/`weights`.
- `Orchestrator` (faixa fina): valida request, escolhe engine, instrumenta memória/logs/progress.
- `PresetService`: carrega/valida presets e mapeia p/ requests tipados.

**Directory Layout (novo)**
- `backend/core/engine_interface.py` — ABCs, tipos e eventos.
- `backend/core/requests.py` — dataclasses p/ `Txt2ImgRequest`, `Img2ImgRequest`, `Txt2VidRequest`, `Img2VidRequest` e respostas.
- `backend/core/registry.py` — registro/descoberta de engines.
- `backend/core/orchestrator.py` — fluxo único de execução com métricas/progress.
- `backend/engines/sd15/engine.py` — implementação SD 1.5.
- `backend/engines/sdxl/engine.py` — implementação SDXL.
- `backend/engines/flux/engine.py` — implementação FLUX.
- `backend/engines/video/txt2vid/engine.py` — implementação de texto→vídeo.
- `backend/engines/video/img2vid/engine.py` — implementação de imagem→vídeo.
- `configs/presets/` — presets versionados por tarefa (txt2img, img2img, txt2vid, img2vid, inpaint, upscale).

**UI Integration (Gradio)**
- Tabs existentes `txt2img`/`img2img` passam a enviar `Request` tipado via orquestrador.
- Novas tabs: `Txt2Vid` e `Img2Vid` (controles p/ duração, fps, resolução, seed, sampler, motion strength).
- Logs/Toasts: versão de preset aplicada, engine selecionado, e warnings de campos ignorados.

**Error Handling & Logging**
- Toda chamada de engine deve:
  - validar inputs com mensagens acionáveis;
  - logar chave do engine, modelo, sampler/scheduler, seed e tempo total;
  - emitir eventos de progresso com etapas (prepare, encode, denoise, decode, save).

**Performance & Memory**
- Pré‑carregamento opcional por engine; cache LRU de pesos e VAE.
- Política de descarte: hooks `on_idle` e `on_low_memory` no orquestrador.
- Métricas: tempo por etapa, VRAM/CPU, throughput (it/s, frames/s).

**Compatibility**
- Adapters de migração: camada fina traduz antiga assinatura para nova API (deprecada com aviso). Onde impossível, renomear explicitamente (registro em `migration-plan.md`).
