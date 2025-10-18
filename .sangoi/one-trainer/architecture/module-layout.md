**Target Layout**
- `backend/core/`
  - `engine_interface.py`
  - `requests.py`
  - `registry.py`
  - `orchestrator.py`
- `backend/engines/`
  - `sd15/engine.py`
  - `sdxl/engine.py`
  - `flux/engine.py`
  - `video/txt2vid/engine.py`
  - `video/img2vid/engine.py`
- `configs/presets/`
  - `txt2img/*.yaml`, `img2img/*.yaml`, `txt2vid/*.yaml`, `img2vid/*.yaml`
- `resources/prompt_templates/` (opcional)

**Naming**
- Engines: `XxxEngine` com atributo `engine_id` (ex.: `sd15`, `sdxl`, `flux`).
- Registries: chaves iguais ao `engine_id`.
- Requests/Responses: nomes estáveis por tarefa.

**Integration Points**
- `modules/ui.py`, `modules/txt2img.py`, `modules/img2img.py`: chamar `orchestrator.run()` em vez de códigos específicos antigos.
- Novos `modules/txt2vid.py` e `modules/img2vid.py` (UI) chamando tarefas de vídeo.

