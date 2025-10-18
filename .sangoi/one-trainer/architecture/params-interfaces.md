# Engine Parameter Interfaces (per model/task)

Purpose
- Provide explicit, typed interfaces for required/optional parameters per Engine × Task, similar ao padrão do OneTrainer, com validação antecipada.

Location
- Code: `backend/core/params/*` — dataclasses por engine/tarefa.
- Enumerações: `backend/core/enums.py` — `Mode`, `EngineKey`, `SamplerName`, `SchedulerName`.
- Registry: `backend/core/param_registry.py` — mapeia `(engine_key, task)` → parser que valida/normaliza `Request` e retorna a interface tipada.

Usage (engines)
- Engines chamam `parse_params(engine_id, task, request)` no início do handler para garantir shape/erros explícitos.
- Engines que aplicam presets por modo devem ler `request.metadata.get('mode')` e decidir, sem sobrescrever valores definidos pelo usuário.

Guarantees
- Campos obrigatórios ausentes geram `ValidationError` com mensagem acionável (engine/task/nome do campo).
- Campos não suportados podem ser ignorados com log; não há fallback silencioso.

