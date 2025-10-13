Gradio 5.49 Migration Notes (stable-diffusion-webui-codex)
==========================================================

Scope
- Target Gradio: 5.49.1 (pinned in `requirements_versions.txt`).
- Goals: remove brittle JS hooks, prefer Gradio components and Python updates, enable SSR, keep functional parity with A1111/Forge UX.

Key Changes
- SSR default on
  - `webui.py`: `shared.demo.launch(..., ssr_mode=True)` by default.
  - Override: `GRADIO_SSR_MODE=0` (accepted truthy: 1/true/yes/on; falsy: 0/false/no/off).

- JS injection policy (head=)
  - `modules/ui_gradio_extensions.py` injects CSS/JS via `Blocks(head=...)`.
  - New env: `GRADIO_JS_ALLOWLIST` controls which JS files from `javascript/` e extensões são injetados.
    - Unset: injeta todos (comportamento legado).
    - "", "1", "true", "yes", "on", "auto": allowlist curada (exclui `token-counters.js` e `settings.js`).
    - Outra string: lista de basenames separados por vírgula.
  - Novo env: `GRADIO_JS_DENYLIST` (sempre exclui os basenames listados).

- InputAccordion sem JS
  - `modules/ui_components.py`: `InputAccordionImpl` agora usa somente Python para sincronizar `Checkbox` -> `Accordion.open`.
  - Remove dependência de `javascript/inputAccordion.js`.

- Settings search sem JS
  - `modules/ui_settings.py`: `search_input.change()` chama `self.search` e retorna `gr.update(visible=...)` por componente.
  - Remove dependência de `javascript/settings.js`.

- Extra Networks – filtro/ordem server-side (fase 1)
  - `modules/ui_extra_networks.py`:
    - Para cada aba de extra networks, adicionamos `Textbox` (Filter…), `Dropdown` (sort field) e `Dropdown` (sort dir).
  - `create_card_view_html` e `create_html` aceitam `search`, `sort_field`, `sort_dir` e renderizam HTML no servidor.
  - JS legado de filtro/ordem permanece por compatibilidade, mas já não é necessário para o fluxo básico.
  - Opcional: `GRADIO_EXTRA_NETWORKS_DATASET=1` usa `gr.Gallery` nativo na aba Checkpoints. Filtro/ordem continuam server-side; selecionar um item aplica o checkpoint (via `main_entry.checkpoint_change`).
  - Textual Inversion: com a mesma flag, usa `gr.Gallery`; seletor "Positive/Negative" define o alvo; ao selecionar um card, o token é adicionado ao prompt escolhido da aba ativa.
  - LoRA: com a mesma flag, usa `gr.Gallery`; seletor "Positive/Negative"; ao selecionar, insere `<lora:alias_or_name:weight>` no prompt alvo. `alias_or_name` respeita `lora_preferred_name` e `weight` usa o preferido do metadata ou `extra_networks_default_multiplier`.
  - Hypernetworks: idem; insere `<hypernet:name:weight>` no prompt alvo; `weight` vem de `extra_networks_default_multiplier`.

Racionais Técnicos
- Gradio 5 reorganiza marcação e timing de hidratação; scripts baseados em `querySelector` e `MutationObserver` quebram com SSR e alteração de DOM.
- Eventos Gradio 5 suportam updates puramente Python e `js=` client-side quando necessário, reduzindo manipulação direta do DOM.
- Server-side render em Extra Networks evita reordenar/filtrar via `innerHTML` no cliente.

Environment Variables
- `GRADIO_SSR_MODE` – default on; defina `0` para desligar.
- `GRADIO_JS_ALLOWLIST` – controlar JS injetado; usar `auto` para a allowlist curada.
- `GRADIO_JS_DENYLIST` – sempre excluir basenames listados.

Backward Compatibility
- Mantivemos `script.js`, `ui.js`, `progressbar.js` e utilitários visuais por padrão.
- `reload_javascript()` permanece no-op para compat scripts legados.
- A troca de abas de “send to …” foi migrada progressivamente para `gr.Tabs.update`; fluxo existente de cópia para img2img já usa Python-only.

Validation Checklist
1) SSR on/off: `GRADIO_SSR_MODE=1` e `=0`.
2) Geração txt2img/img2img; restauração de progresso.
3) Settings search filtra sem console errors.
4) Acordeões abrem/fecham sem JS.
5) Extra Networks: filter e sort funcionam sem flicker.

Known Gaps / Next Steps
- Remover gradualmente `applyExtraNetworkFilter()` e outros seletores DOM, substituindo por updates Python e/ou Dataset/Gallery nativos.
- Substituir cliques em DOM remanescentes por `gr.Tabs.update` (onde existirem).
- Migrar Extra Networks para `gr.Dataset`/`gr.Gallery` (fase 2), reduzindo HTML manual e JS.

Notes for Extension Authors
- Evite supor estrutura de DOM do Gradio; prefira eventos (`.click/.change`) com `js=` e atualizações Python.
- Não redefina prototypes (ex.: Array.prototype).
- Para “send to …”, prefira updates com `gr.Tabs.update` ao invés de `.click()` em botões.
- Offline mode (tokenizers/configs)
  - Flag: `--disable-online-tokenizer` (strict). Se faltarem `model_index.json`/`config.json` ou arquivos essenciais de tokenizer (tokenizer/tokenizer.json ou vocab/merges; idem para tokenizer_2), o loader aborta com erro claro e não tenta baixar nada. Preencha `backend/huggingface/<repo>/...` manualmente.
- Sem a flag: o loader tenta baixar somente `*.json` e `*.txt` (config/tokenizer) via `huggingface_hub` ou HTTP direto, e segue robusto caso já existam localmente.
  - Robustez: downloads HTTP têm retry/backoff (429/5xx) com timeouts curtos; não impacta quando arquivos já existem.
