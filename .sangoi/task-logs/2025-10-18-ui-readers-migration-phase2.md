Task: Readers migration phase 2 (txt2img/img2img)
Date: 2025-10-18

Changes
- javascript/ui.js: `buildNamedTxt2img` e `buildNamedImg2img` agora utilizam `Codex.Components.Readers` (com fallback) para ler campos de texto, numéricos, dropdowns e seed.

Notes
- Sem alteração de contrato — apenas swap de helpers de leitura. Próximas etapas: mover mais utilitários (checkbox/radio) e reduzir ui.js.

