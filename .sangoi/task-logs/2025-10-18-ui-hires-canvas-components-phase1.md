Task: Componentize Hires/Canvas usage in builders
Date: 2025-10-18

Changes
- javascript/codex.components.hires.js: `Hires.get('txt2img')` retorna objeto normalizado com todos os campos hr_*.
- javascript/codex.components.canvas.js: `Canvas.getInpaint()` retorna configurações de inpaint (mask, fill, invert, etc.).
- javascript/ui.js: builders `buildNamedTxt2img` e `buildNamedImg2img` usam Readers e, se presentes, sobrescrevem campos com os valores normalizados dos componentes Hires/Canvas.

Notes
- Sem quebra de contrato. UI continua enviando o mesmo JSON estrito; apenas a leitura ficou modular e mais previsível.

