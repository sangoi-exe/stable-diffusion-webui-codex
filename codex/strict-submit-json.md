Strict Submit JSON – Generate Flows
===================================

Why
- Listas posicionais do Gradio são frágeis: ordem muda com UI, extensões inserem inputs, erros ficam opacos.
- JSON estrito envia apenas o que está ATIVO, com nomes estáveis (elem_id), e falha alto quando algo está errado.

Escopo (fase atual)
- txt2img: 100% escalars via JSON.
- img2img: escalars via JSON; imagens/máscaras/arquivos permanecem como inputs (próxima fase pode migrar também).

Formato
- Campo de versão: `__strict_version: 1` (obrigatório).
- Ativos: `__active_ids: string[]` (lista de elem_ids incluídos no JSON).
- Nomes: iguais ao `elem_id` de cada componente relevante.

txt2img – Campos
- Core:
  - `txt2img_prompt: string`
  - `txt2img_neg_prompt: string`
  - `txt2img_styles: string[]`
  - `txt2img_batch_count: int`, `txt2img_batch_size: int`
  - `txt2img_cfg_scale: float`, `txt2img_distilled_cfg_scale: float`
  - `txt2img_height: int`, `txt2img_width: int`
  - `txt2img_hr_enable: boolean`
- Hires (somente quando `txt2img_hr_enable=true`):
  - `txt2img_denoising_strength: float`, `txt2img_hr_scale: float`
  - `txt2img_hr_upscaler: string`, `txt2img_hires_steps: int`
  - `txt2img_hr_resize_x: int`, `txt2img_hr_resize_y: int`
  - `hr_checkpoint: string`, `hr_vae_te: string[]`, `hr_sampler: string`, `hr_scheduler: string`
  - `txt2img_hr_prompt: string`, `txt2img_hr_neg_prompt: string`
  - `txt2img_hr_cfg: float`, `txt2img_hr_distilled_cfg: float`

img2img – Campos (parcial)
- Core:
  - `img2img_prompt: string`, `img2img_neg_prompt: string`, `img2img_styles: string[]`
  - `img2img_batch_count: int`, `img2img_batch_size: int`
  - `img2img_cfg_scale: float`, `img2img_distilled_cfg_scale: float`, `img2img_image_cfg_scale: float`
  - `img2img_denoising_strength: float`
  - `img2img_selected_scale_tab: int` (id do tab de resize)
  - `img2img_height: int`, `img2img_width: int`, `img2img_scale_by: float`
  - `img2img_resize_mode: int` (índice do Radio)
- Inpaint (somente nas abas inpaint):
  - `img2img_mask_blur: int`, `img2img_mask_alpha: float`, `img2img_inpainting_fill: int`
  - `img2img_inpaint_full_res: boolean`, `img2img_inpaint_full_res_padding: int`, `img2img_inpainting_mask_invert: int`
- Inputs posicionais mantidos (por enquanto): imagens/máscaras, diretórios/listas de arquivos de batch.

Validação (servidor)
- Tipagem estrita: `int/float/bool/list` com mensagens claras (ex.: “Field txt2img_cfg_scale must be float”).
- Gating por ativo: campos de Hires obrigatórios apenas quando `txt2img_hr_enable=true`.
- Sliders/Dropdowns: erros do Gradio são enriquecidos com `label`, `elem_id`, `min/max/step`, `payload/coerced`.

Compatibilidade
- O array posicional ainda existe para manter o Queue/Progress do Gradio, mas é ignorado para escalares com `__strict_version==1`.
- Elemento hidden renomeado: `*_named_override` → `*_named_active` (o servidor aceita ambos; novo preferido).

Exemplo (txt2img)
```json
{
  "__strict_version": 1,
  "__active_ids": [
    "txt2img_prompt", "txt2img_neg_prompt", "txt2img_width", "txt2img_height",
    "txt2img_cfg_scale", "txt2img_distilled_cfg_scale", "txt2img_batch_count",
    "txt2img_batch_size", "txt2img_hr_enable"
  ],
  "txt2img_prompt": "a photo of a dog",
  "txt2img_neg_prompt": "blurry",
  "txt2img_styles": [],
  "txt2img_batch_count": 1,
  "txt2img_batch_size": 1,
  "txt2img_cfg_scale": 7.0,
  "txt2img_distilled_cfg_scale": 3.5,
  "txt2img_width": 768,
  "txt2img_height": 1024,
  "txt2img_hr_enable": false
}
```

Como adicionar um novo campo
1) Dê um `elem_id` claro ao componente.
2) Leia-o no builder JS (funções `readText/readNumber/readDropdownValue/readRadioIndex`) e inclua no JSON + em `__active_ids`.
3) Parse no servidor (txt2img_from_json/img2img_from_json) com validadores.
4) Sem fallback: se o campo for obrigatório em certo estado, falhe com erro objetivo.

Próximas fases
- Migrar campos de batch do img2img para JSON (ou encapsular imagens/arquivos).
- Padronizar JSON estrito para extensões (ex.: ControlNet Units) e consumir apenas os campos ON.

