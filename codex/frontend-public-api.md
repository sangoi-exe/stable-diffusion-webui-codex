Frontend Public API (JS) – Reference
====================================

Overview
- The WebUI exposes a small set of global functions consumed by Python `_js=` hooks and extensions.
- All IDs are resolved under Gradio’s root (`gradioApp()`), which may be a `ShadowRoot`.
- Prefer using the helpers listed below to avoid brittle DOM assumptions.

Ambient globals (from javascript-src/shims.d.ts)
- gradioApp(): Document | ShadowRoot | HTMLElement
- onUiLoaded(cb), onUiUpdate(cb), onAfterUiUpdate(cb), onUiTabChange(cb)
- onOptionsChanged(cb), onOptionsAvailable(cb)
- opts: StableDiffusionOptions (populated from `#settings_json` textarea)

Core helpers (defined in first‑party JS)
- getAppElementById(id: string): HTMLElement | null
- updateInput(target: HTMLInputElement | HTMLTextAreaElement): void
- onEdit(id: string, elem: HTMLInputElement|HTMLTextAreaElement|null, afterMs: number, fn: () => void): () => void
- submitWithProgress(args: IArguments, containerId: string, galleryId: string): unknown[]
- normalizeSubmitArgs(tab: string, res: unknown[]): unknown[]
- requestProgress(id: string, container: HTMLElement, gallery: HTMLElement|null, atEnd: () => void, onProgress?: (r) => void, inactivityTimeout?: number): void
- randomId(): string
- localSet(key: string, value: string), localGet(key: string, fallback?: string|null): string|null, localRemove(key: string): void

Common UI hooks (globals on window)
- submit(...): unknown[]
- submit_img2img(...): unknown[]
- submit_txt2img_upscale(...): unknown[]
- submit_extras(...): unknown[]
- restoreProgressTxt2img(): string | null
- restoreProgressImg2img(): string | null
- get_tab_index(tabId: string): number
- set_theme(theme: string): void
- switch_to_txt2img/img2img/sketch/inpaint/inpaint_sketch/extras(...): unknown[]
- ask_for_style_name(_: unknown, prompt: string, negativePrompt: string): [string, string, string]
- confirm_clear_prompt(prompt: string, negativePrompt: string): [string, string]

Strict Submit JSON (Generate flows)
- txt2img and img2img now ship a compact strict JSON alongside (or instead of) positional args.
- Hidden JSON components:
  - `txt2img_named_active` – contains only active fields, plus `__strict_version=1` and `__active_ids`.
  - `img2img_named_active` – idem para img2img (escalares; imagens/arquivos seguem como inputs).
- Builders (client side): `buildNamedTxt2img` e `buildNamedImg2img` leem os valores do DOM (via elem_id) e populam o JSON.
- Servidor: ignora escalares posicionais quando `__strict_version==1` está presente; apenas o JSON é fonte‑de‑verdade.
- Erros são “fail‑fast”: campos ausentes ou tipo errado levantam `ValueError` com mensagem clara; sliders fora de faixa levantam `gradio.exceptions.Error` enriquecida com label/elem_id e bounds.

Extensions page hooks
- extensions_apply(disableList: unknown, updateList: unknown, disableAll: boolean): [string, string, boolean]
- extensions_check(): [id: string, disabledJson: string]
- install_extension_from_index(button: HTMLInputElement|HTMLButtonElement, url: string): void
- config_state_confirm_restore(_: unknown, name: string, type: string): [boolean, string, string]
- toggle_all_extensions(event: Event): void
- toggle_extension(): void

Token counter hooks
- update_txt2img_tokens(...): unknown
- update_img2img_tokens(...): unknown
- recalculate_prompts_txt2img(...): unknown
- recalculate_prompts_img2img(...): unknown

Textual inversion
- start_training_textual_inversion(...): unknown[] (returns args with task id at index 0)

Usage notes
- Do not bypass helpers with `document.getElementById` — the UI runs inside a shadow root in Gradio 5.
- When programmatically editing inputs, call `updateInput` so Gradio state updates.
- Keep submit arg order; construct new arrays with `Array.from(arguments)` rather than mutating.
- Para novos campos em txt2img/img2img, adicione o elem_id no builder JS e parse no servidor. Não dependa de índices.
