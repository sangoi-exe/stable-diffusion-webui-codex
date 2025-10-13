Gradio 5 Migration (Internal)
=============================

Purpose
- Stabilize Forge UI on Gradio 5.49.x without heavy client JS; prefer Python-first wiring.
- Only use TS/JS where Gradio lacks an equivalent event/state API.

Constraints
- Python ≥ 3.10, Gradio 5.49.x, Windows support (WSL and native), minimal dependencies.
- Avoid breaking existing UX; preserve features: Send-to, token counters, PNG Info paste, extras/upscalers, ControlNet pages.

Key Differences vs 4.x
- Stricter component preprocessing (sliders/numbers must satisfy min/max before Python runs).
- Tabs selection can be updated programmatically; regressions existed earlier but are fixed in late 5.x — still return a clean `gr.Tabs(selected=...)` update in a single hop.
- Queue uses `concurrency_limit` at listener level (not `concurrency_count`).
- Optional SSR (Node ≥ 20) — keep off by default for now.

Detailed API/Syntax Notes (5.49.x)
----------------------------------

Components

- Update pattern: always return `gr.update(...)` for partial updates. Common keys:
  - `value=...`, `visible=True|False`, `interactive=True|False`, `choices=[...]`, `minimum=...`, `maximum=...`, `label=...`.
  - Sliders/Number: front‑end enforces `[minimum, maximum]` before Python runs. Do not clamp on the server — send valid values and fail fast with clear errors (augmented with component context).

Submit Payloads (strict JSON)

- txt2img: send only `[id_task, ...custom_script_inputs, payload_json]`, where `payload_json` contains only active fields and carries `__strict_version` and `__active_ids`.
- img2img: keep image/mask and batch file inputs positional; move scalars to the strict JSON. The server ignores positional scalars.
- Hidden JSON elem ids: `txt2img_named_active`, `img2img_named_active`.

Events

- Standard: `component.change(fn, inputs=[...], outputs=[...], queue=True|False, concurrency_limit=N)`
- Buttons: `component.click(...)`
- Tabs: `tabs.select(fn, inputs=[...], outputs=[...])` (fires when user changes tabs)

Queue/Concurrency

- Prefer `concurrency_limit` on each listener. If needed, a global default via `blocks.queue(concurrency_limit=N)`.

State

- Use `gr.State(value)` to hold internal UI state across events; pass as input and (optionally) return an updated state as output.

Tabs: programmatic selection

- Pattern 1 (direct): return only a Tabs update in the first hop.
  ```python
  def to_txt2img():
      return gr.Tabs.update(selected="txt2img")

  btn.click(to_txt2img, outputs=[tabs])
  ```

- Pattern 2 (chain): first hop switches tabs; subsequent UI updates in `.then(...)`.
  ```python
  btn.click(lambda: gr.Tabs.update(selected="img2img"), outputs=[tabs]) \
     .then(update_fields, inputs=[...], outputs=[...])
  ```

Paste/Send‑to (validate sliders)

- Validate inputs and raise on invalid values instead of clamping silently:
  ```python
  def _parse_int(x):
      if isinstance(x, (int, float)):
          return int(x)
      if isinstance(x, str) and x.strip().replace(".", "", 1).isdigit():
          return int(float(x))
      raise ValueError(f"invalid numeric payload: {x!r}")

  def paste_from_pnginfo(params):
      w = _parse_int(params.get("Size-1", 512))
      h = _parse_int(params.get("Size-2", 512))
      if not (64 <= w <= 2048) or not (64 <= h <= 2048):
          raise ValueError(f"out-of-bounds size: {w}x{h}")
      return gr.update(value=w), gr.update(value=h)
  ```

Gallery → Image

- Prefer retornos Python (PIL `Image` ou `None`) em vez de manipulação DOM. Use um helper JS mínimo só quando inevitável.

SSR (optional)

- Launch with: `demo.launch(ssr_mode=True)` or set `GRADIO_SSR_MODE=true`. Requires Node ≥ 20. Keep disabled by default on Windows.

API endpoints (optional)

- Minimal typed route without explicit components:
  ```python
  import gradio as gr

  def ping():
      return {"ok": True}

  with gr.Blocks() as demo:
      gr.api(ping, api_name="ping")
  ```

Policy
- No DOM manipulation if a Gradio 5 API exists.
- TypeScript only where needed (Canvas overlays, legacy extension UIs, hotkeys).
- JS/TS must be null-safe (no DOM assumptions, idempotent attach, run in onAfterUiUpdate/onUiLoaded only).

Implemented Fixes
- Static shim: `/file=…` route in FastAPI to serve legacy JS/CSS assets.
- PNG Info paste: clamp sliders and avoid empty updates (prevent `-1 < 64`).
- Legacy JS: null checks in `inputAccordion.js`, `token-counters.js`, `ui.js`.
- Tokenizer: fallback to `tokenizer.json` fast tokenizer when `merges.txt` absent.

Upcoming Replacements (Python-first)
- Send-to tab switch: return `gr.Tabs(selected='…')` updates instead of `_js`.
- Token counters: `.change/.click` wiring with optional lightweight TS for local debounce.
- Accordions: use `gr.Accordion(open=…)`, drop DOM-proxy checkbox pattern.

SSR (Optional Later)
- Gate behind `launch(ssr_mode=True)` when Node 20+ exists; keep CSR default.

Validation Checklist
- PNG Info → Send to txt2img/img2img (no errors, tab switches, sliders valid).
- Generation hotkeys (Ctrl+Enter/Esc) work; if kept JS, ensure guards.
- ControlNet pages: no recurring console errors; fallbacks idempotent.
