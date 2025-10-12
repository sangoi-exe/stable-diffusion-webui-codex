About Gradio 5: will try to upgrade to Gradio 5 at about 2025 March. If failed, then will try again on about 2025 June. relatively positive that we can have Gradio5 before next summer.

2025 Oct 11: Removed client-side `compat_gradio5.js` wrapper that suppressed UI callback errors. Errors will now surface in console for proper fixes; no behavior changes intended beyond eliminating silent failures.
2025 Oct 11: Improved callback robustness in `script.js` — invalid (non-function) registrations are ignored with a warning and callback errors are logged with context, reducing confusing messages like "error running callback :" and aiding root-cause fixes.
2025 Oct 11: Begin Gradio 5 migration per "Guia de Migração do Gradio 4.40 para 5" — set expected Gradio to 5.49.1 and use `concurrency_limit` in `.queue()`.
2025 Oct 11: JS → Python: img2img copy buttons now switch tabs via server updates (no DOM clicks). Added optional SSR gating with `GRADIO_SSR_MODE` (off by default).
2025 Oct 11: Switched asset injection to Gradio 5-friendly Blocks(head=...) (no more TemplateResponse monkeypatch). Kept existing reload button behavior; `reload_javascript()` remains a no-op.
2025 Oct 12: Default SSR enabled (can disable with GRADIO_SSR_MODE=0). Added env-driven JS allowlist (GRADIO_JS_ALLOWLIST). When allowlist is active (''/auto/true), token-counters.js and settings.js are not injected by default.
2025 Oct 12: Extra Networks (Checkpoints) – added optional native Gallery (enable with GRADIO_EXTRA_NETWORKS_DATASET=1). Server-side filter/sort retained; selection applies checkpoint immediately.
2025 Oct 12: Extra Networks (Textual Inversion) – optional native Gallery under same flag; selector to choose Positive/Negative prompt target; selecting a card appends the token to the chosen prompt.
2025 Oct 12: Extra Networks (LoRA) – optional native Gallery under same flag; selector Positive/Negative; selecting a card inserts `<lora:alias_or_name:weight>` in the chosen prompt, weight defaults to preferred weight or `extra_networks_default_multiplier`.
2025 Oct 12: Extra Networks (Hypernetworks) – optional native Gallery under same flag; selector Positive/Negative; selecting a card inserts `<hypernet:name:weight>` using `extra_networks_default_multiplier`.
2025 Oct 11: Primeira geração mais rápida (lado cliente). A injeção via Blocks(head=...) e a opção de SSR reduzem o overhead inicial de UI, mitigando aquele “delay” chato na 1ª geração após abrir a página. Observação: o aquecimento de backend (carregar pesos/compilar kernels) ainda pode impactar a 1ª execução.

2024 Oct 28: A new branch `sd35` is contributed by [#2183](https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2183) . I will take a look at quants and sampling and transformer's clip-g vs that clip-g rewrite before merging to main ... (Oct 29: okay maybe medium also need to take a look later)

About Flux ControlNet (sync [here](https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/932)): The rewrite of ControlNet Intergrated will ~start at about Sep 29~ (delayed) ~start at about Oct 15~  (delayed) ~start at about Oct 30~ (delayed) start at about Nov 20. (When this note is announced, the main targets include some diffusers formatted Flux ControlNets and some community implementation of Union ControlNets. However, this may be extended if stronger models come out after this note.)

2024 Sep 7: New sampler `Flux Realistic` is available now! Recommended scheduler is "simple".
