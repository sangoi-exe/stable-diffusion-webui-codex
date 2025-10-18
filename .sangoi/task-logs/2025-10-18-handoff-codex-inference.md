Task: Detailed handoff update for Codex modular inference (engines + presets + video)
Date: 2025-10-18

Context
- User requested a detailed handoff (where we stopped + next steps) and confirmation to proceed with WAN 2.2 TI2V‑5B integration.

What I did
- Verified existing handoff at `.sangoi/handoffs/2025-10-18-codex-inference-rewrite.md`.
- Appended an "Solution Paths — WAN 2.2 Integration" section (6 paths), picked Path A (Native PyTorch) as intended solution, and outlined MVP deliverables, validation, risks, and fallbacks.
- Left code unchanged (docs-only), per directive to present the intended solution before implementation.

Files touched
- .sangoi/handoffs/2025-10-18-codex-inference-rewrite.md (append sections)
- .sangoi/CHANGELOG.md (append internal docs entry)

Commands run
- ls/rg/sed to inspect repo and handoff files.

Open points / Next steps
- Smoke test tornou-se opcional (não bloqueia). Validação principal será via UI (Txt2Vid/Img2Vid) com logs.
- Implementar WAN 2.2 TI2V‑5B via Path A.
- Após o MVP, expandir presets (sd15/sdxl img2img/inpaint/upscale; flux LCM/Lightning; vídeo Img2Vid + SVD/Hunyuan).
- Adicionar avisos de modo incompatível; métricas de VRAM/tempo; exemplos de JSON estrito.

Risks
- VRAM constraints for video models; ensure robust logs and fail-fast errors.
