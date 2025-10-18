**New Tabs**
- `Txt2Vid` (texto→vídeo)
  - Campos: `prompt`, `negative`, `width`, `height`, `fps`, `num_frames`, `sampler`, `scheduler`, `seed`, `motion_strength`, `vae`, `lora`, `extras`.
  - Botões: `Generate`, `Stop`, `Preview` (mostra primeiros N frames), `Save` (mp4/webm).
- `Img2Vid` (imagem→vídeo)
  - Campos: acima + `init_image`/`frames`, `denoise_strength`.

**UX Behavior**
- `Generate` chama `orchestrator.run("txt2vid"|"img2vid")` e consome eventos de progresso; UI mostra barra percent/eta, thumb de frames parciais.
- `Preview` solicita apenas K frames iniciais (flag no request).

**Wiring (alta‑nível)**
- `modules/ui.py`: registrar duas abas e controles (seguir estilo atual), com handlers que constroem `Txt2VidRequest`/`Img2VidRequest` a partir do estado.
- `modules/ui_loadsave.py`: suportar presets de vídeo.

**Validation & Errors**
- Regras de resolução/fps compatíveis com a engine; mensagens direto na UI.
- Logs incluem engine, sampler, duração e tamanho.

