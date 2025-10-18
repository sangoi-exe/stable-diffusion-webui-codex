Handoff: Quicksettings VAE ~ Checkpoint parity
Date: 2025-10-18

What changed
- VAE dropdown no Quicksettings passa a usar o mesmo padrão do checkpoint (mesmo builder/helper; refresh compartilhado; eventos `change`).
- Descoberta de VAEs: `refresh_models()` agora mescla o resultado de `sd_vae.refresh_vae_list()` com a varredura de diretórios.
- Resolução de nomes: `_resolve_module_path()` aceita rótulos com sufixo `[hash]` e tenta `sd_vae.vae_dict` como fallback.

Files touched
- modules_forge/main_entry.py

Commands
- rg, sed (inspeção)
- apply_patch (edições)

Validation
- Arquivo compila estaticamente; não há mudanças em IDs usados externamente (mantidos: `sd_checkpoint`, `sd_vae`, `sd_text_encoders`).
- Botão de refresh continua atualizando os três dropdowns.

Next steps / TODOs
- (Opcional) Adicionar cálculo de hash para todos os VAEs com caching (paridade total com checkpoints).
- (Opcional) Exibir hash do VAE carregado no topo (como o hash do checkpoint).

