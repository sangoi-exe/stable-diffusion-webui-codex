Task: Quicksettings VAE igual ao seletor de Checkpoint (substituir o atual)
Date: 2025-10-18

Summary
- Alinhei o seletor de VAE ao de checkpoint na barra de Quicksettings.
- Centralizei a construção dos dropdowns para manter UX idêntica (tamanho, busca Gradio, botão de atualizar, eventos).
- Tornei a descoberta de VAEs mais robusta unificando `os.walk` com `sd_vae.refresh_vae_list()` e adicionei fallback na resolução nome→caminho.

Files touched
- modules_forge/main_entry.py

Key changes
- refresh_models(): mescla `sd_vae.vae_dict` aos VAEs descobertos por pasta.
- _resolve_module_path(): aceita rótulos com sufixo ` [hash]`, remove brackets e tenta `sd_vae.vae_dict` como fallback.
- _dropdown_model_selector(): helper para construir dropdowns de modelo com mesmos parâmetros (checkpoint, VAE e TE).
- Deixa explícito `interactive=True` e `allow_custom_value=False` também para o checkpoint, igual ao VAE.

Commands run
- rg -n "quicksettings|sd_model_checkpoint|sd_vae|model_selection" -S
- sed -n to inspect modules/ui_settings.py, modules_forge/main_entry.py, style.css, javascript/ui.js
- apply_patch to modify files

Validation
- Build estático: grep por referências aos IDs/elem_classes; nenhum duplicado de VAE.
- Verifiquei que `ui_settings.add_quicksettings()` ignora chaves gerenciadas (`forge_selected_vae`, `forge_additional_modules`), evitando duplicatas.
- O botão de refresh (`forge_refresh_checkpoint`) continua atualizando os três dropdowns (ckpt/vae/TE).

Risks/Notes
- A função `_resolve_module_path` agora lida com rótulos contendo ` [hash]`; caso nomes colidam, a resolução continua baseada em basename primeiro.
- O cálculo de hash de VAE não foi adicionado (mantido simples para não introduzir custo de I/O); se necessário, pode-se adicionar cache em follow-up.

Next steps (opcional)
- Adicionar comando “Calculate hash for all VAEs” análogo ao de checkpoints.
- Exibir hash atual do VAE carregado ao lado do dropdown (como no checkpoint).

