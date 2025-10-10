# Extensions & Integrations

Forge inherits the AUTOMATIC1111 extension system while adding curated integrations and replacement lists.

## Extension Directories
- `extensions/`: User-managed extensions cloned from Git repositories. Each folder contains a `scripts/` directory that Gradio auto-loads.
- `extensions-builtin/`: Bundled extensions maintained alongside Forge (e.g., ControlNet, FreeU). These are versioned with the core codebase.
- `modules/script_callbacks.py`: Registration hub where extensions plug into WebUI lifecycle events.
- `modules_forge/extension_adapter/`: Compatibility layer to reconcile upstream expectations with Forge back-end improvements.

## Loading Lifecycle
1. `launch.py` resolves `--disable-extension`/`--enable-insecure-extension-access` flags.
2. `modules/script_loading.py` scans extension folders, loading Python entry points and optional requirements.
3. Extensions declare components (`Script`, `AfterUiCallback`, API routes) that hook into pipelines via `modules/scripts.py`.
4. Forge adapters ensure backend changes (e.g., new sampler interfaces) remain compatible by providing shims in `backend/modules/`.

## Integration Highlights
- **ControlNet**: Bundled under `extensions-builtin/sd-webui-controlnet`, integrates deeply with `modules_forge/controlnet/` and backend graph execution.
- **LoRA enhancements**: Tools in `sd-webui-lora-layer-weight/` and `modules/lora.py` collaborate with backend attention patches.
- **Flux support**: Tutorials referenced in `README.md` map to runtime switches implemented via backend engines.

## Maintenance Guidelines
- When updating bundled extensions, document version bumps and compatibility notes in `codex/refactor-roadmap.md`.
- Avoid direct modification of third-party code unless necessary. Prefer patches layered through Forge adapters.
- Keep extension-specific settings names stable; UI migrations should include localization updates and API compatibility checks.
