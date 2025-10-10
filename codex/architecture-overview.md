# Architecture Overview

Stable Diffusion WebUI Forge combines a Python back end with a Gradio-powered user interface. The repository can be conceptually divided into the following layers:

## Entry Points
- `webui.py` & `launch.py`: bootstrap environment variables, dependency checks, and finally launch the Gradio app defined in `modules/ui.py`.
- Platform launchers (`webui-user.sh`, `webui-user.bat`, `webui.sh`, `webui.bat`, `webui-macos-env.sh`) configure runtime flags, Python paths, and GPU-specific options before delegating to `launch.py`.

## Core Application
- `modules/`: Legacy AUTOMATIC1111 core implementing pipelines, samplers, model loading, shared state, API routes, and UI definitions.
- `modules_forge/`: Forge-specific augmentations that replace or extend functionality from `modules/`, especially around scheduling, memory, and UI enhancements.
- `backend/`: New Forge backend focused on modular diffusion engines, attention optimizations, and memory management abstractions. It will gradually supersede pieces of `modules/` as refactors land.
- `javascript/`, `html/`, `style.css`, `script.js`: Static assets that enhance or replace default Gradio behaviour.

## Data & Model Assets
- `models/`, `sd-webui-lora-layer-weight/`, and subfolders manage checkpoints, LoRA weights, VAE files, and optimization artifacts. Configuration helpers live in `download_supported_configs.py` and `requirements_versions.txt`.

## Extensions
- `extensions/` and `extensions-builtin/` load external or bundled plugins. Forge also includes integration helpers under `modules_forge/extension_adapter` and `backend/modules` to keep compatibility.

## Tooling & Utilities
- `scripts/` and `modules/scripts.py` expose automation utilities (e.g., batch processing, post-processing). Many scripts rely on `modules/shared.py` state and the WebUI API defined in `modules/api`.
- `packages_3rdparty/` and `node_modules/` contain vendored assets used by the UI and advanced features.

## Execution Flow Summary
1. Launcher resolves environment and calls `launch.py`.
2. `launch.py` parses command-line arguments (`modules/cmd_args.py`), prepares Torch/extension environment, and invokes `webui.py`.
3. `webui.py` initializes global state, registers samplers, loads models via `modules/sd_models.py`, and builds the Gradio interface.
4. Endpoints rely on either the legacy pipeline in `modules/processing.py` or Forge backends in `backend/` (when enabled) to execute diffusion passes.
5. Results pass through post-processing pipelines and are delivered to the UI, where custom JavaScript enhances interactivity and canvas experiences.

Understanding which layer a change impacts is essential when planning refactors: isolate edits within a layer whenever possible, and document any cross-layer dependencies inside the `codex/` guides.
