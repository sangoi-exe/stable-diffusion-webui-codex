<!-- tags: backend, huggingface, microsoft, lens, metadata-mirror -->

# apps/backend/huggingface/microsoft Overview
Date: 2026-05-24
Last Review: 2026-05-24
Status: Active

## Purpose
- Stores metadata-only local mirrors for public Microsoft Hugging Face repositories used by Codex planning.
- These mirrors are source snapshots and metadata only; Codex runtime support is not implemented merely because metadata exists here.

## Key Files
- `apps/backend/huggingface/microsoft/Lens/` - metadata-only mirror for the default RL-tuned Microsoft Lens text-to-image variant.
- `apps/backend/huggingface/microsoft/Lens-Turbo/` - metadata-only mirror for the distilled fast Microsoft Lens variant.
- `apps/backend/huggingface/microsoft/Lens-Base/` - metadata-only mirror for the supervised base Microsoft Lens variant.

## Notes
- Do not add `*.safetensors`, `*.bin`, `*.pth`, `*.pt`, `*.ckpt`, `*.gguf`, or `*.onnx` here.
- Do not keep upstream `.gitattributes` files in this mirror unless this repository intentionally adopts Git LFS for the mirror.
- `*.safetensors.index.json` files are metadata and are allowed; they do not contain tensor payloads.
- The vendored Lens files support future implementation research only. Engine registration, runtime support, asset contracts, API surfaces, and frontend controls must land in a separate implementation tranche.
