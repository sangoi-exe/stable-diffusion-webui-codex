# AGENT — Runtime State Dict Helpers

Purpose: Lightweight state-dict mapping views + small state-dict utilities used by loaders and runtime codepaths.

Key files:
- `apps/backend/runtime/state_dict/keymap_flux_transformer.py`: Flux transformer key-style resolver (native/source Diffusers or internal fused layout → Flux runtime lookup space).
- `apps/backend/runtime/state_dict/keymap_flux2_transformer.py`: FLUX.2 transformer key-style resolver (legacy fused core layout or native/source Diffusers → Diffusers `Flux2Transformer2DModel` lookup space).
- `apps/backend/runtime/state_dict/key_mapping.py`: Strict key-style detection + keyspace resolver core (fail loud; collision/ambiguity checks).
- `apps/backend/runtime/state_dict/keymap_gemma3_text_encoder.py`: Gemma3 text-only GGUF key-style resolver (llama.cpp GGUF or native HF text-backbone layout → `Gemma3TextModel` lookup space).
- `apps/backend/runtime/state_dict/keymap_clip_vision.py`: CLIP vision image-encoder key-style resolver (HF `vision_model.*`, wrapped `image_encoder.vision_model.*`, or explicit OpenCLIP `visual.*` → canonical HF `CLIPVisionModelWithProjection` lookup space).
- `apps/backend/runtime/state_dict/keymap_anima_transformer.py`: Anima transformer key-style resolver (raw `net.*` or already-canonical runtime keys → canonical Anima runtime lookup space).
- `apps/backend/runtime/state_dict/keymap_llama_gguf.py`: llama.cpp-style GGUF tensor-name resolver for text models (HF key layout).
- `apps/backend/runtime/state_dict/keymap_qwen_text_encoder.py`: Qwen text-encoder key-style resolver (native HF backbone or known wrapper/container surfaces -> canonical `model.*` lookup space; optional aux heads handled explicitly by policy).
- `apps/backend/runtime/state_dict/keymap_qwen_image_transformer.py`: Exact Edit-2511 1,933-key transformer topology validator and identity lookup view; stored native keys remain unchanged.
- `apps/backend/runtime/state_dict/keymap_sdxl_clip.py`: SDXL base text-encoder key mapping (CLIP-L/CLIP-G → Codex IntegratedCLIP layout).
- `apps/backend/runtime/state_dict/keymap_sdxl_checkpoint.py`: SDXL checkpoint keyspace resolver for original-format checkpoints; known wrapper/container source styles map explicitly into canonical parser lookup keys.
- `apps/backend/runtime/state_dict/keymap_sdxl_vae.py`: SDXL/Flow16 VAE key-style resolver (LDM-style → diffusers AutoencoderKL; wrapper-rewrite inputs fail loud).
- `apps/backend/runtime/state_dict/keymap_t5_text_encoder.py`: T5 text-encoder key-style resolver (HF `encoder.*`/`shared.weight` → IntegratedT5 `transformer.*`).
- `apps/backend/runtime/state_dict/keymap_zimage_transformer.py`: Z Image transformer key-style resolver (native/source Diffusers or internal fused layout → Z Image runtime lookup space).
- `apps/backend/runtime/state_dict/keymap_wan21_vae.py`: WAN2.1 VAE key-style resolver with strict canonical validation and no runtime key rewriting.
- `apps/backend/runtime/state_dict/keymap_wan22_vae.py`: WAN22 VAE key-style resolvers for 2D native and 3D diffusers/codex lanes (source-key validation + mixed-style/collision fail-loud).
- `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`: WAN22 transformer key-style resolver (Diffusers/WAN-export/Codex).
- `apps/backend/runtime/state_dict/tools.py`: Small state-dict utilities and diagnostics helpers.
- `apps/backend/runtime/state_dict/views.py`: Mapping views (prefix/filter/keyspace-lookup/computed-keyspace/cast) + `LazySafetensorsDict`.

Notes:
- Views should stay lightweight and avoid eagerly materializing large state dicts.
- 2026-03-21: `key_mapping.py` now fail-loud rejects any attempt to rewrite stored layer names during generic preprocessing (including prefix stripping / punctuation edits); keymaps must map source keyspaces explicitly through lookup views or computed views instead of mutating key strings.
- 2026-03-29: `keymap_clip_vision.py` now owns IP-Adapter CLIP vision image-encoder source-style resolution (bare HF including the known full-CLIP extra text/logit keys, explicit `image_encoder.*` wrapped image-encoder checkpoints, and explicit OpenCLIP including the known full-CLIP extra text/logit keys) into the canonical HF `CLIPVisionModelWithProjection` keyspace; adapter-local filtering/rekey/conversion glue is forbidden.
- 2026-03-28: `keymap_wan22_transformer.py` now owns explicit Wan LoRA wrapper-keyspace families (`diffusion_model.*`, `model.diffusion_model.*`, `model.model.diffusion_model.*`, `transformer.*`, `transformer_2.*`, `model.*`) for logical-key resolution; `stage_lora.py` no longer owns target-resolution fallback stripping for those families.
- 2026-03-20: `keymap_sdxl_vae.py` now exposes the shared metadata-filter seam used by external override prep; only known SDXL/Flow bookkeeping keys are dropped before strict keyspace resolution.
- 2026-03-25: `keymap_sdxl_clip.py` now treats missing CLIP `logit_scale` as an omitted-source case and lazily synthesizes the canonical `ln(100)` default, while duplicate native `logit_scale` sources still fail loud.
- 2026-01-25: `LazySafetensorsDict` is now truly lazy on non-Windows (persistent `safe_open` handle) and implements `__contains__` so key checks don’t load tensors; `KeyspaceLookupView` also implements `__contains__` for the same reason.
- Helpers should remain generic and not import model-family runtime code.
- Keyspace resolution must be explicit and strict: unknown/ambiguous layouts raise (no silent fallbacks). Use the family-specific keymap modules from loaders.
- 2026-02-10: Added canonical T5 text-encoder keymap (`keymap_t5_text_encoder.py`) so loader paths no longer perform ad-hoc inline key preprocessing.
- 2026-03-21: `keymap_sdxl_checkpoint.py` now resolves original-format SDXL checkpoints through explicit source-style mapping, including nested UNet label-embedding aliases (`label_emb.0.0.*` → `label_emb.0.*`) and only the known wrapper/container surfaces; detection now also recognizes VAE-only `first_stage_model.*` checkpoint surfaces instead of routing them into a no-sentinel dead end.
- 2026-03-21: `keymap_sdxl_clip.py` and `keymap_qwen_text_encoder.py` now restore their still-owned SDXL/Qwen wrapper/container source styles directly inside the owning keymaps through explicit one-hop source-style mapping; mixed/repeated wrapper chains stay invalid and active loader paths no longer depend on the removed generic rewrite seam.
- 2026-02-10: Structural conversion seams in keymaps are globally policy-gated by `CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto` fail-loud / `convert` explicit opt-in): SDXL CLIP blocks split↔fused QKV/projection conversion in `auto`, and SDXL VAE blocks 1x1-conv flattening in `auto`.
- 2026-02-11: `keymap_sdxl_clip.py` now exposes generic CLIP layout detection + resolver APIs (`detect_clip_layout_metadata`, `resolve_clip_keyspace_with_layout`) and SDXL wrappers with cache-hint support (`*_with_layout`) to avoid repeated style detection on warm SHA layout cache hits.
- 2026-02-11: SDXL CLIP projection handling is orientation-aware (`auto|linear|matmul`) instead of hard-coded transpose; AUTO keeps native orientation and only transposes when explicitly requested (and policy allows structural conversion).
- 2026-02-11: `keymap_sdxl_vae.py` now resolves mid-attention aliases under `encoder/decoder.mid.block_1.{q,k,v,proj_out,norm}.*`, `mid.block_1.attn_1.*`, and prefixed legacy `mid.attn_1.to_{q,k,v,out}.*` to canonical `mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}.*`, preventing SDXL VAE missing mid-attention keys on alias-style exports while preserving resnet-key lookup parity.
- 2026-02-11: `keymap_sdxl_vae.py` now also canonicalizes DIFFUSERS mid-attention legacy aliases (`*.mid_block.attentions.*.{query,key,value,proj_attn}.*` → `*.mid_block.attentions.*.{to_q,to_k,to_v,to_out.0}.*`) and fail-loud rejects any leftover alias outputs in validation.
- 2026-02-11: `keymap_sdxl_vae.py` now uses explicit projection lanes for SDXL VAE mid-attention weights independent of global structural-conversion policy: canonical 2D linear weights pass through, native 1x1 conv 4D weights pass through unchanged, and any non-canonical shape fails loud with key+shape context.
- 2026-02-11: Supersedes the SDXL VAE portion of the 2026-02-10 structural-policy note: SDXL VAE mid-attention projection lanes are now native (`linear_2d`/`conv1x1_4d`) and no longer use keymap flatten gating by `CODEX_WEIGHT_STRUCTURAL_CONVERSION`.
- 2026-02-15: `views.LazySafetensorsDict` now explicitly documents device-targeted lazy loads (`device` controls produced tensor placement; no CPU-only assumption).
- 2026-03-06: `views.py` now also exposes `ComputedKeyspaceView`, which mixes direct lookups with quantization-aware row concat/split/swap transforms for fused/unfused runtime keyspaces without mutating source checkpoints.
- 2026-02-28: `keymap_wan22_transformer.py` now exposes `resolve_wan22_lora_logical_key(...)` as canonical WAN22 LoRA logical-key → transformer target mapping authority.
- 2026-03-07: `resolve_wan22_lora_logical_key(...)` now reuses the current WAN22 export/native vocabulary for stage-LoRA norms, top-level alias families, and modulation targets (`blocks.N.modulation` / `head.modulation`); keep WAN LoRA support inside this interpretation seam only—no runtime remap or compatibility shims elsewhere.
- 2026-03-07: WAN22 LoRA logical-key coverage now also includes local norm families (`self_attn.norm_q/norm_k`, `cross_attn.norm_q/norm_k`, `norm3`) and WAN export/native top-level aliases (`patch_embedding|patch_embed`, `time_embedding|time_embed`, `time_projection|time_proj`, `text_embedding|text_embed`, `head.head|head`) without any runtime key renaming. Upstream image-branch families absent from the local WAN22 runtime remain explicitly unsupported at the stage-LoRA seam and must surface through diagnostics/partial coverage instead of silent acceptance.
- 2026-03-06: WAN video request allowlists were moved out of `keymap_wan22_transformer.py` into `apps/backend/interfaces/api/wan_video_request_keys.py`; the WAN22 keymap module now owns only transformer keyspace understanding and LoRA logical-key resolution.
- 2026-03-03: Added strict generic Qwen text-encoder keymap (`keymap_qwen_text_encoder.py`) covering the native HF backbone keys plus known auxiliary heads (`lm_head.*`, `visual.*`) as the canonical destination contract.
- 2026-03-06: Added family-scoped Flux / FLUX.2 / Z Image GGUF keyspace resolvers so native/source checkpoints can be interpreted through lazy lookup views (including fused/unfused tensor conventions) without materializing remapped state dicts.
- 2026-03-12: Added strict Gemma3 text-only GGUF keyspace resolution in `keymap_gemma3_text_encoder.py` for the LTX2 Gemma3 external loader path; it accepts llama.cpp GGUF text keys or already-native `Gemma3TextModel` keys and exposes only a lookup view.
- 2026-03-21: `keymap_gemma3_text_encoder.py` now rejects documented wrapper-prefix rewrite inputs (`model.`, `language_model.`, `base_text_encoder.`) until the source layout is modeled explicitly; the strict lookup view remains canonical GGUF-or-native-HF only.
- 2026-03-31: Added `keymap_anima_transformer.py` as the explicit raw `net.*`/canonical Anima transformer keyspace owner; parser and loader now keep stored core keys native and resolve them through lazy lookup views instead of parser-side prefix stripping or eager normalization.
- 2026-07-30: `keymap_qwen_text_encoder.py` adds the exact 729-key Qwen2.5-VL multimodal source contract and exposes a 728-key runtime lookup view (`model.*` stored keys -> `language_model.*` runtime names; `visual.*` unchanged) while separately validating and excluding stored `lm_head.weight`.
- 2026-07-30: Added `keymap_qwen_image_transformer.py` for the exact Edit-2511 native transformer keyspace. It accepts no wrappers or alternate layouts and never rewrites stored keys.
- 2026-07-31: The exact Edit-2511 transformer and Qwen2.5-VL keymaps validate every stored logical tensor shape before executable binding; matching names/counts alone are not a strict-load proof for GGUF custom operations.

Last Review: 2026-07-31
