 # Research — ComfyUI and SwarmUI (model families and implications)

 Date: 2025-10-18

 ComfyUI
 - Nature: node-based workflow UI; engine-agnostic with broad model family support. The official README lists support for SD 1.x/2.x/XL, Stable Cascade, SD3/3.5, PixArt Sigma, AuraFlow, HunyuanDiT, Flux, Lumina Image 2.0. Source: comfyorg/comfyui README (Features). [See refs]
 - External/API: Comfy Hub/API documentation shows integration points for many models (Stable Diffusion family, Flux, PixArt, SD3/3.5, etc.). Useful as reference for families and metadata fields. [See refs]

 SwarmUI (aka StableSwarmUI)
 - Nature: A1111-like multi-backend UI with broad, rapidly expanding model support across image and video. Recent release notes and docs list detailed model support pages for image and video.
 - Video: Docs mention support for Qwen Image (for image tasks), OmniGen 2, Flux Kontext, Wan 2.1/2.2 (text/video), Genmo Mochi 1, Lightricks LTX Video, NVIDIA Cosmos, Hunyuan Video, SVD, and more, with guidance per model. [See refs]

 Key takeaways for our plan
 - Families to plan first-class support for (Image): SD15, SD2.x, SDXL(+Turbo), SD3/3.5, Flux 1.*, PixArt Sigma, AuraFlow, HunyuanDiT, Lumina Image 2.0, Stable Cascade.
 - Families to plan for Video: SVD (img2vid), Hunyuan Video, Mochi 1, LTX, NVIDIA Cosmos (Text2World/Video), Wan 2.1/2.2; evaluate Flux video variants as they stabilize.
 - UI impact: “inference type” selection will grow; replace radio buttons with dropdown and dynamic fields per engine/task.
 - Extra networks/LoRAs: follow A1111/Forge token-in-prompt convention; engines that can’t apply must log a clear warning and proceed.

 References (opened on Oct 18, 2025)
 - ComfyUI GitHub features list with supported model families. 
 - ComfyUI Hub/API docs with model catalog categories. 
 - SwarmUI releases referencing detailed docs (Model Support.md, Video Model Support.md) with lists like Qwen Image, OmniGen 2, Flux Kontext, Wan 2.2, SVD, Hunyuan Video, Mochi 1, LTX, Cosmos, etc. 

 Notes
 - We will not mirror node graphs; we use the families and feature patterns for engine modules and presets.

