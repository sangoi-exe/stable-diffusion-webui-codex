# ComfyUI & SwarmUI – Activity Since the Forge Freeze (After 2025-07-31)

## Reference Point
- Forge’s last upstream push landed on **2025-07-31** (`ae278f79`). All items below track what the two leading alternatives – **ComfyUI** and **SwarmUI** – have shipped between **2025-07-31** and **2025-10-10**.

---

## ComfyUI (comfyanonymous/ComfyUI)

### Release cadence
- 2025-08-02 – v0.3.48 ([bff60b5c](https://github.com/comfyanonymous/ComfyUI/commit/bff60b5cfc10d1b037a95746226ac6698dc3e373))
- 2025-08-05 – v0.3.49 ([32a95bba](https://github.com/comfyanonymous/ComfyUI/commit/32a95bba8ac91e8a610c35ce4d9963d2453118c1))
- 2025-08-13 – v0.3.50 ([d5c1954d](https://github.com/comfyanonymous/ComfyUI/commit/d5c1954d5cd4a789bbf84d2b75a955a5a3f93de8))
- 2025-08-20 – v0.3.51 ([7139d6d9](https://github.com/comfyanonymous/ComfyUI/commit/7139d6d93fc7b5481a69b687080bd36f7b531c46))
- 2025-08-23 – v0.3.52 ([71ed4a39](https://github.com/comfyanonymous/ComfyUI/commit/71ed4a399ec76a75aa2870b772d2022e4b9a69a3))
- 2025-08-28 – v0.3.54 ([00636101](https://github.com/comfyanonymous/ComfyUI/commit/00636101771cb373354d6294cc6567deda2635f6))
- 2025-08-30 – v0.3.56 ([4449e147](https://github.com/comfyanonymous/ComfyUI/commit/4449e147692366ac8b9bd3b8834c771bc81e91ac))
- 2025-09-10 – v0.3.58 ([8d7c9302](https://github.com/comfyanonymous/ComfyUI/commit/8d7c930246bd33c32eb957b01ab0d364af6e81c0))
- 2025-09-10 – v0.3.59 ([72212fef](https://github.com/comfyanonymous/ComfyUI/commit/72212fef660bcd7d9702fa52011d089c027a64d8))
- 2025-09-23 – v0.3.60 ([b8730510](https://github.com/comfyanonymous/ComfyUI/commit/b8730510db30c8858e1e5d8e126ef19eac395560))
- 2025-09-30 – v0.3.61 ([977a4ed8](https://github.com/comfyanonymous/ComfyUI/commit/977a4ed8c55ade53d0d6cfe1fe8a6396ee35a2ec)) and v0.3.62 ([bab8ba20](https://github.com/comfyanonymous/ComfyUI/commit/bab8ba20bf47d985d6b1d73627c2add76bd4e716))
- 2025-10-06 – v0.3.63 ([6bd3f8eb](https://github.com/comfyanonymous/ComfyUI/commit/6bd3f8eb9ff2d7c74e8ca75ad1f854a6b256b714))
- 2025-10-08 – v0.3.64 ([63722199](https://github.com/comfyanonymous/ComfyUI/commit/637221995f7424a561bd825de3e61ea117dfe1e3))

### Platform evolution
- **V3 node schema rollout** – kicked off with the base schema definition ([4887743a](https://github.com/comfyanonymous/ComfyUI/commit/4887743a2aef67e05909aeea61f6cdc93e269de3)) and followed by an aggressive migration covering WAN, Stable Cascade, Video, String, CFG, audio encoders, API wrappers (Ideogram, Veo, Moonvalley, ByteDance, PixVerse, Rodin, Pika, Kling, Luma, Flux, SD3, etc.) through early October (e.g. [9201](https://github.com/comfyanonymous/ComfyUI/commit/9201), [9721](https://github.com/comfyanonymous/ComfyUI/commit/9721), [10019](https://github.com/comfyanonymous/ComfyUI/commit/10019), [10160](https://github.com/comfyanonymous/ComfyUI/commit/cbee7d33909f168a08ab7e53d897ea284a304d84), [10199](https://github.com/comfyanonymous/ComfyUI/commit/2ba8d7cce8b6d78efa4b853ae8df187bb13061a3)).
- **Partial graph execution** – backend now accepts `partial_execution_targets`, enabling incremental reruns without replaying entire workflows ([97eb256a](https://github.com/comfyanonymous/ComfyUI/commit/97eb256a355b434bbc96ec27bbce33dd10273857)).
- **Async API nodes** – the entire remote API catalog (OpenAI, Gemini, Moonvalley, Pika, Recraft, Luma, Veo, Tripo, etc.) was converted to async execution for better concurrency and error handling ([bf2a1b5b](https://github.com/comfyanonymous/ComfyUI/commit/bf2a1b5b1ef72b240454f3ac44f5209af45efe00)).

### Model & workflow support
- **USO pipeline** – added subject identity LoRA & projector support ([3412d53b](https://github.com/comfyanonymous/ComfyUI/commit/3412d53b1d69e4dfedf7e86c3092cea085094053), [27e067ce](https://github.com/comfyanonymous/ComfyUI/commit/27e067ce505c102fd0f2be0f1242016c59a6816f)).
- **Wan 2.2 / 5B and WanAnimate** – support for fun control, inpaint, memory adjustments, and bug fixes for animation/video nodes ([e80a14ad](https://github.com/comfyanonymous/ComfyUI/commit/e80a14ad5073d9eba175c2d2c768a5ca8e4c63ea), [c7bb3e2b](https://github.com/comfyanonymous/ComfyUI/commit/c7bb3e2bceaad7accd52c23d22b97a1b6808304b), [707b2638](https://github.com/comfyanonymous/ComfyUI/commit/707b2638ecd82360c0a67e1d86cc4fdeae218d03)).
- **Qwen Image** – initial model loader, merging utilities, LyCORIS training support and doc updates landed in early August ([c0124002](https://github.com/comfyanonymous/ComfyUI/commit/c012400240d4867cd63a45220eb791b91ad47617), [9126c0cf](https://github.com/comfyanonymous/ComfyUI/commit/9126c0cfe49508a64c429f97b45664b241aab3f2), [2208aa61](https://github.com/comfyanonymous/ComfyUI/commit/2208aa616d3ad193cd37ef57076d4f5243cecdd3)).
- **ByteDance, Rodin3D, Wan T2I/T2V, Gemma 3** – expanded third-party integrations: ByteDance image nodes ([26d5b86d](https://github.com/comfyanonymous/ComfyUI/commit/26d5b86da8ceb4589ee70f12ff2209b93a2d99e0)), Rodin3D Gen-2 API ([341b4ade](https://github.com/comfyanonymous/ComfyUI/commit/341b4adefd308cbcf82c07effc255f2770b3b3e2)), Wan T2I/T2V/I2V nodes ([e8087907](https://github.com/comfyanonymous/ComfyUI/commit/e8087907995497c6971ee64bd5fa02cb49c1eda6)), and Gemma 3 text encoder support ([8aea7462](https://github.com/comfyanonymous/ComfyUI/commit/8aea746212dc1bb1601b4dc5e8c8093d2221d89c)).
- **Veo3 video** – new node with audio generation and API path updates ([f69609bb](https://github.com/comfyanonymous/ComfyUI/commit/f69609bbd6c20f4814e313f8974656b187a9bee2)).

### Audio / video tooling
- Added audio trimming, channel splitting, concatenation, merging, volume control and empty-audio primitives to native nodes ([fd79d32f](https://github.com/comfyanonymous/ComfyUI/commit/fd79d32f38fd24adca5a6e8214f05050f287c9db)).
- Latent editing helpers (cut, concat) and S2V improvements for long videos ([d28b39d9](https://github.com/comfyanonymous/ComfyUI/commit/d28b39d93dc498110e28ca32c8f39e6de631aa42), [38f697d9](https://github.com/comfyanonymous/ComfyUI/commit/38f697d953c3989db67e543795768bf954ae0231), [496888fd](https://github.com/comfyanonymous/ComfyUI/commit/496888fd68813033c260195bf70e4d11181e5454)).
- Veo3 audio toggle and USO style references extend multi-modal workflows ([f69609bb](https://github.com/comfyanonymous/ComfyUI/commit/f69609bbd6c20f4814e313f8974656b187a9bee2), [3412d53b](https://github.com/comfyanonymous/ComfyUI/commit/3412d53b1d69e4dfedf7e86c3092cea085094053)).

### Performance & infrastructure
- Convolution autotuning by default to improve inference throughput ([e2d1e5da](https://github.com/comfyanonymous/ComfyUI/commit/e2d1e5dad98dbbcf505703ea8663f20101e6570a)).
- Reduced RAM/VRA​M footprint for Windows and Flux workflows ([885015ee](https://github.com/comfyanonymous/ComfyUI/commit/885015eecf649d6e49e1ade68e4475b434517b82), [4aa79dbf](https://github.com/comfyanonymous/ComfyUI/commit/4aa79dbf2c5118853659fc7f7f8590594ab72417)).
- Memory leak fix when swapping checkpoints by detaching finalizers correctly ([c8d2117f](https://github.com/comfyanonymous/ComfyUI/commit/c8d2117f02bcad6d8316ffd8273bdc27adf83b44)).
- Torchaudio & soundfile dependency removal to slim installs ([caf07331](https://github.com/comfyanonymous/ComfyUI/commit/caf07331ff1b20f4104b9693ed244d6e22f80b5a)).
- Release tooling migrated to Python **3.13** + CUDA **12.9** for the official builds ([5ca8e2fa](https://github.com/comfyanonymous/ComfyUI/commit/5ca8e2fac3b6826261c5991b0663b69eff60b3a1)).

### Developer & ecosystem quality
- Recently-used asset API for UI hints ([3dfefc88](https://github.com/comfyanonymous/ComfyUI/commit/3dfefc88d00bde744b729b073058a57e149cddc1)), new CODEOWNERS entries ([c4a46e94](https://github.com/comfyanonymous/ComfyUI/commit/c4a46e943c12c7f3f6ac72f8fb51caad514ec9b6)), stricter pylint rules ([6ae35158](https://github.com/comfyanonymous/ComfyUI/commit/6ae35158013e50698e680344ab1f54de0d59fef0)) and lazy logging ([51fb505f](https://github.com/comfyanonymous/ComfyUI/commit/51fb505ffa7cdae113ef4303f9ef45a06d668a90)).
- Added Sampler DPM++ 2M SDE Heun (RES) ([3aad339b](https://github.com/comfyanonymous/ComfyUI/commit/3aad339b63f03e17dc6ebae035b90afc2fefb627)) and LoRA trainer fixes for FP8 models ([7be2b49b](https://github.com/comfyanonymous/ComfyUI/commit/7be2b49b6b3430783555bc6bc8fcb3f46d5392e7)).

---

## SwarmUI (mcmonkeyprojects/SwarmUI)

### Auto-scaling & backend orchestration
- Introduced the `AutoScalingBackend`, including backend-aware scaling triggers for direct Comfy queues, decimal-capable settings, new CLI knobs (`no-persist`, `require_control_within`, `AllBackendsLoadFast`) and comprehensive docs ([35ef87e7](https://github.com/mcmonkeyprojects/SwarmUI/commit/35ef87e75a4ea228e570a380afe7713956250f1c), [7e06e6cb](https://github.com/mcmonkeyprojects/SwarmUI/commit/7e06e6cb06be77046a87a8e5e038b2598fc3e0b3), [6a8936e3](https://github.com/mcmonkeyprojects/SwarmUI/commit/6a8936e334ccb98330d9d2508b8239272ad94b6e), [2ed14355](https://github.com/mcmonkeyprojects/SwarmUI/commit/2ed14355b972ab8ee7af6540d63c1b66296e11e5), [0b24b3cf](https://github.com/mcmonkeyprojects/SwarmUI/commit/0b24b3cfee317d0da900b2842ea64fa7ef82260f), [8dc3cf15](https://github.com/mcmonkeyprojects/SwarmUI/commit/8dc3cf151b628da7e10f1dcb44961b5e115a746e)).
- Tightened swarm-of-swarms control awareness and silenced overly chatty Comfy polling ([82dd7828](https://github.com/mcmonkeyprojects/SwarmUI/commit/82dd7828bbc77f21f1db243f197b8f83a87b91ee), [130e7c29](https://github.com/mcmonkeyprojects/SwarmUI/commit/130e7c29ea0aad960327a90521269f7ea4489ebe)).

### Comfy integration enhancements
- Added `ComfyGetNodeTypesForBackend`, nodelist awareness on remotes, and workflow failure handling for smoother front-end UX ([186fe3d2](https://github.com/mcmonkeyprojects/SwarmUI/commit/186fe3d251aff4f5472139ef8a8f7e24b6192760), [cb753d21](https://github.com/mcmonkeyprojects/SwarmUI/commit/cb753d2148d6e3750896abd85971c382da44e53d), [1188cf00](https://github.com/mcmonkeyprojects/SwarmUI/commit/1188cf000af6d4fe0e99b7872ca100b2e72a8f42)).
- Provided smoother loading and adaptive workflow tab visibility tied to feature flags ([f7892300](https://github.com/mcmonkeyprojects/SwarmUI/commit/f78923009be776ef959395819e6f972e7563d33f), [f6d7f0ac](https://github.com/mcmonkeyprojects/SwarmUI/commit/f6d7f0acbf159701e19e6b376ef89e5ae43192b0)).

### UX & productivity updates
- Smart Image Prompt Resizing toggle, explicit icon removal, mouse/keyboard improvements in full-view mode, and clickable trigger phrases / prompt-style copy buttons ([ed48f1a6](https://github.com/mcmonkeyprojects/SwarmUI/commit/ed48f1a6dfb5a2f6ec2acace76f0912ab53bed82), [571f9e26](https://github.com/mcmonkeyprojects/SwarmUI/commit/571f9e264acd815836343724388ecb4cee6b8776), [1db0e7af](https://github.com/mcmonkeyprojects/SwarmUI/commit/1db0e7af5fb82db99ebe1b351003acd928f66228), [ba61a22a](https://github.com/mcmonkeyprojects/SwarmUI/commit/ba61a22a4cdb58c82b91d5d6d5976c9f8aeed03b)).
- Status bar fixes and theme cleanups to stabilise the modern UI skin ([c9f8feef](https://github.com/mcmonkeyprojects/SwarmUI/commit/c9f8feef1224ad74121aecd47ff7a5f64e52cb2b), [49559168](https://github.com/mcmonkeyprojects/SwarmUI/commit/4955916859df50a7d01319dde15a54921cde835f), [dc2c039e](https://github.com/mcmonkeyprojects/SwarmUI/commit/dc2c039e799c3bdc5957c8766d3a0ae896774891)).

### Model & pipeline support
- Qwen LoRA detection patches and new Nunchaku node wiring keep pace with upstream formats ([bc02c401](https://github.com/mcmonkeyprojects/SwarmUI/commit/bc02c40147c211b46da8e5eb67c2335a717cf3c1), [d375d00c](https://github.com/mcmonkeyprojects/SwarmUI/commit/d375d00c755a4dbcaecb2d665746ec868aa06355)).
- SD3 (RF) sampler option, Wan 2.2 VideoSwap improvements, swap-model controls for Video Extend, and resumable model downloads bolster generation ergonomics ([287bd733](https://github.com/mcmonkeyprojects/SwarmUI/commit/287bd7332c9986289d5e7821042a00eba0d3a984), [3a3ee551](https://github.com/mcmonkeyprojects/SwarmUI/commit/3a3ee551c9a799755b60afa38cc4714e1bf4ddaf), [dc9e51d5](https://github.com/mcmonkeyprojects/SwarmUI/commit/dc9e51d5ab84f9f7ff38aa83ce4737b2175006f0), [a017df37](https://github.com/mcmonkeyprojects/SwarmUI/commit/a017df375e7bf8224e861c61452dfde4d2b4ade0)).
- Added global status polling API and better handling of prompt interruptions & queue hygiene ([84996c70](https://github.com/mcmonkeyprojects/SwarmUI/commit/84996c70c861c4c917ec71be35e27b769798fcad), [1cbde37b](https://github.com/mcmonkeyprojects/SwarmUI/commit/1cbde37b72ffd62f50ad6c441400e697f5e7a137)).

### Documentation & configuration
- Autoscaling, WAN 2.2, light 2.2 references and Linux build tweaks updated the public docs/build scripts ([8dc3cf15](https://github.com/mcmonkeyprojects/SwarmUI/commit/8dc3cf151b628da7e10f1dcb44961b5e115a746e), [fe05bc41](https://github.com/mcmonkeyprojects/SwarmUI/commit/fe05bc41d30784893e566695b4af66d80048c781), [b3c54606](https://github.com/mcmonkeyprojects/SwarmUI/commit/b3c54606ad4a7cb2b067140c0ffe20744bc3a5ce)).

---

Both projects have moved rapidly in the two months following Forge’s freeze, with ComfyUI delivering a comprehensive node-schema refresh plus broad model/API support, and SwarmUI investing heavily in autoscaling, Comfy integration depth, and day-to-day operator ergonomics. This gap analysis should feed directly into Forge’s refactor scope and documentation.***
