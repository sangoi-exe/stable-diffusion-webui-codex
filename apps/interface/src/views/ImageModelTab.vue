<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Image model tab view (txt2img/img2img/inpaint) UI for SD/Flux/ZImage/Z-Image L2P/Qwen Image-family engines.
Owns prompt + parameter controls, init-image + mask handling for img2img/inpaint, per-tab history, and integrates with the generation composable to
submit `/api/txt2img`/`/api/img2img` tasks and render progress/results (Z-Image Turbo/Base and FLUX.2 Klein distilled/base-4B are variant-dependent:
CFG label + negative prompt gating follow the selected checkpoint/tab state, while img2img denoise + hires visibility stay truthful to the active capability/mask contract).
The RUN surface now owns a split-button `Generate` / `Infinite` action selector, the Initial Image seam owns the `DIR|IMG` source switch for img2img automation,
IP-Adapter stays on its dedicated nested-owner card, and SUPIR mode now lives on the header toggle plus a split body surface: `Img2ImgBasicParametersCard` owns the SUPIR sampler/scheduler row,
while `SupirModeCard` owns the remaining SUPIR-specific controls with parent-owned blocking/gating/readiness.
When inpaint masking is active, it also forwards natural init-image dimensions, current processing target dimensions, and the current invert-mask state to the shared card/editor preview seam, treats unresolved natural dims as unavailable instead of falling back to processing dims, and normalizes the invalid `maskInvert + maskRegionSplit` pair in the shared parent-owned param path.
Exact-engine inpaint-mode availability now comes from `/api/engines/capabilities` (`exact_engine_inpaint_modes`), so SDXL can expose `Fooocus Inpaint` truthfully without laundering that mode onto `sdxl_refiner` or non-SDXL engines; stale unsupported values stay visible/blocking until the user reselects a supported mode.
When `useInitImage=true`, generation parameters render through `Img2ImgBasicParametersCard` (shared layout with honest img2img control visibility).
CFG Advanced/APG controls are capability-gated (`engineSurface.guidance_advanced`) and persist through tab params/profile snapshots.
Hires settings list upscalers from `/api/upscalers` and share tile controls with `/upscale`.
Also shares the global `min_tile` preference (tiled lower bound) with `/upscale`.
The generic first-pass model-swap stage lives under `params.swapModel`, the generic second-pass model selector lives under `params.hires.swapModel`,
and global + hires refiner cards stay on the SDXL-native `refiner` seams with the shared capability-gated advanced guidance/APG state surface.
Sampler/scheduler selectors normalize current/backend-capability selections against the executable `/api/samplers` + `/api/schedulers` inventory, keep base sampler/scheduler real without frontend fallback defaults, and scrub invalid hires overrides while still using backend recommendation lists for grouped option rendering (`Recommended` vs `Use at your own risk`) with inline technical warnings on out-of-recommendation selections.
Surfaces a one-shot toast when the generation composable auto-reattaches to an in-flight task after a reload/crash.
Generate CTA and run preflight are capability-driven (`/api/engines/capabilities`) plus asset-contract-aware, failing loud in the UI when the
current checkpoint/text-encoder/VAE contract is not runnable.
Run status in the RUN card is centralized via `RunProgressStatus` variants (progress/error/info/success/warning), including dual progress bars (total pipeline + sampling steps), so errors are visible even when Prompt is off-screen.
When XYZ workflow is enabled, RUN header shows an `XYZ` badge beside `Generate` via the run-card center-adjacent slot while keeping the primary CTA label stable as `Generate`.
Qwen Image uses the same image workspace shell but hides unsupported generic controls and the misleading text-only prompt-token counter, keeps batch/action state forced to one-shot Generate,
forces single-image Edit-2511 img2img, omits TXT2IMG/XYZ/source-mode/inpaint/CLIP Skip/swap/refiner/hires/IP-Adapter/advanced-guidance controls, and treats edit output dimensions as backend-derived from the init image.
Z-Image L2P uses the same image workspace shell but hides unsupported generic controls, keeps batch/action state forced to one-shot Generate,
forces 1024x1024 txt2img, omits VAE/CLIP Skip/swap/refiner/hires/IP-Adapter/advanced-guidance controls, and relies on the QuickSettings L2P TEnc selector.

Symbols (top-level; keep in sync; no ghosts):
- `ImageModelTab` (component): Main image model tab view; handles prompt/params/profile persistence, init-image UX, history reuse, and actions.
- `sendToWorkflows` (function): Sends the current params snapshot to the workflows subsystem (async).
- `copyCurrentParams` (function): Copies current params snapshot to clipboard (async).
- `onCancelRun` (function): Cancels the active run (XYZ sweep immediate stop or current image task cancel).
- `showSupirModeCard` / `supportsSupirModeSurface` / `supirSelectionState` / `supirBlockingReason` (const): Shared SUPIR discoverability/readiness contract for the header toggle and split img2img parameter surface.
- `availableInpaintModes` / `availableInpaintModeOptions` / `unsupportedInpaintMode` / `unsupportedInpaintModeMessage` (const): Exact-engine-owned inpaint-mode availability plus explicit invalid-submit blocking for the shared inpaint card.
- `onInpaintModeChange` (function): Applies one validated inpaint-mode update and clears hidden per-step sliders when leaving `Per-step blend`.
- `copyHistoryParams` (function): Copies a history entry’s params snapshot to clipboard (async).
- `applyHistory` (function): Applies a history entry back into current state (prompt/params/assets).
- `formatHistoryTitle` (function): Builds a human-friendly history title from a run entry.
- `profileStorageKeyFor` (function): Computes the localStorage key for saving/loading per-engine profiles.
- `loadProfile` (function): Loads a saved profile into current params (with validation/defaulting).
- `saveProfile` (function): Saves current params as a profile in localStorage.
- `setParams` (function): Applies partial updates to the current tab params state.
- `normalizeImageDimension` (function): Snaps width/height updates to the active engine grid before they reach tab state.
- `normalizeImageParamPatch` (function): Applies engine-aware width/height + img2img resize-mode normalization plus inpaint toggle interlock cleanup to partial param patches.
- `syncImageContractToEngine` (function): Reconciles persisted width/height/resize-mode state with the active engine contract.
- `normalizeGuidanceAdvancedPatch` (function): Sanitizes/normalizes advanced-guidance payload fragments (profile + UI patch merges).
- `setGuidanceAdvanced` (function): Applies partial advanced-guidance updates into `params.guidanceAdvanced`.
- `setHires` (function): Applies partial updates to the hires config object.
- `setHiresSwapModel` (function): Applies the canonical nested hires `swapModel` selection without leaking flat alias fields.
- `setHiresRefiner` (function): Applies partial updates to the hires-refiner config object.
- `setRefiner` (function): Applies partial updates to the refiner config object.
- `setRunAction` (function): Persists the Run-card primary action mode (`generate` vs `infinite`) and normalizes automation batch constraints.
- `setInitSource` (function): Applies Initial Image `DIR|IMG` source changes and clears stale mask / same-as-init dependents on DIR mode.
- `setIpAdapter` (function): Applies partial updates to the dedicated IP-Adapter owner (including nested source patches) and enforces automation batch constraints.
- `setIpAdapterSource` (function): Thin helper for nested IP-Adapter source patches.
- `clampFloat` (function): Clamps a float to `[min, max]` (input sanitation).
- `setMinTile` (function): Updates the global `min_tile` preference used as the tiled OOM fallback lower bound (hires-fix + `/upscale`).
- `snapInitImageDim` (function): Snaps init-image derived dimensions to the active engine grid before reuse/sync.
- `onInitFileSet` (function): Reads an init image file into a data URL and stores name/data, then syncs dims (async).
- `onInitImageRejected` (function): Surfaces dropzone reject reasons for init-image input.
- `clearInit` (function): Clears init image fields.
- `clearMask` (function): Clears mask fields.
- `onIpAdapterReferenceFileSet` (function): Reads an IP-Adapter reference image into the dedicated card source state (async).
- `clearIpAdapterReference` (function): Clears the dedicated IP-Adapter reference image fields.
- `onIpAdapterReferenceRejected` (function): Surfaces dropzone reject reasons for the IP-Adapter reference input.
- `onMaskEditorApply` (function): Validates and stores an edited mask exported from the inpaint mask editor overlay.
- `onMaskEditorResetNotice` (function): Surfaces inpaint mask editor source-reset notices as toasts.
- `toDataUrl` (function): Converts a generated image payload to a data URL for preview.
- `randomizeSeed` (function): Randomizes the seed field for the current tab params.
- `reuseSeed` (function): Reuses the last seed from history/current run as the next seed.
- `download` (function): Downloads a generated image artifact to disk.
- `sendToImg2Img` (function): Sends a generated image back into img2img init-image fields (async).
- `syncInitImageDims` (function): Synchronizes init-image derived dimensions into width/height params (async).
- `maskEditorImageWidth`/`maskEditorImageHeight` (const): Derived init-image dimensions used by the inpaint mask editor canvas (keeps backend mask-dimension contract).
- `maybeApplyKontextDefaults` (function): Applies FLUX.1 Kontext-specific default params when relevant to the current engine/tab.
- `onGenerate` (function): Run handler for the Run card; dispatches standard generation or XYZ sweep depending on XYZ enable state.
- `runGenerateDisabled`/`runGenerateTitle` (const): Run CTA state/title derived from capabilities + active mode + XYZ running/enabled state.
- `assetContractBlockingReason` / `workflowParamsSnapshot` (const): Current image asset-contract gating reason plus workflow snapshot carrying the
  active family-scoped VAE owner for VAE-owning image families.
- `usesImageAutomation` / `infiniteXyzConflict` / `automationBatchConflict` (const): Derived automation guards for the split-button and backend-owned automation route.
- `showIpAdapterCard` / `initFolderMissingPath` / `dirInitMaskConflict` / `ipAdapterBlockingReason` (const): Card-visibility + preflight guards for Initial Image DIR mode and the dedicated IP-Adapter owner card.
- `isQwenImageRequest` / `isZImageL2PRequest` / `showRunBatchControls` (const): Exact one-shot surface guards that hide unsupported generic controls and force one-shot request state.
- `promptTokenEngine` (const): Omits token-count requests for Qwen Edit because its executable conditioning count includes image-token expansion rather than a truthful text-only total.
- `missingInpaintMask` (const): Derived guard flag used to disable generation when INPAINT is enabled without an applied mask.
- `supportsImg2ImgMasking` (const): Truthful backend-capability-driven mask/inpaint support gate for img2img engines.
- `hideNegativePrompt` (const): Hides the base Negative Prompt field when the active checkpoint/model does not support it or effective base CFG is `<= 1`.
- `recommendedSamplers` / `recommendedSchedulers` (const): Sanitized recommendation lists passed into sampler/scheduler selectors.
- `resolveLiveSamplingDefaults` (function): Resolves executable sampler/scheduler defaults from backend capabilities only.
- `normalizeLiveSamplingSelection` (function): Normalizes sampler/scheduler pairs against live executable catalog, family capability constraints, and sampler-allowed schedulers.
- `normalizedBaseSampling` (const): Live-normalized base sampler/scheduler pair used by selector state and hires override cleanup.
- `xyzSamplerChoices`/`xyzSchedulerChoices` (const): Sampler/scheduler names passed to embedded XYZ autofill (scheduler list is sampler-compatible).
- `normalizeXyzSamplingAxisText` (function): Scrubs XYZ sampler/scheduler axis values to current family-compatible choices.
-->

<template>
  <section v-if="tab" class="panels">
    <!-- Left column: Prompt + Parameters -->
    <div class="panel-stack">
      <PromptCard
        v-model:prompt="promptText"
        v-model:negative="negativeText"
        :supportsNegative="supportsNegative"
        :hide-negative="hideNegativePrompt"
        :token-engine="promptTokenEngine"
        :enableAssets="enableAssets"
        :enableStyles="enableStyles"
        :toolbarLabel="toolbarLabel"
        :fieldsId="`image-modeltab-prompt-${tabId}`"
      >
        <div v-if="supportsImg2Img && params.useInitImage" class="panel-section">
          <InitialImageBlock
            :disabled="isRunning"
            :showSourceModeToggle="!isQwenImageRequest"
            :showInpaintControls="!isQwenImageRequest"
            :initSource="params.initSource"
            :initImageData="params.initImageData"
            :initImageName="params.initImageName"
            :imageWidth="maskEditorImageWidth"
            :imageHeight="maskEditorImageHeight"
            :processingWidth="params.width"
            :processingHeight="params.height"
            :useMask="supportsImg2ImgMasking && params.initSource.mode === 'img' ? params.useMask : false"
            :maskImageData="params.maskImageData"
            :maskImageName="params.maskImageName"
            :inpaintMode="params.inpaintMode"
            :inpaintModeOptions="availableInpaintModeOptions"
            :perStepBlendStrength="params.perStepBlendStrength"
            :perStepBlendSteps="params.perStepBlendSteps"
            :inpaintingFill="params.inpaintingFill"
            :inpaintFullResPadding="params.inpaintFullResPadding"
            :maskBlur="params.maskBlur"
            :maskInvert="params.maskInvert"
            :maskRegionSplit="params.maskRegionSplit"
            @patch:initSource="setInitSource"
            @set:initImage="onInitFileSet"
            @clear:initImage="clearInit"
            @reject:initImage="onInitImageRejected"
            @clear:maskImage="clearMask"
            @apply:maskImageData="onMaskEditorApply"
            @notice:maskEditorReset="onMaskEditorResetNotice"
            @update:inpaintMode="onInpaintModeChange"
            @update:perStepBlendStrength="(v) => setParams({ perStepBlendStrength: clampFloat(v, 0, 1) })"
            @update:perStepBlendSteps="(v) => setParams({ perStepBlendSteps: normalizeNonNegativeInt(v) })"
            @update:inpaintingFill="(v) => setParams({ inpaintingFill: normalizeInpaintingFill(v) })"
            @update:inpaintFullResPadding="(v) => setParams({ inpaintFullResPadding: normalizeNonNegativeInt(v) })"
            @update:maskBlur="(v) => setParams({ maskBlur: normalizeNonNegativeInt(v) })"
            @toggle:maskInvert="setParams({ maskInvert: !params.maskInvert })"
            @toggle:maskRegionSplit="setParams({ maskRegionSplit: !params.maskRegionSplit })"
          />
        </div>
      </PromptCard>

      <IpAdapterCard
        v-if="showIpAdapterCard"
        :disabled="isRunning"
        :img2imgMode="params.useInitImage"
        :ipAdapter="params.ipAdapter"
        :modelChoices="ipAdapterModelChoices"
        :imageEncoderChoices="ipAdapterImageEncoderChoices"
        :blockingReason="ipAdapterBlockingReason"
        @patch:ipAdapter="setIpAdapter"
        @set:referenceImage="onIpAdapterReferenceFileSet"
        @clear:referenceImage="clearIpAdapterReference"
        @reject:referenceImage="onIpAdapterReferenceRejected"
		          />

      <div class="panel">
        <div class="panel-header">
          Generation Parameters
          <div class="toolbar">
            <button class="btn btn-sm btn-secondary" type="button" :disabled="isRunning" @click="loadProfile">Load profile</button>
            <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="saveProfile">Save profile</button>
          </div>
        </div>
        <div class="panel-body">
          <template v-if="params.useInitImage">
            <Img2ImgBasicParametersCard
              :samplers="filteredSamplers"
              :schedulers="filteredSchedulers"
              :recommended-samplers="recommendedSamplers"
              :recommended-schedulers="recommendedSchedulers"
              :upscalers="upscalers"
              :upscalersLoading="upscalersLoading"
              :upscalersError="upscalersError"
              :sampler="params.sampler"
              :scheduler="params.scheduler"
              :steps="params.steps"
              :min-steps="isQwenImageRequest ? 2 : 1"
              :width="params.width"
              :height="params.height"
              :cfg-scale="params.cfgScale"
              :cfg-label="cfgLabel"
              :denoise-strength="params.denoiseStrength"
              :show-denoise="showImg2ImgDenoise"
              :seed="params.seed"
              :clip-skip="params.clipSkip"
              :show-clip-skip="showImg2ImgClipSkip"
              :min-clip-skip="minClipSkip"
              :max-clip-skip="12"
              :guidance-advanced="params.guidanceAdvanced"
              :guidance-support="guidanceAdvancedSupport"
              :supir="params.supir"
              :supir-sampler-choices="supirSamplerChoices"
              :supir-selected-sampler-info="supirSelectedSamplerInfo"
              :supir-blocking-reason="supirBlockingReason"
              :upscaler="params.img2imgUpscaler"
              :resize-mode="params.img2imgResizeMode"
              :resize-mode-options="img2imgResizeModeOptions"
              :show-resize-mode="showImg2ImgResizeMode"
              :show-dimensions="showImg2ImgDimensions"
              :dimensions-hidden-hint="img2imgDimensionsHiddenHint"
              :dimension-snap-mode="resolvedEngineForMode === 'zimage' ? 'floor' : 'nearest'"
              :show-init-image-dims="Boolean(params.initImageData)"
              :width-step="imageDimensionSliderStep"
              :width-input-step="imageDimensionInputStep"
              :height-step="imageDimensionSliderStep"
              :height-input-step="imageDimensionInputStep"
              :disabled="isRunning"
              @update:sampler="onSamplerChange"
              @update:scheduler="(v) => setParams({ scheduler: v })"
              @update:steps="(v) => setParams({ steps: Math.max(isQwenImageRequest ? 2 : 1, Math.trunc(v)) })"
              @update:width="(v) => setParams({ width: normalizeImageDimension(v) })"
              @update:height="(v) => setParams({ height: normalizeImageDimension(v) })"
              @update:cfgScale="(v) => setParams({ cfgScale: v })"
              @update:denoiseStrength="(v) => setParams({ denoiseStrength: clampFloat(v, 0, 1) })"
              @update:seed="(v) => setParams({ seed: Math.trunc(v) })"
              @update:clipSkip="(v) => setParams({ clipSkip: Math.max(minClipSkip, Math.trunc(v)) })"
              @update:guidanceAdvanced="setGuidanceAdvanced"
              @patch:supir="setSupir"
              @update:upscaler="(v) => setParams({ img2imgUpscaler: String(v || '').trim() })"
              @update:resizeMode="(v) => setParams({ img2imgResizeMode: normalizeImg2ImgResizeModeForEngine(resolvedEngineForMode, v) })"
              @random-seed="randomizeSeed"
              @reuse-seed="reuseSeed"
              @sync-init-image-dims="syncInitImageDims"
            />

            <SupirModeCard
              v-if="showSupirModeCard"
              :disabled="isRunning"
              :supir="params.supir"
              :variant-choices="supirVariantChoices"
              :blocking-reason="supirBlockingReason"
              @patch:supir="setSupir"
            />
          </template>

          <BasicParametersCard
            v-else
            :samplers="filteredSamplers"
            :schedulers="filteredSchedulers"
            :recommended-samplers="recommendedSamplers"
            :recommended-schedulers="recommendedSchedulers"
            :sampler="params.sampler"
            :scheduler="params.scheduler"
            :steps="params.steps"
            :min-steps="isQwenImageRequest ? 2 : 1"
            :width="params.width"
            :height="params.height"
            :cfg-scale="params.cfgScale"
            :seed="params.seed"
            :clip-skip="params.clipSkip"
            section-title="Basic Parameters"
            :resolutionPresets="resolutionPresets"
            :show-cfg="true"
            :show-denoise="false"
            :denoise-strength="params.denoiseStrength"
            :cfg-label="cfgLabel"
            :show-clip-skip="showClipSkip"
            :min-clip-skip="minClipSkip"
            :max-clip-skip="12"
            :guidance-advanced="params.guidanceAdvanced"
            :guidance-support="guidanceAdvancedSupport"
            :show-init-image-dims="false"
            :width-step="imageDimensionSliderStep"
            :width-input-step="imageDimensionInputStep"
            :height-step="imageDimensionSliderStep"
            :height-input-step="imageDimensionInputStep"
            :disabled="isRunning"
            @update:sampler="onSamplerChange"
            @update:scheduler="(v) => setParams({ scheduler: v })"
            @update:steps="(v) => setParams({ steps: Math.max(isQwenImageRequest ? 2 : 1, Math.trunc(v)) })"
            @update:width="(v) => setParams({ width: normalizeImageDimension(v) })"
            @update:height="(v) => setParams({ height: normalizeImageDimension(v) })"
            @update:cfgScale="(v) => setParams({ cfgScale: v })"
            @update:seed="(v) => setParams({ seed: Math.trunc(v) })"
            @update:clipSkip="(v) => setParams({ clipSkip: Math.max(minClipSkip, Math.trunc(v)) })"
            @update:guidanceAdvanced="setGuidanceAdvanced"
            @random-seed="randomizeSeed"
	            @reuse-seed="reuseSeed"
	          />

					          <SwapStageSettingsCard
			            v-if="showGlobalSwapModel"
		            :enabled="params.swapModel.enabled"
		            :swapAtStep="params.swapModel.swapAtStep"
		            :maxSteps="Math.max(1, params.steps - 1)"
		            :cfg="params.swapModel.cfg"
		            :model="params.swapModel.model"
		            :modelChoices="swapModelChoices"
	            @update:enabled="(v) => setSwapModel({ enabled: v })"
	            @update:swapAtStep="(v) => setSwapModel({ swapAtStep: Math.max(1, Math.trunc(v)) })"
	            @update:cfg="(v) => setSwapModel({ cfg: v })"
	            @update:model="(v) => setSwapModel({ model: v })"
	          />

	          <HiresSettingsCard
	            v-if="showHires"
            :disabled="isRunning"
            :enabled="params.hires.enabled"
            :samplers="filteredSamplers"
            :schedulers="filteredHiresSchedulers"
            :recommended-samplers="recommendedSamplers"
            :recommended-schedulers="recommendedSchedulers"
            :sampler="hiresSampler"
            :scheduler="hiresScheduler"
            :denoise="params.hires.denoise"
            :scale="params.hires.scale"
            :steps="params.hires.steps"
            :cfg-label="cfgLabel"
            :cfg="hiresCfgValue"
            :resize-x="params.hires.resizeX"
            :resize-y="params.hires.resizeY"
            :swap-model="params.hires.swapModel?.model"
            :swap-model-choices="swapModelChoices"
            :show-swap-model="!params.useInitImage"
            :prompt="params.hires.prompt ?? ''"
            :negative-prompt="params.hires.negativePrompt ?? ''"
            :supports-negative="supportsNegative"
            :upscaler="params.hires.upscaler"
            :tile="params.hires.tile"
            :minTile="minTile"
            :upscalers="upscalers"
            :upscalersLoading="upscalersLoading"
            :upscalersError="upscalersError"
            :base-width="params.width"
            :base-height="params.height"
            :refinerEnabled="showHiresRefiner ? params.hires.refiner?.enabled : undefined"
            :refinerSwapAtStep="showHiresRefiner ? params.hires.refiner?.swapAtStep : undefined"
            :refinerCfg="showHiresRefiner ? params.hires.refiner?.cfg : undefined"
            :refinerModel="showHiresRefiner ? params.hires.refiner?.model : undefined"
            :refinerModelChoices="showHiresRefiner ? swapModelChoices : undefined"
            :refinerMaxSteps="showHiresRefiner ? Math.max(1, (params.hires.steps > 0 ? params.hires.steps : params.steps) - 1) : undefined"
            :guidanceAdvanced="params.guidanceAdvanced"
            :guidanceSupport="guidanceAdvancedSupport"
            @update:enabled="(v) => setHires({ enabled: v })"
            @update:denoise="(v) => setHires({ denoise: clampFloat(v, 0, 1) })"
            @update:scale="(v) => setHires({ scale: clampFloat(v, 1, 4) })"
            @update:steps="(v) => setHires({ steps: Math.max(0, Math.trunc(v)) })"
            @update:cfg="onHiresCfgChange"
            @update:resizeX="(v) => setHires({ resizeX: Math.max(0, Math.trunc(v)) })"
            @update:resizeY="(v) => setHires({ resizeY: Math.max(0, Math.trunc(v)) })"
            @update:swapModel="setHiresSwapModel"
            @update:prompt="(v) => setHires({ prompt: String(v || '') })"
            @update:negativePrompt="(v) => setHires({ negativePrompt: String(v || '') })"
            @update:sampler="onHiresSamplerChange"
            @update:scheduler="onHiresSchedulerChange"
            @update:upscaler="(v) => setHires({ upscaler: v })"
            @update:tile="(v) => setHires({ tile: v })"
            @update:minTile="setMinTile"
            @update:refinerEnabled="(v) => setHiresRefiner({ enabled: v })"
            @update:refinerSwapAtStep="(v) => setHiresRefiner({ swapAtStep: Math.max(1, Math.trunc(v)) })"
            @update:refinerCfg="(v) => setHiresRefiner({ cfg: v })"
            @update:refinerModel="(v) => setHiresRefiner({ model: v })"
            @update:guidanceAdvanced="setGuidanceAdvanced"
          />

          <RefinerSettingsCard
            v-if="showGlobalRefiner"
            :enabled="params.refiner.enabled"
            :swapAtStep="params.refiner.swapAtStep"
            :maxSteps="Math.max(1, params.steps - 1)"
            :cfg="params.refiner.cfg"
            :model="params.refiner.model"
            :modelChoices="swapModelChoices"
            :guidanceAdvanced="params.guidanceAdvanced"
            :guidanceSupport="guidanceAdvancedSupport"
            @update:enabled="(v) => setRefiner({ enabled: v })"
            @update:swapAtStep="(v) => setRefiner({ swapAtStep: Math.max(1, Math.trunc(v)) })"
            @update:cfg="(v) => setRefiner({ cfg: v })"
            @update:model="(v) => setRefiner({ model: v })"
            @update:guidanceAdvanced="setGuidanceAdvanced"
          />

          <XyzSweepCard
            v-if="!isQwenImageRequest"
            :samplers="xyzSamplerChoices"
            :schedulers="xyzSchedulerChoices"
          />
        </div>
      </div>
    </div>

    <!-- Right column: Run + Results -->
    <div class="panel-stack panel-stack--sticky">
      <RunCard
        :generateLabel="generateLabel"
        :generateDisabled="runGenerateDisabled"
        :generateTitle="runGenerateTitle"
        :actionMode="params.runAction"
        :showActionMenu="!xyzStore.enabled && !isExactOneShotImageRequest"
        :isRunning="isRunBusy"
        :showBatchControls="showRunBatchControls"
        :batchCount="params.batchCount"
        :batchSize="params.batchSize"
        :disabled="isRunBusy"
        @update:actionMode="setRunAction"
        @generate="onGenerate"
        @cancel="onCancelRun"
        @update:batchCount="(v) => setParams({ batchCount: Math.max(1, Math.trunc(v)) })"
        @update:batchSize="(v) => setParams({ batchSize: Math.max(1, Math.trunc(v)) })"
      >
        <template #header-center-after>
          <span v-if="xyzStore.enabled && !isQwenImageRequest" class="run-badge-xyz">XYZ</span>
        </template>

        <RunProgressStatus
          v-if="isRunning"
          :stage="progress.stage"
          :message="progress.message || ''"
          :percent="progressPercent"
          :step="progress.step"
          :total-steps="progress.totalSteps"
          :eta-seconds="progress.etaSeconds"
          :total-percent="progress.totalPercent"
          :total-phase="progress.totalPhase"
          :total-phase-step="progress.totalPhaseStep"
          :total-phase-total-steps="progress.totalPhaseTotalSteps"
        />
        <RunProgressStatus
          v-else-if="xyzRunning"
          stage="xyz sweep"
          title="XYZ sweep"
          :message="xyzStore.progress.current || ''"
          :percent="xyzProgressPercent"
          :step="xyzStore.progress.total ? xyzStore.progress.completed : null"
          :total-steps="xyzStore.progress.total || null"
          :show-progress-bar="Boolean(xyzStore.progress.total)"
        />
        <RunProgressStatus
          v-else-if="errorMessage"
          variant="error"
          title="Run failed"
          :message="errorMessage"
          :show-progress-bar="false"
        />
        <div v-if="copyNotice" class="caption">{{ copyNotice }}</div>
        <RunSummaryChips :text="runSummary" />
      </RunCard>

      <GenerationResultsPanel showHistory :showInfo="Boolean(info)">
        <template #header-right>
          <div v-if="gentimeSeconds !== null">
            <span class="caption">Time: {{ gentimeSeconds.toFixed(2) }}s</span>
          </div>
          <button class="btn btn-sm btn-outline" type="button" :disabled="workflowBusy" @click="sendToWorkflows">
            {{ workflowBusy ? 'Saving…' : 'Save snapshot' }}
          </button>
          <button class="btn btn-sm btn-outline" type="button" @click="copyCurrentParams">Copy params</button>
        </template>

        <template #history-actions>
          <button class="btn btn-sm btn-ghost" type="button" title="Clear history" :disabled="!history.length || isRunning" @click="clearHistory">Clear</button>
        </template>

        <template #history>
          <ResultsHistoryStrip
            :items="history"
            :selectedTaskId="selectedTaskId"
            :formatTitle="formatHistoryTitle"
            :toDataUrl="toDataUrl"
            @select="onSelectImageHistoryItem"
          />
        </template>

        <template #viewer>
          <ResultViewer
            mode="image"
            :images="images"
            :previewImage="previewImage"
            :previewCaption="previewCaption"
            :isRunning="isRunning"
            :width="params.width"
            :height="params.height"
            :emptyText="resultsEmptyText"
          >
            <template #empty>
              <div class="results-empty-state">
                <div class="results-empty-title">
                  <template v-if="isRunning">{{ resultsEmptyText }}</template>
                  <template v-else>No images yet</template>
                </div>
                <div v-if="!isRunning" class="caption">Generate to see results here.</div>
              </div>
            </template>
            <template #image-actions="{ image, index }">
              <button
                v-if="supportsImg2Img"
                class="gallery-action"
                type="button"
                title="Send to Img2Img"
                @click="sendToImg2Img(image)"
              >
                Send to Img2Img
              </button>
              <button class="gallery-action" type="button" title="Download Image" @click="download(image, index)">
                Download
              </button>
            </template>
          </ResultViewer>
        </template>

        <template #info>
          <pre class="text-xs break-words">{{ formatJson(info) }}</pre>
        </template>
      </GenerationResultsPanel>
    </div>

    <RunHistoryDetailsModal
      v-model="historyDetailsOpen"
      :title="historyDetailsTitle"
      :preview-url="historyDetailsImageUrl"
      :preview-alt="historyDetailsTitle"
      :mode-label="historyDetailsModeLabel"
      :created-at-label="historyDetailsCreatedAtLabel"
      :status="historyDetailsItem?.status || ''"
      :task-id="historyDetailsItem?.taskId || ''"
      :summary="historyDetailsItem?.summary || ''"
      :error-message="historyDetailsItem?.errorMessage || ''"
      :params-snapshot="historyDetailsItem?.paramsSnapshot"
      :sections="historyDetailsSections"
      :load-disabled="!historyDetailsItem || isRunning || historyLoadingTaskId === historyDetailsItem.taskId"
      :load-label="historyDetailsItem && historyLoadingTaskId === historyDetailsItem.taskId ? 'Loading…' : 'Load'"
      :apply-disabled="!historyDetailsItem || isRunning || Boolean(historyApplyingTaskId)"
      :copy-disabled="!historyDetailsItem || isRunning"
      @load="onLoadHistoryDetails"
      @apply="onApplyHistoryDetails"
      @copy="onCopyHistoryDetails"
    />
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { fetchFreshModelInventory, fetchFreshPaths, fetchSamplers, fetchSchedulers, invalidateModelCatalogCaches } from '../api/client'
import type {
  GeneratedImage,
  GuidanceAdvancedCapabilities,
  SamplerInfo,
  SchedulerInfo,
} from '../api/types'
import { useWorkflowSnapshotActions } from '../composables/useWorkflowSnapshotActions'
import { resolveEngineForRequest, useGeneration, type ImageRunHistoryItem } from '../composables/useGeneration'
import { resolveSupirSelectionState, useSupirDiagnostics } from '../composables/useSupirDiagnostics'
import {
  defaultImageParamsForType,
  normalizeSupirSamplerSelection,
  useModelTabsStore,
  type GuidanceAdvancedParams,
  type ImageBaseParams,
  type ImageRunAction,
  type ImageTabType,
  type SupirModeFormState,
  type TabByType,
} from '../stores/model_tabs'
import { getEngineConfig, getEngineDefaults } from '../stores/engine_config'
import {
  filterSamplersForFamilyCapabilities,
  filterSchedulersForFamilyCapabilities,
  filterSchedulersForSampler,
  normalizeSamplerSchedulerSelection,
  useEngineCapabilitiesStore,
} from '../stores/engine_capabilities'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useBootstrapStore } from '../stores/bootstrap'
import { useUpscalersStore } from '../stores/upscalers'
import { useXyzStore } from '../stores/xyz'
import { isWanTabFamily, normalizeTabFamily } from '../utils/engine_taxonomy'
import { buildExplicitImageRequestContract } from '../utils/image_request_contract'
import { filterModelTitlesForFamily } from '../utils/model_family_filters'
import { buildFamilyVaeValidationChoices, canonicalizeVaeChoice, resolveInventoryVaeSha } from '../utils/vae_choices'
import {
  img2imgResizeModeOptionsForEngine,
  normalizeImg2ImgResizeModeForEngine,
} from '../utils/img2img_resize'
import {
  normalizeInpaintMaskToggleState,
  normalizeInpaintingFill,
  parseInpaintMode,
  normalizeNonNegativeInt,
  resolveHiresModePolicy,
} from '../utils/image_params'
import { readFileAsDataURL, readImageDimensions } from '../utils/image_io'
import BasicParametersCard from '../components/BasicParametersCard.vue'
import HiresSettingsCard from '../components/HiresSettingsCard.vue'
import Img2ImgBasicParametersCard from '../components/Img2ImgBasicParametersCard.vue'
import InitialImageBlock from '../components/InitialImageBlock.vue'
import IpAdapterCard from '../components/IpAdapterCard.vue'
import RunHistoryDetailsModal from '../components/modals/RunHistoryDetailsModal.vue'
import PromptCard from '../components/prompt/PromptCard.vue'
import RefinerSettingsCard from '../components/RefinerSettingsCard.vue'
import SupirModeCard from '../components/SupirModeCard.vue'
import SwapStageSettingsCard from '../components/SwapStageSettingsCard.vue'
import ResultViewer from '../components/ResultViewer.vue'
import GenerationResultsPanel from '../components/results/GenerationResultsPanel.vue'
import ResultsHistoryStrip from '../components/results/ResultsHistoryStrip.vue'
import RunCard from '../components/results/RunCard.vue'
import RunProgressStatus from '../components/results/RunProgressStatus.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import XyzSweepCard from '../components/XyzSweepCard.vue'

const props = defineProps<{ tabId: string; type: ImageTabType }>()
type IpAdapterPatch = Partial<Omit<ImageBaseParams['ipAdapter'], 'source'>> & {
  source?: Partial<ImageBaseParams['ipAdapter']['source']>
}
type SupirPatch = Partial<ImageBaseParams['supir']>

const store = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()
const quicksettingsStore = useQuicksettingsStore()
const bootstrap = useBootstrapStore()
const upscalersStore = useUpscalersStore()
const xyzStore = useXyzStore()
const { upscalers, loading: upscalersLoading, error: upscalersError, minTile } = storeToRefs(upscalersStore)

// Use unified generation composable
const {
  generate: generateBase,
  cancel: cancelBase,
  stopStream,
  gallery,
  progress,
  previewImage,
  previewStep,
  errorMessage,
  isRunning,
  lastSeed,
  history,
  selectedTaskId,
  historyLoadingTaskId,
  tab,
  info,
  gentimeMs,
  loadHistory,
  clearHistory,
  resumeNotice,
} = useGeneration(props.tabId)

const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])
const historyDetailsOpen = ref(false)
const historyDetailsItem = ref<ImageRunHistoryItem | null>(null)
const historyApplyingTaskId = ref('')
const { ensureSupirDiagnosticsLoaded } = useSupirDiagnostics()
const ASSET_RESTORE_SUPERSEDED_ERROR = 'A newer image asset restore/apply operation superseded this one.'

function onSelectImageHistoryItem(item: { taskId: string }): void {
  const match = history.value.find((entry) => entry.taskId === item.taskId)
  if (!match) return
  openHistoryDetails(match)
}

onMounted(() => {
  bootstrap
    .runRequired('Failed to initialize image tab controls', async () => {
      await upscalersStore.load({ refresh: true })
      await quicksettingsStore.init()
      const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
      samplers.value = samp.samplers
      schedulers.value = sched.schedulers
    })
    .catch(() => {
      // Fatal state is already set by bootstrap store.
    })
})

onBeforeUnmount(() => {
  stopStream()
})

const workflowParamsSnapshot = computed<Record<string, unknown> | null>(() => {
  const currentTab = tab.value
  if (!currentTab) return null
  const baseSnapshot = currentTab.params as unknown as Record<string, unknown>
  if (props.type === 'zimage_l2p') return { ...baseSnapshot }
  const familyOwnedVae = String(quicksettingsStore.getVaeForFamily(props.type) || '').trim()
  return {
    ...baseSnapshot,
    vae: familyOwnedVae,
  }
})

const {
  notice: copyNotice,
  toast,
  copyJson,
  formatJson,
  workflowBusy,
  sendToWorkflows,
  copyCurrentParams,
} = useWorkflowSnapshotActions({
  getTab: () => tab.value ?? null,
  getWorkflowParamsSnapshot: () => workflowParamsSnapshot.value,
})
type ImageTab = TabByType<ImageTabType>

const historyDetailsTitle = computed(() => (historyDetailsItem.value ? formatHistoryTitle(historyDetailsItem.value) : 'History details'))
const historyDetailsCreatedAtLabel = computed(() => {
  const timestamp = historyDetailsItem.value?.createdAtMs
  if (!timestamp) return '—'
  return new Date(timestamp).toLocaleString()
})
const historyDetailsModeLabel = computed(() => {
  const mode = historyDetailsItem.value?.mode
  return mode === 'img2img' ? 'Img2Img' : 'Txt2Img'
})
const historyDetailsImageUrl = computed(() => {
  const thumbnail = historyDetailsItem.value?.thumbnail
  return thumbnail ? toDataUrl(thumbnail) : ''
})
const historyDetailsPrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  const prompt = readHistorySnapshotText(item, 'prompt')
  if (prompt) return prompt
  return item.promptPreview || ''
})
const historyDetailsNegativePrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  return readHistorySnapshotText(item, 'negativePrompt')
})
const historyDetailsSections = computed(() => [
  { key: 'prompt', label: 'Prompt', text: historyDetailsPrompt.value },
  { key: 'negativePrompt', label: 'Negative Prompt', text: historyDetailsNegativePrompt.value },
])

watch(
  resumeNotice,
  (msg) => {
    const text = String(msg || '').trim()
    if (!text) return
    toast(text)
    resumeNotice.value = ''
  },
  { immediate: true },
)

const imageTab = computed<ImageTab | null>(() => {
  const candidate = tab.value
  if (!candidate || isWanTabFamily(candidate.type) || candidate.type === 'ltx2') return null
  return candidate as unknown as ImageTab
})
const fallbackParams = computed<ImageBaseParams>(() => defaultImageParamsForType(props.type))
const params = computed<ImageBaseParams>(() => imageTab.value?.params ?? fallbackParams.value)

const initImageNaturalWidth = ref(0)
const initImageNaturalHeight = ref(0)
const initImageDimsToken = ref(0)

const maskEditorImageWidth = computed(() => {
  if (!String(params.value.initImageData || '').trim()) return 0
  const w = Math.trunc(Number(initImageNaturalWidth.value))
  return Number.isFinite(w) && w > 0 ? w : 0
})
const maskEditorImageHeight = computed(() => {
  if (!String(params.value.initImageData || '').trim()) return 0
  const h = Math.trunc(Number(initImageNaturalHeight.value))
  return Number.isFinite(h) && h > 0 ? h : 0
})

watch(
  () => String(params.value.initImageData || '').trim(),
  (src) => {
    const token = (initImageDimsToken.value += 1)
    initImageNaturalWidth.value = 0
    initImageNaturalHeight.value = 0
    if (!src) {
      return
    }

    readImageDimensions(src)
      .then(({ width, height }) => {
        if (initImageDimsToken.value !== token) return
        const initW = Math.max(0, Math.trunc(width))
        const initH = Math.max(0, Math.trunc(height))
        initImageNaturalWidth.value = initW
        initImageNaturalHeight.value = initH

        const maskSrc = String(params.value.maskImageData || '').trim()
        if (!maskSrc) return
        readImageDimensions(maskSrc)
          .then(({ width: maskW, height: maskH }) => {
            if (initImageDimsToken.value !== token) return
            if (Math.trunc(maskW) === initW && Math.trunc(maskH) === initH) return
            toast(
              `Mask cleared: init image size is ${initW}×${initH}, but mask is ${maskW}×${maskH}. Re-open the editor to reapply.`,
            )
            setParams({ maskImageData: '', maskImageName: '' })
          })
          .catch(() => {
            if (initImageDimsToken.value !== token) return
            toast('Mask cleared: failed to load the stored mask image.')
            setParams({ maskImageData: '', maskImageName: '' })
          })
      })
      .catch(() => {
        if (initImageDimsToken.value !== token) return
        initImageNaturalWidth.value = 0
        initImageNaturalHeight.value = 0
      })
  },
  { immediate: true },
)

const engineConfig = computed(() => getEngineConfig(props.type))
const resolvedEngineForMode = computed(() => resolveEngineForRequest(props.type, Boolean(params.value.useInitImage)))
const isQwenImageRequest = computed(() => resolvedEngineForMode.value === 'qwen_image')
const isZImageL2PRequest = computed(() => resolvedEngineForMode.value === 'zimage_l2p')
const promptTokenEngine = computed(() => isQwenImageRequest.value ? '' : resolvedEngineForMode.value)
const isExactOneShotImageRequest = computed(() => isQwenImageRequest.value || isZImageL2PRequest.value)
const img2imgResizeModeOptions = computed(() => img2imgResizeModeOptionsForEngine(resolvedEngineForMode.value))
const engineSurface = computed(() => engineCaps.get(resolvedEngineForMode.value))
const engineFamilySurface = computed(() => engineCaps.getFamilyForEngine(resolvedEngineForMode.value))
const imageDimensionInputStep = computed(() => {
  const resolutionStep = Number(engineFamilySurface.value?.resolution_step)
  if (
    (resolvedEngineForMode.value === 'qwen_image' || resolvedEngineForMode.value === 'zimage_l2p')
    && Number.isFinite(resolutionStep)
    && Math.trunc(resolutionStep) > 0
  ) {
    return Math.trunc(resolutionStep)
  }
  return resolvedEngineForMode.value === 'zimage' || resolvedEngineForMode.value === 'qwen_image' || resolvedEngineForMode.value === 'zimage_l2p' ? 16 : 8
})
const imageDimensionSliderStep = computed(() =>
  (resolvedEngineForMode.value === 'qwen_image' || resolvedEngineForMode.value === 'zimage_l2p')
    ? imageDimensionInputStep.value
    : (resolvedEngineForMode.value === 'zimage' ? 16 : 64),
)
const availableInpaintModes = computed(() => engineCaps.getInpaintModes(resolvedEngineForMode.value))
const availableInpaintModeOptions = computed(() =>
  availableInpaintModes.value.map((value) => ({
    value: value as ImageBaseParams['inpaintMode'],
    label: {
      per_step_blend: 'Per-step blend',
      post_sample_blend: 'Post-sample blend',
      fooocus_inpaint: 'Fooocus Inpaint',
      brushnet: 'BrushNet',
    }[value] ?? value,
  })),
)
const guidanceAdvancedSupport = computed<GuidanceAdvancedCapabilities | null>(() => {
  const guidance = engineSurface.value?.guidance_advanced
  return guidance ?? null
})
const familyCapabilities = computed(() => engineCaps.getFamilyForEngine(resolvedEngineForMode.value))
const activeInpaintDependencyMode = computed(() => (
  params.value.useInitImage && params.value.useMask && params.value.initSource.mode === 'img'
    ? params.value.inpaintMode
    : null
))
const dependencyStatus = computed(() => engineCaps.getDependencyStatus(resolvedEngineForMode.value))
const dependencyError = computed(() => engineCaps.firstDependencyError(resolvedEngineForMode.value, {
  inpaintMode: activeInpaintDependencyMode.value,
}))
const dependencyReady = computed(() => engineCaps.isDependencyReady(resolvedEngineForMode.value, {
  inpaintMode: activeInpaintDependencyMode.value,
}))

const zimageTurbo = computed(() => props.type === 'zimage' ? Boolean(params.value.zimageTurbo ?? true) : false)
const flux2Variant = computed(() => (
  props.type === 'flux2'
    ? quicksettingsStore.resolveFlux2CheckpointVariant(String(params.value.checkpoint || '').trim())
    : null
))
const fallbackRequestGuidanceMode = computed<'cfg' | 'distilled_cfg'>(() => (
  Boolean(engineConfig.value.capabilities.usesDistilledCfg) && !Boolean(engineConfig.value.capabilities.usesCfg)
    ? 'distilled_cfg'
    : 'cfg'
))
const usesDistilledCfgModel = computed(() => {
  if (props.type === 'flux2') return flux2Variant.value === 'distilled'
  return Boolean(engineConfig.value.capabilities.usesDistilledCfg) && !engineConfig.value.capabilities.usesCfg
})
const supportsNegative = computed(() => {
  if (!familyCapabilities.value?.supports_negative_prompt) return false
  return !usesDistilledCfgModel.value
})
const hideNegativePrompt = computed(() => {
  if (!supportsNegative.value) return true
  const cfg = Number(params.value.cfgScale)
  return Number.isFinite(cfg) && cfg <= 1
})
const supportsTxt2Img = computed(() => {
  const surf = engineSurface.value
  if (!surf) return false
  return Boolean(surf.supports_txt2img)
})
const supportsImg2Img = computed(() => {
  const surf = engineSurface.value
  if (!surf) return false
  return Boolean(surf.supports_img2img)
})
const canGenerateForCurrentMode = computed(() =>
  dependencyReady.value
  && Boolean(familyCapabilities.value)
  && filteredSamplers.value.length > 0
  && filteredSchedulers.value.length > 0
  && (params.value.useInitImage ? supportsImg2Img.value : supportsTxt2Img.value),
)
const assetContractBlockingReason = computed(() => {
  const checkpoint = String(params.value.checkpoint || '').trim()
  if (!checkpoint) return 'Select a checkpoint to generate.'
  const familyOwnedVae = isZImageL2PRequest.value ? '' : String(quicksettingsStore.getVaeForFamily(props.type) || '').trim()
  try {
    buildExplicitImageRequestContract({
      modelLabel: checkpoint,
      engineKey: resolvedEngineForMode.value,
      textEncoderLabels: params.value.textEncoders,
      selectedVaeLabel: familyOwnedVae,
      zimageTurbo: resolvedEngineForMode.value === 'zimage'
        ? Boolean((params.value as unknown as Record<string, unknown>).zimageTurbo ?? true)
        : false,
      fallbackGuidanceMode: fallbackRequestGuidanceMode.value,
      resolvers: {
        requireModelInfo: quicksettingsStore.requireModelInfo,
        resolveFlux2CheckpointVariant: quicksettingsStore.resolveFlux2CheckpointVariant,
        resolveTextEncoderSha: quicksettingsStore.resolveTextEncoderSha,
        resolveTextEncoderSlot: quicksettingsStore.resolveTextEncoderSlot,
        requireVaeSelection: quicksettingsStore.requireVaeSelection,
        resolveVaeSha: quicksettingsStore.resolveVaeSha,
        getAssetContract: engineCaps.getAssetContract,
      },
    })
    return ''
  } catch (error) {
    return error instanceof Error ? error.message : String(error)
  }
})
const generateDisabledReason = computed(() => {
  if (isRunning.value) return ''
  if (!dependencyStatus.value) return `Dependency checks for '${resolvedEngineForMode.value}' are not available.`
  if (!dependencyReady.value) return dependencyError.value || `Dependencies for '${resolvedEngineForMode.value}' are not ready.`
  if (!engineSurface.value) return `Capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (!familyCapabilities.value) return `Family capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (params.value.useInitImage && !supportsImg2Img.value) return `${engineConfig.value.label} does not support img2img.`
  if (!params.value.useInitImage && !supportsTxt2Img.value) return `${engineConfig.value.label} does not support txt2img.`
  if (assetContractBlockingReason.value) return assetContractBlockingReason.value
  if (filteredSamplers.value.length === 0) return `${engineConfig.value.label} has no family-compatible samplers available.`
  if (filteredSchedulers.value.length === 0) return `${engineConfig.value.label} has no family-compatible schedulers available.`
  return ''
})
const xyzSamplerChoices = computed(() => filteredSamplers.value.map((entry) => entry.name))
const xyzSchedulerChoices = computed(() => filteredSchedulers.value.map((entry) => entry.name))
const xyzRunning = computed(() => xyzStore.status === 'running')
const isRunBusy = computed(() => isRunning.value || xyzRunning.value)
const generateLabel = 'Generate'
const supportsImg2ImgMasking = computed(() => Boolean(engineSurface.value?.supports_img2img_masking))
const usesImageAutomation = computed(() => (
  !isExactOneShotImageRequest.value
  && (params.value.runAction === 'infinite'
  || (params.value.useInitImage && params.value.initSource.mode === 'dir')
  || (params.value.ipAdapter.enabled && params.value.ipAdapter.source.mode === 'dir'))
))
const showRunBatchControls = computed(() => !usesImageAutomation.value && !isExactOneShotImageRequest.value)
const ipAdapterSupported = computed(() => {
  if (isExactOneShotImageRequest.value) return false
  if (engineSurface.value) return Boolean(engineSurface.value.supports_ip_adapter)
  return props.type === 'sd15' || props.type === 'sdxl'
})
const toInventoryChoiceLabel = (value: string): string => {
  const normalized = String(value || '').trim().replace(/\\/g, '/')
  if (!normalized) return ''
  return normalized.split('/').filter(Boolean).pop() || normalized
}
const ipAdapterModelChoices = computed(() => quicksettingsStore.ipAdapterModelChoices.map((value) => ({
  value,
  label: toInventoryChoiceLabel(value),
})))
const ipAdapterImageEncoderChoices = computed(() => quicksettingsStore.ipAdapterImageEncoderChoices.map((value) => ({
  value,
  label: toInventoryChoiceLabel(value),
})))
const supirEnabled = computed(() => Boolean(params.value.supir.enabled))
const supportsSupirModeSurface = computed(() => (
  props.type === 'sdxl'
  && Boolean(engineSurface.value?.supports_supir_mode)
))
const showSupirModeCard = computed(() => supportsSupirModeSurface.value && params.value.useInitImage)
const supirSelectionState = computed(() => resolveSupirSelectionState({
  supported: supportsSupirModeSurface.value,
  selectedVariant: params.value.supir.variant,
  selectedSampler: params.value.supir.sampler,
  guidanceAdvancedEnabled: params.value.guidanceAdvanced.enabled,
}))
const supirVariantChoices = computed(() => supirSelectionState.value.variantChoices)
const supirSamplerChoices = computed(() => supirSelectionState.value.samplerChoices)
const supirSelectionValid = computed(() => supirSelectionState.value.selectionValid)
const supirSelectedSamplerInfo = computed(() => supirSelectionState.value.selectedSamplerInfo)
const supirBlockingReason = computed(() => (
  supportsSupirModeSurface.value ? supirSelectionState.value.blockingReason : ''
))

function getSupirRestoreBlockingReason(candidate: Pick<ImageBaseParams, 'useInitImage' | 'guidanceAdvanced' | 'supir'>): string {
  if (!candidate.supir.enabled) return ''
  if (!candidate.useInitImage) return 'SUPIR mode requires Img2Img/Inpaint mode.'
  if (!supportsSupirModeSurface.value) return 'SUPIR mode is unavailable for the active engine.'
  if (candidate.guidanceAdvanced.enabled) {
    return 'SUPIR mode cannot be enabled while Advanced Guidance/APG is active. Disable Advanced Guidance first.'
  }
  return ''
}

const showIpAdapterCard = computed(() => !isExactOneShotImageRequest.value && !supirEnabled.value && (ipAdapterSupported.value || params.value.ipAdapter.enabled))
const infiniteXyzConflict = computed(() => params.value.runAction === 'infinite' && xyzStore.enabled)
const automationBatchConflict = computed(() => (
  usesImageAutomation.value
  && (params.value.batchCount !== 1 || params.value.batchSize !== 1)
))
const initFolderMissingPath = computed(() => (
  Boolean(params.value.useInitImage)
  && params.value.initSource.mode === 'dir'
  && !String(params.value.initSource.folderPath || '').trim()
))
const dirInitMaskConflict = computed(() => (
  Boolean(params.value.useInitImage)
  && params.value.initSource.mode === 'dir'
  && Boolean(params.value.useMask)
))
const ipAdapterBlockingReason = computed(() => {
  if (!params.value.ipAdapter.enabled) return ''
  if (!ipAdapterSupported.value) return `${engineConfig.value.label} does not support IP-Adapter.`
  if (!String(params.value.ipAdapter.model || '').trim()) return 'Select an IP-Adapter model.'
  if (!String(params.value.ipAdapter.imageEncoder || '').trim()) return 'Select an IP-Adapter image encoder.'
  if (params.value.ipAdapter.source.mode === 'dir') {
    if (!String(params.value.ipAdapter.source.folderPath || '').trim()) return 'IP-Adapter folder mode requires a folder path.'
    return ''
  }
  if (params.value.ipAdapter.source.sameAsInit && !params.value.useInitImage) {
    return 'Same as init image is only available for img2img runs.'
  }
  if (!params.value.ipAdapter.source.sameAsInit && !String(params.value.ipAdapter.source.referenceImageData || '').trim()) {
    return 'Select an IP-Adapter reference image.'
  }
  return ''
})
const xyzProgressPercent = computed(() => {
  if (!xyzStore.progress.total) return null
  return (xyzStore.progress.completed / xyzStore.progress.total) * 100
})
const missingInpaintMask = computed(() =>
  Boolean(params.value.useInitImage)
  && supportsImg2ImgMasking.value
  && params.value.initSource.mode === 'img'
  && Boolean(params.value.useMask)
  && !String(params.value.maskImageData || '').trim(),
)
const unsupportedInpaintMode = computed(() =>
  Boolean(params.value.useInitImage)
  && Boolean(params.value.useMask)
  && params.value.initSource.mode === 'img'
  && !availableInpaintModes.value.includes(params.value.inpaintMode),
)
const unsupportedInpaintModeMessage = computed(() => {
  if (!unsupportedInpaintMode.value) return ''
  return `Inpaint mode '${String(params.value.inpaintMode)}' is not available for ${engineConfig.value.label}. Reselect a supported mode.`
})
const runGenerateDisabled = computed(() => {
  if (isRunBusy.value) return true
  if (infiniteXyzConflict.value) return true
  if (automationBatchConflict.value) return true
  if (supirEnabled.value && supirBlockingReason.value) return true
  if (assetContractBlockingReason.value) return true
  if (xyzStore.enabled) {
    return !(dependencyReady.value && Boolean(familyCapabilities.value) && supportsTxt2Img.value && filteredSamplers.value.length > 0 && filteredSchedulers.value.length > 0)
  }
  if (initFolderMissingPath.value || dirInitMaskConflict.value || ipAdapterBlockingReason.value) return true
  if (missingInpaintMask.value) return true
  if (unsupportedInpaintMode.value) return true
  return !canGenerateForCurrentMode.value
})
const runGenerateTitle = computed(() => {
  if (xyzRunning.value) return 'XYZ sweep is running.'
  if (infiniteXyzConflict.value) return 'Infinite generate cannot run while XYZ is enabled.'
  if (!xyzStore.enabled) {
    if (automationBatchConflict.value) return 'Image automation requires batch count = 1 and batch size = 1.'
    if (initFolderMissingPath.value) return 'Initial image folder mode requires a folder path.'
    if (dirInitMaskConflict.value) return 'Mask editing is only available while the initial image source is set to IMG.'
    if (supirEnabled.value && supirBlockingReason.value) return supirBlockingReason.value
    if (ipAdapterBlockingReason.value) return ipAdapterBlockingReason.value
    if (missingInpaintMask.value) return 'INPAINT is enabled but no mask is applied. Open the mask editor and apply a mask.'
    if (unsupportedInpaintMode.value) return unsupportedInpaintModeMessage.value
    return generateDisabledReason.value
  }
  if (!dependencyStatus.value) return `Dependency checks for '${resolvedEngineForMode.value}' are not available.`
  if (!dependencyReady.value) return dependencyError.value || `Dependencies for '${resolvedEngineForMode.value}' are not ready.`
  if (!engineSurface.value) return `Capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (!familyCapabilities.value) return `Family capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (assetContractBlockingReason.value) return assetContractBlockingReason.value
  if (filteredSamplers.value.length === 0) return `${engineConfig.value.label} has no family-compatible samplers available.`
  if (filteredSchedulers.value.length === 0) return `${engineConfig.value.label} has no family-compatible schedulers available.`
  if (!supportsTxt2Img.value) return `${engineConfig.value.label} does not support txt2img.`
  return ''
})

const enableAssets = computed(() => !isExactOneShotImageRequest.value)
const enableStyles = computed(() => true)
const toolbarLabel = computed(() => {
  if (props.type !== 'zimage') return ''
  return zimageTurbo.value ? 'Z Image Turbo' : 'Z Image Base'
})

const cfgLabel = computed(() => (usesDistilledCfgModel.value ? 'Distilled CFG' : 'CFG'))
const showClipSkip = computed(() => !isExactOneShotImageRequest.value && Boolean(familyCapabilities.value?.shows_clip_skip))
const minClipSkip = computed(() => 0)
const swapModelChoices = computed(() => {
  const family = normalizeTabFamily(props.type)
  if (!family || isWanTabFamily(family)) return []
  return filterModelTitlesForFamily(quicksettingsStore.models, family, quicksettingsStore.pathsConfigSnapshot)
})

const supportsHiresForEngine = computed(() => {
  if (isExactOneShotImageRequest.value) return false
  if (props.type === 'zimage') return false
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_hires
})
const hiresModePolicy = computed(() => resolveHiresModePolicy(
  Boolean(params.value.useInitImage),
  supportsHiresForEngine.value,
  Boolean(params.value.useMask),
))
const showHires = computed(() => !supirEnabled.value && hiresModePolicy.value.showCard)

const showGlobalSwapModel = computed(() => !isExactOneShotImageRequest.value && !Boolean(params.value.useInitImage))

const showHiresRefiner = computed(() => {
  if (isExactOneShotImageRequest.value) return false
  if (params.value.useInitImage) return false
  if (props.type === 'zimage') return false
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_refiner
})

const showGlobalRefiner = computed(() => {
  if (isExactOneShotImageRequest.value) return false
  if (params.value.useInitImage) return false
  if (props.type === 'zimage') return false
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_refiner
})
const showImg2ImgDimensions = computed(() => !isQwenImageRequest.value)
const img2imgDimensionsHiddenHint = computed(() => (
  isQwenImageRequest.value ? 'Qwen Image Edit derives output size from the init image.' : ''
))
const showImg2ImgDenoise = computed(() => !isQwenImageRequest.value)
const showImg2ImgClipSkip = computed(() => showClipSkip.value && !isQwenImageRequest.value)
const showImg2ImgResizeMode = computed(() => (
  !isQwenImageRequest.value
  && !(resolvedEngineForMode.value === 'zimage' && params.value.useMask)
))

function normalizeRecommendedList(values: string[] | null | undefined): string[] | null {
  if (!Array.isArray(values)) return null
  const normalized = Array.from(new Set(values
    .map((value) => String(value || '').trim())
    .filter((value) => value.length > 0)))
  if (normalized.length === 0) return null
  return normalized
}

const recommendedSamplers = computed(() =>
  normalizeRecommendedList(engineSurface.value?.recommended_samplers),
)

const recommendedSchedulers = computed(() =>
  normalizeRecommendedList(engineSurface.value?.recommended_schedulers),
)

function resolveLiveSamplingDefaults(): { sampler: string; scheduler: string } {
  return engineCaps.resolveSamplingDefaults(resolvedEngineForMode.value) ?? { sampler: '', scheduler: '' }
}

function normalizeLiveSamplingSelection(rawSampler: string, rawScheduler: string): { sampler: string; scheduler: string } | null {
  const defaults = resolveLiveSamplingDefaults()
  return normalizeSamplerSchedulerSelection({
    samplers: samplers.value,
    schedulers: schedulers.value,
    familyCapabilities: familyCapabilities.value,
    sampler: rawSampler,
    scheduler: rawScheduler,
    preferredSamplers: [defaults.sampler],
    preferredSchedulers: [defaults.scheduler],
  })
}

const filteredSamplers = computed(() => {
  return filterSamplersForFamilyCapabilities(samplers.value, familyCapabilities.value)
})

const normalizedBaseSampling = computed(() =>
  normalizeLiveSamplingSelection(params.value.sampler, params.value.scheduler),
)

const activeSamplerSpec = computed(() => {
  const normalized = normalizedBaseSampling.value
  if (normalized) {
    return filteredSamplers.value.find((entry) => entry.name === normalized.sampler) ?? null
  }
  return filteredSamplers.value.find((entry) => entry.name === params.value.sampler) ?? null
})

const filteredSchedulers = computed(() => {
  const familyScoped = filterSchedulersForFamilyCapabilities(schedulers.value, familyCapabilities.value)
  return filterSchedulersForSampler(familyScoped, activeSamplerSpec.value)
})

const hiresSampler = computed(() => {
  const normalizedBase = normalizedBaseSampling.value
  if (normalizedBase) {
    const normalizedHires = normalizeLiveSamplingSelection(
      String(params.value.hires.sampler || '').trim() || normalizedBase.sampler,
      String(params.value.hires.scheduler || '').trim() || normalizedBase.scheduler,
    )
    if (normalizedHires) return normalizedHires.sampler
  }
  const override = String(params.value.hires.sampler || '').trim()
  if (override) return override
  return params.value.sampler
})

const hiresScheduler = computed(() => {
  const normalizedBase = normalizedBaseSampling.value
  if (normalizedBase) {
    const normalizedHires = normalizeLiveSamplingSelection(
      String(params.value.hires.sampler || '').trim() || normalizedBase.sampler,
      String(params.value.hires.scheduler || '').trim() || normalizedBase.scheduler,
    )
    if (normalizedHires) return normalizedHires.scheduler
  }
  const override = String(params.value.hires.scheduler || '').trim()
  if (override) return override
  return params.value.scheduler
})

const hiresCfgValue = computed(() => {
  if (usesDistilledCfgModel.value) {
    const value = Number(params.value.hires.distilledCfg)
    if (Number.isFinite(value)) return value
    return params.value.cfgScale
  }
  const value = Number(params.value.hires.cfg)
  if (Number.isFinite(value)) return value
  return params.value.cfgScale
})

const filteredHiresSchedulers = computed(() => {
  const familyScoped = filterSchedulersForFamilyCapabilities(schedulers.value, familyCapabilities.value)
  const spec = filteredSamplers.value.find((entry) => entry.name === hiresSampler.value) ?? null
  return filterSchedulersForSampler(familyScoped, spec)
})

function normalizeXyzSamplingAxisText(axisParam: string, axisValuesText: string): string {
  if (axisParam !== 'sampler' && axisParam !== 'scheduler') return axisValuesText
  const choices = axisParam === 'sampler' ? xyzSamplerChoices.value : xyzSchedulerChoices.value
  if (choices.length === 0) return ''
  const allowed = new Set(choices)
  const values = String(axisValuesText || '')
    .split(/[\n\r,]+/g)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0 && allowed.has(entry))
  const deduped = Array.from(new Set(values))
  return deduped.join(', ')
}

function onSamplerChange(value: string): void {
  const normalized = normalizeLiveSamplingSelection(value, params.value.scheduler)
  if (!normalized) {
    setParams({ sampler: value })
    return
  }
  setParams({
    sampler: normalized.sampler,
    scheduler: normalized.scheduler,
  })
}

function onHiresSamplerChange(value: string): void {
  const normalizedBase = normalizedBaseSampling.value
  if (!normalizedBase) {
    setHires({ sampler: value })
    return
  }
  const normalized = normalizeLiveSamplingSelection(value, hiresScheduler.value)
  if (!normalized) {
    setHires({ sampler: value })
    return
  }
  setHires({
    sampler: normalized.sampler === normalizedBase.sampler ? '' : normalized.sampler,
    scheduler: normalized.scheduler === normalizedBase.scheduler ? '' : normalized.scheduler,
  })
}

function onHiresSchedulerChange(value: string): void {
  const normalizedBase = normalizedBaseSampling.value
  if (!normalizedBase) {
    setHires({ scheduler: value })
    return
  }
  const normalized = normalizeLiveSamplingSelection(hiresSampler.value, value)
  if (!normalized) {
    setHires({ scheduler: value })
    return
  }
  setHires({
    sampler: normalized.sampler === normalizedBase.sampler ? '' : normalized.sampler,
    scheduler: normalized.scheduler === normalizedBase.scheduler ? '' : normalized.scheduler,
  })
}

function onHiresCfgChange(value: number): void {
  const normalized = clampFloat(value, 0, 30)
  if (usesDistilledCfgModel.value) {
    setHires({ distilledCfg: normalized, cfg: undefined })
    return
  }
  setHires({ cfg: normalized, distilledCfg: undefined })
}

watch([() => params.value.sampler, () => params.value.scheduler, samplers, schedulers, familyCapabilities], () => {
  const normalized = normalizeLiveSamplingSelection(params.value.sampler, params.value.scheduler)
  if (!normalized) return
  if (normalized.sampler === params.value.sampler && normalized.scheduler === params.value.scheduler) return
  setParams({
    sampler: normalized.sampler,
    scheduler: normalized.scheduler,
  })
}, { immediate: true })

watch(
  [
    () => params.value.sampler,
    () => params.value.scheduler,
    () => params.value.hires.sampler,
    () => params.value.hires.scheduler,
    samplers,
    schedulers,
    familyCapabilities,
  ],
  () => {
    const normalizedBase = normalizeLiveSamplingSelection(params.value.sampler, params.value.scheduler)
    if (!normalizedBase) return

    const rawHiresSampler = String(params.value.hires.sampler || '').trim()
    const rawHiresScheduler = String(params.value.hires.scheduler || '').trim()
    if (!rawHiresSampler && !rawHiresScheduler) return

    const normalizedHires = normalizeLiveSamplingSelection(
      rawHiresSampler || normalizedBase.sampler,
      rawHiresScheduler || normalizedBase.scheduler,
    )
    if (!normalizedHires) return

    const nextSamplerOverride = normalizedHires.sampler === normalizedBase.sampler ? '' : normalizedHires.sampler
    const nextSchedulerOverride = normalizedHires.scheduler === normalizedBase.scheduler ? '' : normalizedHires.scheduler
    if (
      nextSamplerOverride === params.value.hires.sampler
      && nextSchedulerOverride === params.value.hires.scheduler
    ) {
      return
    }
    setHires({
      sampler: nextSamplerOverride,
      scheduler: nextSchedulerOverride,
    })
  },
  { immediate: true },
)

watch(
  [() => xyzStore.xParam, () => xyzStore.xValuesText, xyzSamplerChoices, xyzSchedulerChoices],
  () => {
    const normalized = normalizeXyzSamplingAxisText(xyzStore.xParam, xyzStore.xValuesText)
    if (normalized === xyzStore.xValuesText) return
    xyzStore.xValuesText = normalized
  },
  { immediate: true },
)

watch(
  [() => xyzStore.yParam, () => xyzStore.yValuesText, xyzSamplerChoices, xyzSchedulerChoices],
  () => {
    const normalized = normalizeXyzSamplingAxisText(xyzStore.yParam, xyzStore.yValuesText)
    if (normalized === xyzStore.yValuesText) return
    xyzStore.yValuesText = normalized
  },
  { immediate: true },
)

watch(
  [() => xyzStore.zParam, () => xyzStore.zValuesText, xyzSamplerChoices, xyzSchedulerChoices],
  () => {
    const normalized = normalizeXyzSamplingAxisText(xyzStore.zParam, xyzStore.zValuesText)
    if (normalized === xyzStore.zValuesText) return
    xyzStore.zValuesText = normalized
  },
  { immediate: true },
)

const promptText = computed({
  get: () => params.value.prompt,
  set: (value: string) => setParams({ prompt: value }),
})

const negativeText = computed({
  get: () => params.value.negativePrompt,
  set: (value: string) => {
    if (!supportsNegative.value) return
    setParams({ negativePrompt: value })
  },
})

watch([supportsImg2Img, () => engineCaps.loaded], ([supported, capsLoaded]) => {
  if (!capsLoaded || supported) return
  if (!params.value.useInitImage) return
  setParams({
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  })
}, { immediate: true })

watch([supportsImg2ImgMasking, () => params.value.useMask], ([supported, useMask]) => {
  if (supported || !useMask) return
  setParams({
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  })
}, { immediate: true })

watch([availableInpaintModes, () => params.value.inpaintMode], ([modes, activeMode]) => {
  if (!Array.isArray(modes) || modes.length === 0) return
  if (modes.includes(activeMode)) return
  if (params.value.perStepBlendStrength !== fallbackParams.value.perStepBlendStrength || params.value.perStepBlendSteps !== fallbackParams.value.perStepBlendSteps) {
    setParams({
      perStepBlendStrength: fallbackParams.value.perStepBlendStrength,
      perStepBlendSteps: fallbackParams.value.perStepBlendSteps,
    })
  }
}, { immediate: true })

watch(() => hiresModePolicy.value.resetState, (shouldReset) => {
  if (!shouldReset) return
  if (!params.value.hires.enabled && !params.value.hires.refiner?.enabled) return
  setHires({ enabled: false })
  setHiresRefiner({ enabled: false })
})

watch(showGlobalRefiner, (show) => {
  if (show) return
  if (!params.value.refiner.enabled) return
  setRefiner({ enabled: false })
})

watch(showGlobalSwapModel, (show) => {
  if (show) return
  if (!params.value.swapModel.enabled) return
  setSwapModel({ enabled: false })
})

watch(showHiresRefiner, (show) => {
  if (show) return
  if (!params.value.hires.refiner?.enabled) return
  setHiresRefiner({ enabled: false })
})

watch(
  isExactOneShotImageRequest,
  (active) => {
    if (!active) return
    if (isQwenImageRequest.value && xyzStore.enabled) {
      xyzStore.enabled = false
    }
    const nextPatch: Partial<ImageBaseParams> = {}
    let needsPatch = false
    if (isQwenImageRequest.value && !params.value.useInitImage) {
      nextPatch.useInitImage = true
      needsPatch = true
    }
    if (isQwenImageRequest.value && params.value.denoiseStrength !== 1) {
      nextPatch.denoiseStrength = 1
      needsPatch = true
    }
    if (isQwenImageRequest.value && params.value.steps < 2) {
      nextPatch.steps = 2
      needsPatch = true
    }
    if (isZImageL2PRequest.value && (params.value.useInitImage || params.value.initImageData || params.value.initImageName)) {
      nextPatch.useInitImage = false
      nextPatch.initImageData = ''
      nextPatch.initImageName = ''
      needsPatch = true
    }
    if (isZImageL2PRequest.value && (params.value.width !== 1024 || params.value.height !== 1024)) {
      nextPatch.width = 1024
      nextPatch.height = 1024
      needsPatch = true
    }
    if (params.value.runAction !== 'generate') {
      nextPatch.runAction = 'generate'
      needsPatch = true
    }
    if (params.value.batchCount !== 1 || params.value.batchSize !== 1) {
      nextPatch.batchCount = 1
      nextPatch.batchSize = 1
      needsPatch = true
    }
    if (params.value.clipSkip !== 0) {
      nextPatch.clipSkip = 0
      needsPatch = true
    }
    if (params.value.useMask || params.value.maskImageData || params.value.maskImageName) {
      nextPatch.useMask = false
      nextPatch.maskImageData = ''
      nextPatch.maskImageName = ''
      needsPatch = true
    }
    if (params.value.initSource.mode !== 'img') {
      nextPatch.initSource = {
        ...params.value.initSource,
        mode: 'img',
        count: 1,
      }
      needsPatch = true
    }
    if (
      params.value.hires.enabled
      || Boolean(params.value.hires.swapModel?.model)
      || Boolean(params.value.hires.refiner?.enabled)
    ) {
      nextPatch.hires = {
        ...params.value.hires,
        enabled: false,
        swapModel: undefined,
        refiner: params.value.hires.refiner
          ? { ...params.value.hires.refiner, enabled: false }
          : params.value.hires.refiner,
      }
      needsPatch = true
    }
    if (params.value.swapModel.enabled) {
      nextPatch.swapModel = { ...params.value.swapModel, enabled: false }
      needsPatch = true
    }
    if (params.value.refiner.enabled) {
      nextPatch.refiner = { ...params.value.refiner, enabled: false }
      needsPatch = true
    }
    if (params.value.supir.enabled) {
      nextPatch.supir = { ...params.value.supir, enabled: false }
      needsPatch = true
    }
    if (params.value.ipAdapter.enabled || params.value.ipAdapter.source.sameAsInit || params.value.ipAdapter.source.mode !== 'img') {
      nextPatch.ipAdapter = {
        ...params.value.ipAdapter,
        enabled: false,
        source: {
          ...params.value.ipAdapter.source,
          mode: 'img',
          sameAsInit: false,
          count: 1,
        },
      }
      needsPatch = true
    }
    if (params.value.guidanceAdvanced.enabled) {
      nextPatch.guidanceAdvanced = {
        ...params.value.guidanceAdvanced,
        enabled: false,
        apgEnabled: false,
        cfgTruncEnabled: false,
      }
      needsPatch = true
    }
    if (needsPatch) {
      setParams(nextPatch)
    }
  },
  { immediate: true },
)

watch(
  () => params.value.useInitImage,
  (enabled, wasEnabled) => {
    if (enabled && params.value.hires.swapModel?.model) {
      setHires({ swapModel: undefined })
    }
    if (!enabled && params.value.ipAdapter.source.sameAsInit) {
      setIpAdapterSource({ sameAsInit: false })
    }
    if (!enabled || wasEnabled) return
    maybeApplyKontextDefaults()
  },
)

watch(
  supportsSupirModeSurface,
  (supported) => {
    if (!supported) {
      if (params.value.supir.enabled) {
        setSupir({ enabled: false })
      }
      return
    }
    void ensureSupirDiagnosticsLoaded()
  },
  { immediate: true },
)

watch(
  () => params.value.useInitImage,
  (enabled) => {
    if (enabled || !params.value.supir.enabled) return
    setSupir({ enabled: false })
  },
)

watch(
  () => [
    params.value.supir.enabled,
    supirSelectionValid.value,
    params.value.hires.enabled,
    Boolean(params.value.hires.refiner?.enabled),
    Boolean(params.value.hires.swapModel?.model),
    params.value.ipAdapter.enabled,
  ] as const,
  ([enabled, selectionValid, hiresEnabled, hiresRefinerEnabled, hiresSwapModelEnabled, ipAdapterEnabled]) => {
    if (!enabled || !selectionValid) return
    const nextPatch: Partial<ImageBaseParams> = {}
    let needsPatch = false
    if (hiresEnabled || hiresRefinerEnabled || hiresSwapModelEnabled) {
      nextPatch.hires = {
        ...params.value.hires,
        enabled: false,
        swapModel: undefined,
        refiner: params.value.hires.refiner
          ? { ...params.value.hires.refiner, enabled: false }
          : params.value.hires.refiner,
      }
      needsPatch = true
    }
    if (ipAdapterEnabled) {
      nextPatch.ipAdapter = {
        ...params.value.ipAdapter,
        enabled: false,
      }
      needsPatch = true
    }
    if (needsPatch) {
      setParams(nextPatch)
    }
  },
  { immediate: true },
)

watch(
  () => params.value.runAction,
  (actionMode) => {
    if (actionMode !== 'infinite') return
    if (params.value.batchCount === 1 && params.value.batchSize === 1) return
    setParams({ batchCount: 1, batchSize: 1 })
  },
  { immediate: true },
)

watch(
  [() => xyzStore.enabled, () => params.value.runAction],
  ([xyzEnabled, actionMode], previous) => {
    if (!xyzEnabled || actionMode !== 'infinite') return
    setRunAction('generate')
    if (previous?.[0] === false) {
      toast('XYZ uses Generate. Infinite was reset to Generate.')
    }
  },
  { immediate: true },
)

const images = computed(() => gallery.value)

const gentimeSeconds = computed(() => {
  if (gentimeMs.value == null) return null
  return gentimeMs.value / 1000
})

const progressPercent = computed(() => {
  if (progress.value.percent !== null) return progress.value.percent
  if (!progress.value.totalSteps || progress.value.step === null) return null
  return (progress.value.step / progress.value.totalSteps) * 100
})

const resultsEmptyText = computed(() => {
  if (!isRunning.value) return 'No images yet. Generate to see results here.'
  const stage = String(progress.value.stage || 'starting')
  if (stage === 'starting' || stage === 'submitted' || stage === 'queued') return 'Starting inference…'
  if (progressPercent.value !== null) return `Generating… (${progressPercent.value.toFixed(1)}%)`
  return `Generating… (${stage})`
})

const previewCaption = computed(() => {
  const step = previewStep.value
  if (step !== null && progress.value.totalSteps) return `Live preview · step ${step}/${progress.value.totalSteps}`
  if (step !== null) return `Live preview · step ${step}`
  return 'Live preview'
})

const resolutionPresets = computed((): [number, number][] => {
  if (isZImageL2PRequest.value) return [[1024, 1024]]
  if (props.type === 'sd15') return [[512, 512], [512, 768], [768, 512]]
  return [[1024, 1024], [1152, 896], [1216, 832], [1344, 768]]
})

const runSummary = computed(() => {
  const normalizedSampling = normalizeLiveSamplingSelection(params.value.sampler, params.value.scheduler)
  const defaults = resolveLiveSamplingDefaults()
  const sampler = normalizedSampling?.sampler || defaults.sampler
  const scheduler = normalizedSampling?.scheduler || defaults.scheduler
  const seedLabel = params.value.seed === -1 ? 'seed random' : `seed ${params.value.seed}`
  const sizeLabel = isQwenImageRequest.value && params.value.useInitImage
    ? 'init-derived size'
    : `${params.value.width}×${params.value.height} px`
  const parts = [
    sizeLabel,
    `${params.value.steps} steps`,
    `${cfgLabel.value} ${params.value.cfgScale}`,
    `${sampler} / ${scheduler}`,
    seedLabel,
  ]
  if (!isExactOneShotImageRequest.value) {
    parts.push(`batch ${params.value.batchCount}×${params.value.batchSize}`)
  }
  return parts.join(' · ')
})

async function onGenerate(actionMode: ImageRunAction = params.value.runAction): Promise<void> {
  if (actionMode !== params.value.runAction) {
    setRunAction(actionMode)
  }
  if (infiniteXyzConflict.value) {
    toast('Infinite generate cannot run while XYZ is enabled.')
    return
  }
  if (unsupportedInpaintMode.value) {
    toast(unsupportedInpaintModeMessage.value)
    return
  }
  if (isQwenImageRequest.value && xyzStore.enabled) {
    xyzStore.enabled = false
    toast('Qwen Image Edit-2511 does not support XYZ.')
    return
  }
  if (xyzStore.enabled) {
    await xyzStore.run()
    return
  }
  await generateBase()
}

async function onCancelRun(): Promise<void> {
  try {
    if (xyzRunning.value) {
      await xyzStore.stop('immediate')
      return
    }
    await cancelBase()
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  }
}

async function copyHistoryParams(item: ImageRunHistoryItem): Promise<void> {
  await copyJson(item.paramsSnapshot, 'Copied history params.')
}

function openHistoryDetails(item: ImageRunHistoryItem): void {
  historyDetailsItem.value = item
  historyDetailsOpen.value = true
}

async function onLoadHistoryDetails(): Promise<void> {
  const item = historyDetailsItem.value
  if (!item) return
  await loadHistory(item.taskId)
}

async function onApplyHistoryDetails(): Promise<void> {
  const item = historyDetailsItem.value
  if (!item) return
  await applyHistory(item)
}

async function onCopyHistoryDetails(): Promise<void> {
  const item = historyDetailsItem.value
  if (!item) return
  await copyHistoryParams(item)
}

async function applyHistory(item: ImageRunHistoryItem): Promise<void> {
  if (historyApplyingTaskId.value) return
  historyApplyingTaskId.value = item.taskId
  const rawSnapshot = item.paramsSnapshot as Record<string, unknown>
  const snapshotVae = typeof rawSnapshot.vae === 'string' ? rawSnapshot.vae.trim() : ''
  let previousPersistedFamilyVae = ''
  const { vae: _ignoredSnapshotVae, ...snapshotWithoutAssets } = rawSnapshot
  const snap = snapshotWithoutAssets as Partial<ImageBaseParams>
  const snapshotUseInitImage = Boolean(snap.useInitImage ?? (item.mode === 'img2img' || snap.supir?.enabled))
  const snapshotUseMask = snapshotUseInitImage && Boolean(snap.useMask)
  const snapshotGuidanceAdvanced = (snap.guidanceAdvanced && typeof snap.guidanceAdvanced === 'object')
    ? normalizeGuidanceAdvancedPatch(snap.guidanceAdvanced, fallbackParams.value.guidanceAdvanced)
    : fallbackParams.value.guidanceAdvanced
  const rawSupir = (snap.supir && typeof snap.supir === 'object')
    ? snap.supir as Partial<SupirModeFormState>
    : null
  const snapshotSupir: SupirModeFormState = rawSupir
    ? {
        ...fallbackParams.value.supir,
        ...rawSupir,
        sampler: normalizeSupirSamplerSelection(rawSupir.sampler, fallbackParams.value.supir.sampler),
      }
    : fallbackParams.value.supir
  const nextPatch: Partial<ImageBaseParams> = {
    ...snap,
    useInitImage: snapshotUseInitImage,
    initImageData: '',
    initImageName: '',
    useMask: snapshotUseMask,
    maskImageData: '',
    maskImageName: '',
    guidanceAdvanced: snapshotGuidanceAdvanced,
    supir: snapshotSupir,
  }
  const blockingReason = getSupirRestoreBlockingReason({
    useInitImage: snapshotUseInitImage,
    guidanceAdvanced: snapshotGuidanceAdvanced,
    supir: snapshotSupir,
  })
  if (blockingReason) {
    toast(`Cannot apply history params: ${blockingReason}`)
    historyApplyingTaskId.value = ''
    return
  }
  let restoredFamilyVae = false
  const snapshotEpoch = quicksettingsStore.bumpAssetSnapshotEpoch()
  const assertSnapshotEpochCurrent = (): void => {
    if (quicksettingsStore.getAssetSnapshotEpoch() !== snapshotEpoch) {
      throw new Error(ASSET_RESTORE_SUPERSEDED_ERROR)
    }
  }
  try {
    if (snapshotVae) {
      await quicksettingsStore.init()
      assertSnapshotEpochCurrent()
      previousPersistedFamilyVae = quicksettingsStore.getPersistedVaeForFamily(props.type)
      invalidateModelCatalogCaches()
      const inventory = await fetchFreshModelInventory()
      const pathsConfig = (await fetchFreshPaths()).paths || {}
      assertSnapshotEpochCurrent()
      quicksettingsStore.hydrateVaeInventorySnapshot(inventory)
      quicksettingsStore.hydratePathsSnapshot(pathsConfig)
      invalidateModelCatalogCaches()
      const canonicalVae = canonicalizeVaeChoice(
        snapshotVae,
        buildFamilyVaeValidationChoices(
          props.type,
          inventory.vaes,
          pathsConfig,
        ),
        (label) => resolveInventoryVaeSha(label, inventory.vaes),
      )
      if (!canonicalVae || canonicalVae.reason === 'fallback') {
        throw new Error(`History VAE '${snapshotVae}' is no longer available for '${props.type}'.`)
      }
      assertSnapshotEpochCurrent()
      await quicksettingsStore.setVaeForFamily(props.type, canonicalVae.value, {
        expectedAssetSnapshotEpoch: snapshotEpoch,
      })
      assertSnapshotEpochCurrent()
      restoredFamilyVae = true
    }
    const normalizedPatch = normalizeImageParamPatch(nextPatch)
    assertSnapshotEpochCurrent()
    await store.updateParams(props.tabId, normalizedPatch as Partial<Record<string, unknown>>)
    assertSnapshotEpochCurrent()
    toast('Applied history params.')
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    if (message === ASSET_RESTORE_SUPERSEDED_ERROR) {
      return
    }
    if (restoredFamilyVae && quicksettingsStore.getAssetSnapshotEpoch() === snapshotEpoch) {
      try {
        await quicksettingsStore.restoreVaeForFamilyOwner(props.type, previousPersistedFamilyVae, {
          expectedAssetSnapshotEpoch: snapshotEpoch,
        })
      } catch (rollbackError) {
        const rollbackMessage = rollbackError instanceof Error ? rollbackError.message : String(rollbackError)
        toast(`${message} VAE rollback failed for '${props.type}': ${rollbackMessage}`)
        return
      }
    }
    toast(message)
  } finally {
    if (historyApplyingTaskId.value === item.taskId) {
      historyApplyingTaskId.value = ''
    }
  }
}

function formatHistoryTitle(item: { mode: string; createdAtMs: number; taskId: string }): string {
  const dt = new Date(item.createdAtMs || Date.now())
  const hh = dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const label = item.mode === 'img2img' ? 'Img2Img' : 'Txt2Img'
  return `${label} · ${hh}`
}

function readHistorySnapshotText(item: ImageRunHistoryItem, key: string): string {
  const snapshot = item.paramsSnapshot
  if (!snapshot || typeof snapshot !== 'object') return ''
  const value = (snapshot as Record<string, unknown>)[key]
  if (typeof value !== 'string') return ''
  return value.trim()
}

function profileStorageKeyFor(type: ImageTabType): string {
  if (type === 'flux1') return 'codex.flux1.profile.v1'
  if (type === 'flux2') return 'codex.flux2.profile.v1'
  if (type === 'sdxl') return 'codex.sdxl.profile.v1'
  if (type === 'zimage') return 'codex.zimage.profile'
  if (type === 'sd15') return 'codex.sd15.profile.v1'
  return `codex.${type}.profile.v1`
}

function normalizeGuidanceAdvancedPatch(raw: unknown, base: GuidanceAdvancedParams): GuidanceAdvancedParams {
  const source = raw && typeof raw === 'object' && !Array.isArray(raw)
    ? (raw as Record<string, unknown>)
    : {}
  const toFinite = (value: unknown, fallback: number): number => {
    const numeric = Number(value)
    return Number.isFinite(numeric) ? numeric : fallback
  }
  const clamp = (value: unknown, fallback: number, min?: number, max?: number): number => {
    const numeric = toFinite(value, fallback)
    if (min !== undefined && numeric < min) return min
    if (max !== undefined && numeric > max) return max
    return numeric
  }
  const clampInt = (value: unknown, fallback: number, min?: number): number => {
    const numeric = Math.trunc(toFinite(value, fallback))
    if (min !== undefined && numeric < min) return min
    return numeric
  }
  return {
    enabled: typeof source.enabled === 'boolean' ? source.enabled : base.enabled,
    apgEnabled: typeof source.apgEnabled === 'boolean' ? source.apgEnabled : base.apgEnabled,
    apgStartStep: clampInt(source.apgStartStep, base.apgStartStep, 0),
    apgEta: clamp(source.apgEta, base.apgEta),
    apgMomentum: clamp(source.apgMomentum, base.apgMomentum, 0, 0.99),
    apgNormThreshold: clamp(source.apgNormThreshold, base.apgNormThreshold, 0),
    apgRescale: clamp(source.apgRescale, base.apgRescale, 0, 1),
    guidanceRescale: clamp(source.guidanceRescale, base.guidanceRescale, 0, 1),
    cfgTruncEnabled: typeof source.cfgTruncEnabled === 'boolean' ? source.cfgTruncEnabled : base.cfgTruncEnabled,
    cfgTruncRatio: clamp(source.cfgTruncRatio, base.cfgTruncRatio, 0, 1),
    renormCfg: clamp(source.renormCfg, base.renormCfg, 0),
  }
}

function loadProfile(): void {
  const key = profileStorageKeyFor(props.type)
  try {
    const raw = localStorage.getItem(key)
    if (!raw) {
      toast('No saved profile found.')
      return
    }

    const snapshot = JSON.parse(raw) as Record<string, unknown>
    const next: Partial<ImageBaseParams> = {}

    const numberOrNull = (value: unknown): number | null => {
      const n = Number(value)
      return Number.isFinite(n) ? n : null
    }

    if (typeof snapshot.prompt === 'string') next.prompt = snapshot.prompt
    if (supportsNegative.value && typeof snapshot.negativePrompt === 'string') next.negativePrompt = snapshot.negativePrompt
    const steps = numberOrNull(snapshot.steps); if (steps !== null) next.steps = Math.max(1, Math.trunc(steps))
    const cfgScale = numberOrNull(snapshot.cfgScale); if (cfgScale !== null) next.cfgScale = cfgScale
    const width = numberOrNull(snapshot.width); if (width !== null) next.width = Math.max(64, Math.trunc(width))
    const height = numberOrNull(snapshot.height); if (height !== null) next.height = Math.max(64, Math.trunc(height))
    const seed = numberOrNull(snapshot.seed); if (seed !== null) next.seed = Math.trunc(seed)
    const clipSkip = numberOrNull(snapshot.clipSkip); if (clipSkip !== null) next.clipSkip = Math.max(minClipSkip.value, Math.trunc(clipSkip))
    const batchSize = numberOrNull(snapshot.batchSize); if (batchSize !== null) next.batchSize = Math.max(1, Math.trunc(batchSize))
    const batchCount = numberOrNull(snapshot.batchCount); if (batchCount !== null) next.batchCount = Math.max(1, Math.trunc(batchCount))
    const snapshotGuidanceAdvanced = (snapshot.guidanceAdvanced && typeof snapshot.guidanceAdvanced === 'object')
      ? normalizeGuidanceAdvancedPatch(snapshot.guidanceAdvanced, fallbackParams.value.guidanceAdvanced)
      : fallbackParams.value.guidanceAdvanced

    const selectedModel = typeof snapshot.selectedModel === 'string' ? snapshot.selectedModel : ''
    const selectedSampler = typeof snapshot.selectedSampler === 'string' ? snapshot.selectedSampler : ''
    const selectedScheduler = typeof snapshot.selectedScheduler === 'string' ? snapshot.selectedScheduler : ''
    const hasUseInitImage = Object.prototype.hasOwnProperty.call(snapshot, 'useInitImage')
    const hasUseMask = Object.prototype.hasOwnProperty.call(snapshot, 'useMask')
    const hasDenoiseStrength = Object.prototype.hasOwnProperty.call(snapshot, 'denoiseStrength')
    const hasResizeMode = Object.prototype.hasOwnProperty.call(snapshot, 'img2imgResizeMode')
    const hasUpscaler = Object.prototype.hasOwnProperty.call(snapshot, 'img2imgUpscaler')
    const hasSupirSnapshot = Object.prototype.hasOwnProperty.call(snapshot, 'supir')
    const denoiseStrength = numberOrNull(snapshot.denoiseStrength)
    const supirSnapshot = (snapshot.supir && typeof snapshot.supir === 'object')
      ? snapshot.supir as Partial<SupirModeFormState>
      : null
    const useInitImage = hasUseInitImage
      ? Boolean(snapshot.useInitImage)
      : Boolean(supirSnapshot?.enabled || params.value.useInitImage)
    const useMask = useInitImage && (hasUseMask ? Boolean(snapshot.useMask) : params.value.useMask)

    if (selectedModel) next.checkpoint = selectedModel
    if (selectedSampler) next.sampler = selectedSampler
    if (selectedScheduler) next.scheduler = selectedScheduler
    if (hasDenoiseStrength && denoiseStrength !== null) next.denoiseStrength = clampFloat(denoiseStrength, 0, 1)
    if (hasResizeMode && typeof snapshot.img2imgResizeMode === 'string') next.img2imgResizeMode = snapshot.img2imgResizeMode as ImageBaseParams['img2imgResizeMode']
    if (hasUpscaler && typeof snapshot.img2imgUpscaler === 'string') next.img2imgUpscaler = snapshot.img2imgUpscaler
    if (hasUseInitImage || hasUseMask || hasSupirSnapshot) {
      next.useInitImage = useInitImage
      next.useMask = useMask
      next.initImageData = ''
      next.initImageName = ''
      next.maskImageData = ''
      next.maskImageName = ''
    }
    const snapshotSupir: SupirModeFormState = hasSupirSnapshot
      ? {
          ...fallbackParams.value.supir,
          ...(supirSnapshot ?? {}),
          sampler: normalizeSupirSamplerSelection(supirSnapshot?.sampler, fallbackParams.value.supir.sampler),
        }
      : fallbackParams.value.supir
    next.guidanceAdvanced = snapshotGuidanceAdvanced
    next.supir = snapshotSupir

    const blockingReason = getSupirRestoreBlockingReason({
      useInitImage: next.useInitImage ?? fallbackParams.value.useInitImage,
      guidanceAdvanced: snapshotGuidanceAdvanced,
      supir: snapshotSupir,
    })
    if (blockingReason) {
      toast(`Cannot load saved profile: ${blockingReason}`)
      return
    }

    setParams(next)
    toast('Loaded saved profile.')
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function saveProfile(): void {
  const key = profileStorageKeyFor(props.type)
  try {
    const snapshot = {
      prompt: params.value.prompt,
      negativePrompt: supportsNegative.value ? params.value.negativePrompt : '',
      steps: params.value.steps,
      cfgScale: params.value.cfgScale,
      width: params.value.width,
      height: params.value.height,
      seed: params.value.seed,
      clipSkip: params.value.clipSkip,
      batchSize: params.value.batchSize,
      batchCount: params.value.batchCount,
      guidanceAdvanced: params.value.guidanceAdvanced,
      selectedModel: params.value.checkpoint,
      selectedSampler: params.value.sampler,
      selectedScheduler: params.value.scheduler,
      useInitImage: params.value.useInitImage,
      useMask: params.value.useMask,
      denoiseStrength: params.value.denoiseStrength,
      img2imgResizeMode: params.value.img2imgResizeMode,
      img2imgUpscaler: params.value.img2imgUpscaler,
      supir: params.value.supir,
    }
    localStorage.setItem(key, JSON.stringify(snapshot))
    toast('Profile saved.')
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function setParams(patch: Partial<ImageBaseParams>): void {
  if (!tab.value) return
  const normalizedPatch = normalizeImageParamPatch(patch)
  store.updateParams(props.tabId, normalizedPatch as Partial<Record<string, unknown>>).catch((error) => {
    toast(error instanceof Error ? error.message : String(error))
  })
}

function setRunAction(actionMode: ImageRunAction): void {
  if (isExactOneShotImageRequest.value) {
    setParams({ runAction: 'generate', batchCount: 1, batchSize: 1 })
    return
  }
  if (actionMode === 'infinite') {
    setParams({ runAction: actionMode, batchCount: 1, batchSize: 1 })
    return
  }
  setParams({ runAction: actionMode })
}

function setGuidanceAdvanced(patch: Partial<GuidanceAdvancedParams>): void {
  const next = normalizeGuidanceAdvancedPatch(
    patch,
    params.value.guidanceAdvanced,
  )
  setParams({ guidanceAdvanced: next })
}

function setHires(patch: Partial<ImageBaseParams['hires']>): void {
  setParams({ hires: { ...params.value.hires, ...patch } })
}

function setHiresSwapModel(value: string): void {
  const model = String(value || '').trim()
  setHires({ swapModel: model ? { model } : undefined })
}

function setSwapModel(patch: Partial<ImageBaseParams['swapModel']>): void {
  const nextSwapModel = {
    ...params.value.swapModel,
    ...patch,
  }
  if (patch.enabled === true && !params.value.swapModel.enabled && patch.cfg === undefined) {
    nextSwapModel.cfg = params.value.cfgScale
  }
  const swapAtStep = Number(nextSwapModel.swapAtStep)
  nextSwapModel.swapAtStep = Number.isFinite(swapAtStep) && swapAtStep >= 1 ? Math.trunc(swapAtStep) : 1
  setParams({ swapModel: nextSwapModel })
}

function setHiresRefiner(patch: Partial<NonNullable<ImageBaseParams['hires']['refiner']>>): void {
  const nextRefiner = {
    enabled: false,
    swapAtStep: 1,
    cfg: 3.5,
    seed: -1,
    model: undefined,
    ...(params.value.hires.refiner || {}),
    ...patch,
  }
  const swapAtStep = Number(nextRefiner.swapAtStep)
  nextRefiner.swapAtStep = Number.isFinite(swapAtStep) && swapAtStep >= 1 ? Math.trunc(swapAtStep) : 1
  setHires({ refiner: nextRefiner })
}

function setRefiner(patch: Partial<ImageBaseParams['refiner']>): void {
  const nextRefiner = { ...params.value.refiner, ...patch }
  const swapAtStep = Number(nextRefiner.swapAtStep)
  nextRefiner.swapAtStep = Number.isFinite(swapAtStep) && swapAtStep >= 1 ? Math.trunc(swapAtStep) : 1
  setParams({ refiner: nextRefiner })
}

function setInitSource(patch: Partial<ImageBaseParams['initSource']>): void {
  const nextInitSource = {
    ...params.value.initSource,
    ...patch,
  }
  const nextInitCount = Math.trunc(Number(nextInitSource.count))
  nextInitSource.count = Number.isFinite(nextInitCount) && nextInitCount >= 1 ? nextInitCount : 1
  const nextPatch: Partial<ImageBaseParams> = { initSource: nextInitSource }
  if (nextInitSource.mode !== 'img') {
    nextPatch.batchCount = 1
    nextPatch.batchSize = 1
    nextPatch.useMask = false
    nextPatch.maskImageData = ''
    nextPatch.maskImageName = ''
    if (params.value.ipAdapter.source.sameAsInit) {
      nextPatch.ipAdapter = {
        ...params.value.ipAdapter,
        source: {
          ...params.value.ipAdapter.source,
          sameAsInit: false,
        },
      }
    }
  }
  setParams(nextPatch)
}

function normalizeSupirColorFix(value: unknown, fallback: SupirModeFormState['colorFix']): SupirModeFormState['colorFix'] {
  return value === 'AdaIN' || value === 'Wavelet' || value === 'None' ? value : fallback
}

function setSupir(patch: SupirPatch): void {
  const nextSupir: ImageBaseParams['supir'] = {
    ...params.value.supir,
    ...patch,
  }
  nextSupir.enabled = Boolean(nextSupir.enabled)
  nextSupir.variant = nextSupir.variant === 'v0F' ? 'v0F' : 'v0Q'
  nextSupir.sampler = String(nextSupir.sampler || '').trim()
  nextSupir.controlScale = clampFloat(Number(nextSupir.controlScale), 0.01, 2)
  nextSupir.restorationScale = clampFloat(Number(nextSupir.restorationScale), 0.01, 6)
  nextSupir.restoreCfgSTmin = clampFloat(Number(nextSupir.restoreCfgSTmin), 0, 5)
  nextSupir.colorFix = normalizeSupirColorFix(nextSupir.colorFix, params.value.supir.colorFix)
  setParams({ supir: nextSupir })
}

function setIpAdapter(patch: IpAdapterPatch): void {
  const nextSource = patch.source
    ? {
        ...params.value.ipAdapter.source,
        ...patch.source,
      }
    : params.value.ipAdapter.source
  const nextIpAdapter: ImageBaseParams['ipAdapter'] = {
    ...params.value.ipAdapter,
    ...patch,
    source: nextSource,
  }
  nextIpAdapter.weight = clampFloat(Number(nextIpAdapter.weight), 0, 2)
  nextIpAdapter.startAt = clampFloat(Number(nextIpAdapter.startAt), 0, 1)
  nextIpAdapter.endAt = clampFloat(Number(nextIpAdapter.endAt), 0, 1)
  const nextSourceCount = Math.trunc(Number(nextIpAdapter.source.count))
  nextIpAdapter.source.count = Number.isFinite(nextSourceCount) && nextSourceCount >= 1 ? nextSourceCount : 1
  const nextPatch: Partial<ImageBaseParams> = { ipAdapter: nextIpAdapter }
  if (nextIpAdapter.enabled && nextIpAdapter.source.mode === 'dir') {
    nextPatch.batchCount = 1
    nextPatch.batchSize = 1
  }
  if (nextIpAdapter.source.mode !== 'img' || !params.value.useInitImage) {
    nextIpAdapter.source.sameAsInit = false
  }
  if (nextIpAdapter.endAt < nextIpAdapter.startAt) {
    nextIpAdapter.endAt = nextIpAdapter.startAt
  }
  setParams(nextPatch)
}

function setIpAdapterSource(patch: Partial<ImageBaseParams['ipAdapter']['source']>): void {
  setIpAdapter({ source: patch })
}

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function setMinTile(value: number): void {
  const v = Math.max(1, Math.trunc(Number(value)))
  if (!Number.isFinite(v)) return
  minTile.value = v
}

const _KONTEXT_DEFAULT_STEPS = 28
const _KONTEXT_DEFAULT_DISTILLED_CFG = 2.5
const _INIT_IMAGE_DIM_MIN = 64
const _INIT_IMAGE_DIM_MAX = 8192

function snapInitImageDim(value: number, step: number): number {
  if (isZImageL2PRequest.value) return 1024
  const clamped = Math.max(_INIT_IMAGE_DIM_MIN, Math.min(_INIT_IMAGE_DIM_MAX, Math.trunc(value)))
  const safeStep = Number.isFinite(step) && step > 0 ? Math.trunc(step) : 8
  const snapped = (resolvedEngineForMode.value === 'zimage' ? Math.floor(clamped / safeStep) : Math.round(clamped / safeStep)) * safeStep
  return Math.max(_INIT_IMAGE_DIM_MIN, Math.min(_INIT_IMAGE_DIM_MAX, snapped))
}

function normalizeImageDimension(value: unknown): number {
  if (isZImageL2PRequest.value) return 1024
  const numeric = Number(value)
  const fallback = Number.isFinite(numeric) ? numeric : _INIT_IMAGE_DIM_MIN
  return snapInitImageDim(fallback, imageDimensionInputStep.value)
}

function normalizeImageParamPatch(patch: Partial<ImageBaseParams>): Partial<ImageBaseParams> {
  const next: Partial<ImageBaseParams> = { ...patch }
  if (patch.width !== undefined) next.width = normalizeImageDimension(patch.width)
  if (patch.height !== undefined) next.height = normalizeImageDimension(patch.height)
  if (patch.perStepBlendStrength !== undefined) {
    next.perStepBlendStrength = clampFloat(Number(patch.perStepBlendStrength), 0, 1)
  }
  if (patch.perStepBlendSteps !== undefined) {
    next.perStepBlendSteps = normalizeNonNegativeInt(patch.perStepBlendSteps)
  }
  if (patch.inpaintMode !== undefined) {
    const nextInpaintMode = parseInpaintMode(patch.inpaintMode)
    if (nextInpaintMode === null) {
      throw new Error(`ImageModelTab received invalid inpaintMode patch '${String(patch.inpaintMode)}'.`)
    }
    next.inpaintMode = nextInpaintMode
    if (next.inpaintMode !== 'per_step_blend') {
      next.perStepBlendStrength = fallbackParams.value.perStepBlendStrength
      next.perStepBlendSteps = fallbackParams.value.perStepBlendSteps
    }
  }
  if (patch.img2imgResizeMode !== undefined) {
    next.img2imgResizeMode = normalizeImg2ImgResizeModeForEngine(resolvedEngineForMode.value, patch.img2imgResizeMode)
  }
  if (patch.maskInvert !== undefined || patch.maskRegionSplit !== undefined) {
    const normalizedMaskToggles = normalizeInpaintMaskToggleState(
      patch.maskInvert ?? params.value.maskInvert,
      patch.maskRegionSplit ?? params.value.maskRegionSplit,
    )
    next.maskInvert = normalizedMaskToggles.maskInvert
    next.maskRegionSplit = normalizedMaskToggles.maskRegionSplit
  }
  if (patch.supir && typeof patch.supir === 'object') {
    const supirPatch = patch.supir as Partial<SupirModeFormState>
    next.supir = {
      ...params.value.supir,
      ...supirPatch,
      sampler: normalizeSupirSamplerSelection(supirPatch.sampler, params.value.supir.sampler),
    }
  }
  return next
}

function syncImageContractToEngine(): void {
  const patch = normalizeImageParamPatch({
    width: params.value.width,
    height: params.value.height,
    img2imgResizeMode: params.value.img2imgResizeMode,
  })
  const needsUpdate = (
    patch.width !== params.value.width
    || patch.height !== params.value.height
    || patch.img2imgResizeMode !== params.value.img2imgResizeMode
  )
  if (!needsUpdate) return
  setParams(patch)
}

function onInpaintModeChange(rawValue: string): void {
  const nextMode = parseInpaintMode(rawValue)
  if (nextMode === null) {
    throw new Error(`ImageModelTab received invalid inpaintMode '${rawValue}'.`)
  }
  const patch: Partial<ImageBaseParams> = { inpaintMode: nextMode }
  if (nextMode !== 'per_step_blend') {
    patch.perStepBlendStrength = fallbackParams.value.perStepBlendStrength
    patch.perStepBlendSteps = fallbackParams.value.perStepBlendSteps
  }
  setParams(patch)
}

watch(
  resolvedEngineForMode,
  () => {
    syncImageContractToEngine()
  },
  { immediate: true },
)

watch(
  () => [params.value.maskInvert, params.value.maskRegionSplit] as const,
  ([maskInvert, maskRegionSplit]) => {
    const normalizedMaskToggles = normalizeInpaintMaskToggleState(maskInvert, maskRegionSplit)
    if (
      normalizedMaskToggles.maskInvert === maskInvert
      && normalizedMaskToggles.maskRegionSplit === maskRegionSplit
    ) {
      return
    }
    setParams(normalizedMaskToggles)
  },
  { immediate: true },
)

async function onInitFileSet(file: File): Promise<void> {
  const dataUrl = await readFileAsDataURL(file)
  const patch: Partial<ImageBaseParams> = {
    initSource: {
      ...params.value.initSource,
      mode: 'img',
    },
    initImageData: dataUrl,
    initImageName: file.name,
    useInitImage: true,
    useMask: Boolean(params.value.useMask),
    maskImageData: '',
    maskImageName: '',
  }
  try {
    const { width, height } = await readImageDimensions(dataUrl)
    patch.width = normalizeImageDimension(width)
    patch.height = normalizeImageDimension(height)
  } catch {
    // ignore: keep current dims
  }
  setParams(patch)
}

function onInitImageRejected(payload: { reason: string; files: File[] }): void {
  const fileName = payload.files[0]?.name || 'file'
  toast(`Init image rejected (${fileName}): ${payload.reason}`)
}

function clearInit(): void {
  const nextPatch: Partial<ImageBaseParams> = {
    initImageData: '',
    initImageName: '',
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  }
  if (params.value.ipAdapter.source.sameAsInit) {
    nextPatch.ipAdapter = {
      ...params.value.ipAdapter,
      source: {
        ...params.value.ipAdapter.source,
        sameAsInit: false,
      },
    }
  }
  setParams(nextPatch)
}

function clearMask(): void {
  setParams({ maskImageData: '', maskImageName: '' })
}

async function onIpAdapterReferenceFileSet(file: File): Promise<void> {
  const dataUrl = await readFileAsDataURL(file)
  setIpAdapterSource({
    mode: 'img',
    sameAsInit: false,
    referenceImageData: dataUrl,
    referenceImageName: file.name,
  })
}

function clearIpAdapterReference(): void {
  setIpAdapterSource({
    referenceImageData: '',
    referenceImageName: '',
    sameAsInit: false,
  })
}

function onIpAdapterReferenceRejected(payload: { reason: string; files: File[] }): void {
  const fileName = payload.files[0]?.name || 'file'
  toast(`IP-Adapter reference rejected (${fileName}): ${payload.reason}`)
}

async function onMaskEditorApply(maskDataUrl: string): Promise<void> {
  if (!params.value.initImageData) {
    toast('Select an initial image before editing a mask.')
    return
  }

  let initDims: { width: number; height: number }
  try {
    initDims = await readImageDimensions(params.value.initImageData)
  } catch {
    toast('Failed to load init image for mask validation.')
    return
  }

  try {
    const { width, height } = await readImageDimensions(maskDataUrl)
    if (width !== initDims.width || height !== initDims.height) {
      toast(`Mask size must match init image size: expected ${initDims.width}×${initDims.height}, got ${width}×${height}.`)
      return
    }
  } catch {
    toast('Failed to load edited mask image.')
    return
  }

  setParams({ useMask: true, maskImageData: maskDataUrl, maskImageName: 'edited-mask.png' })
}

function onMaskEditorResetNotice(message: string): void {
  const text = String(message || '').trim()
  if (!text) return
  toast(text)
}

function toDataUrl(image: GeneratedImage): string { return `data:image/${image.format};base64,${image.data}` }

function randomizeSeed(): void {
  setParams({ seed: -1 })
}

function reuseSeed(): void {
  if (lastSeed.value !== null) setParams({ seed: lastSeed.value })
}

function download(image: GeneratedImage, index: number): void {
  const link = document.createElement('a')
  link.href = toDataUrl(image)
  link.download = `${props.type}_${index + 1}.png`
  link.click()
}

async function sendToImg2Img(image: GeneratedImage): Promise<void> {
  if (!supportsImg2Img.value) return
  const dataUrl = toDataUrl(image)
  const patch: Partial<ImageBaseParams> = {
    useInitImage: true,
    initSource: {
      ...params.value.initSource,
      mode: 'img',
    },
    initImageData: dataUrl,
    initImageName: `from_${props.type}.png`,
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  }
  try {
    const { width, height } = await readImageDimensions(dataUrl)
    patch.width = normalizeImageDimension(width)
    patch.height = normalizeImageDimension(height)
  } catch {
    // ignore
  }
  setParams(patch)
}

async function syncInitImageDims(): Promise<void> {
  const src = String(params.value.initImageData || '')
  if (!src) return
  try {
    const { width, height } = await readImageDimensions(src)
    setParams({ width: normalizeImageDimension(width), height: normalizeImageDimension(height) })
  } catch {
    // ignore
  }
}

function maybeApplyKontextDefaults(): void {
  if (props.type !== 'flux1') return
  const defaults = getEngineDefaults(props.type)
  const defaultCfg = defaults.distilledCfg ?? defaults.cfg
  // Only apply when user hasn't customized away from the Flux defaults.
  if (params.value.steps === defaults.steps) setParams({ steps: _KONTEXT_DEFAULT_STEPS })
  if (params.value.cfgScale === defaultCfg) setParams({ cfgScale: _KONTEXT_DEFAULT_DISTILLED_CFG })
}

defineExpose({ generate: onGenerate })
</script>
