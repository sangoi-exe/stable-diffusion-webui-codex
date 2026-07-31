<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared QuickSettings top bar for Model Tabs (SD/Flux/Qwen Image/Chroma/ZImage/Z-Image L2P/LTX/WAN).
Loads `/api/options`, `/api/models`, `/api/models/inventory`, and `/api/paths`, then filters/presents per-family selectors (models/TE/VAE)
through the shared non-WAN `QuickSettingsAssetBlock.vue` owner plus the specialized exact WAN branches, and commits overrides (device + runtime flags + tab-scoped Z-Image variant) used by generation payload builders. Asset-contract-derived selector
hints now disappear when checkpoint inventory metadata lacks a valid `core_only` flag, preventing stale UI contract display. FLUX.2 stays
first-class as the current Klein 4B / base-4B slice (single Qwen3-4B selector, backend-capability-driven img2img/inpaint gating, no FLUX.1 aliasing).
For LTX, QuickSettings remains the owner of mode + checkpoint/VAE/text-encoder selection only; execution-profile defaults are checkpoint-aware
workspace state, not a second raw sampler/scheduler control surface in the shared header. Native SDXL SUPIR mode now also exposes a shared-header toggle here,
with readiness/blocking resolved from the same diagnostics owner used by the body surface. Outside `/models/:tabId`, model-asset selectors stay summary-only/read-only
and redirect the user back to a real model-tab owner instead of mutating misleading global checkpoint/VAE/text-encoder state. The non-WAN VAE selector
also keeps Anima choices scoped to the truthful WanVAE-compatible root instead of laundering unrelated family VAEs into the Anima lane. Z-Image L2P quicksettings
expose only the denoiser checkpoint and shared `zimage/<path>` Qwen3-4B text-encoder selector; no VAE selector is rendered for that no-VAE engine.
Qwen Image exposes its exact Edit-2511 IMG2IMG mode as locked-on, does not render a TXT2IMG or INPAINT toggle, and seeds its required
external VAE fallback into the family-scoped owner before generation readiness is evaluated.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsBar` (component): Main QuickSettings SFC; includes “advanced” UI, per-family subcomponents, and selector filtering logic.
- `syncAdvancedTargetHeight` (function): Measures/synchronizes the advanced-row `--qs-advanced-target-height` CSS variable for class-driven collapse transitions.
- `toggleAdvancedRow` (function): Toggles the advanced row persisted UI state.
- `currentTab` (function): Determines the current tab kind (`txt2img`/`img2img`/`txt2vid`/`img2vid`) from routing/state.
- `tabFamilyFromStorage` (function): Loads persisted per-tab family from local storage (used to keep UI consistent on reload).
- `resolvedRouteTabFamily` (computed): Resolves the route-tab family from hydrated tab state or persisted tab refs.
- `modelAssetSelectorsReadOnly` (computed): Marks checkpoint/VAE/text-encoder selectors as summary-only outside `/models/:tabId`.
- `modelAssetOwnerRoute` (computed): Resolves the navigation target for reopening the real model-tab owner from read-only routes.
- `routeTabHydrating` (computed): Tracks whether `/models/:tabId` is still waiting on the hydrated tab object and must not render any family branch yet.
- `routeTabLoadFailed` (computed): Tracks whether `/models/:tabId` failed to load tab state and must show an explicit load-failure placeholder.
- `routeTabMissing` (computed): Tracks whether `/models/:tabId` finished syncing without a matching tab and must show an explicit not-found placeholder.
- `normalizePath` (function): Normalizes paths for stable comparisons (slash/case handling).
- `MetadataKind` (type): Discriminant for inline metadata popups (checkpoint/TE/VAE/WAN stage).
- `isRecordObject` (function): Type guard for plain object payloads used by metadata parsers.
- `parseMetadataKind` (function): Narrows a raw metadata kind string to supported `MetadataKind` values.
- `parseMetadataPayload` (function): Parses/validates dynamic metadata event payload into `{ kind, value }`.
- `extractSizeBytes` (function): Reads validated file size bytes from `/models/file-metadata` summary payload.
- `onShowMetadata` (function): Resolves selection metadata and opens a modal.
- `fileInPaths` (function): Checks whether a file path belongs to the configured roots for a key from `/api/paths` (drives selector filtering).
- `isQuicksettingsReady` (ref): Becomes true only after component-local inventory/paths initialization completes; gates mount-time VAE canonicalization.
- `normalizeTextEncoderLabels` (function): Normalizes raw TE values into a stable label list (used for Flux/WAN multi-TE cases).
- `WanAssetsParams` (type): Minimal WAN assets triple used for payload building (metadata dir + TE + VAE).
- `currentWanAssetsFor` (function): Builds `WanAssetsParams` from the active exact WAN tab selections (used by WAN payload generation).
- `flux2TextEncoderFieldLabel` / `qwenImageTextEncoderFieldLabel` / `zimageL2PTextEncoderFieldLabel` (computed): Resolves truthful Qwen-backed text-encoder selector labels from backend asset contracts.
- `toastModelAssetOwnerRequired` (function): Explains that model-asset selectors are read-only outside `/models/:tabId`.
- `onPrimaryTextEncoderChange` (function): Applies primary text-encoder selection changes (and triggers dependent updates).
- `onSecondaryTextEncoderChange` (function): Applies secondary text-encoder selection changes (FLUX.1 dual-encoder workflow only).
- `onSmartOffloadChange` (function): Updates Smart Offload toggle (impacts per-request memory behavior).
- `onSmartFallbackChange` (function): Updates Smart Fallback toggle (best-effort OOM fallback behavior).
- `onSmartCacheChange` (function): Updates Smart Cache toggle (conditioning caching behavior).
- `onCoreStreamingChange` (function): Updates core streaming toggle (runtime streaming behavior).
- `isObliteratingVram` (ref): Tracks in-flight `/api/obliterate-vram` requests to prevent repeated fire.
- `onObliterateVram` (function): Triggers safe VRAM cleanup and surfaces fail-loud status in quicksettings toasts/logs.
- `resolveWan14bFlowShift` (function): Resolves automatic WAN 14B stage `flowShift` policy for the selected input mode + LightX2V toggle.
- `patchWanStageFlowShift` (function): Applies/removes managed WAN stage `flowShift` values without clobbering unrelated manual overrides.
- `finiteStageFlowShift` (function): Normalizes a stage `flowShift` into a finite number or `undefined` for stable policy comparisons.
- `ensureWan14bFlowShiftPolicy` (function): Enforces managed WAN 14B `flowShift` policy on the active tab (including initial load) without update loops.
- `onWanInputModeChange` (function): Updates the exact WAN input mode selection (`TXT2VID|IMG2VID`) and derived controls.
- `onWanBrowseModels` (function): Opens the shared add-path modal for WAN model roots (`wan22_ckpt`) from the WAN quicksettings `+` action.
- `activeLtxMode` (computed): Resolves the authoritative LTX `txt2vid|img2vid` mode from the active tab params.
- `activeLtxRouteTabId` (computed): Resolves the route-scoped LTX tab id even before full tab hydration completes.
- `isActiveLtxTabRunning` (computed): Tracks whether the route-scoped LTX tab currently has an in-flight generation task.
- `ltxRouteHydrating` (computed): Tracks whether the LTX quicksettings row is waiting for route-tab hydration.
- `ltxQuicksettingsDisabled` (computed): Freezes the LTX quicksettings row during active runs and hydration gaps.
- `ltxModeToggleTitle` (computed): Tooltip reason for the LTX mode toggle enabled/disabled state.
- `ltxRefreshTitle` (computed): Tooltip reason for the LTX Refresh button enabled/disabled state.
- `onLtxModeChange` (function): Toggles the active LTX tab between `txt2vid` and `img2vid` from quick settings.
- `onUseInitImageChange` (function): Toggles active image-tab mode between txt2img and img2img from quick settings.
- `canShowModeToggles` (computed): Enables shared IMG2IMG/INPAINT quicksettings controls when the active image tab supports img2img; Qwen renders its locked Edit-only mode separately.
- `useInitImage` / `useMask` / `hasInitImage` / `initSourceIsImg` (computed): Shared-header mode/source/materialized-image state for the active image tab.
- `activeImageRequestEngineId` (computed): Resolves the active image tab to the exact backend request id used for capability and asset-contract lookups.
- `supirEnabled` / `canShowSupirToggle` / `supirSelectionState` (computed): Shared-header SUPIR toggle state, discoverability, and blocking contract for SDXL img2img/inpaint.
- `supportsInpaint` (computed): Flags whether the active image-tab semantic capability truthfully supports mask/inpaint semantics.
- `isActiveImageTabRunning` (computed): Tracks whether the active image tab currently has an in-flight generation task.
- `inpaintToggleDisabled` (computed): Disables INPAINT when the current state cannot be changed safely from quick settings.
- `inpaintToggleTitle` (computed): Tooltip reason for INPAINT enabled/disabled state.
- `onUseMaskChange` (function): Toggles inpaint mode (`useMask`) from quick settings with shared-engine support guards.
- `onSupirModeChange` (function): Toggles native SDXL SUPIR mode from quick settings and forces img2img entry when enabling.
- `zimageTurbo` (computed): Returns the current Z-Image Turbo toggle state for the active tab.
- `zimageTurboLocked` (ref): When true, the Z-Image Turbo toggle is fixed by trusted checkpoint metadata.
- `_trustedZImageVariantFromCheckpointMeta` (function): Extracts `codex.zimage.variant` when metadata is trusted (Codex provenance).
- `onZImageTurboChange` (function): Applies Turbo toggle updates to the active Z-Image tab (with default migration).
- `enginePrefixForFamily` (function): Maps a `TabFamily` to the engine prefix used in options/labels.
- `openAddPathModal` (function): Opens the reusable add-path modal for checkpoint/VAE/text-encoder library keys.
- `onAddPathModalAdded` (function): Refreshes quicksettings lists after add-path operations mutate library paths.
- `onAddPathModalError` (function): Surfaces add-path scan/add failures through quicksettings toasts.
- `applyInventorySnapshot` (function): Applies one inventory payload to local quicksettings selector sources.
- `refreshAll` (function): Refreshes models/paths/inventory and reloads shared SUPIR diagnostics when that surface is available.
- `openPathInputModal` (function): Opens the in-app path input modal and registers async apply behavior.
- `confirmPathInputModal` (function): Validates/applies modal-entered path values.
- `closePathInputModal` (function): Closes and clears the in-app path input modal state.
- `openOverrides` (function): Opens the overrides UI surface (advanced controls entrypoint).
-->

<template>
  <section :class="['quicksettings', { 'quicksettings-loading': isLoadingQuicksettings }]">
    <div class="quicksettings-row">
      <div class="quicksettings-group qs-group-advanced-toggle">
        <div class="qs-row">
          <button
            class="btn qs-btn-outline qs-advanced-handle"
            type="button"
            :aria-expanded="advancedOpen ? 'true' : 'false'"
            :aria-label="advancedOpen ? 'Collapse options' : 'Expand options'"
            :title="advancedOpen ? 'Collapse options' : 'Expand options'"
            @click="toggleAdvancedRow"
          >
            <svg v-if="!advancedOpen" class="qs-advanced-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M6 10L12 16L18 10"
                fill="none"
                stroke="currentColor"
                stroke-width="2.5"
                stroke-linecap="round"
                stroke-linejoin="round"
                transform="rotate(-90 12 12)"
              />
            </svg>
            <svg v-else class="qs-advanced-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 10L12 16L18 10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
          </button>
        </div>
      </div>
      <template v-if="routeTabHydrating">
        <div class="quicksettings-group">
          <label class="label-muted">Model Tab</label>
          <div class="qs-row">
            <span class="caption">Loading tab settings...</span>
          </div>
        </div>
      </template>
      <template v-else-if="routeTabLoadFailed">
        <div class="quicksettings-group">
          <label class="label-muted">Model Tab</label>
          <div class="qs-row">
            <span class="caption" :title="routeTabSyncError">Failed to load tab settings.</span>
          </div>
        </div>
      </template>
      <template v-else-if="routeTabMissing">
        <div class="quicksettings-group">
          <label class="label-muted">Model Tab</label>
          <div class="qs-row">
            <span class="caption">Tab não encontrada.</span>
          </div>
        </div>
      </template>
      <!-- WAN-specific quicksettings -->
      <template v-else-if="activeFamily === 'wan22_14b'">
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint, VAE, and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <QuickSettingsWan
            :mode="wanInputMode"
            :lightx2v="wanLightx2v"
            :high-model="wanHighModel"
            :high-choices="wanHighDirChoices"
            :low-model="wanLowModel"
            :low-choices="wanLowDirChoices"
            :text-encoder="wanTextEncoder"
            :text-encoder-choices="wanTextEncoderChoices"
            :vae="wanVae"
            :vae-choices="wanVaeChoices"
            @update:mode="onWanInputModeChange"
            @update:lightx2v="onWanLightx2vChange"
            @update:highModel="onWanHighModelChange"
            @update:lowModel="onWanLowModelChange"
            @update:textEncoder="onWanTextEncoderChange"
            @update:vae="onWanVaeChange"
            @browseModels="onWanBrowseModels"
            @browseTe="onWanBrowseTe"
            @browseVae="onWanBrowseVae"
            @refresh="refreshAll"
            @showMetadata="onShowMetadata"
          />
        </fieldset>
      </template>
      <template v-else-if="activeFamily === 'wan22_5b'">
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint, VAE, and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <QuickSettingsWan22_5b
            :mode="wanInputMode"
            :model="wan5bModel"
            :model-choices="wan5bModelChoices"
            :text-encoder="wanTextEncoder"
            :text-encoder-choices="wanTextEncoderChoices"
            :vae="wanVae"
            :vae-choices="wanVaeChoices"
            @update:mode="onWanInputModeChange"
            @update:model="onWan5bModelChange"
            @update:textEncoder="onWanTextEncoderChange"
            @update:vae="onWanVaeChange"
            @browseModels="onWanBrowseModels"
            @browseTe="onWanBrowseTe"
            @browseVae="onWanBrowseVae"
            @refresh="refreshAll"
            @showMetadata="onShowMetadata"
          />
        </fieldset>
      </template>

      <!-- FLUX-family-specific quicksettings -->
      <template v-else-if="activeFamily === 'flux1' || activeFamily === 'flux2'">
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint, VAE, and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <QuickSettingsAssetBlock
            v-if="activeFamily === 'flux1'"
            :checkpoint="effectiveCheckpoint"
            :checkpoints="filteredModelTitles"
            checkpoint-choice-mode="truncate"
            :vae="store.currentVae"
            :vae-choices="filteredVaeChoices"
            vae-choice-mode="truncate"
            vae-placeholder-label="Select VAE"
            :text-encoder="flux1TextEncoderPrimary"
            :text-encoder-choices="filteredTextEncoderChoices"
            text-encoder-group-label="Text Encoders"
            text-encoder-group-class="qs-group-flux-tenc"
            text-encoder-automatic-label="Select CLIP"
            text-encoder-metadata-kind="text_encoder_primary"
            show-text-encoder-actions
            :secondary-text-encoder="flux1TextEncoderSecondary"
            :secondary-text-encoder-choices="filteredTextEncoderChoices"
            secondary-text-encoder-automatic-label="Select T5"
            secondary-text-encoder-metadata-kind="text_encoder_secondary"
            show-secondary-text-encoder
            show-secondary-text-encoder-actions
            @update:checkpoint="onModelChange"
            @update:vae="onVaeChange"
            @update:textEncoder="onPrimaryTextEncoderChange"
            @update:secondaryTextEncoder="onSecondaryTextEncoderChange"
            @addCheckpointPath="onAddCheckpointPath"
            @addVaePath="onAddVaePath"
            @addTencPath="onAddTencPath"
            @showMetadata="onShowMetadata"
          />
          <QuickSettingsAssetBlock
            v-else
            :checkpoint="effectiveCheckpoint"
            :checkpoints="filteredModelTitles"
            checkpoint-choice-mode="truncate"
            :vae="store.currentVae"
            :vae-choices="filteredVaeChoices"
            vae-choice-mode="truncate"
            vae-placeholder-label="Select VAE"
            :text-encoder="flux2TextEncoder"
            :text-encoder-choices="filteredTextEncoderChoices"
            :text-encoder-group-label="flux2TextEncoderFieldLabel"
            text-encoder-group-class="qs-group-flux-tenc"
            text-encoder-automatic-label="Select Qwen3-4B"
            show-text-encoder-actions
            @update:checkpoint="onModelChange"
            @update:vae="onVaeChange"
            @update:textEncoder="onPrimaryTextEncoderChange"
            @addCheckpointPath="onAddCheckpointPath"
            @addVaePath="onAddVaePath"
            @addTencPath="onAddTencPath"
            @showMetadata="onShowMetadata"
          />
          <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle">
            <label class="label-muted">Mode</label>
            <div class="qs-row">
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="useInitImage"
                @click="onUseInitImageChange(!useInitImage)"
              >
                IMG2IMG
              </button>
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="useMask"
                :disabled="inpaintToggleDisabled"
                :title="inpaintToggleTitle"
                @click="onUseMaskChange(!useMask)"
              >
                INPAINT
              </button>
            </div>
          </div>
          <div class="quicksettings-group qs-group-models">
            <label class="label-muted">Models</label>
            <div class="qs-row">
              <button
                class="btn qs-btn-secondary qs-refresh-btn"
                type="button"
                :disabled="isLoadingQuicksettings"
                title="Refresh lists"
                @click="refreshAll"
              >
                Refresh
              </button>
            </div>
          </div>
        </fieldset>
      </template>

      <!-- Z Image-specific quicksettings -->
      <template v-else-if="activeFamily === 'zimage'">
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint, VAE, and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <QuickSettingsAssetBlock
            :checkpoint="effectiveCheckpoint"
            :checkpoints="filteredModelTitles"
            checkpoint-label="Model"
            checkpoint-choice-mode="truncate"
            :vae="store.currentVae"
            :vae-choices="filteredVaeChoices"
            vae-choice-mode="truncate"
            vae-placeholder-label="Select VAE"
            :text-encoder="primaryTextEncoder"
            :text-encoder-choices="filteredTextEncoderChoices"
            text-encoder-group-label="Text Encoder (Qwen3)"
            text-encoder-automatic-label="Select Text Encoder"
            show-text-encoder-actions
            @update:checkpoint="onModelChange"
            @update:vae="onVaeChange"
            @update:textEncoder="onPrimaryTextEncoderChange"
            @addCheckpointPath="onAddCheckpointPath"
            @addVaePath="onAddVaePath"
            @addTencPath="onAddTencPath"
            @showMetadata="onShowMetadata"
          >
            <template #after-checkpoint>
              <div class="quicksettings-group qs-group-zimage-turbo">
                <div class="qs-row">
                  <button
                    :class="['btn', 'qs-toggle-btn', zimageTurbo ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                    type="button"
                    :aria-pressed="zimageTurbo"
                    :disabled="zimageTurboLocked"
                    :title="zimageTurboLocked ? 'Turbo variant is fixed by model metadata' : 'Toggle Turbo variant'"
                    @click="onZImageTurboChange(!zimageTurbo)"
                  >
                    Turbo
                  </button>
                </div>
              </div>
            </template>
          </QuickSettingsAssetBlock>
          <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle qs-group-mode-toggle--end">
            <label class="label-muted">Mode</label>
            <div class="qs-row">
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="useInitImage"
                @click="onUseInitImageChange(!useInitImage)"
              >
                IMG2IMG
              </button>
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="useMask"
                :disabled="inpaintToggleDisabled"
                :title="inpaintToggleTitle"
                @click="onUseMaskChange(!useMask)"
              >
                INPAINT
              </button>
            </div>
          </div>
          <div class="quicksettings-group qs-group-models">
            <label class="label-muted">Models</label>
            <div class="qs-row">
              <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
            </div>
          </div>
        </fieldset>
      </template>

      <!-- Z-Image L2P-specific quicksettings -->
      <template v-else-if="activeFamily === 'zimage_l2p'">
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <QuickSettingsAssetBlock
            :checkpoint="effectiveCheckpoint"
            :checkpoints="filteredModelTitles"
            checkpoint-label="Model"
            checkpoint-choice-mode="truncate"
            vae=""
            :vae-choices="[]"
            :show-vae="false"
            :text-encoder="primaryTextEncoder"
            :text-encoder-choices="filteredTextEncoderChoices"
            :text-encoder-group-label="zimageL2PTextEncoderFieldLabel"
            text-encoder-automatic-label="Select Qwen3-4B"
            show-text-encoder-actions
            @update:checkpoint="onModelChange"
            @update:textEncoder="onPrimaryTextEncoderChange"
            @addCheckpointPath="onAddCheckpointPath"
            @addTencPath="onAddTencPath"
            @showMetadata="onShowMetadata"
          />
          <div class="quicksettings-group qs-group-models">
            <label class="label-muted">Models</label>
            <div class="qs-row">
              <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
            </div>
          </div>
        </fieldset>
      </template>

      <!-- Qwen Image-specific quicksettings -->
      <template v-else-if="activeFamily === 'qwen_image'">
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint, VAE, and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <QuickSettingsAssetBlock
            :checkpoint="effectiveCheckpoint"
            :checkpoints="filteredModelTitles"
            checkpoint-label="Model"
            checkpoint-choice-mode="truncate"
            :vae="qwenImageVae"
            :vae-choices="filteredVaeChoices"
            vae-choice-mode="truncate"
            vae-placeholder-label="Select Qwen Image VAE"
            :text-encoder="primaryTextEncoder"
            :text-encoder-choices="filteredTextEncoderChoices"
            :text-encoder-group-label="qwenImageTextEncoderFieldLabel"
            text-encoder-automatic-label="Select Text Encoder"
            show-text-encoder-actions
            @update:checkpoint="onModelChange"
            @update:vae="onVaeChange"
            @update:textEncoder="onPrimaryTextEncoderChange"
            @addCheckpointPath="onAddCheckpointPath"
            @addVaePath="onAddVaePath"
            @addTencPath="onAddTencPath"
            @showMetadata="onShowMetadata"
          />
          <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle">
            <label class="label-muted">Mode</label>
            <div class="qs-row">
              <button
                class="btn qs-toggle-btn qs-toggle-btn--sm qs-toggle-btn--on"
                type="button"
                aria-pressed="true"
                disabled
                title="Qwen Image Edit-2511 is IMG2IMG-only."
              >
                IMG2IMG
              </button>
            </div>
          </div>
          <div class="quicksettings-group qs-group-models">
            <label class="label-muted">Models</label>
            <div class="qs-row">
              <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
            </div>
          </div>
        </fieldset>
      </template>

      <!-- Chroma-specific quicksettings -->
      <template v-else-if="activeFamily === 'chroma'">
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint, VAE, and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <QuickSettingsAssetBlock
            :checkpoint="effectiveCheckpoint"
            :checkpoints="filteredModelTitles"
            checkpoint-label="Model"
            checkpoint-choice-mode="truncate"
            :vae="store.currentVae"
            :vae-choices="filteredVaeChoices"
            vae-choice-mode="truncate"
            vae-placeholder-label="Select VAE"
            :text-encoder="primaryTextEncoder"
            :text-encoder-choices="filteredTextEncoderChoices"
            text-encoder-group-label="Text Encoder (T5)"
            text-encoder-automatic-label="Select Text Encoder"
            :show-text-encoder="store.isModelCoreOnly(effectiveCheckpoint)"
            show-text-encoder-actions
            @update:checkpoint="onModelChange"
            @update:vae="onVaeChange"
            @update:textEncoder="onPrimaryTextEncoderChange"
            @addCheckpointPath="onAddCheckpointPath"
            @addVaePath="onAddVaePath"
            @addTencPath="onAddTencPath"
            @showMetadata="onShowMetadata"
          />
          <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle">
            <label class="label-muted">Mode</label>
            <div class="qs-row">
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="useInitImage"
                @click="onUseInitImageChange(!useInitImage)"
              >
                IMG2IMG
              </button>
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="useMask"
                :disabled="inpaintToggleDisabled"
                :title="inpaintToggleTitle"
                @click="onUseMaskChange(!useMask)"
              >
                INPAINT
              </button>
            </div>
          </div>
          <div class="quicksettings-group qs-group-models">
            <label class="label-muted">Models</label>
            <div class="qs-row">
              <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
            </div>
          </div>
        </fieldset>
      </template>

      <!-- LTX quicksettings -->
      <template v-else-if="activeFamily === 'ltx2'">
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint, VAE, and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <div class="quicksettings-group qs-group-mode-toggle">
            <label class="label-muted">Mode</label>
            <div class="qs-row">
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', activeLtxMode === 'txt2vid' ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :disabled="ltxQuicksettingsDisabled"
                :title="ltxModeToggleTitle"
                :aria-pressed="activeLtxMode === 'txt2vid'"
                @click="onLtxModeChange('txt2vid')"
              >
                TXT2VID
              </button>
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', activeLtxMode === 'img2vid' ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :disabled="ltxQuicksettingsDisabled"
                :title="ltxModeToggleTitle"
                :aria-pressed="activeLtxMode === 'img2vid'"
                @click="onLtxModeChange('img2vid')"
              >
                IMG2VID
              </button>
            </div>
          </div>
          <QuickSettingsAssetBlock
            :checkpoint="effectiveCheckpoint"
            :checkpoints="filteredModelTitles"
            :vae="effectiveVae"
            :vae-choices="filteredVaeChoices"
            :text-encoder="primaryTextEncoder"
            :text-encoder-choices="filteredTextEncoderChoices"
            text-encoder-automatic-label="Select text encoder"
            :show-text-encoder="true"
            :show-text-encoder-actions="true"
            :disabled="ltxQuicksettingsDisabled"
            @update:checkpoint="onModelChange"
            @update:vae="onVaeChange"
            @update:textEncoder="onPrimaryTextEncoderChange"
            @addCheckpointPath="onAddCheckpointPath"
            @addVaePath="onAddVaePath"
            @addTencPath="onAddTencPath"
            @showMetadata="onShowMetadata"
          />
          <div v-if="ltxRouteHydrating" class="caption">Loading LTX tab settings...</div>
          <div class="quicksettings-group qs-group-models">
            <label class="label-muted">Models</label>
            <div class="qs-row">
              <button
                class="btn qs-btn-secondary qs-refresh-btn"
                type="button"
                :disabled="isLoadingQuicksettings || ltxQuicksettingsDisabled"
                :title="ltxRefreshTitle"
                @click="refreshAll"
              >
                Refresh
              </button>
            </div>
          </div>
        </fieldset>
      </template>

      <!-- Default (SD15/SDXL) quicksettings -->
      <template v-else>
        <div v-if="modelAssetSelectorsReadOnly" class="quicksettings-group qs-group-owner-note">
          <label class="label-muted">Model Tab Owner</label>
          <div class="qs-row qs-row-wrap">
            <span class="caption">Checkpoint, VAE, and Text Encoder are read-only here.</span>
            <RouterLink class="btn qs-btn-outline qs-inline-btn" :to="modelAssetOwnerRoute">Open model tab</RouterLink>
          </div>
        </div>
        <fieldset class="qs-readonly-fieldset" :disabled="modelAssetSelectorsReadOnly">
          <QuickSettingsAssetBlock
            :checkpoint="effectiveCheckpoint"
            :checkpoints="filteredModelTitles"
            :vae="store.currentVae"
            :vae-choices="filteredVaeChoices"
            :text-encoder="primaryTextEncoder"
            :text-encoder-choices="filteredTextEncoderChoices"
            text-encoder-automatic-label="Built-in"
            :show-text-encoder="activeFamily !== 'sd15' && activeFamily !== 'sdxl'"
            @update:checkpoint="onModelChange"
            @update:vae="onVaeChange"
            @update:textEncoder="onPrimaryTextEncoderChange"
            @addCheckpointPath="onAddCheckpointPath"
            @addVaePath="onAddVaePath"
            @showMetadata="onShowMetadata"
          />
          <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle">
            <label class="label-muted">Mode</label>
            <div class="qs-row">
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="useInitImage"
                @click="onUseInitImageChange(!useInitImage)"
              >
                IMG2IMG
              </button>
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="useMask"
                :disabled="inpaintToggleDisabled"
                :title="inpaintToggleTitle"
                @click="onUseMaskChange(!useMask)"
              >
                INPAINT
              </button>
              <button
                v-if="canShowSupirToggle"
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', supirEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="supirEnabled"
                :disabled="supirToggleDisabled"
                :title="supirToggleTitle"
                @click="onSupirModeChange(!supirEnabled)"
              >
                SUPIR
              </button>
            </div>
          </div>
          <div class="quicksettings-group qs-group-models">
            <label class="label-muted">Models</label>
            <div class="qs-row">
              <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
            </div>
          </div>
        </fieldset>
      </template>
    </div>

    <div v-if="qsNotice" class="caption">{{ qsNotice }}</div>

    <div
      ref="advancedRowEl"
      class="quicksettings-advanced-collapse"
      :data-ready="advancedRowReady ? 'true' : 'false'"
      :data-state="advancedOpen ? 'open' : 'closed'"
    >
      <div ref="advancedRowInnerEl" class="quicksettings-row quicksettings-row--advanced-inner">
        <QuickSettingsPerf
          :smart-offload="store.smartOffload"
          :smart-fallback="store.smartFallback"
          :smart-cache="store.smartCache"
          :core-streaming="store.coreStreaming"
          :obliterate-busy="isObliteratingVram"
          @update:smartOffload="onSmartOffloadChange"
          @update:smartFallback="onSmartFallbackChange"
          @update:smartCache="onSmartCacheChange"
          @update:coreStreaming="onCoreStreamingChange"
          @obliterateVram="onObliterateVram"
        />

        <div class="quicksettings-group qs-group-overrides">
          <label class="label-muted">Overrides</label>
          <div class="qs-row">
            <button class="btn qs-btn-secondary qs-overrides-btn" type="button" @click="openOverrides">
              Set overrides
            </button>
          </div>
        </div>
      </div>
    </div>

    <QuickSettingsOverridesModal v-model="showOverridesModal" />
    <QuickSettingsAddPathModal
      v-model="showAddPathModal"
      :title="addPathModalTitle"
      :label="addPathModalLabel"
      :target-key="addPathModalTargetKey"
      :target-kind="addPathModalTargetKind"
      :placeholder="addPathModalPlaceholder"
      @added="onAddPathModalAdded"
      @error="onAddPathModalError"
    />
    <AssetMetadataModal v-model="showMetadataModal" :title="metadataModalTitle" :subtitle="metadataModalSubtitle" :payload="metadataModalPayload" />
    <Modal v-model="showPathInputModal" :title="pathInputModalTitle">
      <div class="quicksettings-path-modal">
        <label class="label-muted" for="quicksettings-path-input">{{ pathInputModalLabel }}</label>
        <input
          id="quicksettings-path-input"
          ref="pathInputEl"
          class="ui-input"
          type="text"
          :placeholder="pathInputModalPlaceholder"
          v-model="pathInputModalValue"
          @keydown.enter.prevent="confirmPathInputModal"
        />
      </div>
      <template #footer>
        <button class="btn btn-md btn-outline" type="button" @click="closePathInputModal">Cancel</button>
        <button class="btn btn-md btn-secondary" type="button" @click="confirmPathInputModal">Apply</button>
      </template>
    </Modal>
  </section>
</template>


<script setup lang="ts">
import { onBeforeUnmount, onMounted, computed, nextTick, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useUiPresetsStore } from '../stores/ui_presets'
import { useUiBlocksStore } from '../stores/ui_blocks'
import { MODEL_TABS_STORAGE_KEY, useModelTabsStore, type ImageBaseParams, type LtxGenerationMode, type TabByType, type Wan5bStageParams, type WanAssetsParams, type WanStageParams, type WanTabType } from '../stores/model_tabs'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import {
  fetchCheckpointMetadata,
  fetchFileMetadata,
  fetchFreshPaths,
  fetchObliterateVram,
} from '../api/client'
import type { InventoryResponse, ModelInfo } from '../api/types'
import { isGenerationRunningForTab } from '../composables/useGeneration'
import { isLtxGenerationRunningForTab } from '../composables/useLtxVideoGeneration'
import { useResultsCard } from '../composables/useResultsCard'
import { useSupirDiagnostics, resolveSupirSelectionState } from '../composables/useSupirDiagnostics'
import {
  isWanTabFamily,
  normalizeTabFamily,
  normalizeSemanticEngine,
  resolveImageRequestEngineId,
  tabFamilyFromSemanticEngine,
  type TabFamily,
} from '../utils/engine_taxonomy'
import { buildUseInitImagePatch } from '../utils/image_params'
import { filterModelTitlesForFamily, enginePrefixForFamily } from '../utils/model_family_filters'
import { buildFamilyVaeChoices, canonicalizeVaeChoice, type InventoryVaeChoice } from '../utils/vae_choices'
import QuickSettingsAssetBlock from './quicksettings/QuickSettingsAssetBlock.vue'
import QuickSettingsPerf from './quicksettings/QuickSettingsPerf.vue'
import QuickSettingsWan from './quicksettings/QuickSettingsWan.vue'
import QuickSettingsWan22_5b from './quicksettings/QuickSettingsWan22_5b.vue'
import QuickSettingsOverridesModal from './modals/QuickSettingsOverridesModal.vue'
import QuickSettingsAddPathModal from './modals/QuickSettingsAddPathModal.vue'
import AssetMetadataModal from './modals/AssetMetadataModal.vue'
import Modal from './ui/Modal.vue'

const store = useQuicksettingsStore()
const presets = useUiPresetsStore()
const route = useRoute()
const uiBlocks = useUiBlocksStore()
const tabsStore = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()
type WanInventoryVariant = 'wan22_5b' | 'wan22_14b' | 'wan22_14b_animate'
type InventoryWanGguf = { name: string; path: string; sha256?: string; stage: string; variant?: WanInventoryVariant; repoHint?: string }
type InventoryTextEncoder = { name: string; path: string; sha256?: string }
type ImageTab = TabByType<'sd15' | 'sdxl' | 'flux1' | 'flux2' | 'qwen_image' | 'zimage' | 'zimage_l2p' | 'chroma' | 'anima'>
type LtxTab = TabByType<'ltx2'>
type Wan14bTab = TabByType<'wan22_14b'>
type Wan5bTab = TabByType<'wan22_5b'>
type AddPathTargetKind = 'checkpoint' | 'vae' | 'text_encoder'
const inventoryWan = ref<InventoryWanGguf[]>([])
const inventoryTextEncoders = ref<InventoryTextEncoder[]>([])
const showOverridesModal = ref(false)
const showMetadataModal = ref(false)
const showAddPathModal = ref(false)
const showPathInputModal = ref(false)
const metadataModalTitle = ref('Metadata')
const metadataModalSubtitle = ref('')
const metadataModalPayload = ref<unknown>(null)
const addPathModalTitle = ref('Add Model Path')
const addPathModalLabel = ref('Path')
const addPathModalTargetKey = ref('')
const addPathModalTargetKind = ref<AddPathTargetKind>('checkpoint')
const addPathModalPlaceholder = ref('')
const pathInputModalTitle = ref('Update Path')
const pathInputModalLabel = ref('Path')
const pathInputModalPlaceholder = ref('')
const pathInputModalValue = ref('')
const pathInputEl = ref<HTMLInputElement | null>(null)
let pathInputApply: ((value: string) => Promise<void>) | null = null
const { notice: qsNotice, toast: qsToast } = useResultsCard({ noticeDurationMs: 4000 })
const {
  ensureSupirDiagnosticsLoaded: ensureSharedSupirDiagnosticsLoaded,
  reloadSupirDiagnostics: reloadSharedSupirDiagnostics,
} = useSupirDiagnostics()
const isLoadingQuicksettings = ref(false)
const isQuicksettingsReady = ref(false)
const isObliteratingVram = ref(false)
const QUICKSETTINGS_ADVANCED_OPEN_STORAGE_KEY = 'codex.quicksettings.advanced_open'
const advancedOpen = ref(true)
const advancedRowEl = ref<HTMLElement | null>(null)
const advancedRowInnerEl = ref<HTMLElement | null>(null)
const advancedRowReady = ref(false)
let advancedRowResizeObserver: ResizeObserver | null = null

try {
  const stored = localStorage.getItem(QUICKSETTINGS_ADVANCED_OPEN_STORAGE_KEY)
  if (stored === '0') advancedOpen.value = false
  if (stored === '1') advancedOpen.value = true
} catch {
  // ignore localStorage failures
}

watch(advancedOpen, (isOpen) => {
  try {
    localStorage.setItem(QUICKSETTINGS_ADVANCED_OPEN_STORAGE_KEY, isOpen ? '1' : '0')
  } catch {
    // ignore localStorage failures
  }
})

function syncAdvancedTargetHeight(): void {
  const el = advancedRowEl.value
  const inner = advancedRowInnerEl.value
  if (!el || !inner) return

  const nextHeight = Math.ceil(inner.getBoundingClientRect().height)
  el.style.setProperty('--qs-advanced-target-height', `${nextHeight}px`)
  if (!advancedRowReady.value) advancedRowReady.value = true
}

function toggleAdvancedRow(): void {
  advancedOpen.value = !advancedOpen.value
}

type UiPresetTab = 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid'

function currentTab(): UiPresetTab | null {
  const modelTab = activeModelTab.value
  if (route.path.startsWith('/models/')) {
    if (!modelTab) return null
    if (modelTab.type === 'wan22_14b') {
      const wanTab = asWan14bTab(modelTab)
      if (!wanTab) return null
      return wanTab.params.video.useInitImage ? 'img2vid' : 'txt2vid'
    }
    if (modelTab.type === 'wan22_5b') {
      const params = modelTab.params as { video?: { useInitImage?: unknown } }
      return params.video?.useInitImage === true ? 'img2vid' : 'txt2vid'
    }
    if (modelTab.type === 'ltx2') {
      const ltxTab = asLtxTab(modelTab)
      if (!ltxTab) return null
      return ltxTab.params.mode
    }
    const imageTab = asImageTab(modelTab)
    if (imageTab) {
      return imageTab.params.useInitImage ? 'img2img' : 'txt2img'
    }
    return null
  }

  const p = route.path
  if (p.startsWith('/img2img')) return 'img2img'
  if (p.startsWith('/txt2vid')) return 'txt2vid'
  if (p.startsWith('/img2vid')) return 'img2vid'
  return 'txt2img'
}

const resolvedPresetTab = computed<UiPresetTab | null>(() => currentTab())

const routeTabId = computed(() => String(route.params.tabId || ''))
const isModelTabRoute = computed(() => route.path.startsWith('/models/') && Boolean(routeTabId.value))
const modelAssetSelectorsReadOnly = computed(() => !isModelTabRoute.value)
function tabUpdatedAtMs(tab: { meta?: { updatedAt?: string } }): number {
  const parsed = Date.parse(String(tab.meta?.updatedAt || ''))
  return Number.isFinite(parsed) ? parsed : 0
}

const modelAssetOwnerTab = computed(() => {
  const current = activeModelTab.value
  if (current?.id && normalizeTabFamily(current.type) === activeFamily.value) {
    return current
  }
  const active = tabsStore.activeTab
  if (active?.id && normalizeTabFamily(active.type) === activeFamily.value) {
    return active
  }
  let latestCompatible: (typeof tabsStore.tabs)[number] | null = null
  for (const tab of tabsStore.tabs) {
    if (normalizeTabFamily(tab.type) !== activeFamily.value) continue
    if (!latestCompatible || tabUpdatedAtMs(tab) > tabUpdatedAtMs(latestCompatible)) {
      latestCompatible = tab
    }
  }
  return latestCompatible
})

const modelAssetOwnerRoute = computed(() => {
  const owner = modelAssetOwnerTab.value
  if (owner?.id) return `/models/${owner.id}`
  return '/models'
})
const activeModelTab = computed(() => {
  if (!isModelTabRoute.value) return null
  const id = routeTabId.value
  if (!id) return null
  const fromList = tabsStore.tabs.find(t => t.id === id) || null
  if (fromList) return fromList
  const active = tabsStore.activeTab
  if (active && active.id === id) return active
  return null
})

function asImageTab(value: unknown): ImageTab | null {
  if (!value || typeof value !== 'object') return null
  const candidate = value as { type?: unknown }
  const type = normalizeTabFamily(candidate.type)
  if (!type || isWanTabFamily(type) || type === 'ltx2') return null
  return value as ImageTab
}

function asWan14bTab(value: unknown): Wan14bTab | null {
  if (!value || typeof value !== 'object') return null
  const candidate = value as { type?: unknown }
  return normalizeTabFamily(candidate.type) === 'wan22_14b' ? (value as Wan14bTab) : null
}

function asWan5bTab(value: unknown): Wan5bTab | null {
  if (!value || typeof value !== 'object') return null
  const candidate = value as { type?: unknown }
  return normalizeTabFamily(candidate.type) === 'wan22_5b' ? (value as Wan5bTab) : null
}

function asLtxTab(value: unknown): LtxTab | null {
  if (!value || typeof value !== 'object') return null
  const candidate = value as { type?: unknown }
  return normalizeTabFamily(candidate.type) === 'ltx2' ? (value as unknown as LtxTab) : null
}

function tabFamilyFromStorage(tabId: string): TabFamily | null {
  if (!tabId) return null
  try {
    const raw = localStorage.getItem(MODEL_TABS_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { tabs?: unknown[] }
    const list = Array.isArray(parsed.tabs) ? parsed.tabs : []
    const match = list.find((entry) => {
      if (!entry || typeof entry !== 'object') return false
      const id = String((entry as { id?: unknown }).id || '')
      return id === tabId
    }) as { type?: unknown } | undefined
    return normalizeTabFamily(match?.type)
  } catch {
    return null
  }
}

const resolvedRouteTabFamily = computed<TabFamily | null>(() => {
  if (!isModelTabRoute.value) return null
  return normalizeTabFamily(activeModelTab.value?.type) || tabFamilyFromStorage(routeTabId.value)
})

const routeTabSyncPending = ref(isModelTabRoute.value)
const routeTabSyncError = ref('')
const routeTabHydrating = computed(() => isModelTabRoute.value && !activeModelTab.value && routeTabSyncPending.value)
const routeTabLoadFailed = computed(() => isModelTabRoute.value && !activeModelTab.value && !routeTabSyncPending.value && routeTabSyncError.value.length > 0)
const routeTabMissing = computed(() => isModelTabRoute.value && !activeModelTab.value && !routeTabSyncPending.value && routeTabSyncError.value.length === 0)

let routeActiveSyncToken = 0
watch(routeTabId, async (tabId) => {
  const token = ++routeActiveSyncToken
  routeTabSyncPending.value = Boolean(tabId)
  routeTabSyncError.value = ''
  if (!tabId) {
    routeTabSyncPending.value = false
    return
  }
  try {
    if (!tabsStore.tabs.length) {
      await tabsStore.load()
    }
    if (token !== routeActiveSyncToken) return
    tabsStore.setActive(tabId)
  } catch (error) {
    routeTabSyncError.value = error instanceof Error ? (error.message || error.name || 'Unknown error') : String(error)
    toastQuicksettingsError(error)
  } finally {
    if (token === routeActiveSyncToken) {
      routeTabSyncPending.value = false
    }
  }
}, { immediate: true })

const activeFamily = computed<TabFamily>(() => {
  if (isModelTabRoute.value) {
    const type = resolvedRouteTabFamily.value
    if (type) return type
  }

  // Fallback when no model tab is active (settings/tools pages etc.).
  if (!engineCaps.loaded) return 'sd15'
  const semantic = normalizeSemanticEngine(uiBlocks.semanticEngine || 'sd15')
  const family = tabFamilyFromSemanticEngine(semantic)
  if (family) return family

  return 'sd15'
})
const semanticEngine = computed<string>(() => {
  // Prefer semantic engine from UI blocks when available (video tabs etc.).
  if (uiBlocks.semanticEngine) return uiBlocks.semanticEngine
  return 'sd15'
})

async function loadInventory(options?: { forceRefresh?: boolean }): Promise<void> {
  const inv = await store.fetchInventoryWithLoraHydration({
    refresh: options?.forceRefresh === true,
  })
  applyInventorySnapshot(inv)
}

function applyInventorySnapshot(inv: InventoryResponse): void {
  store.hydrateVaeInventorySnapshot(inv)
  inventoryWan.value = (inv.wan22?.gguf ?? []).map((g) => ({
    name: String(g.name),
    path: String(g.path),
    sha256: typeof g?.sha256 === 'string' ? String(g.sha256) : undefined,
    stage: String(g.stage || 'unknown'),
    variant: g?.variant,
    repoHint: typeof g?.repo_hint === 'string' ? String(g.repo_hint) : undefined,
  }))
  // Text encoder files are available via inventory for future use (e.g., Flux overrides).
  inventoryTextEncoders.value = inv.text_encoders ?? []
}

async function loadPaths(): Promise<void> {
  const requestEpoch = store.getAssetSnapshotEpoch()
  const res = await fetchFreshPaths()
  if (requestEpoch !== store.getAssetSnapshotEpoch()) return
  store.hydratePathsSnapshot((res.paths || {}) as Record<string, string[]>)
}

function normalizePath(path: string): string {
  return path.replace(/\\+/g, '/').replace(/\/+$/, '')
}

type MetadataKind =
  | 'checkpoint'
  | 'vae'
  | 'text_encoder'
  | 'text_encoder_primary'
  | 'text_encoder_secondary'
  | 'wan_model'
  | 'wan_high_model'
  | 'wan_low_model'
  | 'wan_text_encoder'
  | 'wan_vae'

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function parseMetadataKind(value: unknown): MetadataKind | null {
  const kind = String(value || '').trim()
  if (
    kind === 'checkpoint'
    || kind === 'vae'
    || kind === 'text_encoder'
    || kind === 'text_encoder_primary'
    || kind === 'text_encoder_secondary'
    || kind === 'wan_model'
    || kind === 'wan_high_model'
    || kind === 'wan_low_model'
    || kind === 'wan_text_encoder'
    || kind === 'wan_vae'
  ) {
    return kind
  }
  return null
}

function parseMetadataPayload(payload: unknown): { kind: string; value: string } | null {
  if (!isRecordObject(payload)) return null
  const kind = String(payload.kind ?? '').trim()
  if (!kind) return null
  const value = String(payload.value ?? '').trim()
  if (!value) return null
  return { kind, value }
}

function extractSizeBytes(summary: Record<string, unknown>): number | null {
  const summaryFile = summary.file
  if (!isRecordObject(summaryFile)) return null
  const size = summaryFile.size_bytes
  if (typeof size !== 'number' || !Number.isFinite(size) || size < 0) return null
  return size
}

function isSha256(value: string): boolean {
  const lower = value.toLowerCase().trim()
  return lower.length === 64 && /^[0-9a-f]+$/.test(lower)
}

function stripFamilyPrefix(label: string): string {
  const norm = label.replace(/\\+/g, '/').trim()
  const idx = norm.indexOf('/')
  if (idx <= 0) return norm
  const prefix = norm.slice(0, idx)
  const rest = norm.slice(idx + 1)
  if (!rest) return norm
  if (['sd15', 'sdxl', 'flux1', 'flux2', 'chroma', 'qwen_image', 'wan22', 'zimage'].includes(prefix)) return rest
  return norm
}

function findModelByTitle(title: string): ModelInfo | undefined {
  const raw = String(title || '').trim()
  if (!raw) return undefined
  for (const m of store.models) {
    if (!m) continue
    if (m.title === raw) return m
  }
  return undefined
}

function findVaeRecord(label: string): InventoryVaeChoice | undefined {
  const raw = String(label || '').trim()
  if (!raw) return undefined
  const norm = normalizePath(raw)
  const tail = norm.split('/').pop() || raw
  return store.inventoryVaesSnapshot.find((v) => {
    if (!v) return false
    if (v.name === raw) return true
    const vPath = normalizePath(String(v.path || ''))
    return vPath === norm || (tail ? vPath.endsWith('/' + tail) : false)
  })
}

function findTextEncoderRecord(label: string): InventoryTextEncoder | undefined {
  const raw = String(label || '').trim()
  if (!raw) return undefined
  const unprefixed = stripFamilyPrefix(raw)
  const candidates = [raw, unprefixed]
  for (const cand of candidates) {
    const norm = normalizePath(cand)
    const tail = norm.split('/').pop() || cand
    const found = inventoryTextEncoders.value.find((te) => {
      if (!te) return false
      if (te.name === cand) return true
      const tePath = normalizePath(String(te.path || ''))
      return tePath === norm || (tail ? tePath.endsWith('/' + tail) : false)
    })
    if (found) return found
  }
  return undefined
}

function findWanGgufRecord(
  label: string,
  options: { stage?: 'high' | 'low'; variant?: WanInventoryVariant } = {},
): InventoryWanGguf | undefined {
  const raw = String(label || '').trim()
  if (!raw) return undefined
  const norm = normalizePath(raw)
  const tail = norm.split('/').pop() || raw
  return inventoryWan.value.find((w) => {
    if (!w) return false
    if (options.stage && String(w.stage || '') !== options.stage) return false
    if (options.variant && w.variant !== options.variant) return false
    if (w.name === raw) return true
    const wPath = normalizePath(String(w.path || ''))
    return wPath === norm || (tail ? wPath.endsWith('/' + tail) : false)
  })
}

function onShowMetadata(payload: unknown): void {
  const parsed = parseMetadataPayload(payload)
  if (!parsed) return
  const { value } = parsed
  const kind = parseMetadataKind(parsed.kind)

  let title = 'Metadata'
  let subtitle = ''
  let out: Record<string, unknown> = {}
  let filePathForMetadata: string | null = null

	  if (kind === 'checkpoint') {
	    title = 'Checkpoint metadata'
	    subtitle = value
	    out = { selection: value, metadata: { status: 'loading' } }
	  } else if (kind === 'vae' || kind === 'wan_vae') {
	    const rec = findVaeRecord(value)
	    const sha = store.resolveVaeSha(value) || (rec?.sha256 ? String(rec.sha256) : undefined)
	    title = kind === 'wan_vae' ? 'WAN VAE metadata' : 'VAE metadata'
	    subtitle = rec?.name ? String(rec.name) : value
    filePathForMetadata = rec?.path ? String(rec.path) : null
    out = {
      selection: value,
      sha256: sha,
      inventory: rec
        ? {
            name: rec.name,
            path: rec.path,
            sha256: rec.sha256,
            format: rec.format,
            latent_channels: rec.latent_channels ?? null,
            scaling_factor: rec.scaling_factor ?? null,
          }
        : null,
    }
  } else if (kind === 'text_encoder' || kind === 'text_encoder_primary' || kind === 'text_encoder_secondary' || kind === 'wan_text_encoder') {
    const rec = findTextEncoderRecord(value)
    const sha = store.resolveTextEncoderSha(value) || (rec?.sha256 ? String(rec.sha256) : undefined)
    if (kind === 'text_encoder_primary') title = 'Text encoder metadata (CLIP)'
    else if (kind === 'text_encoder_secondary') title = 'Text encoder metadata (T5)'
    else if (kind === 'wan_text_encoder') title = 'WAN text encoder metadata'
    else title = 'Text encoder metadata'
    subtitle = rec?.name ? String(rec.name) : value
    filePathForMetadata = rec?.path ? String(rec.path) : null
    out = {
      selection: value,
      sha256: sha || (isSha256(value) ? value.toLowerCase() : undefined),
      inventory: rec ? { name: rec.name, path: rec.path, sha256: rec.sha256 } : null,
    }
  } else if (kind === 'wan_model' || kind === 'wan_high_model' || kind === 'wan_low_model') {
    const stage = kind === 'wan_high_model' ? 'high' : kind === 'wan_low_model' ? 'low' : null
    const rec = findWanGgufRecord(value, stage ? { stage } : { variant: 'wan22_5b' })
    const sha = store.resolveWanGgufSha(value) || (rec?.sha256 ? String(rec.sha256) : undefined)
    title = stage === 'high'
      ? 'WAN high model metadata'
      : stage === 'low'
        ? 'WAN low model metadata'
        : 'WAN model metadata'
    subtitle = rec?.name ? String(rec.name) : value
    filePathForMetadata = rec?.path ? String(rec.path) : null
    out = {
      selection: value,
      stage,
      variant: rec?.variant,
      repo_hint: rec?.repoHint,
      sha256: sha || (isSha256(value) ? value.toLowerCase() : undefined),
      inventory: rec ? { name: rec.name, path: rec.path, sha256: rec.sha256, stage: rec.stage, variant: rec.variant, repoHint: rec.repoHint } : null,
    }
  } else {
    title = 'Metadata'
    subtitle = value
    out = { selection: value, kind: parsed.kind }
  }

  if (filePathForMetadata) {
    out = { ...out, metadata: { status: 'loading' } }
  }

	  metadataModalTitle.value = title
	  metadataModalSubtitle.value = subtitle
	  metadataModalPayload.value = out
	  showMetadataModal.value = true

	  if (kind === 'checkpoint') {
	    void (async () => {
	      try {
	        const payload = await fetchCheckpointMetadata(value)
	        metadataModalPayload.value = payload
	      } catch (error: unknown) {
	        const message = error instanceof Error ? error.message : String(error)
	        metadataModalPayload.value = {
	          selection: value,
	          metadata: { status: 'error', error: message },
	        }
	      }
	    })()
	    return
	  }

	  if (!filePathForMetadata) return

	  void (async () => {
	    try {
	      const res = await fetchFileMetadata(filePathForMetadata)
	      const current = metadataModalPayload.value
	      if (!isRecordObject(current)) return
	      const flat = res.flat
	      const nested = res.nested
	      const sizeBytes = extractSizeBytes(res.summary)

	      const filePatch: Record<string, unknown> = {}
	      if (sizeBytes !== null) {
	        const mb = sizeBytes / 1_000_000
	        const gb = sizeBytes / 1_000_000_000
	        filePatch['file.size.bytes'] = sizeBytes
	        filePatch['file.size.megabytes'] = Number(mb.toFixed(3))
	        filePatch['file.size.gigabytes'] = Number(gb.toFixed(3))
	      }

	      const metaOut: Record<string, unknown> = {
	        raw: isRecordObject(flat) ? flat : (res as unknown as Record<string, unknown>),
	        nested,
	      }
	      metadataModalPayload.value = {
	        ...current,
	        ...filePatch,
	        metadata: metaOut,
	      }
	    } catch (error: unknown) {
      const current = metadataModalPayload.value
      if (!isRecordObject(current)) return
      const message = error instanceof Error ? error.message : String(error)
      metadataModalPayload.value = {
        ...current,
        metadata: { status: 'error', error: message },
      }
    }
  })()
}

function fileInPaths(file: string, key: string): boolean {
  if (!file) return false
  const roots = store.pathsConfigSnapshot[key] || []
  if (!roots.length) return false
  const fNorm = normalizePath(file)
  for (const root of roots) {
    const rNorm = normalizePath(root)
    if (!rNorm) continue
    // Absolute root: direct prefix match.
    if (fNorm === rNorm || fNorm.startsWith(rNorm + '/')) return true
    // Repo-relative root (e.g. 'models/*-tenc'): match by suffix segment.
    const rel = rNorm.startsWith('/') ? rNorm.slice(1) : rNorm
    if (fNorm.includes('/' + rel + '/') || fNorm.endsWith('/' + rel)) return true
  }
  return false
}

const filteredModelTitles = computed(() => filterModelTitlesForFamily(store.models, activeFamily.value, store.pathsConfigSnapshot))

const filteredVaeChoices = computed(() => {
  return buildFamilyVaeChoices(
    activeFamily.value,
    store.inventoryVaesSnapshot,
    store.vaeChoices.length ? store.vaeChoices : ['built-in'],
    store.pathsConfigSnapshot,
  )
})

watch(
  () => [activeFamily.value, store.currentVae, filteredVaeChoices.value, isQuicksettingsReady.value] as const,
  ([family, currentVae, choices, quicksettingsReady]) => {
    if (!route.path.startsWith('/models/')) return
    if (!activeModelTab.value) return
    if (isWanTabFamily(family) || family === 'ltx2' || family === 'zimage_l2p') return
    if (!quicksettingsReady) return
    const persistedFamilyVae = store.getPersistedVaeForFamily(family)
    const sourceVae = String(persistedFamilyVae || '')
    const canonical = canonicalizeVaeChoice(sourceVae, choices, store.resolveVaeSha)
    if (!canonical) return
    const nextVae = canonical.value
    if (canonical.reason === 'fallback') {
      if (sourceVae || family === 'qwen_image') {
        store.setVaeForFamily(family, nextVae, { persist: false }).catch((error: unknown) => {
          toastQuicksettingsError(error)
        })
        return
      }
      if (String(currentVae || '') === nextVae) return
      store.setVae(nextVae).catch((error: unknown) => {
        toastQuicksettingsError(error)
      })
      return
    }
    if (String(persistedFamilyVae || '') === nextVae) {
      if (String(currentVae || '') === nextVae) return
      store.setVae(nextVae).catch((error: unknown) => {
        toastQuicksettingsError(error)
      })
      return
    }
    store.setVaeForFamily(family, nextVae).catch((error: unknown) => {
      toastQuicksettingsError(error)
    })
  },
  { immediate: true },
)

const filteredTextEncoderChoices = computed(() => {
  const fam = activeFamily.value
  if (fam === 'flux1' || fam === 'flux2') {
    const tencKey = fam === 'flux2' ? 'flux2_tenc' : 'flux1_tenc'
    // For Flux-family tabs, derive choices from inventory.text_encoders constrained by family tenc roots.
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, tencKey))
      .map((item) => `${fam}/${item.path}`)
  }
  if (fam === 'chroma') {
    // Chroma uses a single T5 text encoder; roots are shared with Flux.1 (`flux1_tenc`).
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'flux1_tenc'))
      .map((item) => `chroma/${item.path}`)
  }
  if (fam === 'ltx2') {
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'ltx2_tenc'))
      .map((item) => `ltx2/${item.path}`)
  }
  if (fam === 'zimage') {
    // For Z Image, derive choices from inventory.text_encoders constrained by zimage_tenc paths.
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'zimage_tenc'))
      .map((item) => `zimage/${item.path}`)
  }
  if (fam === 'zimage_l2p') {
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'zimage_tenc'))
      .map((item) => `zimage/${item.path}`)
  }
  if (fam === 'qwen_image') {
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'qwen_image_tenc'))
      .map((item) => `qwen_image/${item.path}`)
  }
  const prefix = isWanTabFamily(fam) ? 'wan22/' : `${fam}/`
  return store.textEncoderChoices.filter((name: string) => typeof name === 'string' && typeof prefix === 'string' && name.startsWith(prefix))
})

const activeImageTab = computed(() => {
  if (!isModelTabRoute.value) return null
  return asImageTab(activeModelTab.value)
})
const activeImageRequestEngineId = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return null
  return resolveImageRequestEngineId(tab.type, Boolean(tab.params.useInitImage))
})
const activeLtxTab = computed(() => {
  if (!isModelTabRoute.value) return null
  return asLtxTab(activeModelTab.value)
})
const activeLtxRouteTabId = computed(() => {
  if (!isModelTabRoute.value) return ''
  if (activeLtxTab.value) return activeLtxTab.value.id
  return activeFamily.value === 'ltx2' ? routeTabId.value : ''
})
const ltxRouteHydrating = computed(() => (
  isModelTabRoute.value
  && activeFamily.value === 'ltx2'
  && Boolean(routeTabId.value)
  && !activeLtxTab.value
))
const activeLtxMode = computed<LtxGenerationMode>(() => {
  const tab = activeLtxTab.value
  if (!tab) return 'txt2vid'
  return tab.params.mode
})
const activeImageSurface = computed(() => {
  const engineId = activeImageRequestEngineId.value
  if (!engineId) return null
  return engineCaps.get(engineId)
})
const canToggleInitImage = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  const surface = activeImageSurface.value
  if (!surface) return false
  return Boolean(surface.supports_img2img)
})
const canShowModeToggles = computed(() => canToggleInitImage.value)
const useInitImage = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return Boolean(tab.params.useInitImage)
})
const useMask = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return Boolean(tab.params.useMask)
})
const supirEnabled = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return Boolean(tab.params.supir.enabled)
})
const hasInitImage = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return String(tab.params.initImageData || '').trim().length > 0
})
const initSourceIsImg = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return String(tab.params.initSource.mode || '').trim().toLowerCase() === 'img'
})
const supportsInpaint = computed(() => Boolean(activeImageSurface.value?.supports_img2img_masking))
const canShowSupirToggle = computed(() => (
  canToggleInitImage.value
  && activeFamily.value === 'sdxl'
  && Boolean(activeImageSurface.value?.supports_supir_mode)
))
const guidanceAdvancedEnabled = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return Boolean(tab.params.guidanceAdvanced.enabled)
})
const supirSelectionState = computed(() => resolveSupirSelectionState({
  supported: canShowSupirToggle.value,
  selectedVariant: activeImageTab.value?.params.supir.variant ?? '',
  selectedSampler: activeImageTab.value?.params.supir.sampler ?? '',
  guidanceAdvancedEnabled: guidanceAdvancedEnabled.value,
}))
const isActiveImageTabRunning = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return isGenerationRunningForTab(tab.id)
})
const isActiveLtxTabRunning = computed(() => {
  const tabId = activeLtxRouteTabId.value
  if (!tabId) return false
  return isLtxGenerationRunningForTab(tabId)
})
const ltxQuicksettingsDisabled = computed(() => isActiveLtxTabRunning.value || ltxRouteHydrating.value)
const ltxModeToggleTitle = computed(() => {
  if (ltxRouteHydrating.value) return 'Loading LTX tab settings...'
  if (isActiveLtxTabRunning.value) return 'Cannot change LTX mode while generation is running.'
  return 'Toggle LTX video mode'
})
const ltxRefreshTitle = computed(() => {
  if (isLoadingQuicksettings.value) return 'Refresh already in progress.'
  if (ltxRouteHydrating.value) return 'Loading LTX tab settings...'
  if (isActiveLtxTabRunning.value) return 'Cannot refresh LTX lists while generation is running.'
  return 'Refresh lists'
})
const inpaintToggleDisabled = computed(() => (
  isActiveImageTabRunning.value
  || !useInitImage.value
  || !initSourceIsImg.value
  || !hasInitImage.value
  || (!supportsInpaint.value && !useMask.value)
))
const inpaintToggleTitle = computed(() => {
  if (isActiveImageTabRunning.value) return 'Cannot change INPAINT while generation is running.'
  if (!useInitImage.value) return 'Enable IMG2IMG first.'
  if (!initSourceIsImg.value) return 'INPAINT requires the initial image source to be IMG.'
  if (!hasInitImage.value) return 'Select an init image first.'
  if (!supportsInpaint.value) {
    if (useMask.value) return 'INPAINT is not supported for the active img2img engine. Disable it to clear the stale mask state.'
    return 'INPAINT is not supported for the active img2img engine.'
  }
  return 'Toggle INPAINT'
})
const supirToggleDisabled = computed(() => (
  isActiveImageTabRunning.value
  || (!supirEnabled.value && guidanceAdvancedEnabled.value)
))
const supirToggleTitle = computed(() => {
  if (isActiveImageTabRunning.value) return 'Cannot change SUPIR while generation is running.'
  if (!canShowSupirToggle.value) return 'SUPIR is only available for native SDXL.'
  if (!supirEnabled.value && guidanceAdvancedEnabled.value) {
    return supirSelectionState.value.blockingReason || 'SUPIR mode cannot be enabled while Advanced Guidance/APG is active.'
  }
  if (!supirEnabled.value && supirSelectionState.value.blockingReason) {
    return `Enable SUPIR mode to repair the current selection. ${supirSelectionState.value.blockingReason}`
  }
  return supirEnabled.value ? 'Disable SUPIR mode' : 'Enable SUPIR mode'
})

watch(
  canShowSupirToggle,
  (show) => {
    if (!show) return
    void ensureSharedSupirDiagnosticsLoaded()
  },
  { immediate: true },
)

const activeWan14bTab = computed(() => asWan14bTab(activeModelTab.value))
const activeWan5bTab = computed(() => asWan5bTab(activeModelTab.value))

function normalizeTextEncoderLabels(raw: unknown): string[] {
  if (!Array.isArray(raw)) return []
  return raw.map((it) => String(it || '').trim()).filter((it) => it.length > 0)
}

const effectiveTextEncoders = computed(() => {
  if (isModelTabRoute.value) {
    if (activeFamily.value === 'ltx2') {
      const tab = activeLtxTab.value
      if (!tab) return []
      const textEncoder = String(tab.params.textEncoder || '').trim()
      return textEncoder ? [textEncoder] : []
    }
    const tab = activeImageTab.value
    if (!tab) return []
    return normalizeTextEncoderLabels(tab.params.textEncoders)
  }
  return store.currentTextEncoders
})

const primaryTextEncoder = computed(() => effectiveTextEncoders.value[0] ?? '')
const flux1TextEncoders = computed(() => effectiveTextEncoders.value.filter((label: string) => label.startsWith('flux1/')))
const flux1TextEncoderPrimary = computed(() => flux1TextEncoders.value[0] ?? '')
const flux1TextEncoderSecondary = computed(() => flux1TextEncoders.value[1] ?? '')
const flux2TextEncoder = computed(() => effectiveTextEncoders.value.find((label: string) => label.startsWith('flux2/')) ?? '')
const qwenImageVae = computed(() => {
  const current = String(store.currentVae || '').trim()
  return filteredVaeChoices.value.includes(current) ? current : ''
})

const effectiveCheckpoint = computed(() => {
  if (isModelTabRoute.value) {
    if (activeFamily.value === 'ltx2') {
      const ltxTab = activeLtxTab.value
      if (!ltxTab) return ''
      const checkpoint = String(ltxTab.params.checkpoint || '').trim()
      if (checkpoint) return checkpoint
      return filteredModelTitles.value[0] ?? ''
    }
    const tab = activeImageTab.value
    if (!tab) return ''
    const checkpoint = String(tab.params.checkpoint || '').trim()
    if (checkpoint) return checkpoint
    return filteredModelTitles.value[0] ?? ''
  }
  return store.currentModel
})

const effectiveVae = computed(() => {
  if (isModelTabRoute.value) {
    if (activeFamily.value === 'ltx2') {
      const ltxTab = activeLtxTab.value
      if (!ltxTab) return ''
      const vae = String(ltxTab.params.vae || '').trim()
      if (vae) return vae
      return filteredVaeChoices.value[0] ?? 'built-in'
    }
    if (!activeModelTab.value) return ''
    return store.currentVae
  }
  return store.currentVae
})

const activeImageAssetContract = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return null
  const engineId = activeImageRequestEngineId.value
  if (!engineId) return null
  const checkpoint = String(tab.params.checkpoint || '').trim()
  const modelInfo = checkpoint ? store.resolveModelInfo(checkpoint) : undefined
  if (checkpoint && modelInfo && typeof modelInfo.core_only !== 'boolean') {
    return null
  }
  return engineCaps.getAssetContract(engineId, {
    checkpointCoreOnly: typeof modelInfo?.core_only === 'boolean' ? modelInfo.core_only : false,
  })
})

const flux2TextEncoderFieldLabel = computed(() => {
  const fallback = 'Text Encoder (Qwen3-4B)'
  if (activeFamily.value !== 'flux2') return fallback

  const contract = activeImageAssetContract.value
  const slotLabel = Array.isArray(contract?.tenc_slot_labels)
    ? contract.tenc_slot_labels
        .map((value) => String(value || '').trim())
        .find((value) => value.length > 0)
    : ''
  if (slotLabel) {
    return /text encoder/i.test(slotLabel) ? slotLabel : `Text Encoder (${slotLabel})`
  }

  const kindLabel = String(contract?.tenc_kind_label || '').trim()
  if (kindLabel) {
    return /text encoder/i.test(kindLabel) ? kindLabel : `Text Encoder (${kindLabel})`
  }

  return fallback
})

const qwenImageTextEncoderFieldLabel = computed(() => {
  const fallback = 'Text Encoder (Qwen2.5-VL-7B)'
  if (activeFamily.value !== 'qwen_image') return fallback

  const contract = activeImageAssetContract.value
  const slotLabel = Array.isArray(contract?.tenc_slot_labels)
    ? contract.tenc_slot_labels
        .map((value) => String(value || '').trim())
        .find((value) => value.length > 0)
    : ''
  if (slotLabel) {
    return /text encoder/i.test(slotLabel) ? slotLabel : `Text Encoder (${slotLabel})`
  }

  const kindLabel = String(contract?.tenc_kind_label || '').trim()
  if (kindLabel) {
    return /text encoder/i.test(kindLabel) ? kindLabel : `Text Encoder (${kindLabel})`
  }

  return fallback
})

const zimageL2PTextEncoderFieldLabel = computed(() => {
  const fallback = 'Text Encoder (Qwen3-4B)'
  if (activeFamily.value !== 'zimage_l2p') return fallback

  const contract = activeImageAssetContract.value
  const slotLabel = Array.isArray(contract?.tenc_slot_labels)
    ? contract.tenc_slot_labels
        .map((value) => String(value || '').trim())
        .find((value) => value.length > 0)
    : ''
  if (slotLabel) {
    return /text encoder/i.test(slotLabel) ? slotLabel : `Text Encoder (${slotLabel})`
  }

  const kindLabel = String(contract?.tenc_kind_label || '').trim()
  if (kindLabel) {
    return /text encoder/i.test(kindLabel) ? kindLabel : `Text Encoder (${kindLabel})`
  }

  return fallback
})

const CODEX_REPO_URL = 'https://github.com/sangoi-exe/stable-diffusion-webui-codex'

const zimageTurbo = computed<boolean>(() => {
  const tab = activeImageTab.value
  if (!tab || tab.type !== 'zimage') return true
  const raw = tab.params.zimageTurbo
  return typeof raw === 'boolean' ? raw : true
})

const zimageTurboLocked = ref(false)
let zimageVariantDetectToken = 0

function _trustedZImageVariantFromCheckpointMeta(payload: unknown): 'turbo' | 'base' | null {
  if (!isRecordObject(payload)) return null
  const metadata = payload.metadata
  if (!isRecordObject(metadata)) return null
  const raw = metadata.raw
  if (!isRecordObject(raw)) return null

  const codexRepo = String(raw['codex.repository'] ?? '').trim()
  const codexBy = String(raw['codex.quantized_by'] ?? '').trim()
  if (!codexRepo || !codexBy) return null
  if (codexRepo !== CODEX_REPO_URL) return null

  const variant = String(raw['codex.zimage.variant'] ?? '').trim().toLowerCase()
  if (variant === 'turbo' || variant === 'base') return variant
  return null
}

watch(
  () => [activeFamily.value, activeImageTab.value?.id ?? '', effectiveCheckpoint.value] as const,
  async ([family, tabId, checkpoint]) => {
    zimageTurboLocked.value = false
    if (family !== 'zimage') return
    if (!tabId) return
    if (!checkpoint) return

    const token = ++zimageVariantDetectToken
    try {
      const meta = await fetchCheckpointMetadata(checkpoint)
      if (token !== zimageVariantDetectToken) return
      const variant = _trustedZImageVariantFromCheckpointMeta(meta)
      if (!variant) return
      zimageTurboLocked.value = true

      const turbo = variant === 'turbo'
      const tab = activeImageTab.value
      if (!tab || tab.type !== 'zimage') return
      const current = typeof tab.params.zimageTurbo === 'boolean' ? Boolean(tab.params.zimageTurbo) : true
      if (current !== turbo) {
        await updateImageTabParams(tab.id, { zimageTurbo: turbo })
        qsToast(`Z-Image: Turbo is ${turbo ? 'ON' : 'OFF'} (from model metadata).`)
      }
    } catch {
      // Non-fatal: if metadata can't be read, keep the toggle user-controlled.
    }
  },
  { immediate: true },
)

watch(
  () => [activeImageTab.value?.id ?? '', filteredModelTitles.value] as const,
  ([tabId, models]) => {
    if (!tabId) return
    const tab = activeImageTab.value
    if (!tab) return
    const ckpt = String(tab.params.checkpoint || '').trim()
    if (ckpt) return
    const first = models[0]
    if (!first) return
    updateImageTabParams(tab.id, { checkpoint: first }).catch((error) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  },
  { immediate: true },
)

watch(
  () => [activeLtxTab.value?.id ?? '', filteredModelTitles.value] as const,
  ([tabId, models]) => {
    if (!tabId) return
    const tab = activeLtxTab.value
    if (!tab) return
    const checkpoint = String(tab.params.checkpoint || '').trim()
    if (checkpoint) return
    const first = models[0]
    if (!first) return
    updateLtxTabParams(tab.id, { checkpoint: first }).catch((error) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  },
  { immediate: true },
)

async function initQuicksettings(
  options?: { forceInventoryRefresh?: boolean; forceModelsRefresh?: boolean },
  controls?: { includeStoreInit?: boolean },
): Promise<void> {
  isQuicksettingsReady.value = false
  isLoadingQuicksettings.value = true
  try {
    if (controls?.includeStoreInit !== false) {
      await store.init()
    }
    if (options?.forceModelsRefresh === true) {
      await store.refreshModelsList()
    }
    await Promise.all([
      loadPaths(),
      loadInventory({ forceRefresh: options?.forceInventoryRefresh === true }),
    ])
    isQuicksettingsReady.value = true
  } finally {
    isLoadingQuicksettings.value = false
  }
}

async function refreshAll(): Promise<void> {
  if (isLoadingQuicksettings.value) return
  isLoadingQuicksettings.value = true
  try {
    await Promise.all([store.refreshModelsList(), loadPaths()])
    const refreshedInventory = await store.fetchInventoryWithLoraHydration({ refresh: true })
    applyInventorySnapshot(refreshedInventory)
    await reloadSharedSupirDiagnostics()
  } catch (error) {
    toastQuicksettingsError(error)
  } finally {
    isLoadingQuicksettings.value = false
  }
}

const wanHighDirChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const g of inventoryWan.value) {
    if (g.variant && g.variant !== 'wan22_14b') continue
    const stage = String(g.stage || 'unknown').trim().toLowerCase()
    if (stage !== 'high' && stage !== 'unknown') continue
    const path = String(g.path || '').trim()
    if (!path) continue
    if (!seen.has(path)) { seen.add(path); out.push(path) }
  }
  return out
})

const wanLowDirChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const g of inventoryWan.value) {
    if (g.variant && g.variant !== 'wan22_14b') continue
    const stage = String(g.stage || 'unknown').trim().toLowerCase()
    if (stage !== 'low' && stage !== 'unknown') continue
    const path = String(g.path || '').trim()
    if (!path) continue
    if (!seen.has(path)) { seen.add(path); out.push(path) }
  }
  return out
})

const wan5bModelChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const g of inventoryWan.value) {
    if (g.variant !== 'wan22_5b') continue
    const path = String(g.path || '').trim()
    if (!path || seen.has(path)) continue
    seen.add(path)
    out.push(path)
  }
  return out
})

const WAN_LIGHTX2V_I2V_14B_FLOW_SHIFT = 5.0

function resolveWan14bFlowShift(useInitImage: boolean, lightx2v: boolean): number | null {
  if (!lightx2v) return null
  if (useInitImage) return WAN_LIGHTX2V_I2V_14B_FLOW_SHIFT
  return null
}

function patchWanStageFlowShift(stage: WanStageParams, flowShift: number | null): WanStageParams {
  const next: WanStageParams = { ...stage }
  if (flowShift === null) {
    if (
      typeof next.flowShift === 'number'
      && Number.isFinite(next.flowShift)
      && Math.abs(next.flowShift - WAN_LIGHTX2V_I2V_14B_FLOW_SHIFT) < 1e-9
    ) {
      delete next.flowShift
    }
    return next
  }
  next.flowShift = flowShift
  return next
}

function finiteStageFlowShift(stage: WanStageParams): number | undefined {
  if (typeof stage.flowShift !== 'number') return undefined
  if (!Number.isFinite(stage.flowShift)) return undefined
  return stage.flowShift
}

let syncingWanFlowShiftPolicy = false

async function ensureWan14bFlowShiftPolicy(): Promise<void> {
  if (syncingWanFlowShiftPolicy) return
  const tab = activeWan14bTab.value
  if (!tab) return
  const flowShift = resolveWan14bFlowShift(Boolean(tab.params.video?.useInitImage), Boolean(tab.params.lightx2v))
  const nextHigh = patchWanStageFlowShift(tab.params.high, flowShift)
  const nextLow = patchWanStageFlowShift(tab.params.low, flowShift)
  if (
    finiteStageFlowShift(tab.params.high) === finiteStageFlowShift(nextHigh)
    && finiteStageFlowShift(tab.params.low) === finiteStageFlowShift(nextLow)
  ) {
    return
  }
  syncingWanFlowShiftPolicy = true
  try {
    await tabsStore.updateParams(tab.id, { high: nextHigh, low: nextLow })
  } finally {
    syncingWanFlowShiftPolicy = false
  }
}

const wanLightx2v = computed(() => {
  const tab = activeWan14bTab.value
  if (!tab) return false
  return Boolean(tab.params.lightx2v)
})

const wanHighModel = computed(() => {
  const tab = activeWan14bTab.value
  if (!tab) return ''
  return tab.params.high?.modelDir || ''
})

const wanLowModel = computed(() => {
  const tab = activeWan14bTab.value
  if (!tab) return ''
  return tab.params.low?.modelDir || ''
})

const wan5bModel = computed(() => {
  const tab = activeWan5bTab.value
  if (!tab) return ''
  return tab.params.stage?.modelDir || ''
})

const wanInputMode = computed<'txt2vid' | 'img2vid'>(() => {
  const tab = activeWan14bTab.value ?? activeWan5bTab.value
  if (!tab) return 'txt2vid'
  return tab.params.video?.useInitImage ? 'img2vid' : 'txt2vid'
})

function currentWanAssetsFor(tab: { params?: { assets?: unknown } } | null): WanAssetsParams {
  const base: WanAssetsParams = { metadata: '', textEncoder: '', vae: '' }
  if (!tab) return base
  const raw = tab.params?.assets as Record<string, unknown> | undefined
  return raw ? { ...base, ...raw } : base
}

function resolveWanInventoryRepoHint(
  modelDir: string,
  options: { stage?: 'high' | 'low'; variant?: WanInventoryVariant } = {},
): string | null {
  return findWanGgufRecord(modelDir, options)?.repoHint ?? null
}

const activeWanAssets = computed(() => currentWanAssetsFor(activeWan14bTab.value ?? activeWan5bTab.value))
const wanTextEncoder = computed(() => activeWanAssets.value.textEncoder || '')
const wanVae = computed(() => activeWanAssets.value.vae || '')

const wanTextEncoderChoices = computed(() => {
  // WAN22 GGUF requires an explicit TE weights file (.safetensors or .gguf). Prefer concrete
  // files under the configured wan22_tenc roots (paths.json) rather than root labels from a dedicated endpoint.
  return inventoryTextEncoders.value
    .filter((item) => {
      const path = typeof item.path === 'string' ? item.path : ''
      if (!path) return false
      const lower = path.toLowerCase()
      if (!lower.endsWith('.safetensors') && !lower.endsWith('.gguf')) return false
      return fileInPaths(path, 'wan22_tenc')
    })
    .map((item) => `wan22/${item.path}`)
})

const wanVaeChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const item of store.inventoryVaesSnapshot) {
    const path = String(item.path || '')
    if (!path) continue
    if (!fileInPaths(path, 'wan22_vae')) continue
    if (!seen.has(path)) { seen.add(path); out.push(path) }
  }
  return out
})

// Event handlers
function toastQuicksettingsError(error: unknown): void {
  qsToast(error instanceof Error ? error.message : String(error))
}

function updateImageTabParams(tabId: string, patch: Partial<ImageBaseParams>): Promise<void> {
  return tabsStore.updateParams(tabId, patch as Partial<Record<string, unknown>>)
}

function updateLtxTabParams(tabId: string, patch: Partial<LtxTab['params']>): Promise<void> {
  return tabsStore.updateParams<Record<string, unknown>>(tabId, patch as unknown as Record<string, unknown>)
}

function toastModelTabStillLoading(): void {
  qsToast('Model tab settings are still loading.')
}

function toastModelAssetOwnerRequired(): void {
  qsToast('Checkpoint, VAE, and Text Encoder are read-only here. Open a model tab to edit them.')
}

async function onModelChange(value: string): Promise<void> {
  try {
    if (!isModelTabRoute.value) {
      toastModelAssetOwnerRequired()
      return
    }
    if (isModelTabRoute.value) {
      if (activeFamily.value === 'ltx2') {
        const ltxTab = activeLtxTab.value
        if (!ltxTab) {
          toastModelTabStillLoading()
          return
        }
        await updateLtxTabParams(ltxTab.id, { checkpoint: String(value || '') })
        return
      }
      const tab = activeImageTab.value
      if (!tab) {
        toastModelTabStillLoading()
        return
      }
      await updateImageTabParams(tab.id, { checkpoint: String(value || '') })
      return
    }
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onVaeChange(value: string): Promise<void> {
  try {
    if (!isModelTabRoute.value) {
      toastModelAssetOwnerRequired()
      return
    }
    if (!activeModelTab.value) {
      toastModelTabStillLoading()
      return
    }
    const ltxTab = activeLtxTab.value
    if (activeFamily.value === 'ltx2' && isModelTabRoute.value) {
      if (!ltxTab) {
        toastModelTabStillLoading()
        return
      }
      await updateLtxTabParams(ltxTab.id, { vae: String(value || '') })
      return
    }
    await store.setVaeForFamily(activeFamily.value, value)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onLtxModeChange(value: LtxGenerationMode): Promise<void> {
  try {
    if (ltxQuicksettingsDisabled.value) return
    const tab = activeLtxTab.value
    if (!tab) return
    const nextMode: LtxGenerationMode = value === 'img2vid' ? 'img2vid' : 'txt2vid'
    if (activeLtxMode.value === nextMode) return
    await updateLtxTabParams(tab.id, { mode: nextMode })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onUseInitImageChange(value: boolean): Promise<void> {
  try {
    const tab = activeImageTab.value
    if (!tab) return
    if (tab.type === 'qwen_image' && !value) {
      throw new Error('Qwen Image Edit-2511 supports IMG2IMG only.')
    }
    const patch: Partial<ImageBaseParams> = { ...buildUseInitImagePatch(Boolean(value)) }
    await updateImageTabParams(tab.id, patch)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onUseMaskChange(value: boolean): Promise<void> {
  try {
    const tab = activeImageTab.value
    if (!tab) return
    if (isActiveImageTabRunning.value) return
    if (!useInitImage.value) return
    if (!initSourceIsImg.value) return
    if (!hasInitImage.value) return
    if (value && !supportsInpaint.value) return
    const patch: Partial<ImageBaseParams> = { useMask: Boolean(value) }
    if (!value) {
      patch.maskImageData = ''
      patch.maskImageName = ''
    }
    await updateImageTabParams(tab.id, patch)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onSupirModeChange(value: boolean): Promise<void> {
  try {
    const tab = activeImageTab.value
    if (!tab) return
    if (isActiveImageTabRunning.value) return
    if (!value) {
      await updateImageTabParams(tab.id, {
        supir: {
          ...tab.params.supir,
          enabled: false,
        },
      })
      return
    }
    if (!canShowSupirToggle.value) return
    if (guidanceAdvancedEnabled.value) return
    await updateImageTabParams(tab.id, {
      useInitImage: true,
      supir: {
        ...tab.params.supir,
        enabled: true,
      },
    })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function updatePrefixedTextEncoders(familyPrefix: 'flux1' | 'flux2', labels: string[]): Promise<void> {
  if (!isModelTabRoute.value) {
    throw new Error('Checkpoint, VAE, and Text Encoder are read-only here. Open a model tab to edit them.')
  }
  const labelPrefix = `${familyPrefix}/`
  const normalizedLabels = labels
    .map((label) => String(label || '').trim())
    .filter((label, index, array) => label.startsWith(labelPrefix) && label.length > 0 && array.indexOf(label) === index)
  const tab = activeImageTab.value
  if (!tab) {
    throw new Error('Model tab settings are still loading.')
  }
  await updateImageTabParams(tab.id, { textEncoders: normalizedLabels })
}

async function updateFlux1TextEncoders(primary: string, secondary: string): Promise<void> {
  const flux1Labels: string[] = []
  const normalizedPrimary = String(primary || '').trim()
  const normalizedSecondary = String(secondary || '').trim()
  if (normalizedPrimary) flux1Labels.push(normalizedPrimary)
  if (normalizedSecondary && normalizedSecondary !== normalizedPrimary) flux1Labels.push(normalizedSecondary)
  await updatePrefixedTextEncoders('flux1', flux1Labels)
}

async function updateFlux2TextEncoder(value: string): Promise<void> {
  const selected = String(value || '').trim()
  await updatePrefixedTextEncoders('flux2', selected ? [selected] : [])
}

function onPrimaryTextEncoderChange(value: string): void {
  if (!isModelTabRoute.value) {
    toastModelAssetOwnerRequired()
    return
  }
  const fam = activeFamily.value
  if (fam === 'flux1') {
    const primary = value || ''
    const secondary = flux1TextEncoderSecondary.value || ''
    updateFlux1TextEncoders(primary, secondary).catch((error: unknown) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  } else if (fam === 'flux2') {
    updateFlux2TextEncoder(value).catch((error: unknown) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  } else if (fam === 'ltx2') {
    const tab = activeLtxTab.value
    const payload = String(value || '').trim()
    if (!tab) {
      toastModelTabStillLoading()
      return
    }
    updateLtxTabParams(tab.id, { textEncoder: payload }).catch((error: unknown) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  } else {
    const tab = activeImageTab.value
    const payload = value ? [value] : []
    if (!tab) {
      toastModelTabStillLoading()
      return
    }
    updateImageTabParams(tab.id, { textEncoders: payload }).catch((error: unknown) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  }
}

function onSecondaryTextEncoderChange(value: string): void {
  if (!isModelTabRoute.value) {
    toastModelAssetOwnerRequired()
    return
  }
  const fam = activeFamily.value
  if (fam !== 'flux1') return
  const primary = flux1TextEncoderPrimary.value || ''
  const secondary = value || ''
  updateFlux1TextEncoders(primary, secondary).catch((error: unknown) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

function onSmartOffloadChange(value: boolean): void {
  store.setSmartOffload(value).catch((error: unknown) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

function onSmartFallbackChange(value: boolean): void {
  store.setSmartFallback(value).catch((error: unknown) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

function onSmartCacheChange(value: boolean): void {
  store.setSmartCache(value).catch((error: unknown) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

function onCoreStreamingChange(value: boolean): void {
  store.setCoreStreaming(value).catch((error: unknown) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

async function onObliterateVram(): Promise<void> {
  if (isObliteratingVram.value) return
  isObliteratingVram.value = true
  try {
    const result = await fetchObliterateVram()
    console.info('[QuickSettingsBar] obliterate-vram result', result)
    if (Array.isArray(result.warnings) && result.warnings.length > 0) {
      console.warn('[QuickSettingsBar] obliterate-vram warnings', result.warnings)
    }
    if (!result.ok) {
      console.error('[QuickSettingsBar] obliterate-vram failed', {
        internal_failures: result.internal_failures,
        external_failures: result.external?.failures ?? [],
      })
      throw new Error(result.message || 'Obliterate VRAM finished with failures.')
    }
    const killedCount = Array.isArray(result.external?.terminated_pids)
      ? result.external.terminated_pids.length
      : 0
    const detectedCount = Array.isArray(result.external?.detected_processes)
      ? result.external.detected_processes.length
      : 0
    const disabledExternalKill = Array.isArray(result.warnings)
      && result.warnings.includes('external_gpu_termination_disabled_by_default')
    if (killedCount > 0) {
      qsToast(`Obliterate VRAM done. Killed ${killedCount} external GPU process(es).`)
      return
    }
    if (disabledExternalKill && detectedCount > 0) {
      qsToast(`Obliterate VRAM done (internal only). ${detectedCount} external GPU process(es) detected.`)
      return
    }
    qsToast('Obliterate VRAM done.')
  } catch (error) {
    qsToast(error instanceof Error ? error.message : String(error))
  } finally {
    isObliteratingVram.value = false
  }
}

async function onWanLightx2vChange(value: boolean): Promise<void> {
  try {
    const tab = activeWan14bTab.value
    if (!tab) return
    const nextLightx2v = Boolean(value)
    const flowShift = resolveWan14bFlowShift(Boolean(tab.params.video?.useInitImage), nextLightx2v)
    const nextHigh = patchWanStageFlowShift(tab.params.high, flowShift)
    const nextLow = patchWanStageFlowShift(tab.params.low, flowShift)
    await tabsStore.updateParams(tab.id, { lightx2v: nextLightx2v, high: nextHigh, low: nextLow })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanInputModeChange(value: 'txt2vid' | 'img2vid'): Promise<void> {
  try {
    const tab = activeWan14bTab.value ?? activeWan5bTab.value
    if (!tab) return
    const nextUseInitImage = value === 'img2vid'
    const currentVideo = tab.params.video
    const videoPatch: Record<string, unknown> = { useInitImage: nextUseInitImage }
    if (!nextUseInitImage) {
      videoPatch.initImageData = ''
      videoPatch.initImageName = ''
    }

    if (tab.type === 'wan22_14b') {
      const flowShift = resolveWan14bFlowShift(nextUseInitImage, Boolean(tab.params.lightx2v))
      const nextHigh = patchWanStageFlowShift(tab.params.high, flowShift)
      const nextLow = patchWanStageFlowShift(tab.params.low, flowShift)
      await tabsStore.updateParams(tab.id, {
        video: { ...currentVideo, ...videoPatch },
        high: nextHigh,
        low: nextLow,
      })
      return
    }

    await tabsStore.updateParams(tab.id, {
      video: {
        ...currentVideo,
        ...videoPatch,
        ...(nextUseInitImage ? { img2vidMode: 'solo' } : {}),
      },
    })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

watch(
  () => {
    const tab = activeWan14bTab.value
    if (!tab) return null
    return {
      tabId: tab.id,
      useInitImage: Boolean(tab.params.video?.useInitImage),
      lightx2v: Boolean(tab.params.lightx2v),
      highFlowShift: finiteStageFlowShift(tab.params.high),
      lowFlowShift: finiteStageFlowShift(tab.params.low),
    }
  },
  () => {
    void ensureWan14bFlowShiftPolicy()
  },
  { immediate: true },
)

async function onZImageTurboChange(value: boolean): Promise<void> {
  try {
    const tab = activeImageTab.value
    if (!tab || tab.type !== 'zimage') return
    if (zimageTurboLocked.value) {
      qsToast('Z-Image: Turbo variant is fixed by model metadata.')
      return
    }

    const currentSteps = Number(tab.params.steps)
    const currentCfg = Number(tab.params.cfgScale)

    const turbo = Boolean(value)
    const patch: Record<string, unknown> = { zimageTurbo: turbo }

    // Apply variant-recommended defaults only when the user is still on the previous variant's defaults.
    // Turbo defaults: steps≈9, distilled guidance≈1.0. Base defaults: steps≈30, CFG≈4.0.
    if (turbo) {
      if (Number.isFinite(currentSteps) && (currentSteps === 30)) patch.steps = 9
      if (Number.isFinite(currentCfg) && Math.abs(currentCfg - 4.0) < 1e-6) patch.cfgScale = 1.0
    } else {
      if (Number.isFinite(currentSteps) && (currentSteps === 8 || currentSteps === 9)) patch.steps = 30
      if (Number.isFinite(currentCfg) && Math.abs(currentCfg - 1.0) < 1e-6) patch.cfgScale = 4.0
    }

    await updateImageTabParams(tab.id, patch)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanHighModelChange(value: string): Promise<void> {
  try {
    const tab = activeWan14bTab.value
    if (!tab) return
    const current = tab.params.high || {}
    const repoHint = resolveWanInventoryRepoHint(value, { stage: 'high', variant: 'wan22_14b' })
    const assets = currentWanAssetsFor(tab)
    await tabsStore.updateParams(tab.id, {
      high: { ...current, modelDir: value },
      assets: repoHint ? { ...assets, metadata: repoHint } : assets,
    })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanLowModelChange(value: string): Promise<void> {
  try {
    const tab = activeWan14bTab.value
    if (!tab) return
    const current = tab.params.low || {}
    const repoHint = resolveWanInventoryRepoHint(value, { stage: 'low', variant: 'wan22_14b' })
    const assets = currentWanAssetsFor(tab)
    await tabsStore.updateParams(tab.id, {
      low: { ...current, modelDir: value },
      assets: repoHint ? { ...assets, metadata: repoHint } : assets,
    })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWan5bModelChange(value: string): Promise<void> {
  try {
    const tab = activeWan5bTab.value
    if (!tab) return
    const current = tab.params.stage || ({} as Wan5bStageParams)
    const repoHint = resolveWanInventoryRepoHint(value, { variant: 'wan22_5b' })
    const assets = currentWanAssetsFor(tab)
    await tabsStore.updateParams(tab.id, {
      stage: { ...current, modelDir: value },
      assets: repoHint ? { ...assets, metadata: repoHint } : assets,
    })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanTextEncoderChange(value: string): Promise<void> {
  try {
    const tab = activeWan14bTab.value ?? activeWan5bTab.value
    if (!tab) return
    const current = currentWanAssetsFor(tab)
    await tabsStore.updateParams(tab.id, { assets: { ...current, textEncoder: value } })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanVaeChange(value: string): Promise<void> {
  try {
    const tab = activeWan14bTab.value ?? activeWan5bTab.value
    if (!tab) return
    const current = currentWanAssetsFor(tab)
    await tabsStore.updateParams(tab.id, { assets: { ...current, vae: value } })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

function openAddPathModal(options: {
  title: string
  label: string
  key: string
  kind: AddPathTargetKind
  placeholder?: string
}): void {
  addPathModalTitle.value = options.title
  addPathModalLabel.value = options.label
  addPathModalTargetKey.value = options.key
  addPathModalTargetKind.value = options.kind
  addPathModalPlaceholder.value = String(options.placeholder || '')
  showAddPathModal.value = true
}

function onAddCheckpointPath(): void {
  const prefix = enginePrefixForFamily(activeFamily.value)
  openAddPathModal({
    title: 'Add Checkpoint Directory',
    label: 'Checkpoint path',
    key: `${prefix}_ckpt`,
    kind: 'checkpoint',
  })
}

function onAddVaePath(): void {
  const prefix = enginePrefixForFamily(activeFamily.value)
  openAddPathModal({
    title: 'Add VAE Directory',
    label: 'VAE path',
    key: `${prefix}_vae`,
    kind: 'vae',
  })
}

function onAddTencPath(): void {
  const prefix = enginePrefixForFamily(activeFamily.value)
  const key = activeFamily.value === 'zimage_l2p' ? 'zimage_tenc' : `${prefix}_tenc`
  openAddPathModal({
    title: 'Add Text Encoder Directory',
    label: 'Text encoder path',
    key,
    kind: 'text_encoder',
  })
}

async function onAddPathModalAdded(payload: { addedCount: number }): Promise<void> {
  if (!payload || !Number.isFinite(payload.addedCount) || payload.addedCount <= 0) return
  try {
    await refreshAll()
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

function onAddPathModalError(message: string): void {
  const text = String(message || '').trim()
  if (!text) return
  qsToast(text)
}

function openPathInputModal(
  options: { title: string; label: string; placeholder?: string; initialValue?: string },
  apply: (value: string) => Promise<void>,
): void {
  pathInputModalTitle.value = options.title
  pathInputModalLabel.value = options.label
  pathInputModalPlaceholder.value = options.placeholder || ''
  pathInputModalValue.value = String(options.initialValue || '')
  pathInputApply = apply
  showPathInputModal.value = true
  void nextTick(() => {
    pathInputEl.value?.focus()
    pathInputEl.value?.select()
  })
}

function closePathInputModal(): void {
  showPathInputModal.value = false
  pathInputApply = null
  pathInputModalValue.value = ''
}

async function confirmPathInputModal(): Promise<void> {
  const apply = pathInputApply
  if (!apply) {
    closePathInputModal()
    return
  }
  const trimmed = pathInputModalValue.value.trim()
  if (!trimmed) {
    qsToast('Path is required.')
    return
  }
  try {
    await apply(trimmed)
    closePathInputModal()
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

function onWanBrowseModels(): void {
  openAddPathModal({
    title: 'Add WAN Model Directory',
    label: 'WAN model path',
    key: 'wan22_ckpt',
    kind: 'checkpoint',
  })
}

async function onWanBrowseTe(): Promise<void> {
  openPathInputModal(
    {
      title: 'WAN Text Encoder',
      label: 'WAN Text Encoder (.safetensors or .gguf) path or sha256',
      initialValue: wanTextEncoder.value,
    },
    async (value) => {
      const normalized = value.replace(/\\+/g, '/')
      const stored = normalized.startsWith('wan22/') || !normalized.startsWith('/') ? normalized : `wan22/${normalized}`
      await onWanTextEncoderChange(stored)
    },
  )
}

async function onWanBrowseVae(): Promise<void> {
  openPathInputModal(
    {
      title: 'WAN VAE',
      label: 'WAN VAE path or sha256',
      initialValue: wanVae.value,
    },
    async (value) => {
      await onWanVaeChange(value)
    },
  )
}

function openOverrides(): void {
  showOverridesModal.value = true
}

onMounted(() => {
  const inner = advancedRowInnerEl.value
  if (inner) {
    advancedRowResizeObserver = new ResizeObserver(() => {
      syncAdvancedTargetHeight()
    })
    advancedRowResizeObserver.observe(inner)
  }
  requestAnimationFrame(syncAdvancedTargetHeight)
  void Promise.allSettled([
    initQuicksettings(),
  ]).then((results) => {
    const [quicksettingsResult] = results
    if (quicksettingsResult.status === 'rejected') {
      console.error('[quicksettings] failed to initialize quicksettings', quicksettingsResult.reason)
      toastQuicksettingsError(quicksettingsResult.reason)
    }
  })
})

onBeforeUnmount(() => {
  advancedRowResizeObserver?.disconnect()
  advancedRowResizeObserver = null
  showAddPathModal.value = false
  closePathInputModal()
})

watch(() => route.path, async () => {
  try {
    await loadInventory()
  } catch (error) {
    toastQuicksettingsError(error)
  }
})

watch(resolvedPresetTab, async (tab, previousTab) => {
  if (!tab || tab === previousTab) return
  try {
    await presets.init(tab)
  } catch (error) {
    console.error('[quicksettings] failed to initialize presets', error)
    toastQuicksettingsError(error)
  }
}, { immediate: true })

// random seed button removed from quicksettings; presets applied elsewhere
</script>
