<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Renderless WAN 2.2 14B runtime helper for the canonical video tab view.
Mounts the exact 14B WAN tab runtime under a view-local seam and exposes reactive slot props to `VideoModelTab.vue`,
while keeping the route-owned video view as the only live body/layout owner and routing shared Results/history actions through the current
workflow save-or-update owner seam.

Symbols (top-level; keep in sync; no ghosts):
- `VideoModelTabWanRuntime` (component): Renderless WAN 2.2 14B runtime helper for `VideoModelTab.vue`.
- `GuidedStep` (type): Guided-generation step definition (message + CSS selector to highlight/focus).
- `AspectMode` (type): Aspect ratio mode presets for width/height controls.
- `setHighPromptText` / `setHighNegativeText` / `setLowPromptText` / `setLowNegativeText` (functions): Prompt-field bridge setters exposed to the parent slot.
- `setShowHighPromptLoraModal` / `setShowLowPromptLoraModal` (functions): Parent-facing modal visibility bridge setters.
- `setVideoZoomOpen` / `setHistoryDetailsOpen` (functions): Parent-facing modal/overlay visibility bridge setters.
- `sendToWorkflows` / `copyCurrentParams` / `applySelectedHistory` (functions): Parent-facing Results/history actions, including truthful save-vs-update Workflow notices.
- `setGuidedTooltipEl` (function): Parent-facing tooltip element ref bridge.
- `slotProps` (const): Reactive slot-prop bundle exposed to `VideoModelTab.vue`.
-->

<template>
  <slot v-bind="slotProps" />
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, computed, ref, watch, nextTick } from 'vue'
import { useModelTabsStore, type TabByType, type WanAssetsParams, type WanStageParams, type WanVideoParams } from '../../stores/model_tabs'
import type { SamplerInfo, SchedulerInfo, GeneratedImage } from '../../api/types'
import { fetchSamplers, fetchSchedulers } from '../../api/client'
import { useVideoGeneration, type VideoRunHistoryItem } from '../../composables/useVideoGeneration'
import { useWorkflowSnapshotActions } from '../../composables/useWorkflowSnapshotActions'
import { useEngineCapabilitiesStore } from '../../stores/engine_capabilities'
import { useBootstrapStore } from '../../stores/bootstrap'
import {
  isWanWindowedImg2VidMode,
  normalizeWanImg2VidMode,
  normalizeWanWindowCommit,
  normalizeWanWindowStride,
  WAN_WINDOW_COMMIT_OVERLAP_MIN,
  WAN_WINDOW_STRIDE_ALIGNMENT,
} from '../../utils/wan_img2vid_temporal'
import {
  normalizeWanImg2VidImageScale,
  type WanImg2VidFrameGuideConfig,
} from '../../utils/wan_img2vid_frame_projection'
import { readFileAsDataURL, readImageDimensions } from '../../utils/image_io'

const props = defineProps<{ tabId: string }>()
const store = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()
const bootstrap = useBootstrapStore()

// Load option lists
const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])

onMounted(() => {
  bootstrap
    .runRequired('Failed to initialize WAN tab controls', async () => {
      const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
      samplers.value = samp.samplers
      schedulers.value = sched.schedulers
    })
    .catch(() => {
      // Fatal state is already set by bootstrap store.
    })
})

type WanTab = TabByType<'wan22_14b'>

const tab = computed<WanTab | null>(() => {
  const candidate = store.tabs.find((entry) => entry.id === props.tabId) || null
  if (!candidate || candidate.type !== 'wan22_14b') return null
  return candidate as WanTab
})
const wanParams = computed<WanTab['params'] | null>(() => tab.value?.params || null)
const lightx2v = computed<boolean>(() => Boolean(wanParams.value?.lightx2v))

function defaultStage(): WanStageParams {
  return { modelDir: '', prompt: '', negativePrompt: '', sampler: 'uni-pc bh2', scheduler: 'simple', steps: 30, cfgScale: 7, seed: -1, loras: [], flowShift: undefined }
}
function defaultVideo(): WanVideoParams {
  return {
    width: 768,
    height: 432,
    fps: 15,
    frames: 17,
    attentionMode: 'global',
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    img2vidMode: 'solo',
    img2vidChunkFrames: 13,
    img2vidOverlapFrames: 4,
    img2vidAnchorAlpha: 0.2,
    img2vidResetAnchorToBase: false,
    img2vidChunkSeedMode: 'increment',
    img2vidWindowFrames: 13,
    img2vidWindowStride: 8,
    img2vidWindowCommitFrames: 12,
    img2vidImageScale: 1,
    img2vidCropOffsetX: 0.5,
    img2vidCropOffsetY: 0.5,
    format: 'video/h264-mp4',
    pixFmt: 'yuv420p',
    crf: 15,
    loopCount: 0,
    pingpong: false,
    returnFrames: false,
    interpolationFps: 0,
  }
}

const video = computed<WanVideoParams>(() => wanParams.value?.video || defaultVideo())
const high = computed<WanStageParams>(() => wanParams.value?.high || defaultStage())
const low = computed<WanStageParams>(() => wanParams.value?.low || defaultStage())
const wanInitImageZoomFrameGuide = computed<WanImg2VidFrameGuideConfig>(() => ({
  targetWidth: Number(video.value.width) || 64,
  targetHeight: Number(video.value.height) || 64,
  imageScale: normalizeWanImg2VidImageScale(video.value.img2vidImageScale, 1),
  cropOffsetX: normalizeGuideOffset(video.value.img2vidCropOffsetX, 0.5),
  cropOffsetY: normalizeGuideOffset(video.value.img2vidCropOffsetY, 0.5),
}))

function defaultAssets(): WanAssetsParams { return { metadata: '', textEncoder: '', vae: '' } }

const assets = computed<WanAssetsParams>(() => wanParams.value?.assets || defaultAssets())

const WAN_FRAMES_MIN = 9
const WAN_FRAMES_MAX = 401
const WAN_DIM_MIN = 64
const WAN_DIM_MAX = 2048
const WAN_DIM_STEP_DEFAULT = 16

function normalizeFrameCount(rawValue: number): number {
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : WAN_FRAMES_MIN
  const clamped = Math.min(WAN_FRAMES_MAX, Math.max(WAN_FRAMES_MIN, numeric))
  if ((clamped - 1) % 4 === 0) return clamped

  const down = clamped - (((clamped - 1) % 4 + 4) % 4)
  const up = down + 4
  const downInRange = down >= WAN_FRAMES_MIN
  const upInRange = up <= WAN_FRAMES_MAX
  if (downInRange && upInRange) {
    const downDistance = Math.abs(clamped - down)
    const upDistance = Math.abs(up - clamped)
    return downDistance <= upDistance ? down : up
  }
  if (downInRange) return down
  if (upInRange) return up
  return WAN_FRAMES_MIN
}

function normalizeAttentionMode(rawValue: unknown): 'global' | 'sliding' {
  return String(rawValue || '').trim().toLowerCase() === 'sliding' ? 'sliding' : 'global'
}

function normalizeImg2VidMode(rawValue: unknown): WanVideoParams['img2vidMode'] {
  return normalizeWanImg2VidMode(rawValue)
}

type WanTemporalEnabledMode = Exclude<WanVideoParams['img2vidMode'], 'solo'>

const DEFAULT_TEMPORAL_ENABLED_MODE: WanTemporalEnabledMode = 'sliding'

function normalizeImg2VidTemporalEnabledMode(rawValue: unknown): WanTemporalEnabledMode {
  const mode = normalizeImg2VidMode(rawValue)
  if (mode === 'solo') return DEFAULT_TEMPORAL_ENABLED_MODE
  return mode
}

function isWindowedTemporalMode(rawValue: unknown): boolean {
  return isWanWindowedImg2VidMode(normalizeImg2VidMode(rawValue))
}

function defaultResetAnchorToBase(_mode: WanVideoParams['img2vidMode']): boolean {
  return false
}

function maxAlignedWindowStride(windowFrames: number): number {
  return normalizeWanWindowStride(
    Number(windowFrames) - WAN_WINDOW_COMMIT_OVERLAP_MIN,
    Number(windowFrames),
    Number(windowFrames) - WAN_WINDOW_COMMIT_OVERLAP_MIN,
  )
}

function normalizeChunkSeedMode(rawValue: unknown): 'fixed' | 'increment' | 'random' {
  const v = String(rawValue || '').trim().toLowerCase()
  if (v === 'fixed' || v === 'random') return v
  return 'increment'
}

function normalizeInterpolationTargetFps(rawValue: unknown, fallback: number): number {
  const maxFps = 240
  const fallbackNumeric = Number.isFinite(Number(fallback)) ? Math.trunc(Number(fallback)) : 0
  const fallbackNormalized = Math.max(0, Math.min(maxFps, fallbackNumeric))
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) return fallbackNormalized
  return Math.max(0, Math.min(maxFps, Math.trunc(numeric)))
}

function normalizeGuideOffset(rawValue: unknown, fallback: number): number {
  const numeric = Number(rawValue)
  const safeFallback = Number.isFinite(Number(fallback)) ? Number(fallback) : 0.5
  if (!Number.isFinite(numeric)) return Math.max(0, Math.min(1, safeFallback))
  return Math.max(0, Math.min(1, numeric))
}

function img2vidTemporalEnabledModeStorageKey(): string {
  return `codex.wan.img2vid.temporal.enabled_mode.${props.tabId}`
}

function readImg2VidTemporalEnabledMode(): WanTemporalEnabledMode | null {
  const key = img2vidTemporalEnabledModeStorageKey()
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return null
    return normalizeImg2VidTemporalEnabledMode(raw)
  } catch {
    return null
  }
}

function writeImg2VidTemporalEnabledMode(mode: WanVideoParams['img2vidMode']): void {
  if (mode === 'solo') return
  const key = img2vidTemporalEnabledModeStorageKey()
  try {
    localStorage.setItem(key, normalizeImg2VidTemporalEnabledMode(mode))
  } catch {
    // ignore localStorage failures
  }
}

function img2vidTemporalStorageKey(mode: WanVideoParams['img2vidMode']): string {
  return `codex.wan.img2vid.temporal.${props.tabId}.${mode}`
}

function readImg2VidTemporalSnapshot(mode: WanVideoParams['img2vidMode']): Partial<WanVideoParams> | null {
  const key = img2vidTemporalStorageKey(mode)
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null
    const record = parsed as Record<string, unknown>
    const patch: Partial<WanVideoParams> = {}
    if (record.attentionMode !== undefined) {
      patch.attentionMode = normalizeAttentionMode(record.attentionMode)
    }
    if (record.img2vidChunkSeedMode !== undefined) {
      patch.img2vidChunkSeedMode = normalizeChunkSeedMode(record.img2vidChunkSeedMode)
    }
    if (record.img2vidAnchorAlpha !== undefined) {
      const anchor = Number(record.img2vidAnchorAlpha)
      if (Number.isFinite(anchor)) {
        patch.img2vidAnchorAlpha = Math.min(1, Math.max(0, anchor))
      }
    }
    if (typeof record.img2vidResetAnchorToBase === 'boolean') {
      patch.img2vidResetAnchorToBase = Boolean(record.img2vidResetAnchorToBase)
    }
    if (record.img2vidImageScale !== undefined) {
      patch.img2vidImageScale = normalizeWanImg2VidImageScale(record.img2vidImageScale, 1)
    }
    if (record.img2vidCropOffsetX !== undefined) {
      patch.img2vidCropOffsetX = normalizeGuideOffset(record.img2vidCropOffsetX, 0.5)
    }
    if (record.img2vidCropOffsetY !== undefined) {
      patch.img2vidCropOffsetY = normalizeGuideOffset(record.img2vidCropOffsetY, 0.5)
    }
    if (isWindowedTemporalMode(mode)) {
      if (record.img2vidWindowFrames !== undefined) {
        const windowFrames = Number(record.img2vidWindowFrames)
        if (Number.isFinite(windowFrames) && windowFrames > 0) patch.img2vidWindowFrames = windowFrames
      }
      if (record.img2vidWindowStride !== undefined) {
        const windowStride = Number(record.img2vidWindowStride)
        if (Number.isFinite(windowStride) && windowStride > 0) patch.img2vidWindowStride = Math.trunc(windowStride)
      }
      if (record.img2vidWindowCommitFrames !== undefined) {
        const commitFrames = Number(record.img2vidWindowCommitFrames)
        if (Number.isFinite(commitFrames) && commitFrames > 0) patch.img2vidWindowCommitFrames = Math.trunc(commitFrames)
      }
    }
    return patch
  } catch {
    return null
  }
}

function writeImg2VidTemporalSnapshot(mode: WanVideoParams['img2vidMode'], source: WanVideoParams): void {
  const key = img2vidTemporalStorageKey(mode)
  const payload: Record<string, unknown> = {
    attentionMode: source.attentionMode,
    img2vidChunkSeedMode: source.img2vidChunkSeedMode,
    img2vidAnchorAlpha: source.img2vidAnchorAlpha,
    img2vidResetAnchorToBase: source.img2vidResetAnchorToBase,
    img2vidImageScale: source.img2vidImageScale,
    img2vidCropOffsetX: source.img2vidCropOffsetX,
    img2vidCropOffsetY: source.img2vidCropOffsetY,
  }
  if (isWindowedTemporalMode(mode)) {
    payload.img2vidWindowFrames = source.img2vidWindowFrames
    payload.img2vidWindowStride = source.img2vidWindowStride
    payload.img2vidWindowCommitFrames = source.img2vidWindowCommitFrames
  }
  try {
    localStorage.setItem(key, JSON.stringify(payload))
  } catch {
    // ignore localStorage failures
  }
}

function normalizeVideoPatch(patch: Partial<WanVideoParams>, current: WanVideoParams): Partial<WanVideoParams> {
  const nextPatch: Partial<WanVideoParams> = { ...patch }

  if (Object.prototype.hasOwnProperty.call(nextPatch, 'frames')) {
    nextPatch.frames = normalizeFrameCount(Number(nextPatch.frames))
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'attentionMode')) {
    nextPatch.attentionMode = normalizeAttentionMode(nextPatch.attentionMode)
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidMode')) {
    nextPatch.img2vidMode = normalizeImg2VidMode(nextPatch.img2vidMode)
  }
  const effectiveMode = normalizeImg2VidMode(
    Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidMode')
      ? nextPatch.img2vidMode
      : current.img2vidMode,
  )
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidChunkSeedMode')) {
    nextPatch.img2vidChunkSeedMode = normalizeChunkSeedMode(nextPatch.img2vidChunkSeedMode)
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidImageScale')) {
    nextPatch.img2vidImageScale = normalizeWanImg2VidImageScale(nextPatch.img2vidImageScale, current.img2vidImageScale)
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidCropOffsetX')) {
    nextPatch.img2vidCropOffsetX = normalizeGuideOffset(nextPatch.img2vidCropOffsetX, current.img2vidCropOffsetX)
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidCropOffsetY')) {
    nextPatch.img2vidCropOffsetY = normalizeGuideOffset(nextPatch.img2vidCropOffsetY, current.img2vidCropOffsetY)
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidResetAnchorToBase')) {
    nextPatch.img2vidResetAnchorToBase = Boolean(nextPatch.img2vidResetAnchorToBase)
  }
  if (effectiveMode === 'svi2' || effectiveMode === 'svi2_pro') {
    nextPatch.img2vidResetAnchorToBase = false
  } else if (!Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidResetAnchorToBase')) {
    if (typeof current.img2vidResetAnchorToBase !== 'boolean') {
      nextPatch.img2vidResetAnchorToBase = defaultResetAnchorToBase(effectiveMode)
    }
  }
  const effectiveTotalFrames = Number(
    Object.prototype.hasOwnProperty.call(nextPatch, 'frames')
      ? nextPatch.frames
      : current.frames,
  )
  const normalizedTotalFrames = Number.isFinite(effectiveTotalFrames)
    ? Math.max(WAN_FRAMES_MIN, Math.trunc(effectiveTotalFrames))
    : WAN_FRAMES_MIN
  const temporalUpperBound = normalizeFrameCount(Math.max(WAN_FRAMES_MIN, normalizedTotalFrames - 4))
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidChunkFrames')) {
    const rawChunk = Number(nextPatch.img2vidChunkFrames)
    if (!Number.isFinite(rawChunk) || rawChunk <= 0) {
      nextPatch.img2vidChunkFrames = current.img2vidChunkFrames > 0 ? current.img2vidChunkFrames : 13
    } else {
      nextPatch.img2vidChunkFrames = normalizeFrameCount(rawChunk)
    }
  }
  let effectiveChunkFrames = Number(
    Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidChunkFrames')
      ? nextPatch.img2vidChunkFrames
      : current.img2vidChunkFrames,
  )
  if (temporalUpperBound < normalizedTotalFrames && effectiveChunkFrames >= normalizedTotalFrames) {
    effectiveChunkFrames = temporalUpperBound
    nextPatch.img2vidChunkFrames = temporalUpperBound
  }
  const didAdjustChunkFrames = effectiveChunkFrames !== Number(current.img2vidChunkFrames)
  const overlapSource = Number(
    Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidOverlapFrames')
      ? nextPatch.img2vidOverlapFrames
      : current.img2vidOverlapFrames,
  )
  const overlapInt = Number.isFinite(overlapSource) ? Math.trunc(overlapSource) : Math.trunc(Number(current.img2vidOverlapFrames))
  const overlapMax = Math.max(0, effectiveChunkFrames - 1)
  const normalizedOverlap = Math.min(overlapMax, Math.max(0, overlapInt))
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidOverlapFrames') || didAdjustChunkFrames) {
    nextPatch.img2vidOverlapFrames = normalizedOverlap
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidAnchorAlpha')) {
    const rawAnchor = Number(nextPatch.img2vidAnchorAlpha)
    const fallback = Number(current.img2vidAnchorAlpha)
    nextPatch.img2vidAnchorAlpha = Number.isFinite(rawAnchor) ? Math.min(1, Math.max(0, rawAnchor)) : fallback
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidWindowFrames')) {
    const rawWindow = Number(nextPatch.img2vidWindowFrames)
    if (!Number.isFinite(rawWindow) || rawWindow <= 0) {
      nextPatch.img2vidWindowFrames = current.img2vidWindowFrames > 0 ? current.img2vidWindowFrames : 13
    } else {
      nextPatch.img2vidWindowFrames = normalizeFrameCount(rawWindow)
    }
  }
  let effectiveWindowFrames = Number(
    Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidWindowFrames')
      ? nextPatch.img2vidWindowFrames
      : current.img2vidWindowFrames,
  )
  if (temporalUpperBound < normalizedTotalFrames && effectiveWindowFrames >= normalizedTotalFrames) {
    effectiveWindowFrames = temporalUpperBound
    nextPatch.img2vidWindowFrames = temporalUpperBound
  }
  const didAdjustWindowFrames = effectiveWindowFrames !== Number(current.img2vidWindowFrames)
  const strideSource = Number(
    Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidWindowStride')
      ? nextPatch.img2vidWindowStride
      : current.img2vidWindowStride,
  )
  const normalizedStride = normalizeWanWindowStride(
    strideSource,
    effectiveWindowFrames,
    Number(current.img2vidWindowStride),
  )
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidWindowStride') || didAdjustWindowFrames) {
    nextPatch.img2vidWindowStride = normalizedStride
  }
  const effectiveWindowStride = Number(
    Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidWindowStride')
      ? nextPatch.img2vidWindowStride
      : normalizedStride,
  )
  const didAdjustWindowStride = effectiveWindowStride !== Number(current.img2vidWindowStride)
  const commitSource = Number(
    Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidWindowCommitFrames')
      ? nextPatch.img2vidWindowCommitFrames
      : current.img2vidWindowCommitFrames,
  )
  const normalizedCommit = normalizeWanWindowCommit(
    commitSource,
    effectiveWindowFrames,
    Math.trunc(Math.max(WAN_WINDOW_STRIDE_ALIGNMENT, effectiveWindowStride)),
    Number(current.img2vidWindowCommitFrames),
  )
  if (
    Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidWindowCommitFrames')
    || didAdjustWindowFrames
    || didAdjustWindowStride
  ) {
    nextPatch.img2vidWindowCommitFrames = normalizedCommit
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'interpolationFps')) {
    nextPatch.interpolationFps = normalizeInterpolationTargetFps(
      nextPatch.interpolationFps,
      current.interpolationFps,
    )
  }
  return nextPatch
}

function setVideo(patch: Partial<WanVideoParams>): void {
  if (!tab.value) return
  const current = tab.value.params.video
  const normalizedPatch = normalizeVideoPatch(patch, current)
  store.updateParams(props.tabId, { video: { ...current, ...normalizedPatch } }).catch(reportTabMutationError)
}
function setHigh(patch: Partial<WanStageParams>): void {
  if (!tab.value) return
  const current = tab.value.params.high
  store.updateParams(props.tabId, { high: { ...current, ...patch } }).catch(reportTabMutationError)
}
function setLow(patch: Partial<WanStageParams>): void {
  if (!tab.value) return
  const current = tab.value.params.low
  store.updateParams(props.tabId, { low: { ...current, ...patch } }).catch(reportTabMutationError)
}

function normalizeNonNegativeInteger(value: unknown, fallback: number, max?: number): number {
  const fallbackInt = Number.isFinite(fallback) ? Math.max(0, Math.trunc(fallback)) : 0
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return fallbackInt
  const parsed = Math.max(0, Math.trunc(numeric))
  if (typeof max === 'number' && Number.isFinite(max)) {
    return Math.min(Math.max(0, Math.trunc(max)), parsed)
  }
  return parsed
}

function onLoopCountChange(value: number): void {
  setVideo({ loopCount: normalizeNonNegativeInteger(value, video.value.loopCount, 32) })
}

function onCrfChange(value: number): void {
  setVideo({ crf: normalizeNonNegativeInteger(value, video.value.crf, 51) })
}

function onInterpolationTargetFpsChange(value: number): void {
  setVideo({ interpolationFps: normalizeInterpolationTargetFps(value, video.value.interpolationFps) })
}

const lowFollowsHigh = computed<boolean>(() => Boolean(wanParams.value?.lowFollowsHigh))
const lowNoiseOpen = ref(true)
const highPromptOpen = ref(true)
const lowPromptOpen = ref(true)

function syncLowFromHighIfNeeded(): void {
  const patch: Partial<WanStageParams> = {
    sampler: high.value.sampler,
    scheduler: high.value.scheduler,
    steps: high.value.steps,
    cfgScale: high.value.cfgScale,
    seed: high.value.seed,
    flowShift: high.value.flowShift,
  }
  const keys = Object.keys(patch) as Array<keyof WanStageParams>
  const needsUpdate = keys.some((key) => low.value[key] !== patch[key])
  if (!needsUpdate) return
  setLow(patch)
}

function onLowFollowsHighChange(enabled: boolean): void {
  if (!tab.value) return
  if (!enabled) {
    store.updateParams(props.tabId, { lowFollowsHigh: false }).catch(reportTabMutationError)
    return
  }

  const nextLow: Partial<WanStageParams> = {
    sampler: high.value.sampler,
    scheduler: high.value.scheduler,
    steps: high.value.steps,
    cfgScale: high.value.cfgScale,
    seed: high.value.seed,
    flowShift: high.value.flowShift,
  }
  store.updateParams(props.tabId, { lowFollowsHigh: true, low: { ...low.value, ...nextLow } }).catch(reportTabMutationError)
}

function toggleLowNoise(): void {
  lowNoiseOpen.value = !lowNoiseOpen.value
}

function setImg2VidTemporalMode(nextMode: WanVideoParams['img2vidMode']): void {
  const currentMode = normalizeImg2VidMode(video.value.img2vidMode)
  const targetMode = normalizeImg2VidMode(nextMode)
  if (currentMode === targetMode) return
  if (targetMode !== 'solo') writeImg2VidTemporalEnabledMode(targetMode)
  writeImg2VidTemporalSnapshot(currentMode, video.value)
  const restoredPatch = readImg2VidTemporalSnapshot(targetMode)
  const restoredResetAnchor = restoredPatch?.img2vidResetAnchorToBase
  const nextResetAnchorToBase =
    typeof restoredResetAnchor === 'boolean'
      ? restoredResetAnchor
      : defaultResetAnchorToBase(targetMode)
  setVideo({ img2vidMode: targetMode, img2vidResetAnchorToBase: nextResetAnchorToBase, ...(restoredPatch ?? {}) })
}

const temporalControlsEnabled = computed<boolean>(() => normalizeImg2VidMode(video.value.img2vidMode) !== 'solo')
const temporalEnabledMode = computed<WanTemporalEnabledMode>(() => normalizeImg2VidTemporalEnabledMode(video.value.img2vidMode))

function setImg2VidTemporalEnabled(enabled: boolean): void {
  const currentMode = normalizeImg2VidMode(video.value.img2vidMode)
  if (enabled) {
    if (currentMode !== 'solo') return
    const restoredMode = readImg2VidTemporalEnabledMode() ?? DEFAULT_TEMPORAL_ENABLED_MODE
    setImg2VidTemporalMode(restoredMode)
    return
  }
  if (currentMode === 'solo') return
  writeImg2VidTemporalEnabledMode(currentMode)
  setImg2VidTemporalMode('solo')
}

function toggleHighPrompt(): void {
  highPromptOpen.value = !highPromptOpen.value
}

function toggleLowPrompt(): void {
  lowPromptOpen.value = !lowPromptOpen.value
}

watch(
  () => ([
    lowFollowsHigh.value,
    high.value.sampler,
    high.value.scheduler,
    high.value.steps,
    high.value.cfgScale,
    high.value.seed,
    high.value.flowShift,
  ] as const),
  ([enabled]) => {
    if (!enabled) return
    syncLowFromHighIfNeeded()
  },
)

watch(
  () => ([
    video.value.img2vidMode,
    video.value.attentionMode,
    video.value.img2vidChunkSeedMode,
    video.value.img2vidChunkFrames,
    video.value.img2vidOverlapFrames,
    video.value.img2vidAnchorAlpha,
    video.value.img2vidResetAnchorToBase,
    video.value.img2vidWindowFrames,
    video.value.img2vidWindowStride,
    video.value.img2vidWindowCommitFrames,
  ] as const),
  () => {
    const currentMode = normalizeImg2VidMode(video.value.img2vidMode)
    writeImg2VidTemporalSnapshot(currentMode, video.value)
    if (currentMode !== 'solo') {
      writeImg2VidTemporalEnabledMode(currentMode)
    }
  },
)

watch(
  () => ([
    lowFollowsHigh.value,
    low.value.sampler,
    low.value.scheduler,
    low.value.steps,
    low.value.cfgScale,
    low.value.seed,
    low.value.flowShift,
  ] as const),
  ([enabled]) => {
    if (!enabled) return
    syncLowFromHighIfNeeded()
  },
)

const highPrompt = computed({
  get: () => high.value.prompt,
  set: (value: string) => setHigh({ prompt: value }),
})

const highNegative = computed({
  get: () => high.value.negativePrompt,
  set: (value: string) => setHigh({ negativePrompt: value }),
})

const lowPrompt = computed({
  get: () => low.value.prompt,
  set: (value: string) => setLow({ prompt: value }),
})

const lowNegative = computed({
  get: () => low.value.negativePrompt,
  set: (value: string) => setLow({ negativePrompt: value }),
})

const hideHighNegativePrompt = computed(() => {
  const cfg = Number(high.value.cfgScale)
  return Number.isFinite(cfg) && cfg === 1
})

const hideLowNegativePrompt = computed(() => {
  const cfg = Number(low.value.cfgScale)
  return Number.isFinite(cfg) && cfg === 1
})

const showHighPromptLoraModal = ref(false)
const showLowPromptLoraModal = ref(false)

type PromptTokenInsertPayload = {
  token: string
  target?: 'positive' | 'negative'
  action?: 'add' | 'remove'
}

function splitPromptTokens(current: string): string[] {
  return String(current || '')
    .split(/\s+/)
    .map((part) => part.trim())
    .filter(Boolean)
}

function appendPromptToken(current: string, token: string): string {
  const trimmedToken = String(token || '').trim()
  if (!trimmedToken) return String(current || '')
  const tokens = splitPromptTokens(current)
  if (tokens.includes(trimmedToken)) return tokens.join(' ')
  tokens.push(trimmedToken)
  return tokens.join(' ')
}

function removePromptToken(current: string, token: string): string {
  const trimmedToken = String(token || '').trim()
  if (!trimmedToken) return String(current || '')
  return splitPromptTokens(current)
    .filter((part) => part !== trimmedToken)
    .join(' ')
}

function normalizeLoraSha(rawValue: unknown): string | undefined {
  const normalized = String(rawValue || '').trim().toLowerCase()
  if (!/^[0-9a-f]{64}$/.test(normalized)) return undefined
  return normalized
}

function normalizeStageLoraList(rawValue: unknown): WanStageParams['loras'] {
  if (!Array.isArray(rawValue)) return []

  const normalized: WanStageParams['loras'] = []
  const indexBySha = new Map<string, number>()
  for (const candidate of rawValue) {
    if (!isRecord(candidate)) continue
    const sha = normalizeLoraSha(candidate.sha)
    if (!sha) continue

    const hasWeight = Object.prototype.hasOwnProperty.call(candidate, 'weight')
    let weight: number | undefined
    if (hasWeight) {
      if (typeof candidate.weight !== 'number' || !Number.isFinite(candidate.weight)) continue
      weight = Number(candidate.weight)
    }

    const nextEntry = weight === undefined ? { sha } : { sha, weight }
    const existingIndex = indexBySha.get(sha)
    if (typeof existingIndex === 'number') {
      normalized[existingIndex] = nextEntry
      continue
    }
    indexBySha.set(sha, normalized.length)
    normalized.push(nextEntry)
  }
  return normalized
}

function onHighPromptLoraInsert(payload: PromptTokenInsertPayload): void {
  const target = payload.target === 'negative' ? 'negative' : 'positive'
  const action = payload.action === 'remove' ? 'remove' : 'add'
  if (target === 'negative') {
    const current = high.value.negativePrompt
    const next = action === 'remove' ? removePromptToken(current, payload.token) : appendPromptToken(current, payload.token)
    setHigh({ negativePrompt: next })
    return
  }
  const current = high.value.prompt
  const next = action === 'remove' ? removePromptToken(current, payload.token) : appendPromptToken(current, payload.token)
  setHigh({ prompt: next })
}

function onLowPromptLoraInsert(payload: PromptTokenInsertPayload): void {
  const target = payload.target === 'negative' ? 'negative' : 'positive'
  const action = payload.action === 'remove' ? 'remove' : 'add'
  if (target === 'negative') {
    const current = low.value.negativePrompt
    const next = action === 'remove' ? removePromptToken(current, payload.token) : appendPromptToken(current, payload.token)
    setLow({ negativePrompt: next })
    return
  }
  const current = low.value.prompt
  const next = action === 'remove' ? removePromptToken(current, payload.token) : appendPromptToken(current, payload.token)
  setLow({ prompt: next })
}

async function onInitImageFile(file: File): Promise<void> {
  const dataUrl = await readFileAsDataURL(file)
  setVideo({ initImageData: dataUrl, initImageName: file.name, useInitImage: true })
}

function onInitImageRejected(payload: { reason: string; files: File[] }): void {
  const fileName = payload.files[0]?.name || 'file'
  toast(`Init image rejected (${fileName}): ${payload.reason}`)
}

function onZoomFrameGuideUpdate(guide: WanImg2VidFrameGuideConfig): void {
  const nextWidth = snapDimForAspect(guide.targetWidth)
  const nextHeight = snapDimForAspect(guide.targetHeight)
  setVideo({
    width: nextWidth,
    height: nextHeight,
    img2vidImageScale: normalizeWanImg2VidImageScale(guide.imageScale, video.value.img2vidImageScale),
    img2vidCropOffsetX: normalizeGuideOffset(guide.cropOffsetX, video.value.img2vidCropOffsetX),
    img2vidCropOffsetY: normalizeGuideOffset(guide.cropOffsetY, video.value.img2vidCropOffsetY),
  })
}

function clearInit(): void { setVideo({ initImageData: '', initImageName: '' }) }

// Generation wiring (composable)
const {
  generate,
  isRunning,
  canGenerate,
  cancel,
  progress,
  frames: framesResult,
  info,
  videoUrl,
  errorMessage,
  mode,
  history,
  selectedTaskId,
  historyLoadingTaskId,
  loadHistory,
  clearHistory,
  resumeNotice,
} = useVideoGeneration(props.tabId)

const wanDependencyStatus = computed(() => engineCaps.getDependencyStatus('wan22_14b'))
const wanDependencyReady = computed(() => Boolean(wanDependencyStatus.value?.ready))
const wanDependencyError = computed(() => engineCaps.firstDependencyError('wan22_14b'))
const wanEngineSurface = computed(() => engineCaps.get('wan22_14b'))
const wanStageSamplers = computed(() => {
  const allowedExact = new Set(['euler', 'euler a'])
  return samplers.value.filter((entry) => {
    const normalizedName = String(entry.name || '').trim().toLowerCase()
    if (!normalizedName) return false
    if (normalizedName.startsWith('uni-pc')) return true
    return allowedExact.has(normalizedName)
  })
})
const wanStageSchedulers = computed(() => {
  return schedulers.value.filter((entry) => String(entry.name || '').trim() === 'simple')
})
const wanRecommendedSamplers = computed(() => {
  const available = new Set(wanStageSamplers.value.map((entry) => entry.name))
  const values = wanEngineSurface.value?.recommended_samplers
  const fallback = wanStageSamplers.value.map((entry) => entry.name)
  const source = Array.isArray(values) && values.length > 0 ? values : fallback
  const normalized = Array.from(
    new Set(source.map((value) => String(value || '').trim()).filter((value) => value.length > 0 && available.has(value))),
  )
  return normalized.length > 0 ? normalized : null
})
const wanRecommendedSchedulers = computed(() => {
  const available = new Set(wanStageSchedulers.value.map((entry) => entry.name))
  const values = wanEngineSurface.value?.recommended_schedulers
  const fallback = wanStageSchedulers.value.map((entry) => entry.name)
  const source = Array.isArray(values) && values.length > 0 ? values : fallback
  const normalized = Array.from(
    new Set(source.map((value) => String(value || '').trim()).filter((value) => value.length > 0 && available.has(value))),
  )
  return normalized.length > 0 ? normalized : null
})
const canRunGeneration = computed(() => wanDependencyReady.value && canGenerate.value)
const generateTitle = computed(() => {
  if (!wanDependencyReady.value) {
    return wanDependencyError.value || 'WAN dependencies are not ready.'
  }
  if (!canGenerate.value) return 'Guided gen: click to see what is missing.'
  return ''
})
const interpolationCaption = computed<string>(() => {
  const targetFps = normalizeInterpolationTargetFps(video.value.interpolationFps, 0)
  const baseFps = normalizeNonNegativeInteger(video.value.fps, 0, 240)
  if (targetFps <= 0) {
    return baseFps > 0 ? `Off · Output: ${baseFps} fps` : 'Off'
  }
  if (baseFps <= 0) return `Target: ${targetFps} fps`
  if (targetFps <= baseFps) return `Disabled · Target (${targetFps} fps) <= base (${baseFps} fps)`
  const times = Math.max(2, Math.ceil(targetFps / baseFps))
  const outputFps = baseFps * times
  return `Target: ${targetFps} fps · Output: ${outputFps} fps`
})
const videoZoomOpen = ref(false)

watch(videoUrl, (currentVideoUrl) => {
  if (!currentVideoUrl) videoZoomOpen.value = false
})

function openResultVideoZoom(): void {
  if (!videoUrl.value) return
  videoZoomOpen.value = true
}

function normalizeVideoBeforeSubmit(): void {
  const snappedW = snapDimForAspect(video.value.width)
  const snappedH = snapDimForAspect(video.value.height)
  const snappedFrames = normalizeFrameCount(video.value.frames)
  if (snappedW !== video.value.width || snappedH !== video.value.height || snappedFrames !== video.value.frames) {
    setVideo({ width: snappedW, height: snappedH, frames: snappedFrames })
  }
}

async function onGenerateClick(): Promise<void> {
  if (isRunning.value) return
  const activeElement = document.activeElement
  if (activeElement instanceof HTMLElement) {
    activeElement.blur()
    await nextTick()
  }
  if (!wanDependencyReady.value) {
    toast(wanDependencyError.value || 'WAN dependencies are not ready.')
    return
  }
  if (!canGenerate.value) {
    startGuided()
    return
  }
  stopGuided()
  normalizeVideoBeforeSubmit()
  await generate()
}

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
  getWorkflowParamsSnapshot: () => (tab.value?.params as Record<string, unknown> | null) ?? null,
  getCopyCurrentParamsSnapshot: () => buildCurrentSnapshot(),
  copyCurrentParamsMessage: 'Copied current params JSON.',
})
const historyDetailsOpen = ref(false)
const historyDetailsItem = ref<VideoRunHistoryItem | null>(null)

const historyDetailsTitle = computed(() => (historyDetailsItem.value ? formatHistoryTitle(historyDetailsItem.value) : 'History details'))
const historyDetailsCreatedAtLabel = computed(() => {
  const timestamp = historyDetailsItem.value?.createdAtMs
  if (!timestamp) return '—'
  return new Date(timestamp).toLocaleString()
})
const historyDetailsModeLabel = computed(() => {
  const mode = historyDetailsItem.value?.mode
  return formatVideoModeLabel(mode)
})
const historyDetailsImageUrl = computed(() => {
  const thumbnail = historyDetailsItem.value?.thumbnail
  return thumbnail ? toDataUrl(thumbnail) : ''
})
const historyDetailsHighPrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  const prompt = readHistoryStageSnapshotText(item, 'high', 'prompt')
  if (prompt) return prompt
  const legacyPrompt = readHistorySnapshotText(item, 'prompt')
  if (legacyPrompt) return legacyPrompt
  return item.promptPreview || ''
})

const historyDetailsHighNegativePrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  const negative = readHistoryStageSnapshotText(item, 'high', 'negativePrompt')
  if (negative) return negative
  return readHistorySnapshotText(item, 'negativePrompt')
})

const historyDetailsLowPrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  const prompt = readHistoryStageSnapshotText(item, 'low', 'prompt')
  if (prompt) return prompt
  return readHistorySnapshotText(item, 'prompt')
})

const historyDetailsLowNegativePrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  const negative = readHistoryStageSnapshotText(item, 'low', 'negativePrompt')
  if (negative) return negative
  return readHistorySnapshotText(item, 'negativePrompt')
})
const historyDetailsSections = computed(() => [
  { key: 'highPrompt', label: 'High Prompt', text: historyDetailsHighPrompt.value },
  { key: 'highNegativePrompt', label: 'High Negative Prompt', text: historyDetailsHighNegativePrompt.value },
  { key: 'lowPrompt', label: 'Low Prompt', text: historyDetailsLowPrompt.value },
  { key: 'lowNegativePrompt', label: 'Low Negative Prompt', text: historyDetailsLowNegativePrompt.value },
])

function reportTabMutationError(error: unknown): void {
  toast(error instanceof Error ? error.message : String(error))
}

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

function onSelectHistoryStripItem(item: { taskId: string }): void {
  const match = history.value.find((entry) => entry.taskId === item.taskId)
  if (!match) return
  openHistoryDetails(match)
}

type GuidedStep = { id: string; message: string; selector: string; focusSelector?: string }
const guidedActive = ref(false)
const guidedMessage = ref('')
const guidedRect = ref<DOMRect | null>(null)
const guidedCurrentId = ref('')
let guidedHighlightedEl: HTMLElement | null = null
let guidedRaf: number | null = null
let guidedSettleTimer: number | null = null

const guidedTooltipEl = ref<HTMLElement | null>(null)
const guidedTooltipPos = ref<{ left: number; top: number; placement: 'top' | 'bottom' } | null>(null)

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function computeGuidedTooltipPosition(): void {
  const rect = guidedRect.value
  const el = guidedTooltipEl.value
  if (!rect || !el) {
    guidedTooltipPos.value = null
    return
  }

  const tooltipW = el.offsetWidth || 0
  const tooltipH = el.offsetHeight || 0
  if (tooltipW <= 0 || tooltipH <= 0) {
    guidedTooltipPos.value = null
    return
  }

  const margin = 12
  const spaceAbove = rect.top
  const spaceBelow = window.innerHeight - rect.bottom
  const placement: 'top' | 'bottom' = (spaceBelow >= tooltipH + margin || spaceBelow >= spaceAbove) ? 'bottom' : 'top'

  const centerX = rect.left + rect.width / 2
  const rawLeft = centerX - tooltipW / 2
  const left = clampNumber(rawLeft, margin, window.innerWidth - margin - tooltipW)

  const rawTop = placement === 'bottom' ? (rect.bottom + 10) : (rect.top - 10 - tooltipH)
  const top = clampNumber(rawTop, margin, window.innerHeight - margin - tooltipH)

  guidedTooltipPos.value = { left, top, placement }
}

const guidedTooltipPlacement = computed<'top' | 'bottom'>(() => guidedTooltipPos.value?.placement || 'bottom')
const guidedTooltipStyle = computed<Record<string, string>>(() => {
  const pos = guidedTooltipPos.value
  if (!pos) return { left: '0px', top: '0px', opacity: '0' }
  return { left: `${Math.round(pos.left)}px`, top: `${Math.round(pos.top)}px`, opacity: '1' }
})

function isFocusable(el: Element | null): el is HTMLElement {
  if (!(el instanceof HTMLElement)) return false
  const tag = el.tagName.toLowerCase()
  if (tag === 'input' || tag === 'select' || tag === 'textarea' || tag === 'button') return true
  if (el.getAttribute('contenteditable') === 'true') return true
  return typeof el.focus === 'function'
}

function findFocusTarget(root: Element, selector?: string): HTMLElement | null {
  if (selector) {
    const el = document.querySelector(selector)
    return isFocusable(el) ? el : null
  }
  if (isFocusable(root)) return root
  const inside = root.querySelector('input,select,textarea,button,[contenteditable=\"true\"]')
  return isFocusable(inside) ? inside : null
}

function clearGuidedHighlight(): void {
  if (guidedHighlightedEl) guidedHighlightedEl.classList.remove('codex-guided-attention')
  guidedHighlightedEl = null
}

function updateGuidedRect(): void {
  if (!guidedHighlightedEl) {
    guidedRect.value = null
    return
  }
  guidedRect.value = guidedHighlightedEl.getBoundingClientRect()
}

function scheduleGuidedRectUpdate(): void {
  if (guidedRaf !== null) return
  guidedRaf = window.requestAnimationFrame(() => {
    guidedRaf = null
    updateGuidedRect()
    computeGuidedTooltipPosition()
  })
}

function scheduleGuidedSettleUpdate(): void {
  if (guidedSettleTimer !== null) window.clearTimeout(guidedSettleTimer)
  guidedSettleTimer = window.setTimeout(() => {
    guidedSettleTimer = null
    updateGuidedRect()
    computeGuidedTooltipPosition()
  }, 250)
}

function stopGuided(): void {
  guidedActive.value = false
  guidedMessage.value = ''
  guidedRect.value = null
  guidedTooltipPos.value = null
  guidedCurrentId.value = ''
  clearGuidedHighlight()
  if (guidedSettleTimer !== null) window.clearTimeout(guidedSettleTimer)
  guidedSettleTimer = null
}

function focusGuided(step: GuidedStep): void {
  const target = document.querySelector(step.selector) as HTMLElement | null
  if (!target) return

  const focusEl = findFocusTarget(target, step.focusSelector) || target
  clearGuidedHighlight()
  guidedHighlightedEl = focusEl
  guidedHighlightedEl.classList.add('codex-guided-attention')

  guidedMessage.value = step.message
  guidedCurrentId.value = step.id
  guidedHighlightedEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
  try {
    guidedHighlightedEl.focus({ preventScroll: true })
  } catch {
    try { guidedHighlightedEl.focus() } catch { /* ignore */ }
  }
  updateGuidedRect()
  scheduleGuidedRectUpdate()
  scheduleGuidedSettleUpdate()
}

function startGuided(): void {
  guidedActive.value = true
}

const guidedSteps = computed<GuidedStep[]>(() => {
  const steps: GuidedStep[] = []

  const highStagePrompt = String(high.value.prompt || '').trim()
  if (!highStagePrompt) {
    steps.push({
      id: 'high_prompt',
      message: 'Write the High stage prompt to generate.',
      selector: '#wan-guided-high-prompt',
      focusSelector: '#wan-guided-high-prompt [contenteditable=\"true\"]',
    })
    return steps
  }

  const lowStagePrompt = String(low.value.prompt || '').trim()
  if (!lowStagePrompt) {
    steps.push({
      id: 'low_prompt',
      message: 'Write the Low stage prompt to generate.',
      selector: '#wan-guided-low-prompt',
      focusSelector: '#wan-guided-low-prompt [contenteditable=\"true\"]',
    })
    return steps
  }

  if (!high.value.modelDir && !low.value.modelDir) {
    steps.push({
      id: 'wan_models',
      message: 'Select WAN High/Low models in QuickSettings (header).',
      selector: '#qs-wan-high',
    })
    return steps
  }

  if (mode.value === 'img2vid' && !video.value.initImageData) {
    steps.push({
      id: 'init_image',
      message: 'Image mode needs an input image. Upload one (or switch to Text mode).',
      selector: '#wan-guided-init-image',
      focusSelector: '#wan-guided-init-image .cdx-dropzone',
    })
    return steps
  }

  return steps
})

watch(guidedActive, (active) => {
  if (active) {
    window.addEventListener('scroll', scheduleGuidedRectUpdate, true)
    window.addEventListener('resize', scheduleGuidedRectUpdate)
    scheduleGuidedRectUpdate()
  } else {
    window.removeEventListener('scroll', scheduleGuidedRectUpdate, true)
    window.removeEventListener('resize', scheduleGuidedRectUpdate)
    if (guidedRaf !== null) window.cancelAnimationFrame(guidedRaf)
    guidedRaf = null
  }
})

watch(isRunning, (running) => {
  if (running) stopGuided()
})

watch([guidedActive, guidedSteps], async ([active, steps]) => {
  if (!active) return
  await nextTick()

  if (!steps.length) {
    focusGuided({
      id: 'ready',
      message: 'Ready. Click Generate.',
      selector: '#wan-guided-generate',
      focusSelector: '#wan-guided-generate',
    })
    return
  }

  const step = steps[0]!
  if (step.id === 'high_prompt' && !highPromptOpen.value) {
    highPromptOpen.value = true
    await nextTick()
  }
  if (step.id === 'low_prompt' && !lowPromptOpen.value) {
    lowPromptOpen.value = true
    await nextTick()
  }
  if (step.id === guidedCurrentId.value && guidedRect.value) return
  focusGuided(step)
}, { deep: true })

function onGuidedGenEvent(event: Event): void {
  const e = event as CustomEvent<{ tabId?: string }>
  if (e.detail?.tabId && e.detail.tabId !== props.tabId) return
  startGuided()
}

onMounted(() => {
  window.addEventListener('codex-wan-guided-gen', onGuidedGenEvent as EventListener)
})

onBeforeUnmount(() => {
  window.removeEventListener('codex-wan-guided-gen', onGuidedGenEvent as EventListener)
  stopGuided()
})

function setInputMode(next: 'txt2vid' | 'img2vid'): void {
  if (isRunning.value) return
  if (next === 'txt2vid') {
    setVideo({ useInitImage: false, initImageData: '', initImageName: '' })
    return
  }
  setVideo({ useInitImage: true })
}

const durationLabel = computed(() => {
  const fps = Number(video.value.fps) || 0
  const frames = Number(video.value.frames) || 0
  if (fps <= 0) return '0.00'
  return (frames / fps).toFixed(2)
})

const runSummary = computed(() => {
  const v = video.value
  const highStage = high.value
  const lowStage = low.value
  const base = `${mode.value} · ${v.width}×${v.height} px · ${v.frames} frames @ ${v.fps} fps (~ ${durationLabel.value}s) · High ${highStage.steps} steps · CFG ${highStage.cfgScale} · Low ${lowStage.steps} steps · CFG ${lowStage.cfgScale}`
  return lightx2v.value ? `${base} · lightx2v` : base
})

function buildCurrentSnapshot(): Record<string, unknown> {
  const img2vidMode = normalizeImg2VidMode(video.value.img2vidMode)
  return {
    mode: video.value.useInitImage ? 'img2vid' : 'txt2vid',
    initImageName: video.value.initImageName || '',
    attentionMode: video.value.attentionMode,
    img2vid: {
      mode: img2vidMode,
      anchorAlpha: video.value.img2vidAnchorAlpha,
      resetAnchorToBase: video.value.img2vidResetAnchorToBase,
      chunkSeedMode: video.value.img2vidChunkSeedMode,
      windowFrames: video.value.img2vidWindowFrames,
      windowStride: video.value.img2vidWindowStride,
      windowCommitFrames: video.value.img2vidWindowCommitFrames,
      imageScale: video.value.img2vidImageScale,
      cropOffsetX: video.value.img2vidCropOffsetX,
      cropOffsetY: video.value.img2vidCropOffsetY,
    },
    width: video.value.width,
    height: video.value.height,
    frames: video.value.frames,
    fps: video.value.fps,
    lightx2v: lightx2v.value,
    assets: {
      metadata: String(assets.value.metadata || ''),
      textEncoder: String(assets.value.textEncoder || ''),
      vae: String(assets.value.vae || ''),
    },
    high: {
      modelDir: high.value.modelDir,
      prompt: String(high.value.prompt || ''),
      negativePrompt: String(high.value.negativePrompt || ''),
      sampler: high.value.sampler,
      scheduler: high.value.scheduler,
      steps: high.value.steps,
      cfgScale: high.value.cfgScale,
      seed: high.value.seed,
      loras: normalizeStageLoraList(high.value.loras),
      flowShift: high.value.flowShift,
    },
    low: {
      modelDir: low.value.modelDir,
      prompt: String(low.value.prompt || ''),
      negativePrompt: String(low.value.negativePrompt || ''),
      sampler: low.value.sampler,
      scheduler: low.value.scheduler,
      steps: low.value.steps,
      cfgScale: low.value.cfgScale,
      seed: low.value.seed,
      loras: normalizeStageLoraList(low.value.loras),
      flowShift: low.value.flowShift,
    },
    output: {
      format: video.value.format,
      pixFmt: video.value.pixFmt,
      crf: video.value.crf,
      loopCount: video.value.loopCount,
      pingpong: video.value.pingpong,
      returnFrames: video.value.returnFrames,
    },
    interpolation: {
      targetFps: video.value.interpolationFps,
    },
  }
}

async function copyInfo(): Promise<void> {
  await copyJson(info.value, 'Copied info JSON.')
}

async function copyHistoryParams(item: VideoRunHistoryItem): Promise<void> {
  await copyJson(item.paramsSnapshot, 'Copied history params JSON.')
}

function openHistoryDetails(item: VideoRunHistoryItem): void {
  historyDetailsItem.value = item
  historyDetailsOpen.value = true
}

async function onLoadHistoryDetails(): Promise<void> {
  const item = historyDetailsItem.value
  if (!item) return
  await loadHistory(item.taskId)
}

function onApplyHistoryDetails(): void {
  const item = historyDetailsItem.value
  if (!item) return
  applyHistory(item)
}

async function onCopyHistoryDetails(): Promise<void> {
  const item = historyDetailsItem.value
  if (!item) return
  await copyHistoryParams(item)
}

function applyHistory(item: VideoRunHistoryItem): void {
  const snap = isRecord(item.paramsSnapshot) ? item.paramsSnapshot : {}

  const rawMode = String(snap.mode || '').toLowerCase()
  if (rawMode !== '' && rawMode !== 'txt2vid' && rawMode !== 'img2vid') {
    toast(`Unsupported history mode '${rawMode}'. This run cannot be applied.`)
    return
  }
  const nextMode: 'txt2vid' | 'img2vid' = rawMode === 'img2vid' ? 'img2vid' : 'txt2vid'

  const output = isRecord(snap.output) ? snap.output : {}
  const interpolation = isRecord(snap.interpolation) ? snap.interpolation : {}
  const i2v = isRecord(snap.img2vid) ? snap.img2vid : {}
  const snapshotInitImageName = typeof snap.initImageName === 'string' ? snap.initImageName : ''
  const historyInitImageData = typeof item.initImageData === 'string' ? item.initImageData : ''
  const i2vModeRaw = typeof i2v.mode === 'string' ? i2v.mode : ''
  const hasSnapshotWindowFrames = typeof i2v.windowFrames === 'number' && Number.isFinite(i2v.windowFrames) && Number(i2v.windowFrames) > 0
  if (String(i2vModeRaw || '').trim().toLowerCase() === 'chunk') {
    toast("History snapshot uses removed img2vid_mode='chunk'. Update the snapshot to 'sliding'/'svi2'/'svi2_pro' or 'solo'.")
    return
  }
  if (typeof i2v.enabled === 'boolean' && Boolean(i2v.enabled)) {
    toast("History snapshot uses removed legacy img2vid chunk toggle (img2vid.enabled=true).")
    return
  }
  const nextImg2VidMode = i2vModeRaw
    ? normalizeImg2VidMode(i2vModeRaw)
    : (hasSnapshotWindowFrames
      ? 'sliding'
      : normalizeImg2VidMode(video.value.img2vidMode))
  if (nextMode === 'img2vid' && !historyInitImageData) {
    toast('This history item does not carry the init image bytes anymore. Re-select the init image before applying.')
    return
  }
  setInputMode(nextMode)
  const historyInterpolationFps = (() => {
    if (typeof interpolation.targetFps === 'number' && Number.isFinite(interpolation.targetFps)) {
      return Number(interpolation.targetFps)
    }
    return video.value.interpolationFps
  })()

  setVideo({
    useInitImage: nextMode === 'img2vid',
    initImageData: nextMode === 'img2vid' ? historyInitImageData : '',
    initImageName: nextMode === 'img2vid' ? snapshotInitImageName : '',
    width: Number(snap.width) || video.value.width,
    height: Number(snap.height) || video.value.height,
    frames: Number(snap.frames) || video.value.frames,
    fps: Number(snap.fps) || video.value.fps,
    attentionMode: normalizeAttentionMode(snap.attentionMode),
    img2vidMode: nextImg2VidMode,
    img2vidChunkFrames: typeof i2v.chunkFrames === 'number' && Number.isFinite(i2v.chunkFrames) ? Number(i2v.chunkFrames) : video.value.img2vidChunkFrames,
    img2vidOverlapFrames: typeof i2v.overlapFrames === 'number' && Number.isFinite(i2v.overlapFrames) ? Number(i2v.overlapFrames) : video.value.img2vidOverlapFrames,
    img2vidAnchorAlpha: typeof i2v.anchorAlpha === 'number' && Number.isFinite(i2v.anchorAlpha) ? Number(i2v.anchorAlpha) : video.value.img2vidAnchorAlpha,
    img2vidResetAnchorToBase: typeof i2v.resetAnchorToBase === 'boolean'
      ? Boolean(i2v.resetAnchorToBase)
      : defaultResetAnchorToBase(nextImg2VidMode),
    img2vidChunkSeedMode: normalizeChunkSeedMode(i2v.chunkSeedMode),
    img2vidWindowFrames: typeof i2v.windowFrames === 'number' && Number.isFinite(i2v.windowFrames) ? Number(i2v.windowFrames) : video.value.img2vidWindowFrames,
    img2vidWindowStride: typeof i2v.windowStride === 'number' && Number.isFinite(i2v.windowStride) ? Number(i2v.windowStride) : video.value.img2vidWindowStride,
    img2vidWindowCommitFrames: typeof i2v.windowCommitFrames === 'number' && Number.isFinite(i2v.windowCommitFrames) ? Number(i2v.windowCommitFrames) : video.value.img2vidWindowCommitFrames,
    img2vidImageScale: normalizeWanImg2VidImageScale(i2v.imageScale, video.value.img2vidImageScale),
    img2vidCropOffsetX: normalizeGuideOffset(i2v.cropOffsetX, video.value.img2vidCropOffsetX),
    img2vidCropOffsetY: normalizeGuideOffset(i2v.cropOffsetY, video.value.img2vidCropOffsetY),
    format: String(output.format || video.value.format),
    pixFmt: String(output.pixFmt || video.value.pixFmt),
    crf: typeof output.crf === 'number' && Number.isFinite(output.crf) ? Number(output.crf) : video.value.crf,
    loopCount: typeof output.loopCount === 'number' && Number.isFinite(output.loopCount) ? Number(output.loopCount) : video.value.loopCount,
    pingpong: Boolean(output.pingpong),
    returnFrames: typeof output.returnFrames === 'boolean' ? output.returnFrames : video.value.returnFrames,
    interpolationFps: historyInterpolationFps,
  })

  const hi = isRecord(snap.high) ? snap.high : {}
  const lo = isRecord(snap.low) ? snap.low : {}
  const legacyPrompt = String(snap.prompt || '')
  const legacyNegativePrompt = String(snap.negativePrompt || '')
  const nextHighPrompt = typeof hi.prompt === 'string' ? hi.prompt : legacyPrompt
  const nextHighNegative = typeof hi.negativePrompt === 'string' ? hi.negativePrompt : legacyNegativePrompt
  const nextLowPrompt = typeof lo.prompt === 'string' ? lo.prompt : legacyPrompt
  const nextLowNegative = typeof lo.negativePrompt === 'string' ? lo.negativePrompt : legacyNegativePrompt
  const nextHighLoras = Object.prototype.hasOwnProperty.call(hi, 'loras')
    ? normalizeStageLoraList(hi.loras)
    : normalizeStageLoraList(high.value.loras)
  const nextLowLoras = Object.prototype.hasOwnProperty.call(lo, 'loras')
    ? normalizeStageLoraList(lo.loras)
    : normalizeStageLoraList(low.value.loras)
  const snapLightx2v = typeof snap.lightx2v === 'boolean' ? Boolean(snap.lightx2v) : lightx2v.value
  store.updateParams(props.tabId, { lightx2v: snapLightx2v }).catch(reportTabMutationError)

  const snapAssets = isRecord(snap.assets) ? snap.assets : null
  if (snapAssets) {
    store.updateParams(props.tabId, { assets: { ...assets.value, ...snapAssets } }).catch(reportTabMutationError)
  }

  setHigh({
    modelDir: String(hi.modelDir || ''),
    prompt: String(nextHighPrompt || ''),
    negativePrompt: String(nextHighNegative || ''),
    sampler: String(hi.sampler || ''),
    scheduler: typeof hi.scheduler === 'string' && hi.scheduler.trim() ? hi.scheduler.trim() : high.value.scheduler,
    steps: Number(hi.steps) || high.value.steps,
    cfgScale: typeof hi.cfgScale === 'number' && Number.isFinite(hi.cfgScale) ? Number(hi.cfgScale) : high.value.cfgScale,
    seed: typeof hi.seed === 'number' && Number.isFinite(hi.seed) ? Number(hi.seed) : high.value.seed,
    loras: nextHighLoras,
    flowShift: typeof hi.flowShift === 'number' && Number.isFinite(hi.flowShift) ? Number(hi.flowShift) : high.value.flowShift,
  })

  setLow({
    modelDir: String(lo.modelDir || ''),
    prompt: String(nextLowPrompt || ''),
    negativePrompt: String(nextLowNegative || ''),
    sampler: String(lo.sampler || ''),
    scheduler: typeof lo.scheduler === 'string' && lo.scheduler.trim() ? lo.scheduler.trim() : low.value.scheduler,
    steps: Number(lo.steps) || low.value.steps,
    cfgScale: typeof lo.cfgScale === 'number' && Number.isFinite(lo.cfgScale) ? Number(lo.cfgScale) : low.value.cfgScale,
    seed: typeof lo.seed === 'number' && Number.isFinite(lo.seed) ? Number(lo.seed) : low.value.seed,
    loras: nextLowLoras,
    flowShift: typeof lo.flowShift === 'number' && Number.isFinite(lo.flowShift) ? Number(lo.flowShift) : low.value.flowShift,
  })
  toast('Applied params from history.')
}

function reuseLast(): void {
  if (!history.value.length) return
  applyHistory(history.value[0] as VideoRunHistoryItem)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function formatDiffValue(value: unknown): string {
  if (typeof value === 'string') {
    const v = value.length > 160 ? value.slice(0, 160) + '…' : value
    return JSON.stringify(v)
  }
  if (typeof value === 'number' || typeof value === 'boolean' || value === null || value === undefined) {
    return String(value)
  }
  try {
    const raw = JSON.stringify(value)
    if (raw.length > 180) return raw.slice(0, 180) + '…'
    return raw
  } catch {
    return String(value)
  }
}

function diffObjects(before: unknown, after: unknown, prefix = '', out: Array<{ path: string; before: unknown; after: unknown }> = []): Array<{ path: string; before: unknown; after: unknown }> {
  if (out.length > 80) return out
  if (before === after) return out

  const aObj = isRecord(before)
  const bObj = isRecord(after)
  if (aObj && bObj) {
    const keys = new Set([...Object.keys(before), ...Object.keys(after)])
    for (const k of keys) {
      const nextPrefix = prefix ? `${prefix}.${k}` : k
      diffObjects(before[k], after[k], nextPrefix, out)
      if (out.length > 80) break
    }
    return out
  }

  if (Array.isArray(before) && Array.isArray(after)) {
    const max = Math.max(before.length, after.length)
    for (let i = 0; i < max; i++) {
      const nextPrefix = `${prefix}[${i}]`
      diffObjects(before[i], after[i], nextPrefix, out)
      if (out.length > 80) break
    }
    return out
  }

  out.push({ path: prefix || '(root)', before, after })
  return out
}

const selectedHistoryItem = computed<VideoRunHistoryItem | null>(() => {
  const id = String(selectedTaskId.value || '')
  if (!id) return null
  return (history.value as VideoRunHistoryItem[]).find((h) => h.taskId === id) || null
})

const previousHistoryItem = computed<VideoRunHistoryItem | null>(() => {
  const selected = selectedHistoryItem.value
  if (!selected) return null
  const idx = (history.value as VideoRunHistoryItem[]).findIndex((h) => h.taskId === selected.taskId)
  if (idx < 0) return null
  return (history.value as VideoRunHistoryItem[])[idx + 1] || null
})

const diffText = computed(() => {
  const selected = selectedHistoryItem.value
  const prev = previousHistoryItem.value
  if (!selected || !prev) return ''

  const rows = diffObjects(prev.paramsSnapshot, selected.paramsSnapshot)
  if (!rows.length) return ''

  return rows
    .map((r) => `${r.path}: ${formatDiffValue(r.before)} → ${formatDiffValue(r.after)}`)
    .join('\n')
})

type AspectMode = 'free' | 'current' | 'image' | '16:9' | '1:1' | '9:16' | '4:3' | '3:4'
const aspectMode = ref<AspectMode>('free')
const aspectRatio = ref<number | null>(null)
const initImageAspectRatio = ref<number | null>(null)
let initImageAspectTicket = 0

const dimensionInputStep = computed(() => WAN_DIM_STEP_DEFAULT)

function snapDim(value: number, step: number = WAN_DIM_STEP_DEFAULT): number {
  const safeStep = Math.max(1, Math.trunc(step))
  const v = Number.isFinite(value) ? value : WAN_DIM_MIN
  return Math.min(WAN_DIM_MAX, Math.max(WAN_DIM_MIN, Math.ceil(v / safeStep) * safeStep))
}

function snapDimForAspect(value: number): number {
  return snapDim(value, WAN_DIM_STEP_DEFAULT)
}

function ratioForMode(mode: AspectMode): number | null {
  if (mode === 'current') {
    const w = Number(video.value.width) || 0
    const h = Number(video.value.height) || 0
    return h > 0 ? w / h : null
  }
  if (mode === 'image') return initImageAspectRatio.value
  if (mode === '16:9') return 16 / 9
  if (mode === '1:1') return 1
  if (mode === '9:16') return 9 / 16
  if (mode === '4:3') return 4 / 3
  if (mode === '3:4') return 3 / 4
  return null
}

function onAspectModeChange(e: Event): void {
  const mode = String((e.target as HTMLSelectElement).value || 'free') as AspectMode
  aspectMode.value = mode
  if (mode === 'free') {
    aspectRatio.value = null
    return
  }
  const ratio = ratioForMode(mode)
  aspectRatio.value = ratio
  if (!ratio || ratio <= 0) return

  // For fixed presets, snap the current size into the chosen ratio (preserve width).
  if (mode !== 'current') {
    const w = snapDimForAspect(Number(video.value.width) || WAN_DIM_MIN)
    const h = snapDimForAspect(w / ratio)
    setVideo({ width: w, height: h })
  }
}

function applyWidth(value: number): void {
  const nextW = snapDimForAspect(value)
  const r = aspectRatio.value
  if (r && r > 0) {
    const nextH = snapDimForAspect(nextW / r)
    setVideo({ width: nextW, height: nextH })
    return
  }
  setVideo({ width: nextW })
}

function applyHeight(value: number): void {
  const nextH = snapDimForAspect(value)
  const r = aspectRatio.value
  if (r && r > 0) {
    const nextW = snapDimForAspect(nextH * r)
    setVideo({ width: nextW, height: nextH })
    return
  }
  setVideo({ height: nextH })
}

watch(
  () => video.value.initImageData,
  async (src) => {
    const ticket = ++initImageAspectTicket
    const imageSrc = String(src || '').trim()
    if (!imageSrc) {
      initImageAspectRatio.value = null
      if (aspectMode.value === 'image') {
        aspectMode.value = 'free'
        aspectRatio.value = null
      }
      return
    }

    initImageAspectRatio.value = null
    if (aspectMode.value === 'image') {
      aspectRatio.value = null
    }

    try {
      const { width, height } = await readImageDimensions(imageSrc)
      if (ticket !== initImageAspectTicket) return
      const ratio = width > 0 && height > 0 ? width / height : null
      initImageAspectRatio.value = ratio
      if (aspectMode.value !== 'image') return
      if (!ratio || ratio <= 0) {
        aspectMode.value = 'free'
        aspectRatio.value = null
        return
      }
      aspectRatio.value = ratio
      const w = snapDimForAspect(Number(video.value.width) || WAN_DIM_MIN)
      const h = snapDimForAspect(w / ratio)
      setVideo({ width: w, height: h })
    } catch {
      if (ticket !== initImageAspectTicket) return
      console.warn('[WANTab] Failed to read init image dimensions for Image aspect mode.')
      initImageAspectRatio.value = null
      if (aspectMode.value === 'image') {
        aspectMode.value = 'free'
        aspectRatio.value = null
      }
    }
  },
  { immediate: true },
)

function toDataUrl(image: GeneratedImage): string { return `data:image/${image.format};base64,${image.data}` }

function formatVideoModeLabel(mode: unknown): string {
  const normalized = String(mode ?? '').trim().toLowerCase()
  if (normalized === 'img2vid') return 'Img2Vid'
  if (normalized === 'txt2vid') return 'Txt2Vid'
  return `Unsupported (${normalized || 'unknown'})`
}

function formatHistoryTitle(item: VideoRunHistoryItem): string {
  const dt = new Date(item.createdAtMs || Date.now())
  const hh = dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const label = formatVideoModeLabel(item.mode)
  return `${label} · ${hh}`
}

function readHistorySnapshotText(item: VideoRunHistoryItem, key: string): string {
  const snapshot = item.paramsSnapshot
  if (!snapshot || typeof snapshot !== 'object') return ''
  const value = (snapshot as Record<string, unknown>)[key]
  if (typeof value !== 'string') return ''
  return value.trim()
}

function readHistoryStageSnapshotText(item: VideoRunHistoryItem, stageKey: 'high' | 'low', key: 'prompt' | 'negativePrompt'): string {
  const snapshot = item.paramsSnapshot
  if (!snapshot || typeof snapshot !== 'object') return ''
  const stage = (snapshot as Record<string, unknown>)[stageKey]
  if (!stage || typeof stage !== 'object' || Array.isArray(stage)) return ''
  const value = (stage as Record<string, unknown>)[key]
  if (typeof value !== 'string') return ''
  return value.trim()
}

defineExpose({ generate })

function setHighPromptText(value: string): void {
  setHigh({ prompt: value })
}

function setHighNegativeText(value: string): void {
  setHigh({ negativePrompt: value })
}

function setLowPromptText(value: string): void {
  setLow({ prompt: value })
}

function setLowNegativeText(value: string): void {
  setLow({ negativePrompt: value })
}

const lastHighSeed = ref<number | null>(null)
const lastLowSeed = ref<number | null>(null)

function randomizeHighSeed(): void {
  if (high.value.seed !== -1) lastHighSeed.value = high.value.seed
  setHigh({ seed: -1 })
}

function reuseHighSeed(): void {
  if (lastHighSeed.value !== null) setHigh({ seed: lastHighSeed.value })
}

function randomizeLowSeed(): void {
  if (low.value.seed !== -1) lastLowSeed.value = low.value.seed
  setLow({ seed: -1 })
}

function reuseLowSeed(): void {
  if (lastLowSeed.value !== null) setLow({ seed: lastLowSeed.value })
}

function setShowHighPromptLoraModal(value: boolean): void {
  showHighPromptLoraModal.value = value
}

function setShowLowPromptLoraModal(value: boolean): void {
  showLowPromptLoraModal.value = value
}

function setVideoZoomOpen(value: boolean): void {
  videoZoomOpen.value = value
}

function setHistoryDetailsOpen(value: boolean): void {
  historyDetailsOpen.value = value
}

function setGuidedTooltipEl(element: Element | null): void {
  guidedTooltipEl.value = element instanceof HTMLElement ? element : null
}

const slotProps = computed(() => ({
  tab: tab.value,
  lightx2v: lightx2v.value,
  mode: mode.value,
  video: video.value,
  high: high.value,
  low: low.value,
  assets: assets.value,
  isRunning: isRunning.value,
  highPrompt: highPrompt.value,
  highNegative: highNegative.value,
  lowPrompt: lowPrompt.value,
  lowNegative: lowNegative.value,
  hideHighNegativePrompt: hideHighNegativePrompt.value,
  hideLowNegativePrompt: hideLowNegativePrompt.value,
  showHighPromptLoraModal: showHighPromptLoraModal.value,
  showLowPromptLoraModal: showLowPromptLoraModal.value,
  setShowHighPromptLoraModal,
  setShowLowPromptLoraModal,
  highPromptOpen: highPromptOpen.value,
  lowPromptOpen: lowPromptOpen.value,
  toggleHighPrompt,
  toggleLowPrompt,
  setHighPromptText,
  setHighNegativeText,
  setLowPromptText,
  setLowNegativeText,
  randomizeHighSeed,
  reuseHighSeed,
  canReuseHighSeed: lastHighSeed.value !== null,
  randomizeLowSeed,
  reuseLowSeed,
  canReuseLowSeed: lastLowSeed.value !== null,
  onHighPromptLoraInsert,
  onLowPromptLoraInsert,
  onInitImageFile,
  clearInit,
  onInitImageRejected,
  wanInitImageZoomFrameGuide: wanInitImageZoomFrameGuide.value,
  onZoomFrameGuideUpdate,
  dimensionInputStep: dimensionInputStep.value,
  aspectMode: aspectMode.value,
  initImageAspectRatio: initImageAspectRatio.value,
  onAspectModeChange,
  applyWidth,
  applyHeight,
  setVideo,
  onLoopCountChange,
  onCrfChange,
  onInterpolationTargetFpsChange,
  interpolationCaption: interpolationCaption.value,
  temporalControlsEnabled: temporalControlsEnabled.value,
  temporalEnabledMode: temporalEnabledMode.value,
  setImg2VidTemporalEnabled,
  setImg2VidTemporalMode,
  normalizeImg2VidTemporalEnabledMode,
  normalizeAttentionMode,
  normalizeChunkSeedMode,
  isWindowedTemporalMode,
  maxAlignedWindowStride,
  WAN_WINDOW_COMMIT_OVERLAP_MIN,
  WAN_WINDOW_STRIDE_ALIGNMENT,
  wanStageSamplers: wanStageSamplers.value,
  wanStageSchedulers: wanStageSchedulers.value,
  wanRecommendedSamplers: wanRecommendedSamplers.value,
  wanRecommendedSchedulers: wanRecommendedSchedulers.value,
  setHigh,
  setLow,
  lowFollowsHigh: lowFollowsHigh.value,
  lowNoiseOpen: lowNoiseOpen.value,
  onLowFollowsHighChange,
  toggleLowNoise,
  canRunGeneration: canRunGeneration.value,
  generateTitle: generateTitle.value,
  onGenerateClick,
  cancel,
  history: history.value,
  selectedTaskId: selectedTaskId.value,
  historyLoadingTaskId: historyLoadingTaskId.value,
  clearHistory,
  reuseLast,
  copyNotice: copyNotice.value,
  runSummary: runSummary.value,
  progress: progress.value,
  errorMessage: errorMessage.value,
  framesResult: framesResult.value,
  info: info.value,
  videoUrl: videoUrl.value,
  workflowBusy: workflowBusy.value,
  sendToWorkflows,
  copyCurrentParams,
  copyInfo,
  formatJson,
  toDataUrl,
  openResultVideoZoom,
  videoZoomOpen: videoZoomOpen.value,
  setVideoZoomOpen,
  openHistoryDetails,
  onSelectHistoryStripItem,
  historyDetailsOpen: historyDetailsOpen.value,
  setHistoryDetailsOpen,
  historyDetailsTitle: historyDetailsTitle.value,
  historyDetailsItem: historyDetailsItem.value,
  historyDetailsImageUrl: historyDetailsImageUrl.value,
  historyDetailsModeLabel: historyDetailsModeLabel.value,
  historyDetailsCreatedAtLabel: historyDetailsCreatedAtLabel.value,
  historyDetailsSections: historyDetailsSections.value,
  onLoadHistoryDetails,
  onApplyHistoryDetails,
  onCopyHistoryDetails,
  formatHistoryTitle,
  diffText: diffText.value,
  guidedActive: guidedActive.value,
  guidedRect: guidedRect.value,
  setGuidedTooltipEl,
  guidedTooltipPlacement: guidedTooltipPlacement.value,
  guidedTooltipStyle: guidedTooltipStyle.value,
  guidedMessage: guidedMessage.value,
  stopGuided,
}))

</script>
