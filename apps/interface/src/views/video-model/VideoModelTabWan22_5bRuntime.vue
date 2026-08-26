<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Renderless WAN 2.2 5B runtime helper for the canonical video tab view.
Mounts the exact WAN 2.2 5B single-stage video lane under a view-local seam and exposes reactive slot props to `VideoModelTab.vue`,
while keeping the route-owned video view as the only live body/layout owner and routing shared Results/history actions through the current
workflow save-or-update owner seam.

Symbols (top-level; keep in sync; no ghosts):
- `VideoModelTabWan22_5bRuntime` (component): Renderless WAN 2.2 5B runtime helper for `VideoModelTab.vue`.
- `AspectMode` (type): Aspect ratio mode presets for width/height controls.
- `setPromptText` / `setNegativeText` (functions): Prompt-field bridge setters exposed to the parent slot.
- `setShowPromptLoraModal` / `setVideoZoomOpen` / `setHistoryDetailsOpen` (functions): Parent-facing modal/overlay visibility bridge setters.
- `sendToWorkflows` / `copyCurrentParams` / `applyHistory` (functions): Parent-facing Results/history actions, including truthful save-vs-update Workflow notices.
- `slotProps` (const): Reactive slot-prop bundle exposed to `VideoModelTab.vue`.
-->

<template>
  <slot v-bind="slotProps" />
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, ref, watch } from 'vue'

import { fetchSamplers, fetchSchedulers } from '../../api/client'
import type { GeneratedImage, SamplerInfo, SchedulerInfo } from '../../api/types'
import { useWorkflowSnapshotActions } from '../../composables/useWorkflowSnapshotActions'
import {
  useWan22_5bVideoGeneration,
  type VideoRunHistoryItem,
} from '../../composables/useWan22_5bVideoGeneration'
import { useBootstrapStore } from '../../stores/bootstrap'
import { useEngineCapabilitiesStore } from '../../stores/engine_capabilities'
import {
  useModelTabsStore,
  type TabByType,
  type Wan5bStageParams,
  type Wan5bTabParams,
  type WanAssetsParams,
  type WanVideoParams,
} from '../../stores/model_tabs'
import { readFileAsDataURL, readImageDimensions } from '../../utils/image_io'
import {
  normalizeWanImg2VidImageScale,
  type WanImg2VidFrameGuideConfig,
} from '../../utils/wan_img2vid_frame_projection'
import { normalizeWanImg2VidMode } from '../../utils/wan_img2vid_temporal'

const props = defineProps<{ tabId: string }>()

const store = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()
const bootstrap = useBootstrapStore()

const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])

onMounted(() => {
  bootstrap
    .runRequired('Failed to initialize WAN 2.2 5B tab controls', async () => {
      const [samplerResult, schedulerResult] = await Promise.all([fetchSamplers(), fetchSchedulers()])
      samplers.value = samplerResult.samplers
      schedulers.value = schedulerResult.schedulers
    })
    .catch(() => {
      // Fatal state already handled by bootstrap store.
    })
})

type Wan5bTab = TabByType<'wan22_5b'>
type PromptTokenInsertPayload = {
  token: string
  target?: 'positive' | 'negative'
  action?: 'add' | 'remove'
}
type AspectMode = 'free' | 'current' | 'image' | '16:9' | '1:1' | '9:16' | '4:3' | '3:4'

const WAN_FRAMES_MIN = 9
const WAN_FRAMES_MAX = 401
const WAN_DIM_MIN = 64
const WAN_DIM_MAX = 2048
const WAN_DIM_STEP_DEFAULT = 16

const tab = computed<Wan5bTab | null>(() => {
  const candidate = store.tabs.find((entry) => entry.id === props.tabId) || null
  if (!candidate || candidate.type !== 'wan22_5b') return null
  return candidate as Wan5bTab
})
const params = computed<Wan5bTab['params'] | null>(() => tab.value?.params || null)

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

function defaultStage(): Wan5bStageParams {
  return {
    modelDir: '',
    loras: [],
    flowShift: undefined,
  }
}

function defaultAssets(): WanAssetsParams {
  return { metadata: '', textEncoder: '', vae: '' }
}

const prompt = computed(() => String(params.value?.prompt || ''))
const negativePrompt = computed(() => String(params.value?.negativePrompt || ''))
const sampler = computed(() => String(params.value?.sampler || 'uni-pc bh2'))
const scheduler = computed(() => String(params.value?.scheduler || 'simple'))
const steps = computed(() => Number(params.value?.steps ?? 30))
const cfgScale = computed(() => Number(params.value?.cfgScale ?? 7))
const seed = computed(() => Number(params.value?.seed ?? -1))
const video = computed<WanVideoParams>(() => params.value?.video || defaultVideo())
const stage = computed<Wan5bStageParams>(() => params.value?.stage || defaultStage())
const assets = computed<WanAssetsParams>(() => params.value?.assets || defaultAssets())

const wanInitImageZoomFrameGuide = computed<WanImg2VidFrameGuideConfig>(() => ({
  targetWidth: Number(video.value.width) || WAN_DIM_MIN,
  targetHeight: Number(video.value.height) || WAN_DIM_MIN,
  imageScale: normalizeWanImg2VidImageScale(video.value.img2vidImageScale, 1),
  cropOffsetX: normalizeGuideOffset(video.value.img2vidCropOffsetX, 0.5),
  cropOffsetY: normalizeGuideOffset(video.value.img2vidCropOffsetY, 0.5),
}))

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

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

function snapDim(value: number, step: number = WAN_DIM_STEP_DEFAULT): number {
  const safeStep = Math.max(1, Math.trunc(step))
  const numeric = Number.isFinite(value) ? value : WAN_DIM_MIN
  return Math.min(WAN_DIM_MAX, Math.max(WAN_DIM_MIN, Math.ceil(numeric / safeStep) * safeStep))
}

function snapDimForAspect(value: number): number {
  return snapDim(value, WAN_DIM_STEP_DEFAULT)
}

function normalizeAttentionMode(rawValue: unknown): 'global' | 'sliding' {
  return String(rawValue || '').trim().toLowerCase() === 'sliding' ? 'sliding' : 'global'
}

function normalizeChunkSeedMode(rawValue: unknown): 'fixed' | 'increment' | 'random' {
  const value = String(rawValue || '').trim().toLowerCase()
  if (value === 'fixed' || value === 'random') return value
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

function normalizeLoraSha(rawValue: unknown): string | undefined {
  const normalized = String(rawValue || '').trim().toLowerCase()
  if (!/^[0-9a-f]{64}$/.test(normalized)) return undefined
  return normalized
}

function normalizeStageLoraList(rawValue: unknown): Wan5bStageParams['loras'] {
  if (!Array.isArray(rawValue)) return []

  const normalized: Wan5bStageParams['loras'] = []
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

function normalizeVideoPatch(patch: Partial<WanVideoParams>, current: WanVideoParams): Partial<WanVideoParams> {
  const nextPatch: Partial<WanVideoParams> = { ...patch }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'width')) {
    nextPatch.width = snapDim(Number(nextPatch.width))
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'height')) {
    nextPatch.height = snapDim(Number(nextPatch.height))
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'frames')) {
    nextPatch.frames = normalizeFrameCount(Number(nextPatch.frames))
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'fps')) {
    nextPatch.fps = Math.max(1, Math.trunc(Number(nextPatch.fps) || current.fps || 1))
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'attentionMode')) {
    nextPatch.attentionMode = normalizeAttentionMode(nextPatch.attentionMode)
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'img2vidMode')) {
    nextPatch.img2vidMode = normalizeWanImg2VidMode(nextPatch.img2vidMode)
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
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'interpolationFps')) {
    nextPatch.interpolationFps = normalizeInterpolationTargetFps(nextPatch.interpolationFps, current.interpolationFps)
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'loopCount')) {
    nextPatch.loopCount = normalizeNonNegativeInteger(nextPatch.loopCount, current.loopCount, 32)
  }
  if (Object.prototype.hasOwnProperty.call(nextPatch, 'crf')) {
    nextPatch.crf = normalizeNonNegativeInteger(nextPatch.crf, current.crf, 51)
  }
  return nextPatch
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

function reportTabMutationError(error: unknown): void {
  toast(error instanceof Error ? error.message : String(error))
}

async function updateParamsPatch(patch: Partial<Wan5bTabParams>): Promise<void> {
  if (!tab.value) return
  try {
    await store.updateParams<Wan5bTabParams>(props.tabId, patch)
  } catch (error) {
    reportTabMutationError(error)
  }
}

function setVideo(patch: Partial<WanVideoParams>): void {
  if (!tab.value) return
  const current = tab.value.params.video
  const normalizedPatch = normalizeVideoPatch(patch, current)
  store.updateParams<Wan5bTabParams>(props.tabId, {
    video: { ...current, ...normalizedPatch },
  }).catch(reportTabMutationError)
}

function setStage(patch: Partial<Wan5bStageParams>): void {
  if (!tab.value) return
  const current = tab.value.params.stage
  store.updateParams<Wan5bTabParams>(props.tabId, {
    stage: { ...current, ...patch },
  }).catch(reportTabMutationError)
}

function setPromptText(value: string): void {
  void updateParamsPatch({ prompt: value })
}

function setNegativeText(value: string): void {
  void updateParamsPatch({ negativePrompt: value })
}

function setSamplerValue(value: string): void {
  void updateParamsPatch({ sampler: value })
}

function setSchedulerValue(value: string): void {
  void updateParamsPatch({ scheduler: value })
}

function setStepsValue(value: number): void {
  void updateParamsPatch({ steps: Math.max(1, Math.trunc(value)) })
}

function setCfgScaleValue(value: number): void {
  void updateParamsPatch({ cfgScale: Number.isFinite(value) ? value : cfgScale.value })
}

function setSeedValue(value: number): void {
  void updateParamsPatch({ seed: Math.trunc(value) })
}

function randomizeSeed(): void {
  const nextSeed = Math.floor(Math.random() * 2_147_483_647)
  void updateParamsPatch({ seed: nextSeed })
}

const lastSeed = ref<number | null>(null)

function reuseSeed(): void {
  if (lastSeed.value === null) return
  void updateParamsPatch({ seed: lastSeed.value })
}

function setInputMode(next: 'txt2vid' | 'img2vid'): void {
  if (next === 'txt2vid') {
    setVideo({ useInitImage: false, initImageData: '', initImageName: '' })
    return
  }
  setVideo({ useInitImage: true, img2vidMode: 'solo' })
}

const promptOpen = ref(true)
const showPromptLoraModal = ref(false)
const hideNegativePrompt = computed(() => {
  const value = Number(cfgScale.value)
  return Number.isFinite(value) && value === 1
})

function onPromptLoraInsert(payload: PromptTokenInsertPayload): void {
  const target = payload.target === 'negative' ? 'negative' : 'positive'
  const action = payload.action === 'remove' ? 'remove' : 'add'
  if (target === 'negative') {
    const current = negativePrompt.value
    const next = action === 'remove' ? removePromptToken(current, payload.token) : appendPromptToken(current, payload.token)
    setNegativeText(next)
    return
  }
  const current = prompt.value
  const next = action === 'remove' ? removePromptToken(current, payload.token) : appendPromptToken(current, payload.token)
  setPromptText(next)
}

async function onInitImageFile(file: File): Promise<void> {
  const dataUrl = await readFileAsDataURL(file)
  setVideo({ initImageData: dataUrl, initImageName: file.name, useInitImage: true })
}

function clearInit(): void {
  setVideo({ initImageData: '', initImageName: '' })
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

const {
  generate,
  isRunning,
  canGenerate,
  blockedReason,
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
} = useWan22_5bVideoGeneration(props.tabId)

const wanDependencyStatus = computed(() => engineCaps.getDependencyStatus('wan22_5b'))
const wanDependencyReady = computed(() => Boolean(wanDependencyStatus.value?.ready))
const wanDependencyError = computed(() => engineCaps.firstDependencyError('wan22_5b'))
const wanEngineSurface = computed(() => engineCaps.get('wan22_5b'))
const wanStageSamplers = computed(() => {
  const allowedExact = new Set(['euler', 'euler a'])
  return samplers.value.filter((entry) => {
    const normalizedName = String(entry.name || '').trim().toLowerCase()
    if (!normalizedName) return false
    if (normalizedName.startsWith('uni-pc')) return true
    return allowedExact.has(normalizedName)
  })
})
const wanStageSchedulers = computed(() => schedulers.value.filter((entry) => String(entry.name || '').trim() === 'simple'))
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
  return blockedReason.value || ''
})

const interpolationCaption = computed<string>(() => {
  const targetFps = normalizeInterpolationTargetFps(video.value.interpolationFps, 0)
  const baseFps = Math.max(1, Math.trunc(Number(video.value.fps) || 1))
  if (targetFps <= 0) return 'Disabled'
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
  const snappedWidth = snapDimForAspect(video.value.width)
  const snappedHeight = snapDimForAspect(video.value.height)
  const snappedFrames = normalizeFrameCount(video.value.frames)
  const nextPatch: Partial<WanVideoParams> = {}
  if (snappedWidth !== video.value.width) nextPatch.width = snappedWidth
  if (snappedHeight !== video.value.height) nextPatch.height = snappedHeight
  if (snappedFrames !== video.value.frames) nextPatch.frames = snappedFrames
  if (Object.keys(nextPatch).length > 0) setVideo(nextPatch)
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
    toast(blockedReason.value || 'WAN 2.2 5B is not ready to generate yet.')
    return
  }
  normalizeVideoBeforeSubmit()
  await generate()
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

const durationLabel = computed(() => {
  const fpsValue = Number(video.value.fps) || 0
  const frameValue = Number(video.value.frames) || 0
  if (fpsValue <= 0) return '0.00'
  return (frameValue / fpsValue).toFixed(2)
})

const modeLabel = computed(() => (mode.value === 'img2vid' ? 'Img2Vid' : 'Txt2Vid'))
const runSummary = computed(() => {
  const currentVideo = video.value
  const currentStage = stage.value
  return `${modeLabel.value} · ${currentVideo.width}×${currentVideo.height} px · ${currentVideo.frames} frames @ ${currentVideo.fps} fps (~ ${durationLabel.value}s) · ${steps.value} steps · CFG ${cfgScale.value} · ${sampler.value}${currentStage.modelDir ? ' · model selected' : ''}`
})

function buildCurrentSnapshot(): Record<string, unknown> {
  return {
    mode: mode.value,
    initImageName: video.value.initImageName || '',
    prompt: prompt.value,
    negativePrompt: negativePrompt.value,
    sampler: sampler.value,
    scheduler: scheduler.value,
    steps: steps.value,
    cfgScale: cfgScale.value,
    seed: seed.value,
    width: video.value.width,
    height: video.value.height,
    frames: video.value.frames,
    fps: video.value.fps,
    attentionMode: video.value.attentionMode,
    img2vid: {
      mode: video.value.img2vidMode,
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
    assets: {
      metadata: String(assets.value.metadata || ''),
      textEncoder: String(assets.value.textEncoder || ''),
      vae: String(assets.value.vae || ''),
    },
    stage: {
      modelDir: stage.value.modelDir,
      loras: normalizeStageLoraList(stage.value.loras),
      flowShift: stage.value.flowShift,
    },
    output: {
      format: video.value.format,
      pixFmt: video.value.pixFmt,
      crf: video.value.crf,
      loopCount: video.value.loopCount,
      pingpong: Boolean(video.value.pingpong),
      returnFrames: Boolean(video.value.returnFrames),
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

const historyDetailsOpen = ref(false)
const historyDetailsItem = ref<VideoRunHistoryItem | null>(null)

function openHistoryDetails(item: VideoRunHistoryItem): void {
  historyDetailsItem.value = item
  historyDetailsOpen.value = true
}

function onSelectHistoryStripItem(item: { taskId: string }): void {
  const match = history.value.find((entry) => entry.taskId === item.taskId)
  if (!match) return
  openHistoryDetails(match)
}

const historyDetailsTitle = computed(() => (historyDetailsItem.value ? formatHistoryTitle(historyDetailsItem.value) : 'History details'))
const historyDetailsCreatedAtLabel = computed(() => {
  const timestamp = historyDetailsItem.value?.createdAtMs
  if (!timestamp) return '—'
  return new Date(timestamp).toLocaleString()
})
const historyDetailsModeLabel = computed(() => formatVideoModeLabel(historyDetailsItem.value?.mode))
const historyDetailsImageUrl = computed(() => {
  const thumbnail = historyDetailsItem.value?.thumbnail
  return thumbnail ? toDataUrl(thumbnail) : ''
})
const historyDetailsPrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  const promptText = readHistorySnapshotText(item, 'prompt')
  if (promptText) return promptText
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

async function onLoadHistoryDetails(): Promise<void> {
  const item = historyDetailsItem.value
  if (!item) return
  await loadHistory(item.taskId)
}

async function applyHistory(item: VideoRunHistoryItem): Promise<void> {
  const snapshot = isRecord(item.paramsSnapshot) ? item.paramsSnapshot : {}
  const rawMode = String(snapshot.mode || '').trim().toLowerCase()
  if (rawMode !== '' && rawMode !== 'txt2vid' && rawMode !== 'img2vid') {
    toast(`Unsupported history mode '${rawMode}'. This run cannot be applied.`)
    return
  }
  const nextMode: 'txt2vid' | 'img2vid' = rawMode === 'img2vid' ? 'img2vid' : 'txt2vid'
  const output = isRecord(snapshot.output) ? snapshot.output : {}
  const interpolation = isRecord(snapshot.interpolation) ? snapshot.interpolation : {}
  const img2vid = isRecord(snapshot.img2vid) ? snapshot.img2vid : {}
  const snapshotInitImageName = typeof snapshot.initImageName === 'string' ? snapshot.initImageName : ''
  const historyInitImageData = typeof item.initImageData === 'string' ? item.initImageData : ''
  const snapshotAssets = isRecord(snapshot.assets) ? snapshot.assets : {}
  const snapshotStage = isRecord(snapshot.stage) ? snapshot.stage : {}
  let nextImg2VidMode = video.value.img2vidMode
  try {
    nextImg2VidMode = normalizeWanImg2VidMode(img2vid.mode)
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
    return
  }
  if (nextMode === 'img2vid' && !historyInitImageData) {
    toast('This history item does not carry the init image bytes anymore. Re-select the init image before applying.')
    return
  }

  const nextVideo: WanVideoParams = {
    ...video.value,
    useInitImage: nextMode === 'img2vid',
    initImageData: nextMode === 'img2vid' ? historyInitImageData : '',
    initImageName: nextMode === 'img2vid' ? snapshotInitImageName : '',
    width: Number(snapshot.width) || video.value.width,
    height: Number(snapshot.height) || video.value.height,
    frames: Number(snapshot.frames) || video.value.frames,
    fps: Number(snapshot.fps) || video.value.fps,
    attentionMode: normalizeAttentionMode(snapshot.attentionMode),
    img2vidMode: nextImg2VidMode,
    img2vidAnchorAlpha: typeof img2vid.anchorAlpha === 'number' && Number.isFinite(img2vid.anchorAlpha)
      ? Number(img2vid.anchorAlpha)
      : video.value.img2vidAnchorAlpha,
    img2vidResetAnchorToBase: typeof img2vid.resetAnchorToBase === 'boolean'
      ? Boolean(img2vid.resetAnchorToBase)
      : video.value.img2vidResetAnchorToBase,
    img2vidChunkSeedMode: normalizeChunkSeedMode(img2vid.chunkSeedMode),
    img2vidWindowFrames: typeof img2vid.windowFrames === 'number' && Number.isFinite(img2vid.windowFrames)
      ? Number(img2vid.windowFrames)
      : video.value.img2vidWindowFrames,
    img2vidWindowStride: typeof img2vid.windowStride === 'number' && Number.isFinite(img2vid.windowStride)
      ? Number(img2vid.windowStride)
      : video.value.img2vidWindowStride,
    img2vidWindowCommitFrames: typeof img2vid.windowCommitFrames === 'number' && Number.isFinite(img2vid.windowCommitFrames)
      ? Number(img2vid.windowCommitFrames)
      : video.value.img2vidWindowCommitFrames,
    img2vidImageScale: normalizeWanImg2VidImageScale(img2vid.imageScale, video.value.img2vidImageScale),
    img2vidCropOffsetX: normalizeGuideOffset(img2vid.cropOffsetX, video.value.img2vidCropOffsetX),
    img2vidCropOffsetY: normalizeGuideOffset(img2vid.cropOffsetY, video.value.img2vidCropOffsetY),
    format: String(output.format || video.value.format),
    pixFmt: String(output.pixFmt || video.value.pixFmt),
    crf: typeof output.crf === 'number' && Number.isFinite(output.crf) ? Number(output.crf) : video.value.crf,
    loopCount: typeof output.loopCount === 'number' && Number.isFinite(output.loopCount) ? Number(output.loopCount) : video.value.loopCount,
    pingpong: typeof output.pingpong === 'boolean' ? output.pingpong : video.value.pingpong,
    returnFrames: typeof output.returnFrames === 'boolean' ? output.returnFrames : video.value.returnFrames,
    interpolationFps: typeof interpolation.targetFps === 'number' && Number.isFinite(interpolation.targetFps)
      ? Number(interpolation.targetFps)
      : video.value.interpolationFps,
  }

  const nextStage: Wan5bStageParams = {
    modelDir: typeof snapshotStage.modelDir === 'string' ? snapshotStage.modelDir : stage.value.modelDir,
    loras: Object.prototype.hasOwnProperty.call(snapshotStage, 'loras')
      ? normalizeStageLoraList(snapshotStage.loras)
      : normalizeStageLoraList(stage.value.loras),
  }
  if (typeof snapshotStage.flowShift === 'number' && Number.isFinite(snapshotStage.flowShift)) {
    nextStage.flowShift = Number(snapshotStage.flowShift)
  } else if (typeof stage.value.flowShift === 'number' && Number.isFinite(stage.value.flowShift)) {
    nextStage.flowShift = stage.value.flowShift
  }

  await updateParamsPatch({
    prompt: typeof snapshot.prompt === 'string' ? snapshot.prompt : prompt.value,
    negativePrompt: typeof snapshot.negativePrompt === 'string' ? snapshot.negativePrompt : negativePrompt.value,
    sampler: typeof snapshot.sampler === 'string' && snapshot.sampler.trim() ? snapshot.sampler.trim() : sampler.value,
    scheduler: typeof snapshot.scheduler === 'string' && snapshot.scheduler.trim() ? snapshot.scheduler.trim() : scheduler.value,
    steps: typeof snapshot.steps === 'number' && Number.isFinite(snapshot.steps) ? Math.max(1, Math.trunc(snapshot.steps)) : steps.value,
    cfgScale: typeof snapshot.cfgScale === 'number' && Number.isFinite(snapshot.cfgScale) ? snapshot.cfgScale : cfgScale.value,
    seed: typeof snapshot.seed === 'number' && Number.isFinite(snapshot.seed) ? Math.trunc(snapshot.seed) : seed.value,
    video: nextVideo,
    stage: nextStage,
    assets: {
      ...assets.value,
      ...(typeof snapshotAssets.metadata === 'string' ? { metadata: snapshotAssets.metadata } : {}),
      ...(typeof snapshotAssets.textEncoder === 'string' ? { textEncoder: snapshotAssets.textEncoder } : {}),
      ...(typeof snapshotAssets.vae === 'string' ? { vae: snapshotAssets.vae } : {}),
    },
  })
  toast('Applied params from history.')
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

async function reuseLast(): Promise<void> {
  const item = history.value[0]
  if (!item) return
  await applyHistory(item)
}

function formatDiffValue(value: unknown): string {
  if (typeof value === 'string') {
    const trimmed = value.length > 160 ? `${value.slice(0, 160)}…` : value
    return JSON.stringify(trimmed)
  }
  if (typeof value === 'number' || typeof value === 'boolean' || value === null || value === undefined) {
    return String(value)
  }
  try {
    const raw = JSON.stringify(value)
    if (raw.length > 180) return `${raw.slice(0, 180)}…`
    return raw
  } catch {
    return String(value)
  }
}

function diffObjects(
  before: unknown,
  after: unknown,
  prefix = '',
  out: Array<{ path: string; before: unknown; after: unknown }> = [],
): Array<{ path: string; before: unknown; after: unknown }> {
  if (out.length > 80) return out
  if (before === after) return out

  const beforeRecord = isRecord(before)
  const afterRecord = isRecord(after)
  if (beforeRecord && afterRecord) {
    const keys = new Set([...Object.keys(before), ...Object.keys(after)])
    for (const key of keys) {
      const nextPrefix = prefix ? `${prefix}.${key}` : key
      diffObjects(before[key], after[key], nextPrefix, out)
      if (out.length > 80) break
    }
    return out
  }

  if (Array.isArray(before) && Array.isArray(after)) {
    const max = Math.max(before.length, after.length)
    for (let index = 0; index < max; index += 1) {
      const nextPrefix = `${prefix}[${index}]`
      diffObjects(before[index], after[index], nextPrefix, out)
      if (out.length > 80) break
    }
    return out
  }

  out.push({ path: prefix || '(root)', before, after })
  return out
}

const selectedHistoryItem = computed<VideoRunHistoryItem | null>(() => {
  const taskId = String(selectedTaskId.value || '')
  if (!taskId) return null
  return history.value.find((entry) => entry.taskId === taskId) || null
})

const previousHistoryItem = computed<VideoRunHistoryItem | null>(() => {
  const selected = selectedHistoryItem.value
  if (!selected) return null
  const index = history.value.findIndex((entry) => entry.taskId === selected.taskId)
  if (index < 0) return null
  return history.value[index + 1] || null
})

const diffText = computed(() => {
  const selected = selectedHistoryItem.value
  const previous = previousHistoryItem.value
  if (!selected || !previous) return ''
  const rows = diffObjects(previous.paramsSnapshot, selected.paramsSnapshot)
  if (!rows.length) return ''
  return rows.map((row) => `${row.path}: ${formatDiffValue(row.before)} → ${formatDiffValue(row.after)}`).join('\n')
})

const aspectMode = ref<AspectMode>('free')
const aspectRatio = ref<number | null>(null)
const initImageAspectRatio = ref<number | null>(null)
let initImageAspectTicket = 0

const dimensionInputStep = computed(() => WAN_DIM_STEP_DEFAULT)

function ratioForMode(modeValue: AspectMode): number | null {
  if (modeValue === 'current') {
    const width = Number(video.value.width) || 0
    const height = Number(video.value.height) || 0
    return height > 0 ? width / height : null
  }
  if (modeValue === 'image') return initImageAspectRatio.value
  if (modeValue === '16:9') return 16 / 9
  if (modeValue === '1:1') return 1
  if (modeValue === '9:16') return 9 / 16
  if (modeValue === '4:3') return 4 / 3
  if (modeValue === '3:4') return 3 / 4
  return null
}

function onAspectModeChange(event: Event): void {
  const nextMode = String((event.target as HTMLSelectElement).value || 'free') as AspectMode
  aspectMode.value = nextMode
  if (nextMode === 'free') {
    aspectRatio.value = null
    return
  }
  const ratio = ratioForMode(nextMode)
  aspectRatio.value = ratio
  if (!ratio || ratio <= 0) return
  if (nextMode !== 'current') {
    const width = snapDimForAspect(Number(video.value.width) || WAN_DIM_MIN)
    const height = snapDimForAspect(width / ratio)
    setVideo({ width, height })
  }
}

function applyWidth(value: number): void {
  const nextWidth = snapDimForAspect(value)
  const ratio = aspectRatio.value
  if (ratio && ratio > 0) {
    const nextHeight = snapDimForAspect(nextWidth / ratio)
    setVideo({ width: nextWidth, height: nextHeight })
    return
  }
  setVideo({ width: nextWidth })
}

function applyHeight(value: number): void {
  const nextHeight = snapDimForAspect(value)
  const ratio = aspectRatio.value
  if (ratio && ratio > 0) {
    const nextWidth = snapDimForAspect(nextHeight * ratio)
    setVideo({ width: nextWidth, height: nextHeight })
    return
  }
  setVideo({ height: nextHeight })
}

watch(
  () => video.value.initImageData,
  async (source) => {
    const ticket = ++initImageAspectTicket
    const imageSource = String(source || '').trim()
    if (!imageSource) {
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
      const { width, height } = await readImageDimensions(imageSource)
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
      const nextWidth = snapDimForAspect(Number(video.value.width) || WAN_DIM_MIN)
      const nextHeight = snapDimForAspect(nextWidth / ratio)
      setVideo({ width: nextWidth, height: nextHeight })
    } catch {
      if (ticket !== initImageAspectTicket) return
      console.warn('[WAN22_5bTab] Failed to read init image dimensions for Image aspect mode.')
      initImageAspectRatio.value = null
      if (aspectMode.value === 'image') {
        aspectMode.value = 'free'
        aspectRatio.value = null
      }
    }
  },
  { immediate: true },
)

function toDataUrl(image: GeneratedImage): string {
  return `data:image/${image.format};base64,${image.data}`
}

function formatVideoModeLabel(modeValue: unknown): string {
  const normalized = String(modeValue ?? '').trim().toLowerCase()
  if (normalized === 'img2vid') return 'Img2Vid'
  if (normalized === 'txt2vid') return 'Txt2Vid'
  return `Unsupported (${normalized || 'unknown'})`
}

function formatHistoryTitle(item: VideoRunHistoryItem): string {
  const timestamp = new Date(item.createdAtMs || Date.now())
  const hh = timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
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

function setShowPromptLoraModal(value: boolean): void {
  showPromptLoraModal.value = value
}

function setVideoZoomOpen(value: boolean): void {
  videoZoomOpen.value = value
}

function setHistoryDetailsOpen(value: boolean): void {
  historyDetailsOpen.value = value
}

defineExpose({ generate })

const slotProps = computed(() => ({
  tab: tab.value,
  params: params.value,
  mode: mode.value,
  modeLabel: modeLabel.value,
  prompt: prompt.value,
  negativePrompt: negativePrompt.value,
  hideNegativePrompt: hideNegativePrompt.value,
  promptOpen: promptOpen.value,
  togglePrompt: (value: boolean) => { promptOpen.value = value },
  setPromptText,
  setNegativeText,
  showPromptLoraModal: showPromptLoraModal.value,
  setShowPromptLoraModal,
  onPromptLoraInsert,
  video: video.value,
  stage: stage.value,
  assets: assets.value,
  sampler: sampler.value,
  scheduler: scheduler.value,
  steps: steps.value,
  cfgScale: cfgScale.value,
  seed: seed.value,
  setVideo,
  setStage,
  setSamplerValue,
  setSchedulerValue,
  setStepsValue,
  setCfgScaleValue,
  setSeedValue,
  randomizeSeed,
  reuseSeed,
  canReuseSeed: lastSeed.value !== null,
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
  onLoopCountChange,
  onCrfChange,
  onInterpolationTargetFpsChange,
  interpolationCaption: interpolationCaption.value,
  isRunning: isRunning.value,
  canRunGeneration: canRunGeneration.value,
  generateTitle: generateTitle.value,
  onGenerateClick,
  cancel,
  copyNotice: copyNotice.value,
  resumeNotice: resumeNotice.value,
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
  history: history.value,
  selectedTaskId: selectedTaskId.value,
  historyLoadingTaskId: historyLoadingTaskId.value,
  clearHistory,
  reuseLast,
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
  wanStageSamplers: wanStageSamplers.value,
  wanStageSchedulers: wanStageSchedulers.value,
  wanRecommendedSamplers: wanRecommendedSamplers.value,
  wanRecommendedSchedulers: wanRecommendedSchedulers.value,
  setInputMode,
}))
</script>
