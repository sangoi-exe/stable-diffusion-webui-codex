/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 5B video generation composable (txt2vid/img2vid).
Owns per-tab WAN 2.2 5B runtime state (progress/results/history), builds the exact `wan_single` payload contract,
starts tasks, and consumes task SSE events while delegating shared resume/history plumbing to `useTaskRunLifecycle.ts`.
The 5B contract keeps prompt/sampler/steps/cfg/seed on the top-level request owner and uses `wan_single` only for the
single-stage selector payload (`model_sha`, `loras`, `flow_shift`).

Symbols (top-level; keep in sync; no ghosts):
- `Status` (type): Video generation status state (`idle|running|error|done`).
- `VideoMode` (type): Supported WAN 2.2 5B video modes (`txt2vid|img2vid`).
- `VideoRunStatus` (type): Terminal status for history entries (`completed|error|cancelled`).
- `VideoRunHistoryItem` (interface): Persisted run history entry for the 5B lane.
- `PreparedWan22_5bRun` (type): Prepared 5B txt2vid/img2vid request with exact payload contract.
- `VideoProgressState` (interface): Progress payload shape consumed from task SSE.
- `VideoGenerationState` (interface): Per-tab runtime state for WAN 2.2 5B.
- `useWan22_5bVideoGeneration` (function): Main 5B composable API.
*/

import { computed, ref } from 'vue'

import { cancelTask, fetchTaskResult, getApiErrorStatus, startImg2Vid, startTxt2Vid } from '../api/client'
import { formatZodError } from '../api/payloads'
import {
  buildWan22_5bImg2VidPayload,
  buildWan22_5bTxt2VidPayload,
  type Wan22_5bImg2VidInput,
  type Wan22_5bImg2VidPayload,
  type Wan22_5bStageInput,
  type Wan22_5bTxt2VidPayload,
  type Wan22_5bVideoCommonInput,
} from '../api/payloads_wan22_5b_video'
import type { GeneratedImage, TaskErrorCode, TaskEvent } from '../api/types'
import { useQuicksettingsStore } from '../stores/quicksettings'
import {
  useModelTabsStore,
  type TabByType,
  type Wan5bStageParams,
  type WanAssetsParams,
  type WanVideoParams,
} from '../stores/model_tabs'
import { useTaskRunLifecycle } from './useTaskRunLifecycle'
import {
  isWanWindowedImg2VidMode,
  normalizeWanImg2VidMode,
} from '../utils/wan_img2vid_temporal'
import { normalizeWanImg2VidImageScale } from '../utils/wan_img2vid_frame_projection'
import {
  formatSettingsRevisionConflictMessage,
  resolveSettingsRevisionConflict,
} from './settings_revision_conflict'

type Status = 'idle' | 'running' | 'error' | 'done'
type VideoMode = 'txt2vid' | 'img2vid'
type VideoRunStatus = 'completed' | 'error' | 'cancelled'

export interface VideoRunHistoryItem {
  taskId: string
  mode: VideoMode
  createdAtMs: number
  status: VideoRunStatus
  summary: string
  promptPreview: string
  paramsSnapshot: Record<string, unknown>
  initImageData?: string
  thumbnail: GeneratedImage | null
  errorMessage?: string
}

export type PreparedWan22_5bRun =
  | {
      mode: 'txt2vid'
      createdAtMs: number
      summary: string
      promptPreview: string
      paramsSnapshot: Record<string, unknown>
      payload: Wan22_5bTxt2VidPayload
    }
  | {
      mode: 'img2vid'
      createdAtMs: number
      summary: string
      promptPreview: string
      paramsSnapshot: Record<string, unknown>
      initImageData: string
      payload: Wan22_5bImg2VidPayload
    }

function assertRunPayloadObject(payload: unknown, mode: VideoMode): asserts payload is Record<string, unknown> {
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`useWan22_5bVideoGeneration: invalid payload for mode '${mode}'.`)
  }
}

export interface VideoProgressState {
  stage: string
  percent: number | null
  etaSeconds: number | null
  step: number | null
  totalSteps: number | null
}

export interface VideoGenerationState {
  status: Status
  progress: VideoProgressState
  frames: GeneratedImage[]
  info: unknown | null
  video: { rel_path?: string | null; mime?: string | null } | null
  errorMessage: string
  taskId: string
  cancelRequested: boolean
  currentRun: VideoRunHistoryItem | null
  history: VideoRunHistoryItem[]
  selectedTaskId: string
  historyLoadingTaskId: string
}

const DEFAULT_PROGRESS: VideoProgressState = { stage: 'idle', percent: null, etaSeconds: null, step: null, totalSteps: null }
const MAX_HISTORY = 8
const WAN_LORA_TAG_RE = /<\s*lora\s*:\s*([^:>]+)\s*(?::\s*([^>]*))?\s*>/gi
const WAN22_5B_METADATA_REPO = 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'

const tabStates = new Map<string, VideoGenerationState>()
const unsubscribers = new Map<string, () => void>()
const resumeAttempts = new Set<string>()
const resumeToastShown = new Set<string>()

type ResumeState = {
  taskId: string
  lastEventId: number
  createdAtMs: number
  mode: VideoMode
  summary: string
  promptPreview: string
  paramsSnapshot: Record<string, unknown>
}

type ResumeStateLoad = {
  state: ResumeState | null
  error: string | null
}

function resumeKey(tabId: string): string {
  return `codex.resume.wan22_5b.${tabId}`
}

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function parseResumeMode(value: unknown): VideoMode | null {
  const normalized = String(value ?? '').trim().toLowerCase()
  if (normalized === 'txt2vid' || normalized === 'img2vid') return normalized
  return null
}

function loadResumeState(key: string): ResumeStateLoad {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return { state: null, error: null }
    const parsed: unknown = JSON.parse(raw)
    if (!isRecordObject(parsed)) return { state: null, error: null }
    if (typeof parsed.taskId !== 'string' || !parsed.taskId.trim()) return { state: null, error: null }
    const taskId = String(parsed.taskId).trim()
    const lastEventId = typeof parsed.lastEventId === 'number' && Number.isFinite(parsed.lastEventId) ? Math.trunc(parsed.lastEventId) : 0
    const createdAtMs = typeof parsed.createdAtMs === 'number' && Number.isFinite(parsed.createdAtMs) ? Math.trunc(parsed.createdAtMs) : 0
    const summary = typeof parsed.summary === 'string' ? parsed.summary : ''
    const promptPreview = typeof parsed.promptPreview === 'string' ? parsed.promptPreview : ''
    const paramsSnapshot = isRecordObject(parsed.paramsSnapshot) ? parsed.paramsSnapshot : {}
    const mode = parseResumeMode(parsed.mode)
    if (!mode) {
      const modeLabel = String(parsed.mode ?? '').trim() || 'unknown'
      return {
        state: null,
        error: `Unsupported resume mode '${modeLabel}'. Resume is disabled for this mode.`,
      }
    }
    return {
      state: {
        taskId,
        lastEventId: Math.max(0, lastEventId),
        createdAtMs,
        mode,
        summary,
        promptPreview,
        paramsSnapshot,
      },
      error: null,
    }
  } catch {
    return { state: null, error: null }
  }
}

function saveResumeState(key: string, state: ResumeState): void {
  try {
    localStorage.setItem(key, JSON.stringify(state))
  } catch {
    // ignore localStorage failures
  }
}

function clearResumeState(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    // ignore
  }
}

function updateResumeEventId(key: string, eventId: number): void {
  const value = Math.trunc(Number(eventId))
  if (!Number.isFinite(value) || value <= 0) return
  const current = loadResumeState(key).state
  if (!current || value <= current.lastEventId) return
  saveResumeState(key, { ...current, lastEventId: value })
}

function freshState(): VideoGenerationState {
  return {
    status: 'idle',
    progress: { ...DEFAULT_PROGRESS },
    frames: [],
    info: null,
    video: null,
    errorMessage: '',
    taskId: '',
    cancelRequested: false,
    currentRun: null,
    history: [],
    selectedTaskId: '',
    historyLoadingTaskId: '',
  }
}

function getTabState(tabId: string): VideoGenerationState {
  let current = tabStates.get(tabId)
  if (!current) {
    current = freshState()
    tabStates.set(tabId, current)
  }
  return current
}

function defaultStage(): Wan5bStageParams {
  return {
    modelDir: '',
    loras: [],
    flowShift: undefined,
  }
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

function defaultAssets(): WanAssetsParams {
  return { metadata: '', textEncoder: '', vae: '' }
}

function normalizeImg2VidMode(rawValue: unknown): WanVideoParams['img2vidMode'] {
  return normalizeWanImg2VidMode(rawValue)
}

function normalizeGuideOffset(rawValue: unknown): number {
  if (rawValue === undefined || rawValue === null || rawValue === '') return 0.5
  if (typeof rawValue === 'boolean') {
    throw new Error(`useWan22_5bVideoGeneration: img2vid crop offset must be a finite number in [0,1] (got ${String(rawValue)}).`)
  }
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) {
    throw new Error(`useWan22_5bVideoGeneration: img2vid crop offset must be finite in [0,1] (got ${String(rawValue)}).`)
  }
  if (numeric < 0 || numeric > 1) {
    throw new Error(`useWan22_5bVideoGeneration: img2vid crop offset must be in [0,1] (got ${String(rawValue)}).`)
  }
  return numeric
}

function resolveBooleanWithDefault(rawValue: unknown, fallback: boolean): boolean {
  return typeof rawValue === 'boolean' ? rawValue : fallback
}

export function useWan22_5bVideoGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()

  const state = ref(getTabState(tabId))
  const resumeNotice = ref('')

  type WanTab = TabByType<'wan22_5b'>

  const tab = computed<WanTab | null>(() => {
    const candidate = modelTabs.tabs.find((entry) => entry.id === tabId) || null
    if (!candidate || candidate.type !== 'wan22_5b') return null
    return candidate as WanTab
  })
  const params = computed<WanTab['params'] | null>(() => tab.value?.params || null)

  const prompt = computed(() => String(params.value?.prompt || ''))
  const negativePrompt = computed(() => String(params.value?.negativePrompt || ''))
  const video = computed<WanVideoParams>(() => params.value?.video || defaultVideo())
  const stage = computed<Wan5bStageParams>(() => params.value?.stage || defaultStage())
  const assets = computed<WanAssetsParams>(() => params.value?.assets || defaultAssets())
  const sampler = computed(() => String(params.value?.sampler || 'uni-pc bh2'))
  const scheduler = computed(() => String(params.value?.scheduler || 'simple'))
  const steps = computed(() => Number(params.value?.steps ?? 30))
  const cfgScale = computed(() => Number(params.value?.cfgScale ?? 7))
  const seed = computed(() => Number(params.value?.seed ?? -1))
  const mode = computed<VideoMode>(() => (video.value.useInitImage ? 'img2vid' : 'txt2vid'))

  function normalizeWanMetadataRepo(raw: string): string | null {
    const value = String(raw || '').trim()
    if (!value) return null
    if (value.startsWith('/') || value.includes('\\') || value.includes(':')) return null
    if (!value.includes('/')) return null
    return value
  }

  function effectiveWanMetadataRepo(): string {
    return normalizeWanMetadataRepo(assets.value.metadata) || WAN22_5B_METADATA_REPO
  }

  function blockedReasonFor(
    currentVideo: WanVideoParams,
    currentStage: Wan5bStageParams,
    currentPrompt: string,
  ): string {
    if (!String(currentPrompt || '').trim()) {
      return 'Prompt must not be empty.'
    }
    if (currentVideo.useInitImage && !currentVideo.initImageData) {
      return 'Image mode requires an initial image; select a file or switch to Text mode.'
    }
    if (!currentStage.modelDir) {
      return 'WAN 2.2 5B requires one GGUF model. Set it in QuickSettings.'
    }
    if (!quicksettings.resolveWanGgufSha(currentStage.modelDir)) {
      return 'WAN 2.2 5B model must resolve to a sha256. Click Refresh and re-select the model.'
    }
    if (quicksettings.resolveWanGgufVariant(currentStage.modelDir) !== 'wan22_5b') {
      return 'WAN 2.2 5B model must resolve to a structurally 5B GGUF. Click Refresh and re-select a 5B model.'
    }
    if (currentVideo.useInitImage && normalizeImg2VidMode(currentVideo.img2vidMode) !== 'solo') {
      return 'WAN 2.2 5B img2vid currently supports only solo mode.'
    }

    const textEncoderLabel = String(assets.value.textEncoder || '').trim()
    if (!textEncoderLabel) {
      return 'WAN 2.2 5B requires a text encoder (.safetensors/.gguf). Set WAN Text Encoder in QuickSettings.'
    }
    if (!quicksettings.resolveTextEncoderSha(textEncoderLabel)) {
      return 'WAN Text Encoder must resolve to a sha256. Click Refresh and re-select the text encoder.'
    }

    const vaeLabel = String(assets.value.vae || '').trim()
    if (!vaeLabel) {
      return 'WAN 2.2 5B requires a VAE selection. Set WAN VAE in QuickSettings.'
    }
    if (!quicksettings.resolveVaeSha(vaeLabel)) {
      return 'WAN VAE must resolve to a sha256. Click Refresh and re-select the VAE.'
    }

    const device = String(quicksettings.currentDevice || 'cpu').trim().toLowerCase()
    if (device !== 'cpu' && device !== 'cuda') {
      return `WAN video currently supports only cpu or cuda. Switch QuickSettings device from '${quicksettings.currentDevice}' to a supported backend.`
    }
    return ''
  }

  const blockedReason = computed(() => blockedReasonFor(video.value, stage.value, prompt.value))
  const canGenerate = computed(() => blockedReason.value.length === 0)

  function stopStream(): void {
    taskLifecycle.stopStream()
  }

  function resetProgress(): void {
    state.value.progress = { ...DEFAULT_PROGRESS }
  }

  function setError(message: string): void {
    state.value.status = 'error'
    state.value.errorMessage = message
  }

  function setErrorMessage(message: string): void {
    state.value.errorMessage = message
  }

  function resolveVideoRunStatus(code?: TaskErrorCode, message?: string): VideoRunStatus {
    if (code === 'cancelled') return 'cancelled'
    return String(message || '').trim().toLowerCase() === 'cancelled' ? 'cancelled' : 'error'
  }

  function logRunStartError(run: PreparedWan22_5bRun, err: unknown): void {
    const status = getApiErrorStatus(err)
    const message = err instanceof Error ? err.message : String(err)
    const detail = isRecordObject(err) ? err.detail : undefined
    const body = isRecordObject(err) ? err.body : undefined
    console.error('[useWan22_5bVideoGeneration] failed to start WAN run', {
      tabId,
      mode: run.mode,
      status,
      message,
      detail,
      body,
    })
  }

  function buildRunSummary(currentVideo: WanVideoParams): string {
    const width = Number(currentVideo.width) || 0
    const height = Number(currentVideo.height) || 0
    const frames = Number(currentVideo.frames) || 0
    const fps = Number(currentVideo.fps) || 0
    const seconds = fps > 0 ? frames / fps : 0
    return `${width}×${height} · ${frames}f @ ${fps}fps (~${seconds.toFixed(2)}s) · steps ${steps.value} · cfg ${cfgScale.value}`
  }

  function buildParamsSnapshot(currentVideo: WanVideoParams, currentStage: Wan5bStageParams): Record<string, unknown> {
    const img2vidMode = normalizeImg2VidMode(currentVideo.img2vidMode)
    return {
      mode: currentVideo.useInitImage ? 'img2vid' : 'txt2vid',
      initImageName: currentVideo.initImageName || '',
      prompt: prompt.value,
      negativePrompt: negativePrompt.value,
      sampler: sampler.value,
      scheduler: scheduler.value,
      steps: steps.value,
      cfgScale: cfgScale.value,
      seed: seed.value,
      width: currentVideo.width,
      height: currentVideo.height,
      frames: currentVideo.frames,
      fps: currentVideo.fps,
      attentionMode: currentVideo.attentionMode,
      img2vid: {
        mode: img2vidMode,
        anchorAlpha: currentVideo.img2vidAnchorAlpha,
        resetAnchorToBase: currentVideo.img2vidResetAnchorToBase,
        chunkSeedMode: currentVideo.img2vidChunkSeedMode,
        windowFrames: currentVideo.img2vidWindowFrames,
        windowStride: currentVideo.img2vidWindowStride,
        windowCommitFrames: currentVideo.img2vidWindowCommitFrames,
        imageScale: currentVideo.img2vidImageScale,
        cropOffsetX: currentVideo.img2vidCropOffsetX,
        cropOffsetY: currentVideo.img2vidCropOffsetY,
      },
      assets: {
        metadata: String(assets.value.metadata || ''),
        textEncoder: String(assets.value.textEncoder || ''),
        vae: String(assets.value.vae || ''),
      },
      stage: {
        modelDir: currentStage.modelDir,
        loras: currentStage.loras,
        flowShift: currentStage.flowShift,
      },
      output: {
        format: currentVideo.format,
        pixFmt: currentVideo.pixFmt,
        crf: currentVideo.crf,
        loopCount: currentVideo.loopCount,
        pingpong: resolveBooleanWithDefault(currentVideo.pingpong, false),
        returnFrames: resolveBooleanWithDefault(currentVideo.returnFrames, false),
      },
      interpolation: {
        targetFps: currentVideo.interpolationFps,
      },
    }
  }

  function pushHistory(item: VideoRunHistoryItem): void {
    state.value.history.unshift(item)
    if (state.value.history.length > MAX_HISTORY) state.value.history.length = MAX_HISTORY
  }

  function normalizeWanLoraSha(rawValue: unknown): string | null {
    const normalized = String(rawValue || '').trim().toLowerCase()
    if (!/^[0-9a-f]{64}$/.test(normalized)) return null
    return normalized
  }

  function normalizeWanPromptText(rawValue: unknown): string {
    return String(rawValue || '').replace(/\s{2,}/g, ' ').trim()
  }

  function dedupeWanStageLoras(entries: Wan5bStageParams['loras']): Wan5bStageParams['loras'] {
    const deduped: Wan5bStageParams['loras'] = []
    const indexBySha = new Map<string, number>()
    for (const entry of entries) {
      const normalizedSha = normalizeWanLoraSha(entry?.sha)
      if (!normalizedSha) {
        throw new Error('WAN LoRA SHA must be a 64-character hex string.')
      }
      const rawWeight = entry?.weight
      const weight = rawWeight === undefined ? 1.0 : Number(rawWeight)
      if (!Number.isFinite(weight)) {
        throw new Error(`WAN LoRA weight must be a finite number for sha '${normalizedSha}'.`)
      }
      const normalizedEntry = { sha: normalizedSha, weight }
      const existingIndex = indexBySha.get(normalizedSha)
      if (typeof existingIndex === 'number') {
        deduped[existingIndex] = normalizedEntry
      } else {
        indexBySha.set(normalizedSha, deduped.length)
        deduped.push(normalizedEntry)
      }
    }
    return deduped
  }

  function parseWanPromptLoras(
    promptValue: string,
    negativePromptValue: string,
  ): { prompt: string; negativePrompt: string; loras: Wan5bStageParams['loras'] } {
    const collected: Wan5bStageParams['loras'] = []
    const collectFromText = (field: 'prompt' | 'negative_prompt', rawText: string): string => {
      WAN_LORA_TAG_RE.lastIndex = 0
      return String(rawText || '').replace(WAN_LORA_TAG_RE, (_fullMatch, rawName: string, rawWeight?: string) => {
        const tokenName = String(rawName || '').trim()
        if (!tokenName) {
          throw new Error(`WAN ${field} contains an empty LoRA token name.`)
        }
        const resolvedSha = normalizeWanLoraSha(quicksettings.resolveLoraSha(tokenName))
        if (!resolvedSha) {
          throw new Error(`WAN ${field}: LoRA SHA not found for '${tokenName}'. Refresh inventory and retry.`)
        }
        let weight = 1.0
        if (rawWeight !== undefined) {
          const weightText = String(rawWeight || '').trim()
          if (!weightText) {
            throw new Error(`WAN ${field}: LoRA token '${tokenName}' has an empty weight.`)
          }
          const parsedWeight = Number(weightText)
          if (!Number.isFinite(parsedWeight)) {
            throw new Error(`WAN ${field}: LoRA token '${tokenName}' has invalid weight '${weightText}'.`)
          }
          weight = parsedWeight
        }
        collected.push({ sha: resolvedSha, weight })
        return ''
      })
    }

    const cleanedPrompt = normalizeWanPromptText(collectFromText('prompt', promptValue))
    if (!cleanedPrompt) {
      throw new Error('WAN prompt must not be empty after LoRA token parsing.')
    }
    const cleanedNegative = normalizeWanPromptText(collectFromText('negative_prompt', negativePromptValue))
    return {
      prompt: cleanedPrompt,
      negativePrompt: cleanedNegative,
      loras: dedupeWanStageLoras(collected),
    }
  }

  function buildCommonInput(currentVideo: WanVideoParams, currentStage: Wan5bStageParams): Wan22_5bVideoCommonInput {
    const textEncoderLabel = String(assets.value.textEncoder || '').trim()
    const vaeLabel = String(assets.value.vae || '').trim()
    const modelSha = quicksettings.resolveWanGgufSha(currentStage.modelDir) || ''
    const textEncoderSha = quicksettings.resolveTextEncoderSha(textEncoderLabel) || ''
    const vaeSha = quicksettings.resolveVaeSha(vaeLabel) || ''
    const parsedPrompt = parseWanPromptLoras(prompt.value, negativePrompt.value)
    const explicitStageLoras = Array.isArray(currentStage.loras) ? currentStage.loras : []
    const mergedStageLoras = dedupeWanStageLoras([...explicitStageLoras, ...parsedPrompt.loras])

    return {
      device: quicksettings.currentDevice || 'cpu',
      settingsRevision: quicksettings.getSettingsRevision(),
      width: currentVideo.width,
      height: currentVideo.height,
      fps: currentVideo.fps,
      frames: currentVideo.frames,
      prompt: parsedPrompt.prompt,
      negativePrompt: parsedPrompt.negativePrompt,
      sampler: sampler.value,
      scheduler: scheduler.value,
      steps: steps.value,
      cfgScale: cfgScale.value,
      seed: seed.value,
      attentionMode: currentVideo.attentionMode,
      stage: {
        modelSha,
        loras: mergedStageLoras,
        flowShift: currentStage.flowShift,
      },
      format: 'auto',
      assets: {
        metadataRepo: effectiveWanMetadataRepo(),
        textEncoderSha,
        vaeSha,
      },
      output: {
        format: currentVideo.format,
        pixFmt: currentVideo.pixFmt,
        crf: currentVideo.crf,
        loopCount: currentVideo.loopCount,
        pingpong: resolveBooleanWithDefault(currentVideo.pingpong, false),
        returnFrames: resolveBooleanWithDefault(currentVideo.returnFrames, false),
      },
      interpolation: {
        targetFps: currentVideo.interpolationFps,
      },
    }
  }

  function prepareRunFromValues(currentVideo: WanVideoParams, currentStage: Wan5bStageParams): PreparedWan22_5bRun {
    const promptPreview = String(prompt.value || '').trim().slice(0, 120)
    const createdAtMs = Date.now()
    const summary = buildRunSummary(currentVideo)
    const paramsSnapshot = buildParamsSnapshot(currentVideo, currentStage)
    const common = buildCommonInput(currentVideo, currentStage)

    if (currentVideo.useInitImage) {
      const img2vidMode = normalizeImg2VidMode(currentVideo.img2vidMode)
      const normalizedImageScale = normalizeWanImg2VidImageScale(currentVideo.img2vidImageScale, 1)
      const payloadImageScale = Math.abs(normalizedImageScale - 1) < 1e-9 ? undefined : normalizedImageScale
      const guideInput: Partial<Wan22_5bImg2VidInput> = {
        imageScale: payloadImageScale,
        cropOffsetX: normalizeGuideOffset(currentVideo.img2vidCropOffsetX),
        cropOffsetY: normalizeGuideOffset(currentVideo.img2vidCropOffsetY),
      }
      const payload = buildWan22_5bImg2VidPayload({
        ...common,
        initImageData: currentVideo.initImageData,
        img2vidMode,
        ...guideInput,
      })
      return { mode: 'img2vid', createdAtMs, summary, promptPreview, paramsSnapshot, initImageData: currentVideo.initImageData, payload }
    }

    const payload = buildWan22_5bTxt2VidPayload(common)
    return { mode: 'txt2vid', createdAtMs, summary, promptPreview, paramsSnapshot, payload }
  }

  function onTaskEvent(event: TaskEvent): void {
    const key = resumeKey(tabId)
    switch (event.type) {
      case 'status':
        state.value.progress.stage = event.stage
        break
      case 'progress':
        state.value.progress = {
          stage: event.stage,
          percent: event.percent ?? null,
          etaSeconds: event.eta_seconds ?? null,
          step: event.step ?? null,
          totalSteps: event.total_steps ?? null,
        }
        break
      case 'gap':
        if (state.value.taskId) void refreshTaskSnapshot(state.value.taskId)
        break
      case 'result':
        state.value.frames = Array.isArray(event.images) ? event.images : []
        state.value.info = event.info ?? null
        state.value.video = event.video ?? null
        state.value.status = 'done'
        if (state.value.currentRun && state.value.currentRun.taskId) {
          state.value.currentRun.status = 'completed'
          if (Array.isArray(event.images) && event.images.length > 0) {
            state.value.currentRun.thumbnail = event.images[0]
          }
        }
        break
      case 'error': {
        const terminalStatus = resolveVideoRunStatus(event.code, event.message)
        state.value.status = 'error'
        state.value.errorMessage = event.message
        state.value.frames = []
        state.value.info = null
        state.value.video = null
        clearResumeState(key)
        if (state.value.currentRun && state.value.currentRun.taskId) {
          state.value.currentRun.status = terminalStatus
          state.value.currentRun.errorMessage = event.message
        }
        break
      }
      case 'end':
        clearResumeState(key)
        if (state.value.status !== 'error') state.value.status = 'done'
        if (state.value.currentRun && state.value.currentRun.taskId) {
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        stopStream()
        break
    }
  }

  async function refreshTaskSnapshot(taskId: string): Promise<void> {
    try {
      const result = await fetchTaskResult(taskId)
      if (result.status !== 'running') return
      if (typeof result.stage === 'string' && result.stage.trim()) state.value.progress.stage = result.stage
      const progressPayload = result.progress
      if (progressPayload && typeof progressPayload === 'object') {
        state.value.progress = {
          stage: String(progressPayload.stage ?? state.value.progress.stage),
          percent: progressPayload.percent ?? null,
          etaSeconds: progressPayload.eta_seconds ?? null,
          step: progressPayload.step ?? null,
          totalSteps: progressPayload.total_steps ?? null,
        }
      }
    } catch {
      // ignore snapshot refresh failures
    }
  }

  function handleResumedRunningSnapshot(saved: ResumeState, result: Awaited<ReturnType<typeof fetchTaskResult>>): void {
    if (result.status !== 'running') return
    state.value.status = 'running'
    state.value.taskId = saved.taskId
    state.value.errorMessage = ''
    state.value.frames = []
    state.value.info = null
    state.value.video = null
    resetProgress()
    state.value.cancelRequested = false
    state.value.currentRun = {
      taskId: saved.taskId,
      mode: saved.mode,
      createdAtMs: saved.createdAtMs,
      status: 'completed',
      summary: saved.summary,
      promptPreview: saved.promptPreview,
      paramsSnapshot: saved.paramsSnapshot,
      thumbnail: null,
    }

    if (typeof result.stage === 'string' && result.stage.trim()) state.value.progress.stage = result.stage
    const progressPayload = result.progress
    if (progressPayload && typeof progressPayload === 'object') {
      state.value.progress = {
        stage: String(progressPayload.stage ?? state.value.progress.stage),
        percent: progressPayload.percent ?? null,
        etaSeconds: progressPayload.eta_seconds ?? null,
        step: progressPayload.step ?? null,
        totalSteps: progressPayload.total_steps ?? null,
      }
    }
  }

  function handleResumeTerminalSnapshot(saved: ResumeState, result: Awaited<ReturnType<typeof fetchTaskResult>>): void {
    if (result.status === 'completed' && result.result) {
      state.value.frames = Array.isArray(result.result.images) ? result.result.images : []
      state.value.info = result.result.info ?? null
      state.value.video = result.result.video ?? null
      state.value.errorMessage = ''
      state.value.status = 'done'
      state.value.taskId = saved.taskId
      state.value.cancelRequested = false
      pushHistory({
        taskId: saved.taskId,
        mode: saved.mode,
        createdAtMs: saved.createdAtMs,
        status: 'completed',
        summary: saved.summary,
        promptPreview: saved.promptPreview,
        paramsSnapshot: saved.paramsSnapshot,
        thumbnail: Array.isArray(result.result.images) && result.result.images.length > 0 ? result.result.images[0] : null,
      })
      state.value.selectedTaskId = saved.taskId
      state.value.currentRun = null
      return
    }
    if (result.status === 'error') {
      const terminalStatus = resolveVideoRunStatus(result.error_code, String(result.error || 'Task failed.'))
      state.value.status = 'error'
      state.value.errorMessage = String(result.error || 'Task failed.')
      state.value.frames = []
      state.value.info = null
      state.value.video = null
      state.value.taskId = saved.taskId
      state.value.cancelRequested = false
      pushHistory({
        taskId: saved.taskId,
        mode: saved.mode,
        createdAtMs: saved.createdAtMs,
        status: terminalStatus,
        summary: saved.summary,
        promptPreview: saved.promptPreview,
        paramsSnapshot: saved.paramsSnapshot,
        thumbnail: null,
        errorMessage: String(result.error || 'Task failed.'),
      })
      state.value.selectedTaskId = saved.taskId
      state.value.currentRun = null
    }
  }

  function handleHistorySnapshot(taskId: string, result: Awaited<ReturnType<typeof fetchTaskResult>>): void {
    if (result.status === 'error') {
      state.value.status = 'error'
      state.value.errorMessage = result.error || 'Task failed.'
      state.value.frames = []
      state.value.info = null
      state.value.video = null
      state.value.taskId = taskId
      state.value.selectedTaskId = taskId
      state.value.currentRun = null
      return
    }
    if (result.status === 'completed' && result.result) {
      state.value.frames = Array.isArray(result.result.images) ? result.result.images : []
      state.value.info = result.result.info ?? null
      state.value.video = result.result.video ?? null
      state.value.errorMessage = ''
      state.value.status = 'done'
      state.value.taskId = taskId
      state.value.selectedTaskId = taskId
      state.value.currentRun = null
      return
    }
    setError('Task is still running.')
  }

  const taskLifecycle = useTaskRunLifecycle({
    tabId,
    state,
    resumeKey: resumeKey(tabId),
    unsubscribers,
    resumeAttempts,
    loadResumeState,
    saveResumeState,
    clearResumeState,
    updateResumeEventId,
    onTaskEvent,
    isSnapshotRunning: (snapshot) => snapshot.status === 'running',
    onResumeRunning: handleResumedRunningSnapshot,
    onResumeTerminal: handleResumeTerminalSnapshot,
    onResumeLoadError: (message) => {
      resumeNotice.value = message
    },
    onHistoryLoaded: handleHistorySnapshot,
    onHistoryLoadError: (_taskId, error) => {
      setError(error instanceof Error ? error.message : String(error))
    },
    resumeNotice,
    resumeToastShown,
  })

  async function startPreparedRun(run: PreparedWan22_5bRun): Promise<void> {
    stopStream()
    state.value.errorMessage = ''
    state.value.frames = []
    state.value.info = null
    state.value.video = null
    resetProgress()
    state.value.progress.stage = 'starting'
    state.value.cancelRequested = false
    state.value.currentRun = null
    state.value.status = 'running'

    try {
      const response = await (async () => {
        if (run.mode === 'img2vid') {
          assertRunPayloadObject(run.payload, run.mode)
          return startImg2Vid(run.payload)
        }
        if (run.mode === 'txt2vid') {
          assertRunPayloadObject(run.payload, run.mode)
          return startTxt2Vid(run.payload)
        }
        throw new Error('useWan22_5bVideoGeneration: unsupported prepared run mode.')
      })()
      const taskId = response.task_id
      state.value.taskId = taskId
      state.value.currentRun = {
        taskId,
        mode: run.mode,
        createdAtMs: run.createdAtMs,
        status: 'completed',
        summary: run.summary,
        promptPreview: run.promptPreview,
        paramsSnapshot: run.paramsSnapshot,
        initImageData: run.mode === 'img2vid' ? run.initImageData : '',
        thumbnail: null,
      }
      taskLifecycle.saveResume({
        taskId,
        lastEventId: 0,
        createdAtMs: run.createdAtMs,
        mode: run.mode,
        summary: run.summary,
        promptPreview: run.promptPreview,
        paramsSnapshot: run.paramsSnapshot,
      })
      taskLifecycle.attachStream(taskId)
    } catch (error) {
      logRunStartError(run, error)
      clearResumeState(resumeKey(tabId))
      const conflictRevision = resolveSettingsRevisionConflict(error)
      if (conflictRevision !== null) {
        try {
          await quicksettings.refreshSettingsRevision(conflictRevision)
        } catch {
          // ignore refresh failure; fallback revision already applied
        }
        setError(formatSettingsRevisionConflictMessage(quicksettings.getSettingsRevision()))
        return
      }
      setError(formatZodError(error))
    }
  }

  async function generate(): Promise<void> {
    if (!tab.value) {
      setError(`useWan22_5bVideoGeneration: tab '${tabId}' not found or not available.`)
      return
    }
    if (tab.value.type !== 'wan22_5b') {
      setError(`useWan22_5bVideoGeneration: unsupported tab type '${String(tab.value.type)}'`)
      return
    }

    const blocked = blockedReasonFor(video.value, stage.value, prompt.value)
    if (blocked) {
      setError(blocked)
      return
    }

    try {
      const run = prepareRunFromValues(video.value, stage.value)
      await startPreparedRun(run)
    } catch (error) {
      setError(formatZodError(error))
    }
  }

  void taskLifecycle.tryAutoResume()

  async function cancel(modeValue: 'immediate' | 'after_current' = 'immediate'): Promise<void> {
    const taskId = state.value.taskId
    if (!taskId || state.value.status !== 'running') return
    state.value.cancelRequested = true
    try {
      await cancelTask(taskId, modeValue)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error))
    }
  }

  const loadHistory = taskLifecycle.loadHistory

  function clearHistory(): void {
    taskLifecycle.clearHistory()
  }

  function outputUrl(relPath: string): string {
    const clean = String(relPath || '').replace(/\\+/g, '/').replace(/^\/+/, '')
    const encoded = clean.split('/').map((segment) => encodeURIComponent(segment)).join('/')
    return `/api/output/${encoded}`
  }

  const videoExport = computed(() => state.value.video)
  const videoUrl = computed(() => {
    const relPath = state.value.video?.rel_path
    if (!relPath) return ''
    return outputUrl(relPath)
  })

  return {
    status: computed(() => state.value.status),
    progress: computed(() => state.value.progress),
    frames: computed(() => state.value.frames),
    info: computed(() => state.value.info),
    videoExport,
    videoUrl,
    errorMessage: computed(() => state.value.errorMessage),
    taskId: computed(() => state.value.taskId),
    isRunning: computed(() => state.value.status === 'running'),
    cancelRequested: computed(() => state.value.cancelRequested),
    history: computed(() => state.value.history),
    selectedTaskId: computed(() => state.value.selectedTaskId),
    historyLoadingTaskId: computed(() => state.value.historyLoadingTaskId),
    tab,
    params,
    prompt,
    negativePrompt,
    video,
    stage,
    assets,
    sampler,
    scheduler,
    steps,
    cfgScale,
    seed,
    mode,
    blockedReason,
    canGenerate,
    generate,
    stopStream,
    cancel,
    loadHistory,
    clearHistory,
    resumeNotice,
  }
}
