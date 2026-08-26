/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 14B video generation composable (txt2vid/img2vid).
Owns per-tab video generation state (progress/frames/video result/history/queue), builds typed WAN payloads, starts tasks, and consumes task SSE events
to update UI state and fetch final results. Delegates the shared task-stream/resume/history shell to `useTaskRunLifecycle.ts`, while keeping
WAN 2.2 14B-specific queueing, summary/snapshot builders, and result shaping local. Every start payload includes `settings_revision`, and stale-revision conflicts (`409` + `current_revision`)
trigger revision refresh + manual-retry UX. Persists a minimal resume marker to `localStorage` and auto-reattaches to in-flight tasks after reload
via SSE replay (`after` / `lastEventId`) and snapshot refresh on `gap`. WAN payload ownership is explicit: top-level core owners (`prompt`, `negativePrompt`,
`sampler`, `scheduler`, `steps`, `cfgScale`, `seed`) stay top-level in the typed builder input, while `wan_high` carries only model/Lora/flow overrides and
`wan_low` keeps the live low-stage prompt/sampler fields. Includes compact output pass-through (`format`/`pixFmt`/`crf`/`loopCount`/`pingpong`/`returnFrames`),
interpolation target FPS (`0` disables; payload computes backend interpolation factor from target/base FPS).
Img2vid temporal payload fields are gated by `img2vidMode` (`solo|sliding|svi2|svi2_pro`), and WAN prompt `<lora:...>` tags are parsed client-side into
stage-level LoRA arrays (`wan_high/wan_low.loras[]` with `sha+weight`) before payload dispatch. Start failures now log structured diagnostics to the browser console (status/detail/body/message + mode/tab)
before surfacing UI error text.
Img2vid no-stretch guide fields (`imageScale`, `cropOffsetX`, `cropOffsetY`) are forwarded into payload builders with strict scale/offset validation, and default scale (`1`) is omitted so runtime auto-fit minimum can apply.

Symbols (top-level; keep in sync; no ghosts):
- `Status` (type): Video generation status state (`idle|running|error|done`).
- `VideoMode` (type): Supported video modes (`txt2vid|img2vid`).
- `VideoRunStatus` (type): Terminal status for history entries (`completed|error|cancelled`).
- `VideoRunHistoryItem` (interface): Persisted run history entry (task id, status, summary, params snapshot, error message).
- `PreparedWanRun` (type): Mode-discriminated prepared run payload (`txt2vid|img2vid`) with typed WAN request bodies.
- `VideoQueuedRun` (type): Queued run entry (`PreparedWanRun` + `id`) to support sequential submissions.
- `VideoProgressState` (interface): Progress payload shape (stage/percent/eta/step/totalSteps).
- `VideoGenerationState` (interface): Per-tab runtime state (status/progress/frames/video result/history/queue/cancel flags).
- `ResumeState` (type): Persisted task resume marker (task id + last event id + params snapshot).
- `ResumeStateLoad` (type): Parsed resume-state load result (`state` + optional fail-loud parse error message).
- `freshState` (function): Creates a new `VideoGenerationState` with empty progress/history/queue.
- `getTabState` (function): Returns (and initializes) the per-tab `VideoGenerationState` for a given tab id.
- `defaultStage` (function): Creates default `WanStageParams` for UI state initialization.
- `defaultVideo` (function): Creates default `WanVideoParams` for UI state initialization.
- `defaultAssets` (function): Creates default `WanAssetsParams`.
- `resumeKey` (function): Returns the localStorage key used to persist a WAN task resume marker for a tab.
- `isRecordObject` (function): Type guard for plain-object payloads used in resume-state parsing.
- `parseResumeMode` (function): Strict parser for persisted WAN resume mode (`txt2vid|img2vid`), returns null for unsupported/legacy modes.
- `assertRunPayloadObject` (function): Runtime invariant guard that fails loud when a prepared run carries a non-object payload.
- `assertNeverMode` (function): Exhaustiveness guard for prepared-run dispatch by `mode`.
- `useVideoGeneration` (function): Main 14B composable API; wires payload building, task start/cancel, SSE handling, queued runs, and history updates
  (contains nested handlers for events, queue progression, and per-mode payload assembly).
*/

import { computed, ref } from 'vue'

import { cancelTask, fetchTaskResult, getApiErrorStatus, startImg2Vid, startTxt2Vid } from '../api/client'
import { formatZodError } from '../api/payloads'
import {
  buildWanImg2VidPayload,
  buildWanTxt2VidPayload,
  type WanImg2VidInput,
  type WanImg2VidPayload,
  type WanTxt2VidPayload,
  type WanVideoCommonInput,
} from '../api/payloads_video'
import type { GeneratedImage, TaskErrorCode, TaskEvent } from '../api/types'
import { useModelTabsStore, type TabByType, type WanAssetsParams, type WanStageParams, type WanVideoParams } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { formatSettingsRevisionConflictMessage, resolveSettingsRevisionConflict } from './settings_revision_conflict'
import { useTaskRunLifecycle } from './useTaskRunLifecycle'
import { isWanWindowedImg2VidMode, normalizeWanImg2VidMode } from '../utils/wan_img2vid_temporal'
import { normalizeWanImg2VidImageScale } from '../utils/wan_img2vid_frame_projection'

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
  thumbnail?: GeneratedImage | null
  errorMessage?: string
}

export type PreparedWanRun =
  | {
      mode: 'txt2vid'
      createdAtMs: number
      summary: string
      promptPreview: string
      paramsSnapshot: Record<string, unknown>
      payload: WanTxt2VidPayload
    }
  | {
      mode: 'img2vid'
      createdAtMs: number
      summary: string
      promptPreview: string
      paramsSnapshot: Record<string, unknown>
      initImageData: string
      payload: WanImg2VidPayload
    }

export type VideoQueuedRun = PreparedWanRun & {
  id: string
}

function assertRunPayloadObject(payload: unknown, mode: VideoMode): asserts payload is Record<string, unknown> {
  if (!isRecordObject(payload)) {
    throw new Error(`useVideoGeneration: invalid payload for mode '${mode}'.`)
  }
}

function assertNeverMode(mode: never): never {
  throw new Error(`useVideoGeneration: unsupported prepared run mode '${String(mode)}'.`)
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
  queue: VideoQueuedRun[]
}

const DEFAULT_PROGRESS: VideoProgressState = { stage: 'idle', percent: null, etaSeconds: null, step: null, totalSteps: null }
const MAX_HISTORY = 8
const MAX_QUEUE = 3
const WAN_LORA_TAG_RE = /<\s*lora\s*:\s*([^:>]+)\s*(?::\s*([^>]*))?\s*>/gi

// Per-tab generation state (keyed by tab ID)
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
  return `codex.resume.wan.${tabId}`
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
    // ignore localStorage failures (private mode/quota)
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
  const v = Math.trunc(Number(eventId))
  if (!Number.isFinite(v) || v <= 0) return
  const cur = loadResumeState(key).state
  if (!cur) return
  if (v <= cur.lastEventId) return
  saveResumeState(key, { ...cur, lastEventId: v })
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
    queue: [],
  }
}

function getTabState(tabId: string): VideoGenerationState {
  if (!tabStates.has(tabId)) tabStates.set(tabId, freshState())
  return tabStates.get(tabId)!
}

function defaultStage(): WanStageParams {
  return {
    modelDir: '',
    prompt: '',
    negativePrompt: '',
    sampler: '',
    scheduler: 'simple',
    steps: 30,
    cfgScale: 7,
    seed: -1,
    loras: [],
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
    throw new Error(`useVideoGeneration: img2vid crop offset must be a finite number in [0,1] (got ${String(rawValue)}).`)
  }
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) {
    throw new Error(`useVideoGeneration: img2vid crop offset must be finite in [0,1] (got ${String(rawValue)}).`)
  }
  if (numeric < 0 || numeric > 1) {
    throw new Error(`useVideoGeneration: img2vid crop offset must be in [0,1] (got ${String(rawValue)}).`)
  }
  return numeric
}

function resolveBooleanWithDefault(rawValue: unknown, fallback: boolean): boolean {
  return typeof rawValue === 'boolean' ? rawValue : fallback
}

export function useVideoGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()

  const state = ref(getTabState(tabId))
  const resumeNotice = ref('')

  type WanTab = TabByType<'wan22_14b'>

  const tab = computed<WanTab | null>(() => {
    const candidate = modelTabs.tabs.find((entry) => entry.id === tabId) || null
    if (!candidate || candidate.type !== 'wan22_14b') return null
    return candidate as WanTab
  })
  const params = computed<WanTab['params'] | null>(() => tab.value?.params || null)

  const video = computed<WanVideoParams>(() => params.value?.video || defaultVideo())
  const high = computed<WanStageParams>(() => params.value?.high || defaultStage())
  const low = computed<WanStageParams>(() => params.value?.low || defaultStage())
  const lightx2v = computed<boolean>(() => Boolean(params.value?.lightx2v))
  const assets = computed<WanAssetsParams>(() => params.value?.assets || defaultAssets())
  const mode = computed<VideoMode>(() => {
    if (video.value.useInitImage) return 'img2vid'
    return 'txt2vid'
  })

  function normalizeWanMetadataRepo(raw: string): string | null {
    const v = String(raw || '').trim()
    if (!v) return null
    // Back-compat: ignore legacy path values persisted in older tabs.
    if (v.startsWith('/') || v.includes('\\') || v.includes(':')) return null
    if (!v.includes('/')) return null
    return v
  }

  function inferWanMetadataRepo(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): string {
    const hint = `${hi.modelDir || ''} ${lo.modelDir || ''}`.toLowerCase()
    if (hint.includes('ti2v') || hint.includes('5b')) return 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
    if (v.useInitImage) return 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
    return 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'
  }

  function effectiveWanMetadataRepo(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): string {
    const repo = normalizeWanMetadataRepo(assets.value.metadata)
    const repoLower = (repo || '').toLowerCase()
    const is5b = repoLower.includes('wan2.2-ti2v-5b')
    if (is5b) return 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'

    const isKnown14b = repoLower.includes('wan2.2-i2v-a14b') || repoLower.includes('wan2.2-t2v-a14b')
    if (isKnown14b) {
      if (v.useInitImage) return 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
      return 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'
    }

    // If a different (valid) repo id is pinned in the tab, respect it.
    if (repo) return repo

    return inferWanMetadataRepo(v, hi, lo)
  }

  function blockedReasonFor(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): string {
    const highPrompt = String(hi.prompt || '').trim()
    if (!highPrompt) return 'High stage prompt must not be empty.'
    const lowPrompt = String(lo.prompt || '').trim()
    if (!lowPrompt) return 'Low stage prompt must not be empty.'
    if (v.useInitImage && !v.initImageData) {
      return 'Image mode requires an initial image; select a file or switch to Text mode.'
    }
    if (!hi.modelDir || !lo.modelDir) {
      return 'WAN requires both High and Low stage models. Set them in QuickSettings.'
    }
    if (!quicksettings.resolveWanGgufSha(hi.modelDir)) {
      return 'WAN High model must resolve to a sha256. Click Refresh and re-select the High model.'
    }
    if (!quicksettings.resolveWanGgufSha(lo.modelDir)) {
      return 'WAN Low model must resolve to a sha256. Click Refresh and re-select the Low model.'
    }
    if (quicksettings.resolveWanGgufVariant(hi.modelDir) !== 'wan22_14b') {
      return 'WAN High model must resolve to a structurally 14B GGUF. Click Refresh and re-select the High model.'
    }
    if (quicksettings.resolveWanGgufVariant(lo.modelDir) !== 'wan22_14b') {
      return 'WAN Low model must resolve to a structurally 14B GGUF. Click Refresh and re-select the Low model.'
    }

    const teLabel = String(assets.value.textEncoder || '').trim()
    if (!teLabel) {
      return 'WAN requires a text encoder (.safetensors). Set WAN Text Encoder in QuickSettings.'
    }
    if (!quicksettings.resolveTextEncoderSha(teLabel)) {
      return 'WAN Text Encoder must resolve to a sha256. Click Refresh and re-select the text encoder.'
    }

    const vaeLabel = String(assets.value.vae || '').trim()
    if (!vaeLabel) {
      return 'WAN requires a VAE selection. Set WAN VAE in QuickSettings.'
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

  const blockedReason = computed(() => blockedReasonFor(video.value, high.value, low.value))
  const canGenerate = computed(() => blockedReason.value.length === 0)

  function uuid(): string {
    return `q-${Math.random().toString(36).slice(2, 10)}`
  }

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
    onTaskEvent: onTaskEvent,
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

  function resolveVideoRunStatus(code?: TaskErrorCode, message?: string): VideoRunStatus {
    if (code === 'cancelled') return 'cancelled'
    return String(message || '').trim().toLowerCase() === 'cancelled' ? 'cancelled' : 'error'
  }

  function logRunStartError(run: PreparedWanRun, err: unknown): void {
    const status = getApiErrorStatus(err)
    const message = err instanceof Error ? err.message : String(err)
    const detail = isRecordObject(err) ? err.detail : undefined
    const body = isRecordObject(err) ? err.body : undefined
    console.error('[useVideoGeneration] failed to start WAN run', {
      tabId,
      mode: run.mode,
      status,
      message,
      detail,
      body,
    })
  }

  function buildRunSummary(v: WanVideoParams, hi: WanStageParams): string {
    const w = Number(v.width) || 0
    const h = Number(v.height) || 0
    const frames = Number(v.frames) || 0
    const fps = Number(v.fps) || 0
    const seconds = fps > 0 ? (frames / fps) : 0
    const loraTag = lightx2v.value ? ' · lightx2v' : ''
    return `${w}×${h} · ${frames}f @ ${fps}fps (~${seconds.toFixed(2)}s) · steps ${hi.steps} · cfg ${hi.cfgScale}${loraTag}`
  }

  function buildParamsSnapshot(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): Record<string, unknown> {
    const img2vidMode = normalizeImg2VidMode(v.img2vidMode)
    return {
      mode: v.useInitImage ? 'img2vid' : 'txt2vid',
      initImageName: v.initImageName || '',
      width: v.width,
      height: v.height,
      frames: v.frames,
      fps: v.fps,
      attentionMode: v.attentionMode,
      img2vid: {
        mode: img2vidMode,
        anchorAlpha: v.img2vidAnchorAlpha,
        resetAnchorToBase: v.img2vidResetAnchorToBase,
        chunkSeedMode: v.img2vidChunkSeedMode,
        windowFrames: v.img2vidWindowFrames,
        windowStride: v.img2vidWindowStride,
        windowCommitFrames: v.img2vidWindowCommitFrames,
        imageScale: v.img2vidImageScale,
        cropOffsetX: v.img2vidCropOffsetX,
        cropOffsetY: v.img2vidCropOffsetY,
      },
      lightx2v: lightx2v.value,
      assets: {
        metadata: String(assets.value.metadata || ''),
        textEncoder: String(assets.value.textEncoder || ''),
        vae: String(assets.value.vae || ''),
      },
      high: {
        modelDir: hi.modelDir,
        prompt: hi.prompt,
        negativePrompt: hi.negativePrompt,
        sampler: hi.sampler,
        scheduler: hi.scheduler,
        steps: hi.steps,
        cfgScale: hi.cfgScale,
        seed: hi.seed,
        loras: hi.loras,
        flowShift: hi.flowShift,
      },
      low: {
        modelDir: lo.modelDir,
        prompt: lo.prompt,
        negativePrompt: lo.negativePrompt,
        sampler: lo.sampler,
        scheduler: lo.scheduler,
        steps: lo.steps,
        cfgScale: lo.cfgScale,
        seed: lo.seed,
        loras: lo.loras,
        flowShift: lo.flowShift,
      },
      output: {
        format: v.format,
        pixFmt: v.pixFmt,
        crf: v.crf,
        loopCount: v.loopCount,
        pingpong: resolveBooleanWithDefault(v.pingpong, false),
        returnFrames: resolveBooleanWithDefault(v.returnFrames, false),
      },
      interpolation: {
        targetFps: v.interpolationFps,
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

  function dedupeWanStageLoras(entries: WanStageParams['loras']): WanStageParams['loras'] {
    const deduped: WanStageParams['loras'] = []
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

  function parseWanStagePromptLoras(
    stageName: 'high' | 'low',
    promptValue: string,
    negativePromptValue: string,
  ): { prompt: string; negativePrompt: string; loras: WanStageParams['loras'] } {
    const collected: WanStageParams['loras'] = []
    const collectFromText = (field: 'prompt' | 'negative_prompt', rawText: string): string => {
      WAN_LORA_TAG_RE.lastIndex = 0
      return String(rawText || '').replace(WAN_LORA_TAG_RE, (_fullMatch, rawName: string, rawWeight?: string) => {
        const tokenName = String(rawName || '').trim()
        if (!tokenName) {
          throw new Error(`WAN ${stageName}.${field} contains an empty LoRA token name.`)
        }
        const resolvedSha = normalizeWanLoraSha(quicksettings.resolveLoraSha(tokenName))
        if (!resolvedSha) {
          throw new Error(`WAN ${stageName}.${field}: LoRA SHA not found for '${tokenName}'. Refresh inventory and retry.`)
        }
        let weight = 1.0
        if (rawWeight !== undefined) {
          const weightText = String(rawWeight || '').trim()
          if (!weightText) {
            throw new Error(`WAN ${stageName}.${field}: LoRA token '${tokenName}' has an empty weight.`)
          }
          const parsedWeight = Number(weightText)
          if (!Number.isFinite(parsedWeight)) {
            throw new Error(`WAN ${stageName}.${field}: LoRA token '${tokenName}' has invalid weight '${weightText}'.`)
          }
          weight = parsedWeight
        }
        collected.push({ sha: resolvedSha, weight })
        return ''
      })
    }

    const cleanedPrompt = normalizeWanPromptText(collectFromText('prompt', promptValue))
    if (!cleanedPrompt) {
      throw new Error(`WAN ${stageName}.prompt must not be empty after LoRA token parsing.`)
    }
    const cleanedNegative = normalizeWanPromptText(collectFromText('negative_prompt', negativePromptValue))

    return {
      prompt: cleanedPrompt,
      negativePrompt: cleanedNegative,
      loras: dedupeWanStageLoras(collected),
    }
  }

  function buildCommonInput(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): WanVideoCommonInput {
    const metaRepo = effectiveWanMetadataRepo(v, hi, lo)
    const teLabel = String(assets.value.textEncoder || '').trim()
    const vaeLabel = String(assets.value.vae || '').trim()

    const hiSha = quicksettings.resolveWanGgufSha(hi.modelDir) || ''
    const loSha = quicksettings.resolveWanGgufSha(lo.modelDir) || ''
    const tencSha = quicksettings.resolveTextEncoderSha(teLabel) || ''
    const vaeSha = quicksettings.resolveVaeSha(vaeLabel) || ''
    const parsedHighStage = parseWanStagePromptLoras('high', hi.prompt, hi.negativePrompt)
    const parsedLowStage = parseWanStagePromptLoras('low', lo.prompt, lo.negativePrompt)
    const explicitHighLoras = Array.isArray(hi.loras) ? hi.loras : []
    const explicitLowLoras = Array.isArray(lo.loras) ? lo.loras : []
    const mergedHighLoras = dedupeWanStageLoras([...explicitHighLoras, ...parsedHighStage.loras])
    const mergedLowLoras = dedupeWanStageLoras([...explicitLowLoras, ...parsedLowStage.loras])

    return {
      device: quicksettings.currentDevice || 'cpu',
      settingsRevision: quicksettings.getSettingsRevision(),
      width: v.width,
      height: v.height,
      fps: v.fps,
      frames: v.frames,
      prompt: parsedHighStage.prompt,
      negativePrompt: parsedHighStage.negativePrompt,
      sampler: hi.sampler,
      scheduler: hi.scheduler,
      steps: hi.steps,
      cfgScale: hi.cfgScale,
      seed: hi.seed,
      attentionMode: v.attentionMode,
      high: {
        modelSha: hiSha,
        loras: mergedHighLoras,
        flowShift: hi.flowShift,
      },
      low: {
        modelSha: loSha,
        prompt: parsedLowStage.prompt,
        negativePrompt: parsedLowStage.negativePrompt,
        sampler: lo.sampler,
        scheduler: lo.scheduler,
        steps: lo.steps,
        cfgScale: lo.cfgScale,
        seed: lo.seed,
        loras: mergedLowLoras,
        flowShift: lo.flowShift,
      },
      format: 'auto' as const,
      assets: {
        metadataRepo: metaRepo,
        textEncoderSha: tencSha,
        vaeSha: vaeSha,
      },
      output: {
        format: v.format,
        pixFmt: v.pixFmt,
        crf: v.crf,
        loopCount: v.loopCount,
        pingpong: resolveBooleanWithDefault(v.pingpong, false),
        returnFrames: resolveBooleanWithDefault(v.returnFrames, false),
      },
      interpolation: {
        targetFps: v.interpolationFps,
      },
    }
  }

  function prepareRunFromValues(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): PreparedWanRun {
    const promptPreview = String(hi.prompt || '').trim().slice(0, 120)
    const createdAtMs = Date.now()
    const summary = buildRunSummary(v, hi)
    const paramsSnapshot = buildParamsSnapshot(v, hi, lo)

    const common = buildCommonInput(v, hi, lo)

    if (v.useInitImage) {
      const img2vidMode = normalizeImg2VidMode(v.img2vidMode)
      const normalizedImageScale = normalizeWanImg2VidImageScale(v.img2vidImageScale, 1)
      const payloadImageScale = Math.abs(normalizedImageScale - 1) < 1e-9 ? undefined : normalizedImageScale
      const img2vidTemporalInput: Partial<WanImg2VidInput> = {}
      const img2vidGuideInput: Partial<WanImg2VidInput> = {
        imageScale: payloadImageScale,
        cropOffsetX: normalizeGuideOffset(v.img2vidCropOffsetX),
        cropOffsetY: normalizeGuideOffset(v.img2vidCropOffsetY),
      }
      if (isWanWindowedImg2VidMode(img2vidMode)) {
        img2vidTemporalInput.windowFrames = v.img2vidWindowFrames
        img2vidTemporalInput.windowStride = v.img2vidWindowStride
        img2vidTemporalInput.windowCommitFrames = v.img2vidWindowCommitFrames
        img2vidTemporalInput.anchorAlpha = v.img2vidAnchorAlpha
        img2vidTemporalInput.resetAnchorToBase = v.img2vidResetAnchorToBase
        img2vidTemporalInput.chunkSeedMode = v.img2vidChunkSeedMode
      }
      const payload = buildWanImg2VidPayload({
        ...common,
        initImageData: v.initImageData,
        img2vidMode,
        ...img2vidGuideInput,
        ...img2vidTemporalInput,
      })
      return { mode: 'img2vid', createdAtMs, summary, promptPreview, paramsSnapshot, initImageData: v.initImageData, payload }
    }

    const payload = buildWanTxt2VidPayload(common)
    return { mode: 'txt2vid', createdAtMs, summary, promptPreview, paramsSnapshot, payload }
  }

  async function startPreparedRun(run: PreparedWanRun): Promise<void> {
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
      const res = await (async () => {
        const mode = run.mode
        switch (mode) {
          case 'img2vid':
            assertRunPayloadObject(run.payload, mode)
            return startImg2Vid(run.payload)
          case 'txt2vid':
            assertRunPayloadObject(run.payload, mode)
            return startTxt2Vid(run.payload)
          default:
            return assertNeverMode(mode)
        }
      })()
      const task_id = res.task_id
      state.value.taskId = task_id
      state.value.currentRun = {
        taskId: task_id,
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
        taskId: task_id,
        lastEventId: 0,
        createdAtMs: run.createdAtMs,
        mode: run.mode,
        summary: run.summary,
        promptPreview: run.promptPreview,
        paramsSnapshot: run.paramsSnapshot,
      })
      taskLifecycle.attachStream(task_id)
    } catch (err) {
      logRunStartError(run, err)
      clearResumeState(resumeKey(tabId))
      const conflictRevision = resolveSettingsRevisionConflict(err)
      if (conflictRevision !== null) {
        try {
          await quicksettings.refreshSettingsRevision(conflictRevision)
        } catch {
          // Ignore refresh failures; fallback revision is already applied.
        }
        setError(formatSettingsRevisionConflictMessage(quicksettings.getSettingsRevision()))
        return
      }
      setError(formatZodError(err))
    }
  }

  async function startNextQueued(): Promise<void> {
    if (state.value.status === 'running') return
    const next = state.value.queue.shift()
    if (!next) return
    await startPreparedRun(next)
  }

  async function generate(): Promise<void> {
    if (!tab.value) {
      setError(`useVideoGeneration: tab '${tabId}' not found or not available.`)
      return
    }
    if (tab.value.type !== 'wan22_14b') {
      setError(`useVideoGeneration: unsupported tab type '${String(tab.value.type)}'`)
      return
    }

    const v = video.value
    const hi = high.value
    const lo = low.value

    const blocked = blockedReasonFor(v, hi, lo)
    if (blocked) {
      setError(blocked)
      return
    }

    try {
      const run = prepareRunFromValues(v, hi, lo)
      await startPreparedRun(run)
    } catch (err) {
      setError(formatZodError(err))
    }
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
      case 'error':
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
      case 'end':
        clearResumeState(key)
        if (state.value.status !== 'error') state.value.status = 'done'
        if (state.value.currentRun && state.value.currentRun.taskId) {
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        stopStream()
        void startNextQueued()
        break
    }
  }

  async function refreshTaskSnapshot(taskId: string): Promise<void> {
    try {
      const res = await fetchTaskResult(taskId)
      if (res.status !== 'running') return
      if (typeof res.stage === 'string' && res.stage.trim()) state.value.progress.stage = res.stage
      const p = res.progress
      if (p && typeof p === 'object') {
        state.value.progress = {
          stage: String(p.stage ?? state.value.progress.stage),
          percent: p.percent ?? null,
          etaSeconds: p.eta_seconds ?? null,
          step: p.step ?? null,
          totalSteps: p.total_steps ?? null,
        }
      }
    } catch {
      // ignore snapshot refresh failures
    }
  }

  function handleResumedRunningSnapshot(saved: ResumeState, res: Awaited<ReturnType<typeof fetchTaskResult>>): void {
    if (res.status !== 'running') return
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

    if (typeof res.stage === 'string' && res.stage.trim()) state.value.progress.stage = res.stage
    const progressPayload = res.progress
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

  function handleResumeTerminalSnapshot(saved: ResumeState, res: Awaited<ReturnType<typeof fetchTaskResult>>): void {
    if (res.status === 'completed' && res.result) {
      state.value.frames = Array.isArray(res.result.images) ? res.result.images : []
      state.value.info = res.result.info ?? null
      state.value.video = res.result.video ?? null
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
        thumbnail: Array.isArray(res.result.images) && res.result.images.length > 0 ? res.result.images[0] : null,
      })
      state.value.selectedTaskId = saved.taskId
      state.value.currentRun = null
      return
    }
    if (res.status === 'error') {
      const terminalStatus = resolveVideoRunStatus(res.error_code, String(res.error || 'Task failed.'))
      state.value.status = 'error'
      state.value.errorMessage = String(res.error || 'Task failed.')
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
        errorMessage: String(res.error || 'Task failed.'),
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

  void taskLifecycle.tryAutoResume()

  async function cancel(mode: 'immediate' | 'after_current' = 'immediate'): Promise<void> {
    const taskId = state.value.taskId
    if (!taskId || state.value.status !== 'running') return
    state.value.cancelRequested = true
    try {
      await cancelTask(taskId, mode)
    } catch (err) {
      // Keep running; backend may still send end/error shortly.
      setErrorMessage(err instanceof Error ? err.message : String(err))
    }
  }

  async function enqueue(): Promise<void> {
    if (state.value.queue.length >= MAX_QUEUE) throw new Error(`Queue is full (max ${MAX_QUEUE}).`)

    const v = video.value
    const hi = high.value
    const lo = low.value
    const blocked = blockedReasonFor(v, hi, lo)
    if (blocked) throw new Error(blocked)

    const run = prepareRunFromValues(v, hi, lo)
    state.value.queue.push({ id: uuid(), ...run })
  }

  function clearQueue(): void {
    state.value.queue = []
  }

  const loadHistory = taskLifecycle.loadHistory

  function clearHistory(): void {
    taskLifecycle.clearHistory()
  }

  function outputUrl(relPath: string): string {
    const clean = String(relPath || '').replace(/\\+/g, '/').replace(/^\/+/, '')
    const encoded = clean.split('/').map((p) => encodeURIComponent(p)).join('/')
    return `/api/output/${encoded}`
  }

  const videoExport = computed(() => state.value.video)
  const videoUrl = computed(() => {
    const rel = state.value.video?.rel_path
    if (!rel) return ''
    return outputUrl(rel)
  })

  return {
    // State
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
    queue: computed(() => state.value.queue),
    queueMax: computed(() => MAX_QUEUE),

    // Tab info
    tab,
    params,
    video,
    high,
    low,
    lightx2v,
    assets,
    mode,
    blockedReason,
    canGenerate,

    // Actions
    generate,
    stopStream,
    cancel,
    loadHistory,
    clearHistory,
    enqueue,
    clearQueue,
    resumeNotice,
  }
}
