/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Unified generation composable for image tabs (SD/Flux/Qwen Image/Chroma/ZImage/Z-Image L2P; txt2img/img2img/inpaint + automation).
Owns per-tab generation state (progress/live preview/gallery/history), builds request payloads using Model Tabs + QuickSettings,
starts `/api/txt2img`, `/api/img2img`, and `/api/image-automation` (when run action or source contracts require backend-owned looping),
includes `settings_revision` in payloads, handles stale-revision conflicts (`409` + `current_revision`), and consumes task SSE events to update UI state.
Consumes rich progress payload metadata (`progress.message` + `progress.data`) plus buffered `automation_iteration` events, and derives total-phase progress fields for dual run-card bars.
Exposes task cancellation for active runs (`/api/tasks/:id/cancel`).
Delegates the shared task-stream/resume/history shell to `useTaskRunLifecycle.ts`, while keeping image-specific auto-resume hydration,
automation replay recovery, and result/history shaping local. Persists a per-tab resume marker to `localStorage` and auto-reattaches to
in-flight tasks after reload (SSE replay via `after` / `lastEventId`), reconstructing truthful `currentRun` / history-selection state and
preserving wall-clock gentime for runs that finish after resume.
FLUX.2 img2img guidance emission is variant-aware (`img2img_cfg_scale` xor `img2img_distilled_cfg_scale`), and img2img hires emission uses
the nested `img2img_extras.hires` owner while remaining blocked for masked runs. Native SDXL SUPIR mode stays on the single nested frontend owner
`params.supir` and is emitted only through `img2img_extras.supir` for truthful SDXL img2img/inpaint runs, with diagnostics-backed sampler metadata
owning the effective runtime sampler/scheduler pair and fail-loud rejection of stale APG/advanced-guidance overlap.
Qwen Image Edit-2511 uses a dedicated img2img-only request-state guard and payload allowlist so txt2img plus hidden unsupported state fail loud before `/api/*` submission.
Z-Image L2P uses the shared image shell but is exact 1024x1024 txt2img only, no-VAE, no LoRA/IP/hires/refiner/advanced-guidance, and emits only
denoiser + Qwen3-4B TEnc asset selectors.
Non-SUPIR image runs resolve the executable sampler/scheduler pair from strictly validated current params or backend capability defaults before any task start.
Masked img2img runtime selection now flows through strict `inpaintMode` / `img2img_inpaint_mode`, with exact-engine mode discoverability coming from
`/api/engines/capabilities.exact_engine_inpaint_modes` and invalid stale values failing loud before submit. Image run-history snapshots for VAE-owning
families carry the active family-scoped VAE selection so history replay can restore the same asset owner instead of leaking whichever global VAE happens to be selected later.

Symbols (top-level; keep in sync; no ghosts):
- `ImageRunHistoryItem` (interface): Persisted per-tab run history entry (task id, status, summary, params snapshot, error message).
- `GenerationState` (interface): Per-tab reactive runtime state (status/progress with sampling + total-phase metadata/preview/gallery/history selection).
- `defaultState` (function): Creates a fresh `GenerationState` with empty progress/gallery/history.
- `getTabState` (function): Returns (and initializes) the `GenerationState` for a given tab id from internal maps.
- `usesStaticDistilledCfgEngine` (function): Returns whether an engine id uses a fixed distilled-guidance contract independent of checkpoint metadata.
- `resolveEngineForRequest` (function): Canonical tab-type/mode -> backend engine mapping used for capability checks and request dispatch.
- `resolveEffectiveImageSampling` (function): Resolves a strict current/backend sampler-scheduler pair for image requests.
- `buildImg2ImgGuidanceFields` (function): Emits the single img2img guidance field that matches the resolved guidance mode.
- `BuildImg2ImgPayloadArgs` (interface): Input contract for deterministic img2img payload assembly.
- `buildImg2ImgPayload` (function): Builds img2img start payload at the source, including capability-gated unmasked hires fields.
- `buildGuidancePayload` (function): Builds `extras.guidance` payload from tab state + per-engine advanced-guidance support matrix.
- `buildSupirPayload` (function): Normalizes the nested SUPIR request owner into `img2img_extras.supir` and rejects stale conflicting state.
- `usesImageAutomation` (function): Detects when the current image-tab state must use backend-owned `/api/image-automation`.
- `buildIpAdapterPayload` (function): Normalizes the nested IP-Adapter request owner into the route-local `extras.ip_adapter` carrier.
- `resolveAutomationLoop` (function): Builds the bounded vs `until_cancelled` automation loop contract from tab state.
- `buildImageAutomationRequest` (function): Wraps a one-shot image template payload in the automation request envelope.
- `inferRunModeFromSnapshot` (function): Recovers the truthful txt2img/img2img run mode from a persisted params snapshot.
- `inferRunSummaryFromSnapshot` (function): Rebuilds a fallback run-summary string when older resume markers lack explicit summary metadata.
- `ResumeStateLoad` (type): Parsed image resume-state load result (`state` + optional parse error).
- `buildRunItemFromResumeState` (function): Reconstructs the active/terminal run record from persisted resume metadata.
- `resolveErrorRunStatus` (function): Distinguishes `cancelled` from generic terminal errors for history truthfulness.
- `extractLoraNamesFromPrompt` (function): Extracts LoRA token names from prompt text (`<lora:name:weight>`).
- `isGenerationRunningForTab` (function): Returns whether the cached generation state for a tab id is currently `running`.
- `useGeneration` (function): Main composable API; wires payload building, task start, SSE handling, and history updates, enforcing GGUF-required
  `vae_sha`/`tenc_sha` (core-only checkpoints) and enforcing engine-level external asset requirements via backend `asset_contracts`.
- `cancel` (function): Requests cancellation for the current in-flight image task (`/api/tasks/:id/cancel`).
*/

import { computed, reactive, ref } from 'vue'
import {
  useModelTabsStore,
  type BaseTab,
  type GuidanceAdvancedParams,
  type ImageBaseParams,
  type SupirModeFormState,
} from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { normalizeSamplerSchedulerSelection, useEngineCapabilitiesStore, type SamplingDefaults } from '../stores/engine_capabilities'
import { useUpscalersStore } from '../stores/upscalers'
import {
  buildNormalizedHiresOptions,
  buildTxt2ImgPayload,
  type NestedStageSelectorPayloads,
  type Txt2ImgRequest,
} from '../api/payloads'
import { cancelTask, fetchSamplers, fetchSchedulers, fetchTaskResult, startImageAutomation, startImg2Img, startTxt2Img } from '../api/client'
import type {
  FamilyCapabilities,
  GeneratedImage,
  GuidanceAdvancedCapabilities,
  ImageAutomationRequest,
  SamplerInfo,
  SchedulerInfo,
  SupirSamplerInfo,
  TaskErrorCode,
  TaskEvent,
} from '../api/types'
import { resolveImageRequestEngineId } from '../utils/engine_taxonomy'
import { buildExplicitImageRequestContract } from '../utils/image_request_contract'
import { parseInpaintMode } from '../utils/image_params'
import { normalizeImg2ImgResizeModeForEngine } from '../utils/img2img_resize'
import { formatSettingsRevisionConflictMessage, resolveSettingsRevisionConflict } from './settings_revision_conflict'
import { type ResumeLoadResult, useTaskRunLifecycle } from './useTaskRunLifecycle'
import { resolveSupirSelectionState } from './useSupirDiagnostics'

export type ImageRunStatus = 'running' | 'completed' | 'error' | 'cancelled'
const QWEN_IMAGE_ENGINE_ID = 'qwen_image'
const QWEN_IMAGE_PUBLIC_SAMPLER = 'euler'
const QWEN_IMAGE_PUBLIC_SCHEDULER = 'simple'
const ZIMAGE_L2P_ENGINE_ID = 'zimage_l2p'
const QWEN_IMAGE_ASSET_EXTRA_KEYS = new Set([
  'checkpoint_core_only',
  'model_format',
  'model_sha',
  'tenc_sha',
  'vae_sha',
  'vae_source',
])
const QWEN_IMAGE_IMG2IMG_ALLOWED_KEYS = new Set([
  'device',
  'engine',
  'img2img_cfg_scale',
  'img2img_extras',
  'img2img_init_image',
  'img2img_neg_prompt',
  'img2img_prompt',
  'img2img_sampling',
  'img2img_scheduler',
  'img2img_seed',
  'img2img_steps',
  'img2img_styles',
  'model',
  'settings_revision',
])

export interface ImageRunHistoryItem {
  taskId: string
  mode: 'txt2img' | 'img2img'
  createdAtMs: number
  status: ImageRunStatus
  summary: string
  promptPreview: string
  paramsSnapshot: Record<string, unknown>
  thumbnail?: GeneratedImage | null
  errorMessage?: string
}

export interface GenerationState {
  status: 'idle' | 'running' | 'done' | 'error'
  progress: {
    stage: string
    percent: number | null
    etaSeconds: number | null
    step: number | null
    totalSteps: number | null
    message: string | null
    data: Record<string, unknown> | null
    totalPercent: number | null
    totalPhase: string | null
    totalPhaseStep: number | null
    totalPhaseTotalSteps: number | null
    totalPhaseEtaSeconds: number | null
  }
  previewImage: GeneratedImage | null
  previewStep: number | null
  gallery: GeneratedImage[]
  info: unknown | null
  errorMessage: string
  taskId: string
  lastSeed: number | null
  startedAtMs: number | null
  finishedAtMs: number | null
  history: ImageRunHistoryItem[]
  selectedTaskId: string
  historyLoadingTaskId: string
  currentRun: ImageRunHistoryItem | null
}

const MAX_HISTORY = 8
const RESUME_STORAGE_PREFIX = 'codex.resume.image'
const resumeAttempts = new Set<string>()
const LORA_TAG_RE = /<\s*lora\s*:\s*([^:>]+)\s*(?::[^>]*)?>/gi

type ResumeState = {
  taskId: string
  lastEventId: number
  createdAtMs: number
  paramsSnapshot: Record<string, unknown>
  mode: 'txt2img' | 'img2img'
  promptPreview: string
  summary: string
  finishedAtMs: number | null
  terminalStatus: Exclude<ImageRunStatus, 'running'> | null
}

function resumeKey(tabId: string): string {
  return `${RESUME_STORAGE_PREFIX}.${tabId}`
}

function inferRunModeFromSnapshot(paramsSnapshot: Record<string, unknown>): 'txt2img' | 'img2img' {
  return normalizeBooleanParam(paramsSnapshot.useInitImage, false) ? 'img2img' : 'txt2img'
}

function inferPromptPreviewFromSnapshot(paramsSnapshot: Record<string, unknown>): string {
  return String(paramsSnapshot.prompt || '').trim().slice(0, 120)
}

const SUPIR_SAMPLER_SUMMARY_INFO: Record<string, { label: string; scheduler: string }> = {
  restore_heun_edm_stable: { label: 'Restore Heun EDM (Stable)', scheduler: 'karras' },
  restore_euler_edm_stable: { label: 'Restore Euler EDM (Stable)', scheduler: 'karras' },
  restore_dpmpp_2m_stable: { label: 'Restore DPM++ 2M (Stable)', scheduler: 'karras' },
  restore_heun_edm_dev: { label: 'Restore Heun EDM (Dev)', scheduler: 'karras' },
  restore_euler_edm_dev: { label: 'Restore Euler EDM (Dev)', scheduler: 'karras' },
  restore_dpmpp_2m_dev: { label: 'Restore DPM++ 2M (Dev)', scheduler: 'karras' },
}

function inferRunSummaryFromSnapshot(paramsSnapshot: Record<string, unknown>): string {
  const width = Math.trunc(Number(paramsSnapshot.width))
  const height = Math.trunc(Number(paramsSnapshot.height))
  const steps = Math.trunc(Number(paramsSnapshot.steps))
  const cfgScale = Number(paramsSnapshot.cfgScale)
  const supirSnapshot = paramsSnapshot.supir && typeof paramsSnapshot.supir === 'object'
    ? paramsSnapshot.supir as Record<string, unknown>
    : null
  const supirEnabled = Boolean(supirSnapshot?.enabled)
  const supirSamplerKey = String(supirSnapshot?.sampler || '').trim()
  const supirSummaryInfo = SUPIR_SAMPLER_SUMMARY_INFO[supirSamplerKey] ?? null
  const sampler = supirEnabled
    ? (supirSummaryInfo?.label || supirSamplerKey || String(paramsSnapshot.sampler || '').trim())
    : String(paramsSnapshot.sampler || '').trim()
  const scheduler = supirEnabled
    ? (supirSummaryInfo?.scheduler || '')
    : String(paramsSnapshot.scheduler || '').trim()
  const batchCount = Math.max(1, Math.trunc(Number(paramsSnapshot.batchCount || 1)))
  const batchSize = Math.max(1, Math.trunc(Number(paramsSnapshot.batchSize || 1)))
  const seed = Number(paramsSnapshot.seed)
  if (
    !Number.isFinite(width) || width <= 0
    || !Number.isFinite(height) || height <= 0
    || !Number.isFinite(steps) || steps <= 0
    || !Number.isFinite(cfgScale)
  ) {
    return inferRunModeFromSnapshot(paramsSnapshot) === 'img2img' ? 'Img2Img run' : 'Txt2Img run'
  }
  const seedLabel = seed === -1 ? 'seed random' : `seed ${seed}`
  return `${width}×${height} px · ${steps} steps · cfg ${cfgScale} · ${sampler || 'sampler'} / ${scheduler || 'scheduler'} · ${seedLabel} · batch ${batchCount}×${batchSize}`
}

function normalizeResumeTerminalStatus(value: unknown): Exclude<ImageRunStatus, 'running'> | null {
  if (value === 'completed' || value === 'error' || value === 'cancelled') return value
  return null
}

type ResumeStateLoad = ResumeLoadResult<ResumeState>

function loadResumeState(key: string): ResumeStateLoad {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return { state: null, error: null }
    const obj = JSON.parse(raw) as any
    if (!obj || typeof obj !== 'object') return { state: null, error: null }
    if (typeof obj.taskId !== 'string' || !obj.taskId.trim()) return { state: null, error: null }
    const lastEventId = typeof obj.lastEventId === 'number' && Number.isFinite(obj.lastEventId) ? Math.trunc(obj.lastEventId) : 0
    const createdAtMs = typeof obj.createdAtMs === 'number' && Number.isFinite(obj.createdAtMs) ? Math.trunc(obj.createdAtMs) : 0
    const paramsSnapshot = obj.paramsSnapshot && typeof obj.paramsSnapshot === 'object' ? (obj.paramsSnapshot as Record<string, unknown>) : {}
    const mode = obj.mode === 'img2img' || obj.mode === 'txt2img'
      ? obj.mode
      : inferRunModeFromSnapshot(paramsSnapshot)
    const promptPreview = typeof obj.promptPreview === 'string' && obj.promptPreview.trim()
      ? obj.promptPreview.trim()
      : inferPromptPreviewFromSnapshot(paramsSnapshot)
    const summary = typeof obj.summary === 'string' && obj.summary.trim()
      ? obj.summary.trim()
      : inferRunSummaryFromSnapshot(paramsSnapshot)
    const finishedAtMs = typeof obj.finishedAtMs === 'number' && Number.isFinite(obj.finishedAtMs)
      ? Math.trunc(obj.finishedAtMs)
      : null
    return {
      state: {
        taskId: obj.taskId,
        lastEventId: Math.max(0, lastEventId),
        createdAtMs,
        paramsSnapshot,
        mode,
        promptPreview,
        summary,
        finishedAtMs,
        terminalStatus: normalizeResumeTerminalStatus(obj.terminalStatus),
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

function patchResumeState(key: string, patch: Partial<ResumeState>): void {
  const current = loadResumeState(key).state
  if (!current) return
  saveResumeState(key, { ...current, ...patch })
}

function updateResumeEventId(key: string, eventId: number): void {
  const v = Math.trunc(Number(eventId))
  if (!Number.isFinite(v) || v <= 0) return
  const cur = loadResumeState(key).state
  if (!cur || v <= cur.lastEventId) return
  patchResumeState(key, { lastEventId: v })
}

function buildRunItemFromResumeState(
  saved: ResumeState,
  patch: Partial<ImageRunHistoryItem> = {},
): ImageRunHistoryItem {
  return {
    taskId: saved.taskId,
    mode: saved.mode,
    createdAtMs: saved.createdAtMs,
    status: 'running',
    summary: saved.summary,
    promptPreview: saved.promptPreview,
    paramsSnapshot: saved.paramsSnapshot,
    thumbnail: null,
    ...patch,
  }
}

function resolveErrorRunStatus(
  code?: TaskErrorCode,
  message?: string,
): Exclude<ImageRunStatus, 'running' | 'completed'> {
  if (code === 'cancelled') return 'cancelled'
  return String(message || '').trim().toLowerCase() === 'cancelled' ? 'cancelled' : 'error'
}

function defaultState(): GenerationState {
  return {
    status: 'idle',
    progress: {
      stage: 'none',
      percent: null,
      etaSeconds: null,
      step: null,
      totalSteps: null,
      message: null,
      data: null,
      totalPercent: null,
      totalPhase: null,
      totalPhaseStep: null,
      totalPhaseTotalSteps: null,
      totalPhaseEtaSeconds: null,
    },
    previewImage: null,
    previewStep: null,
    gallery: [],
    info: null,
    errorMessage: '',
    taskId: '',
    lastSeed: null,
    startedAtMs: null,
    finishedAtMs: null,
    history: [],
    selectedTaskId: '',
    historyLoadingTaskId: '',
    currentRun: null,
  }
}

export function resolveEngineForRequest(tabType: string, useInitImage: boolean): string {
  return resolveImageRequestEngineId(tabType, useInitImage)
}

export function resolveEffectiveImageSampling(opts: {
  engineId: string
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  familyCapabilities: FamilyCapabilities
  currentSampler: string | null | undefined
  currentScheduler: string | null | undefined
  backendDefaults: SamplingDefaults | null
}): SamplingDefaults {
  const resolved = normalizeSamplerSchedulerSelection({
    samplers: opts.samplers,
    schedulers: opts.schedulers,
    familyCapabilities: opts.familyCapabilities,
    sampler: opts.currentSampler,
    scheduler: opts.currentScheduler,
    preferredSamplers: opts.backendDefaults ? [opts.backendDefaults.sampler] : [],
    preferredSchedulers: opts.backendDefaults ? [opts.backendDefaults.scheduler] : [],
  })
  if (resolved) return resolved
  throw new Error(`Image generation requires a valid sampler and scheduler for '${opts.engineId}'. Select valid values or refresh backend capabilities.`)
}

function normalizeBooleanParam(rawValue: unknown, fallback: boolean): boolean {
  if (typeof rawValue === 'boolean') return rawValue
  if (typeof rawValue === 'number') {
    if (rawValue === 1) return true
    if (rawValue === 0) return false
  }
  if (typeof rawValue === 'string') {
    const normalized = rawValue.trim().toLowerCase()
    if (normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on') return true
    if (normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'off') return false
  }
  return fallback
}

function toNullableNumber(rawValue: unknown): number | null {
  if (rawValue === null || rawValue === undefined) return null
  const value = typeof rawValue === 'number' ? rawValue : Number(rawValue)
  if (!Number.isFinite(value)) return null
  return value
}

function toNullableInteger(rawValue: unknown): number | null {
  const value = toNullableNumber(rawValue)
  if (value === null) return null
  return Math.trunc(value)
}

function toNullableRecord(rawValue: unknown): Record<string, unknown> | null {
  if (!rawValue || typeof rawValue !== 'object' || Array.isArray(rawValue)) return null
  return rawValue as Record<string, unknown>
}

function buildProgressStateFromPayload(
  payload: {
    stage?: string | null
    percent?: number | null
    eta_seconds?: number | null
    step?: number | null
    total_steps?: number | null
    message?: string | null
    data?: Record<string, unknown> | null
  },
  options: { fallbackStage: string },
): GenerationState['progress'] {
  const fallbackStage = String(options.fallbackStage || 'running')
  const data = toNullableRecord(payload.data)
  return {
    stage: String(payload.stage ?? fallbackStage),
    percent: toNullableNumber(payload.percent),
    etaSeconds: toNullableNumber(payload.eta_seconds),
    step: toNullableInteger(payload.step),
    totalSteps: toNullableInteger(payload.total_steps),
    message: payload.message === null || payload.message === undefined ? null : String(payload.message),
    data,
    totalPercent: toNullableNumber(data?.total_percent),
    totalPhase: data?.total_phase === undefined || data?.total_phase === null ? null : String(data.total_phase),
    totalPhaseStep: toNullableInteger(data?.phase_step),
    totalPhaseTotalSteps: toNullableInteger(data?.phase_total_steps),
    totalPhaseEtaSeconds: toNullableNumber(data?.phase_eta_seconds),
  }
}

export interface BuildImg2ImgPayloadArgs {
  params: ImageBaseParams
  supportsNegativePrompt: boolean
  supportsHires: boolean
  guidanceMode: 'cfg' | 'distilled_cfg'
  batchCount: number
  batchSize: number
  device: string
  settingsRevision: number
  engineId: string
  modelOverride: string
  hiresFallbackOnOom: boolean
  hiresMinTile: number
  extras: Record<string, unknown>
  samplerOverride?: string
  schedulerOverride?: string
}

function buildImg2ImgGuidanceFields(
  guidanceScale: number,
  guidanceMode: 'cfg' | 'distilled_cfg',
): Record<string, number> {
  if (guidanceMode === 'distilled_cfg') {
    return { img2img_distilled_cfg_scale: guidanceScale }
  }
  return { img2img_cfg_scale: guidanceScale }
}

function isQwenImageEngine(engineId: string): boolean {
  return String(engineId || '').trim().toLowerCase() === QWEN_IMAGE_ENGINE_ID
}

function isZImageL2PEngine(engineId: string): boolean {
  return String(engineId || '').trim().toLowerCase() === ZIMAGE_L2P_ENGINE_ID
}

function assertAllowedRecordKeys(record: Record<string, unknown>, allowed: ReadonlySet<string>, context: string): void {
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) {
      throw new Error(`${context}.${key} is unsupported for Qwen Image.`)
    }
  }
}

function buildQwenImageAssetExtras(rawExtras: Record<string, unknown>, context: string): Record<string, unknown> {
  const extras: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(rawExtras)) {
    if (value === undefined) continue
    if (!QWEN_IMAGE_ASSET_EXTRA_KEYS.has(key)) {
      throw new Error(`${context}.${key} is unsupported for Qwen Image.`)
    }
    extras[key] = value
  }
  const missing: string[] = []
  for (const key of QWEN_IMAGE_ASSET_EXTRA_KEYS) {
    if (extras[key] === undefined || extras[key] === null || extras[key] === '') {
      missing.push(key)
    }
  }
  if (missing.length > 0) {
    throw new Error(`Qwen Image requires ${missing.map((key) => `${context}.${key}`).join(', ')}.`)
  }
  if (extras.vae_source !== 'external') {
    throw new Error('Qwen Image requires an external VAE selection.')
  }
  return extras
}

function assertQwenImageRequestState(args: {
  params: ImageBaseParams
  useInitImage: boolean
  batchCount: number
  batchSize: number
  loraNames: readonly string[]
}): void {
  const params = args.params
  if (!args.useInitImage) {
    throw new Error('Qwen Image Edit-2511 supports img2img only. Select an init image.')
  }
  if (params.runAction === 'infinite') {
    throw new Error('Qwen Image does not support Infinite generate. Use Generate.')
  }
  if (args.batchCount !== 1 || args.batchSize !== 1) {
    throw new Error('Qwen Image requests require batch count = 1 and batch size = 1.')
  }
  const clipSkip = Number(params.clipSkip)
  if (Number.isFinite(clipSkip) && Math.trunc(clipSkip) > 0) {
    throw new Error('Qwen Image does not support CLIP Skip. Reset CLIP Skip to 0.')
  }
  if (normalizeBooleanParam(params.hires?.enabled, false)) {
    throw new Error('Qwen Image does not support Hires Fix. Disable Hires before generating.')
  }
  if (normalizeBooleanParam(params.swapModel?.enabled, false)) {
    throw new Error('Qwen Image does not support first-pass model swap. Disable model swap before generating.')
  }
  if (normalizeBooleanParam(params.refiner?.enabled, false)) {
    throw new Error('Qwen Image does not support refiner. Disable refiner before generating.')
  }
  if (normalizeBooleanParam(params.ipAdapter?.enabled, false)) {
    throw new Error('Qwen Image does not support IP-Adapter. Disable IP-Adapter before generating.')
  }
  if (normalizeBooleanParam(params.guidanceAdvanced?.enabled, false)) {
    throw new Error('Qwen Image does not support Advanced Guidance/APG. Disable Advanced Guidance before generating.')
  }
  if (args.loraNames.length > 0) {
    throw new Error('Qwen Image does not support LoRA prompt tags.')
  }
  if (normalizeBooleanParam(params.useMask, false)) {
    throw new Error('Qwen Image Edit does not support masks or inpaint. Disable INPAINT before generating.')
  }
  if (params.initSource.mode !== 'img') {
    throw new Error('Qwen Image Edit requires a single init image. Switch the initial image source to IMG.')
  }
}

function assertZImageL2PRequestState(args: {
  params: ImageBaseParams
  useInitImage: boolean
  batchCount: number
  batchSize: number
  loraNames: readonly string[]
}): void {
  const params = args.params
  if (args.useInitImage) {
    throw new Error('Z-Image L2P supports txt2img only. Disable IMG2IMG.')
  }
  if (params.runAction === 'infinite') {
    throw new Error('Z-Image L2P does not support Infinite generate. Use Generate.')
  }
  if (args.batchCount !== 1 || args.batchSize !== 1) {
    throw new Error('Z-Image L2P requests require batch count = 1 and batch size = 1.')
  }
  if (Math.trunc(Number(params.width)) !== 1024 || Math.trunc(Number(params.height)) !== 1024) {
    throw new Error('Z-Image L2P requires 1024x1024 output.')
  }
  const clipSkip = Number(params.clipSkip)
  if (Number.isFinite(clipSkip) && Math.trunc(clipSkip) !== 0) {
    throw new Error('Z-Image L2P does not support CLIP Skip. Reset CLIP Skip to 0.')
  }
  if (normalizeBooleanParam(params.hires?.enabled, false)) {
    throw new Error('Z-Image L2P does not support Hires Fix. Disable Hires before generating.')
  }
  if (normalizeBooleanParam(params.swapModel?.enabled, false)) {
    throw new Error('Z-Image L2P does not support first-pass model swap. Disable model swap before generating.')
  }
  if (normalizeBooleanParam(params.refiner?.enabled, false)) {
    throw new Error('Z-Image L2P does not support refiner. Disable refiner before generating.')
  }
  if (normalizeBooleanParam(params.ipAdapter?.enabled, false)) {
    throw new Error('Z-Image L2P does not support IP-Adapter. Disable IP-Adapter before generating.')
  }
  if (normalizeBooleanParam(params.guidanceAdvanced?.enabled, false)) {
    throw new Error('Z-Image L2P does not support Advanced Guidance/APG. Disable Advanced Guidance before generating.')
  }
  if (args.loraNames.length > 0) {
    throw new Error('Z-Image L2P does not support LoRA prompt tags.')
  }
}

function buildQwenImageImg2ImgPayload(args: BuildImg2ImgPayloadArgs): Record<string, unknown> {
  const params = args.params
  assertQwenImageRequestState({
    params,
    useInitImage: true,
    batchCount: args.batchCount,
    batchSize: args.batchSize,
    loraNames: [],
  })
  if (args.batchCount !== 1 || args.batchSize !== 1) {
    throw new Error('Qwen Image Edit requires batch count = 1 and batch size = 1.')
  }
  if (normalizeBooleanParam(params.useMask, false)) {
    throw new Error('Qwen Image Edit does not support masks or inpaint. Disable INPAINT before generating.')
  }
  if (!String(params.initImageData || '').trim()) {
    throw new Error('Qwen Image Edit requires an init image.')
  }
  const steps = Number(params.steps)
  if (!Number.isInteger(steps) || steps < 2) {
    throw new Error('Qwen Image Edit requires at least 2 integer denoise steps.')
  }
  const effectiveSampler = String(args.samplerOverride || params.sampler || '').trim()
  const effectiveScheduler = String(args.schedulerOverride || params.scheduler || '').trim()
  if (effectiveSampler.toLowerCase() !== QWEN_IMAGE_PUBLIC_SAMPLER) {
    throw new Error(`Qwen Image Edit requires sampler '${QWEN_IMAGE_PUBLIC_SAMPLER}'.`)
  }
  if (effectiveScheduler.toLowerCase() !== QWEN_IMAGE_PUBLIC_SCHEDULER) {
    throw new Error(`Qwen Image Edit requires scheduler '${QWEN_IMAGE_PUBLIC_SCHEDULER}'.`)
  }
  const payload: Record<string, unknown> = {
    img2img_init_image: params.initImageData,
    img2img_prompt: params.prompt,
    img2img_neg_prompt: args.supportsNegativePrompt ? params.negativePrompt : '',
    img2img_styles: [],
    img2img_steps: steps,
    img2img_cfg_scale: params.cfgScale,
    img2img_sampling: effectiveSampler,
    img2img_scheduler: effectiveScheduler,
    img2img_seed: params.seed,
    device: args.device,
    settings_revision: args.settingsRevision,
    engine: QWEN_IMAGE_ENGINE_ID,
    model: args.modelOverride,
    img2img_extras: buildQwenImageAssetExtras(args.extras, 'img2img_extras'),
  }
  assertAllowedRecordKeys(payload, QWEN_IMAGE_IMG2IMG_ALLOWED_KEYS, 'Qwen Image img2img payload')
  assertAllowedRecordKeys(payload.img2img_extras as Record<string, unknown>, QWEN_IMAGE_ASSET_EXTRA_KEYS, 'img2img_extras')
  return payload
}

export function buildImg2ImgPayload(args: BuildImg2ImgPayloadArgs): Record<string, unknown> {
  if (isQwenImageEngine(args.engineId)) {
    return buildQwenImageImg2ImgPayload(args)
  }

  const params = args.params
  const effectiveSampler = String(args.samplerOverride || params.sampler || '').trim()
  const effectiveScheduler = String(args.schedulerOverride || params.scheduler || '').trim()
  const useMask = normalizeBooleanParam(params.useMask, false)
  const hiresEnabled = normalizeBooleanParam(params.hires?.enabled, false)
  const maskInvert = normalizeBooleanParam(params.maskInvert, false)
  const maskRound = normalizeBooleanParam(params.maskRound, true)
  const maskRegionSplit = normalizeBooleanParam(params.maskRegionSplit, false)
  const resizeMode = normalizeImg2ImgResizeModeForEngine(args.engineId, params.img2imgResizeMode)
  const guidanceFields = buildImg2ImgGuidanceFields(params.cfgScale, args.guidanceMode)
  if (hiresEnabled && !args.supportsHires) {
    throw new Error(`This engine does not support img2img hires (${args.engineId}).`)
  }
  if (Boolean(params.swapModel?.enabled)) {
    throw new Error('img2img swap_model is not supported yet. Disable the generic model swap or switch back to txt2img.')
  }
  if (hiresEnabled && useMask) {
    throw new Error('Masked img2img hires is not supported yet. Disable the mask or hires before generating.')
  }
  if (hiresEnabled && String(params.hires?.swapModel?.model || '').trim()) {
    throw new Error('img2img hires swap_model is not supported yet. Disable the second-pass model or switch back to txt2img.')
  }
  if (hiresEnabled && params.hires?.refiner?.enabled) {
    throw new Error('img2img hires refiner is not supported yet. Disable the hires refiner or switch back to txt2img.')
  }
  if (params.refiner?.enabled) {
    throw new Error('img2img refiner is not supported yet. Disable the refiner or switch back to txt2img.')
  }
  const payload: Record<string, unknown> = {
    img2img_init_image: params.initImageData,
    img2img_prompt: params.prompt,
    img2img_neg_prompt: args.supportsNegativePrompt ? params.negativePrompt : '',
    img2img_styles: [],
    img2img_batch_count: args.batchCount,
    img2img_batch_size: args.batchSize,
    img2img_steps: params.steps,
    ...guidanceFields,
    img2img_denoising_strength: params.denoiseStrength,
    img2img_width: params.width,
    img2img_height: params.height,
    img2img_sampling: effectiveSampler,
    img2img_scheduler: effectiveScheduler,
    img2img_seed: params.seed,
    img2img_clip_skip: params.clipSkip,
    device: args.device,
    settings_revision: args.settingsRevision,
    engine: args.engineId,
    model: args.modelOverride,
    img2img_extras: { ...args.extras },
  }
  if (hiresEnabled) {
    const hiresPayload = buildNormalizedHiresOptions(
      {
        prompt: params.prompt,
        negativePrompt: args.supportsNegativePrompt ? params.negativePrompt : '',
        hires: params.hires,
      },
      args.guidanceMode,
      { hiresFallbackOnOom: args.hiresFallbackOnOom, hiresMinTile: args.hiresMinTile },
    )
    if (hiresPayload) {
      const img2imgExtras = payload.img2img_extras as Record<string, unknown>
      img2imgExtras.hires = hiresPayload
    }
  }
  if (args.engineId === 'zimage' && !useMask) {
    payload.img2img_resize_mode = resizeMode
  }
  if (useMask) {
    const maskData = String(params.maskImageData || '').trim()
    if (!maskData) {
      throw new Error('INPAINT is enabled but no mask is applied. Open the mask editor and apply a mask.')
    }
    payload.img2img_mask = maskData
    const inpaintMode = parseInpaintMode(params.inpaintMode)
    if (inpaintMode === null) {
      throw new Error(`INPAINT is enabled but inpaintMode '${String(params.inpaintMode ?? '')}' is invalid.`)
    }
    payload.img2img_inpaint_mode = inpaintMode
    payload.img2img_inpainting_fill = Math.max(0, Math.min(3, Math.trunc(Number(params.inpaintingFill))))
    payload.img2img_inpaint_full_res_padding = Math.max(0, Math.trunc(Number(params.inpaintFullResPadding)))
    payload.img2img_inpainting_mask_invert = maskInvert ? 1 : 0
    payload.img2img_mask_blur = Math.max(0, Math.trunc(Number(params.maskBlur)))
    payload.img2img_mask_round = maskRound
    if (inpaintMode === 'per_step_blend') {
      const rawPerStepBlendStrength = Number(params.perStepBlendStrength)
      const perStepBlendStrength = Number.isFinite(rawPerStepBlendStrength)
        ? Math.max(0, Math.min(1, rawPerStepBlendStrength))
        : 1
      const rawPerStepBlendSteps = Number(params.perStepBlendSteps)
      const perStepBlendSteps = Number.isFinite(rawPerStepBlendSteps)
        ? Math.max(0, Math.trunc(rawPerStepBlendSteps))
        : 0
      payload.img2img_per_step_blend_strength = perStepBlendStrength
      payload.img2img_per_step_blend_steps = perStepBlendSteps
    }

    const wantsRegionSplit = maskRegionSplit
    if (wantsRegionSplit) {
      if (maskInvert) {
        throw new Error('Mask region splitting is not supported with "Invert mask".')
      }
      if (args.batchSize !== 1) {
        throw new Error('Mask region splitting currently requires batch size = 1.')
      }
    }
    payload.img2img_mask_region_split = wantsRegionSplit
  }
  return payload
}

function buildSupirPayload(
  supir: SupirModeFormState,
  options: {
    engineId: string
    useInitImage: boolean
    supportsSupirMode: boolean
    hiresEnabled: boolean
    ipAdapterEnabled: boolean
    hasLoraSelection: boolean
    guidanceAdvancedEnabled: boolean
  },
): { payload: Record<string, unknown>; samplerInfo: SupirSamplerInfo } | null {
  if (!normalizeBooleanParam(supir.enabled, false)) return null
  if (!options.useInitImage) {
    throw new Error('SUPIR mode is only available for SDXL img2img/inpaint.')
  }
  if (!options.supportsSupirMode || options.engineId !== 'sdxl') {
    throw new Error('SUPIR mode is only available for native SDXL img2img/inpaint.')
  }
  if (options.hiresEnabled) {
    throw new Error('SUPIR mode cannot be combined with img2img hires. Disable SUPIR mode or hires before generating.')
  }
  if (options.ipAdapterEnabled) {
    throw new Error('SUPIR mode cannot be combined with IP-Adapter. Disable SUPIR mode or IP-Adapter before generating.')
  }
  if (options.hasLoraSelection) {
    throw new Error('SUPIR mode cannot be combined with LoRA selections. Remove LoRA tags before generating.')
  }
  const selection = resolveSupirSelectionState({
    supported: true,
    selectedVariant: supir.variant,
    selectedSampler: supir.sampler,
    guidanceAdvancedEnabled: options.guidanceAdvancedEnabled,
  })
  if (selection.blockingReason) {
    throw new Error(selection.blockingReason)
  }
  if (!selection.selectedSamplerInfo) {
    throw new Error(`Selected SUPIR sampler '${String(supir.sampler || '').trim()}' is unavailable.`)
  }
  const variant = supir.variant === 'v0F' ? 'v0F' : 'v0Q'
  const colorFix = supir.colorFix === 'AdaIN' || supir.colorFix === 'Wavelet' ? supir.colorFix : 'None'
  const controlScale = Number.isFinite(Number(supir.controlScale)) ? Number(supir.controlScale) : 0.8
  const restorationScale = Number.isFinite(Number(supir.restorationScale)) ? Number(supir.restorationScale) : 4
  const restoreCfgSTmin = Number.isFinite(Number(supir.restoreCfgSTmin)) ? Number(supir.restoreCfgSTmin) : 0.05
  return {
    payload: {
      enabled: true,
      variant,
      sampler: selection.selectedSamplerInfo.id,
      controlScale: Math.min(2, Math.max(0.01, controlScale)),
      restorationScale: Math.min(6, Math.max(0.01, restorationScale)),
      restoreCfgSTmin: Math.min(5, Math.max(0, restoreCfgSTmin)),
      colorFix,
    },
    samplerInfo: selection.selectedSamplerInfo,
  }
}

export function buildGuidancePayload(
  guidanceAdvanced: GuidanceAdvancedParams,
  support: GuidanceAdvancedCapabilities | null | undefined,
): Record<string, unknown> | null {
  if (!guidanceAdvanced.enabled) return null
  if (!support) return null

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

  const payload: Record<string, unknown> = {}

  const apgEnabled = normalizeBooleanParam(guidanceAdvanced.apgEnabled, false)
  const cfgTruncEnabled = normalizeBooleanParam(guidanceAdvanced.cfgTruncEnabled, false)

  if (support.apg_enabled && apgEnabled) payload.apg_enabled = true
  if (support.apg_start_step) payload.apg_start_step = clampInt(guidanceAdvanced.apgStartStep, 0, 0)
  if (support.apg_eta) payload.apg_eta = toFinite(guidanceAdvanced.apgEta, 0)
  if (support.apg_momentum) payload.apg_momentum = clamp(guidanceAdvanced.apgMomentum, 0, 0, 0.999999)
  if (support.apg_norm_threshold) payload.apg_norm_threshold = clamp(guidanceAdvanced.apgNormThreshold, 15, 0)
  if (support.apg_rescale) payload.apg_rescale = clamp(guidanceAdvanced.apgRescale, 0, 0, 1)
  if (support.guidance_rescale) payload.guidance_rescale = clamp(guidanceAdvanced.guidanceRescale, 0, 0, 1)
  if (support.cfg_trunc_ratio && cfgTruncEnabled) {
    payload.cfg_trunc_ratio = clamp(guidanceAdvanced.cfgTruncRatio, 0.8, 0, 1)
  }
  if (support.renorm_cfg) payload.renorm_cfg = clamp(guidanceAdvanced.renormCfg, 0, 0)

  return Object.keys(payload).length > 0 ? payload : null
}

function extractLoraNamesFromPrompt(prompt: string): string[] {
  const names: string[] = []
  const seen = new Set<string>()
  const text = String(prompt || '')
  let match: RegExpExecArray | null = null
  LORA_TAG_RE.lastIndex = 0
  while ((match = LORA_TAG_RE.exec(text)) !== null) {
    const name = String(match[1] || '').trim()
    if (!name) continue
    const key = name.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    names.push(name)
  }
  return names
}

export function isGenerationRunningForTab(tabId: string): boolean {
  return getTabState(tabId).status === 'running'
}

function usesStaticDistilledCfgEngine(engineId: string): boolean {
  return engineId === 'flux1' || engineId === 'flux1_kontext' || engineId === 'flux1_chroma'
}

// Per-tab generation state (keyed by tab ID)
const tabStates = new Map<string, GenerationState>()
const unsubscribers = new Map<string, () => void>()

function getTabState(tabId: string): GenerationState {
  if (!tabStates.has(tabId)) {
    tabStates.set(tabId, reactive(defaultState()) as GenerationState)
  }
  return tabStates.get(tabId)!
}

export function useGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()
  const backendCaps = useEngineCapabilitiesStore()
  const upscalersStore = useUpscalersStore()
  
  // Reactive state for this tab
  const state = ref(getTabState(tabId))
  let currentEventId = 0
  let skipAutomationReplayThroughEventId = 0
  
  // Tab info
  const tab = computed(() => modelTabs.tabs.find(t => t.id === tabId) as BaseTab | undefined)
  const params = computed(() => tab.value?.params as ImageBaseParams | undefined)
  const engineType = computed(() => tab.value?.type as string | undefined)

  function buildRunSummary(
    p: ImageBaseParams,
    guidanceMode: 'cfg' | 'distilled_cfg',
    effectiveSampling?: { sampler: string; scheduler: string; supirLabel?: string } | null,
    engineId?: string,
  ): string {
    const sampler = p.supir.enabled
      ? String(effectiveSampling?.supirLabel || effectiveSampling?.sampler || p.supir.sampler || p.sampler || '').trim()
      : String(effectiveSampling?.sampler || p.sampler || '').trim()
    const scheduler = String(effectiveSampling?.scheduler || p.scheduler || '').trim()
    const seedLabel = p.seed === -1 ? 'seed random' : `seed ${p.seed}`
    const clipSkipLabel = Number.isFinite(p.clipSkip) && p.clipSkip > 0 && p.clipSkip !== 1 ? ` · clip-skip ${p.clipSkip}` : ''
    const guidanceLabel = guidanceMode === 'distilled_cfg' ? 'distilled cfg' : 'cfg'
    const batchLabel = isZImageL2PEngine(engineId || '') ? '' : ` · batch ${p.batchCount}×${p.batchSize}`
    return `${p.width}×${p.height} px · ${p.steps} steps · ${guidanceLabel} ${p.cfgScale} · ${sampler} / ${scheduler} · ${seedLabel}${clipSkipLabel}${batchLabel}`
  }

  function usesImageAutomation(p: ImageBaseParams, engineId: string): boolean {
    if (isZImageL2PEngine(engineId)) return false
    return (
      p.runAction === 'infinite'
      || (normalizeBooleanParam(p.useInitImage, false) && p.initSource.mode === 'dir')
      || (p.ipAdapter.enabled && p.ipAdapter.source.mode === 'dir')
    )
  }

  function buildIpAdapterPayload(
    p: ImageBaseParams,
    options: { useInitImage: boolean },
  ): Record<string, unknown> | null {
    if (!p.ipAdapter.enabled) return null
    const source = p.ipAdapter.source
    const model = String(p.ipAdapter.model || '').trim()
    if (!model) {
      throw new Error('Select an IP-Adapter model.')
    }
    const imageEncoder = String(p.ipAdapter.imageEncoder || '').trim()
    if (!imageEncoder) {
      throw new Error('Select an IP-Adapter image encoder.')
    }
    const payload: Record<string, unknown> = {
      enabled: true,
      model,
      image_encoder: imageEncoder,
      weight: p.ipAdapter.weight,
      start_at: p.ipAdapter.startAt,
      end_at: p.ipAdapter.endAt,
    }
    if (source.mode === 'dir') {
      const folderPath = String(source.folderPath || '').trim()
      if (!folderPath) {
        throw new Error('IP-Adapter folder mode requires a folder path.')
      }
      payload.source = {
        kind: 'server_folder',
        folder_path: folderPath,
        selection_mode: source.selectionMode,
        count: source.selectionMode === 'count' ? source.count : null,
        order: source.order,
        sort_by: source.sortBy,
      }
      return payload
    }
    if (source.sameAsInit) {
      if (!options.useInitImage) {
        throw new Error('Same as init image is only available for img2img runs.')
      }
      payload.source = { kind: 'same_as_init' }
      return payload
    }
    const referenceImageData = String(source.referenceImageData || '').trim()
    if (!referenceImageData) {
      throw new Error('Select an IP-Adapter reference image.')
    }
    payload.source = {
      kind: 'uploaded',
      reference_image_data: referenceImageData,
    }
    return payload
  }

  function resolveAutomationLoop(p: ImageBaseParams): ImageAutomationRequest['loop'] {
    if (p.runAction === 'infinite') {
      return {
        mode: 'until_cancelled',
        delay_ms: 0,
        stop_on_error: true,
      }
    }

    const candidateCounts: number[] = []
    let hasAllSelection = false
    if (normalizeBooleanParam(p.useInitImage, false) && p.initSource.mode === 'dir') {
      if (p.initSource.selectionMode === 'all') hasAllSelection = true
      else candidateCounts.push(Math.max(1, Math.trunc(Number(p.initSource.count || 1))))
    }
    if (p.ipAdapter.enabled && p.ipAdapter.source.mode === 'dir') {
      if (p.ipAdapter.source.selectionMode === 'all') hasAllSelection = true
      else candidateCounts.push(Math.max(1, Math.trunc(Number(p.ipAdapter.source.count || 1))))
    }

    return {
      mode: 'count',
      count: hasAllSelection ? null : Math.max(1, ...candidateCounts),
      delay_ms: 0,
      stop_on_error: true,
    }
  }

  function buildImageAutomationRequest(args: {
    mode: 'txt2img' | 'img2img'
    template: Record<string, unknown>
    params: ImageBaseParams
  }): ImageAutomationRequest {
    const request: ImageAutomationRequest = {
      mode: args.mode,
      template: args.template,
      loop: resolveAutomationLoop(args.params),
      seed_policy: {
        mode: 'increment',
        increment_step: 1,
      },
      prompt_source: {
        kind: 'current',
        insert_position: 'replace',
        wildcard_mode: 'disabled',
      },
    }
    if (args.mode === 'img2img') {
      request.init_source = args.params.initSource.mode === 'dir'
        ? {
            kind: 'server_folder',
            folder_path: args.params.initSource.folderPath,
            selection_mode: args.params.initSource.selectionMode,
            count: args.params.initSource.selectionMode === 'count' ? args.params.initSource.count : null,
            order: args.params.initSource.order,
            sort_by: args.params.initSource.sortBy,
            use_crop: args.params.initSource.useCrop,
          }
        : {
            kind: 'uploaded_current',
            use_crop: false,
            order: 'sorted',
          }
    }
    return request
  }

  function buildParamsSnapshot(
    p: ImageBaseParams,
    tabType: string,
    effectiveSampling?: { sampler: string; scheduler: string } | null,
    engineId?: string,
  ): Record<string, unknown> {
    const familyOwnedVae = isZImageL2PEngine(engineId || '') ? '' : String(quicksettings.getVaeForFamily(tabType) || '').trim()
    const snapshot: Record<string, unknown> = {
      checkpoint: p.checkpoint,
      textEncoders: p.textEncoders,

      prompt: p.prompt,
      negativePrompt: p.negativePrompt,

      width: p.width,
      height: p.height,
      sampler: String(effectiveSampling?.sampler || p.sampler || '').trim(),
      scheduler: String(effectiveSampling?.scheduler || p.scheduler || '').trim(),
      steps: p.steps,
      cfgScale: p.cfgScale,
      seed: p.seed,
      clipSkip: p.clipSkip,

      batchSize: p.batchSize,
      batchCount: p.batchCount,
      runAction: p.runAction,
      img2imgResizeMode: p.img2imgResizeMode,
      img2imgUpscaler: p.img2imgUpscaler,
      guidanceAdvanced: p.guidanceAdvanced,

      hires: p.hires,
      swapModel: p.swapModel,
      refiner: p.refiner,

      useInitImage: p.useInitImage,
      initSource: p.initSource,
      initImageName: p.initImageName,
      denoiseStrength: p.denoiseStrength,
      useMask: p.useMask,
      maskImageName: p.maskImageName,
      inpaintMode: p.inpaintMode,
      perStepBlendStrength: p.perStepBlendStrength,
      perStepBlendSteps: p.perStepBlendSteps,
      inpaintFullResPadding: p.inpaintFullResPadding,
      inpaintingFill: p.inpaintingFill,
      maskInvert: p.maskInvert,
      maskBlur: p.maskBlur,
      maskRound: p.maskRound,
      maskRegionSplit: p.maskRegionSplit,
      supir: p.supir,
      ipAdapter: p.ipAdapter,
    }
    if (!isZImageL2PEngine(engineId || '')) {
      snapshot.vae = familyOwnedVae
    }
    return snapshot
  }

  function pushHistory(item: ImageRunHistoryItem): void {
    state.value.history = [item, ...state.value.history.filter((entry) => entry.taskId !== item.taskId)]
    if (state.value.history.length > MAX_HISTORY) state.value.history.length = MAX_HISTORY
  }

  function resetAutomationRecoveryReplayState(): void {
    currentEventId = 0
    skipAutomationReplayThroughEventId = 0
  }

  function markAutomationRecoveryWatermark(eventId: unknown): void {
    if (typeof eventId !== 'number' || !Number.isFinite(eventId)) return
    const normalized = Math.max(0, Math.trunc(eventId))
    if (normalized <= 0) return
    skipAutomationReplayThroughEventId = Math.max(skipAutomationReplayThroughEventId, normalized)
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
    onTaskEvent: handleTaskEvent,
    isSnapshotRunning: (snapshot) => snapshot.status === 'running',
    onResumeRunning: handleResumedRunningSnapshot,
    onResumeTerminal: handleResumeTerminalSnapshot,
    onHistoryLoaded: handleHistorySnapshot,
    onHistoryLoadError: (_taskId, error) => {
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
    },
    stopStreamBeforeHistoryLoad: true,
    getResumeAttachOptions: (saved) => ({
      after: saved.lastEventId,
      onEventId: (eventId) => {
        if (!Number.isFinite(eventId)) return
        currentEventId = Math.max(0, Math.trunc(eventId))
      },
    }),
    onStopStream: resetAutomationRecoveryReplayState,
  })
  const resumeNotice = taskLifecycle.resumeNotice
  
  function stopStream(): void {
    taskLifecycle.stopStream()
  }
  
  function resetProgress(): void {
    state.value.progress = {
      stage: 'none',
      percent: null,
      etaSeconds: null,
      step: null,
      totalSteps: null,
      message: null,
      data: null,
      totalPercent: null,
      totalPhase: null,
      totalPhaseStep: null,
      totalPhaseTotalSteps: null,
      totalPhaseEtaSeconds: null,
    }
  }
  
  async function generate(): Promise<void> {
    if (!tab.value || !params.value || !engineType.value) {
      state.value.status = 'error'
      state.value.errorMessage = 'Tab not found'
      return
    }

    await backendCaps.init()
    
    stopStream()
    state.value.status = 'running'
    state.value.errorMessage = ''
    state.value.gallery = []
    state.value.info = null
    state.value.previewImage = null
    state.value.previewStep = null
    state.value.selectedTaskId = ''
    resetProgress()
    state.value.progress.stage = 'starting'
    state.value.startedAtMs = Date.now()
    state.value.finishedAtMs = null

    const p = params.value
    const checkpoint = String((p as any).checkpoint || '').trim()
    const useInitImage = normalizeBooleanParam(p.useInitImage, false)
    const usesInitFolderSource = useInitImage && p.initSource.mode === 'dir'
    const useMask = normalizeBooleanParam(p.useMask, false)
    const tabType = String(engineType.value)
    const engineOverrideForRequest = resolveEngineForRequest(tabType, useInitImage)
    const modelRef = String(checkpoint || '').trim()

    const engineSurface = backendCaps.get(engineOverrideForRequest)
    if (!engineSurface) {
      const message = `Engine capabilities missing for '${engineOverrideForRequest}'.`
      console.error(`[useGeneration] ${message}`)
      state.value.status = 'error'
      state.value.errorMessage = message
      return
    }
    const familyCaps = backendCaps.getFamilyForEngine(engineOverrideForRequest)
    if (!familyCaps) {
      const message = `Family capabilities missing for '${engineOverrideForRequest}'.`
      console.error(`[useGeneration] ${message}`)
      state.value.status = 'error'
      state.value.errorMessage = message
      return
    }
    if (!useInitImage && !engineSurface.supports_txt2img) {
      const message = `This engine does not support txt2img (${engineOverrideForRequest}).`
      console.error(`[useGeneration] ${message}`)
      state.value.status = 'error'
      state.value.errorMessage = message
      return
    }

    if (useInitImage) {
      if (!engineSurface.supports_img2img) {
        const message = `This engine does not support img2img (${engineOverrideForRequest}).`
        console.error(`[useGeneration] ${message}`)
        state.value.status = 'error'
        state.value.errorMessage = message
        return
      }
      if (!usesInitFolderSource && !p.initImageData) {
        state.value.status = 'error'
        state.value.errorMessage = 'Select an initial image for img2img.'
        return
      }
      if (useMask) {
        if (p.initSource.mode !== 'img') {
          state.value.status = 'error'
          state.value.errorMessage = 'Mask editing is only available while the initial image source is set to IMG.'
          return
        }
        if (!engineSurface.supports_img2img_masking) {
          state.value.status = 'error'
          state.value.errorMessage = `Masking is not supported for ${engineOverrideForRequest} img2img yet.`
          return
        }
        if (!p.maskImageData) {
          state.value.status = 'error'
          state.value.errorMessage = 'Select a mask image for inpaint.'
          return
        }
      }
    }

    const textEncoders = Array.isArray((p as any).textEncoders)
      ? (p as any).textEncoders.map((it: unknown) => String(it || '').trim()).filter((it: string) => it.length > 0)
      : []
    const familyOwnedVae = isZImageL2PEngine(engineOverrideForRequest) ? '' : String(quicksettings.getVaeForFamily(tabType) || '').trim()
    const requestContractResolvers = {
      requireModelInfo: quicksettings.requireModelInfo,
      resolveFlux2CheckpointVariant: quicksettings.resolveFlux2CheckpointVariant,
      resolveTextEncoderSha: quicksettings.resolveTextEncoderSha,
      resolveTextEncoderSlot: quicksettings.resolveTextEncoderSlot,
      requireVaeSelection: quicksettings.requireVaeSelection,
      resolveVaeSha: quicksettings.resolveVaeSha,
      getAssetContract: backendCaps.getAssetContract,
    }
    const resolveStageSwapModelPayload = (
      modelLabel: string,
      options: { engineKey: string },
    ): NonNullable<NestedStageSelectorPayloads['swapModel']> => {
      const { engineKey } = options
      const resolvedModelLabel = String(modelLabel || '').trim()
      if (!resolvedModelLabel) {
        throw new Error(`Missing model selection for '${engineKey}'.`)
      }
      const stageContract = buildExplicitImageRequestContract({
        modelLabel: resolvedModelLabel,
        engineKey,
        textEncoderLabels: textEncoders,
        selectedVaeLabel: familyOwnedVae,
        zimageTurbo: engineKey === 'zimage'
          ? Boolean((p as any)?.zimageTurbo ?? true)
          : false,
        fallbackGuidanceMode: usesStaticDistilledCfgEngine(engineKey) ? 'distilled_cfg' : 'cfg',
        resolvers: requestContractResolvers,
      })
      return {
        model: resolvedModelLabel,
        ...stageContract.extras,
      }
    }
    const buildNestedSelectorPayloads = (): NestedStageSelectorPayloads | undefined => {
      const nested: NestedStageSelectorPayloads = {}
      if (p.swapModel?.enabled) {
        const topLevelSwapModel = String(p.swapModel.model || '').trim()
        if (!topLevelSwapModel) {
          throw new Error('Select a first-pass swap model before generating.')
        }
        nested.swapModel = resolveStageSwapModelPayload(topLevelSwapModel, { engineKey: engineOverrideForRequest })
      }
      if (p.refiner?.enabled) {
        if (!engineSurface.supports_refiner) {
          throw new Error(`This engine does not support refiner (${engineOverrideForRequest}).`)
        }
        const refinerModel = String(p.refiner.model || '').trim()
        if (!refinerModel) {
          throw new Error('Select a refiner model before generating.')
        }
        nested.refiner = {
          enable: true,
          switch_at_step: p.refiner.swapAtStep,
          cfg: p.refiner.cfg,
          seed: p.refiner.seed,
          ...resolveStageSwapModelPayload(refinerModel, { engineKey: 'sdxl_refiner' }),
        }
      }
      if (p.hires?.enabled) {
        const hiresSwapModel = String(p.hires.swapModel?.model || '').trim()
        if (hiresSwapModel) {
          nested.hiresSwapModel = resolveStageSwapModelPayload(hiresSwapModel, { engineKey: engineOverrideForRequest })
        }
        if (p.hires.refiner?.enabled) {
          if (!engineSurface.supports_refiner) {
            throw new Error(`This engine does not support refiner (${engineOverrideForRequest}).`)
          }
          const hiresRefinerModel = String(p.hires.refiner.model || '').trim()
          if (!hiresRefinerModel) {
            throw new Error('Select a hires refiner model before generating.')
          }
          nested.hiresRefiner = {
            enable: true,
            switch_at_step: p.hires.refiner.swapAtStep,
            cfg: p.hires.refiner.cfg,
            seed: p.hires.refiner.seed,
            ...resolveStageSwapModelPayload(hiresRefinerModel, { engineKey: 'sdxl_refiner' }),
          }
        }
      }
      return Object.keys(nested).length > 0 ? nested : undefined
    }
    let guidanceMode: 'cfg' | 'distilled_cfg'
    let extras: Record<string, unknown>
    try {
      const requestContract = buildExplicitImageRequestContract({
        modelLabel: modelRef,
        engineKey: engineOverrideForRequest,
        textEncoderLabels: textEncoders,
        selectedVaeLabel: familyOwnedVae,
        zimageTurbo: engineOverrideForRequest === 'zimage'
          ? Boolean((p as any)?.zimageTurbo ?? true)
          : false,
        fallbackGuidanceMode: usesStaticDistilledCfgEngine(engineOverrideForRequest) ? 'distilled_cfg' : 'cfg',
        resolvers: requestContractResolvers,
      })
      if (requestContract.guidanceMode !== 'cfg' && requestContract.guidanceMode !== 'distilled_cfg') {
        throw new Error(`Image guidance mode is missing for '${engineOverrideForRequest}'.`)
      }
      guidanceMode = requestContract.guidanceMode
      extras = { ...requestContract.extras }
    } catch (error) {
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
      return
    }

    const usesDistilledCfgModel = guidanceMode === 'distilled_cfg'
    const createdAtMs = state.value.startedAtMs ?? Date.now()
    const promptPreview = String(p.prompt || '').trim().slice(0, 120)
    const automationRun = usesImageAutomation(p, engineOverrideForRequest)
    const batchSize = Math.max(1, Math.trunc(Number(p.batchSize)))
    const batchCount = Math.max(1, Math.trunc(Number(p.batchCount)))
    const requestBatchSize = automationRun ? 1 : batchSize
    const requestBatchCount = automationRun ? 1 : batchCount
    const settingsRevision = quicksettings.getSettingsRevision()
    const supportsNegative = familyCaps.supports_negative_prompt && !usesDistilledCfgModel
    const guidanceSupport = engineSurface.guidance_advanced ?? null
    const loraNames = [
      ...extractLoraNamesFromPrompt(p.prompt),
      ...(supportsNegative ? extractLoraNamesFromPrompt(p.negativePrompt) : []),
    ]
    if (isQwenImageEngine(engineOverrideForRequest)) {
      try {
        assertQwenImageRequestState({
          params: p,
          useInitImage,
          batchCount: requestBatchCount,
          batchSize: requestBatchSize,
          loraNames,
        })
      } catch (error) {
        state.value.status = 'error'
        state.value.errorMessage = error instanceof Error ? error.message : String(error)
        return
      }
    }
    if (isZImageL2PEngine(engineOverrideForRequest)) {
      try {
        assertZImageL2PRequestState({
          params: p,
          useInitImage,
          batchCount: requestBatchCount,
          batchSize: requestBatchSize,
          loraNames,
        })
      } catch (error) {
        state.value.status = 'error'
        state.value.errorMessage = error instanceof Error ? error.message : String(error)
        return
      }
    }
    const resolveEffectiveSampling = async (): Promise<SamplingDefaults> => {
      const backendDefaults = backendCaps.resolveSamplingDefaults(engineOverrideForRequest)
      const [samplerResponse, schedulerResponse] = await Promise.all([fetchSamplers(), fetchSchedulers()])
      return resolveEffectiveImageSampling({
        engineId: engineOverrideForRequest,
        samplers: samplerResponse.samplers,
        schedulers: schedulerResponse.schedulers,
        familyCapabilities: familyCaps,
        currentSampler: p.sampler,
        currentScheduler: p.scheduler,
        backendDefaults,
      })
    }

    const guidancePayload = buildGuidancePayload(p.guidanceAdvanced, guidanceSupport)
    if (guidancePayload) {
      extras.guidance = guidancePayload
    }

    try {
      const ipAdapterPayload = buildIpAdapterPayload(p, { useInitImage })
      if (ipAdapterPayload) {
        extras.ip_adapter = ipAdapterPayload
      }
    } catch (error) {
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
      return
    }

    let effectiveSampling: { sampler: string; scheduler: string; supirLabel?: string } | null = null
    try {
      const supirPayload = buildSupirPayload(p.supir, {
        engineId: engineOverrideForRequest,
        useInitImage,
        supportsSupirMode: Boolean(engineSurface.supports_supir_mode),
        hiresEnabled: normalizeBooleanParam(p.hires?.enabled, false),
        ipAdapterEnabled: normalizeBooleanParam(p.ipAdapter?.enabled, false),
        hasLoraSelection: loraNames.length > 0,
        guidanceAdvancedEnabled: normalizeBooleanParam(p.guidanceAdvanced.enabled, false),
      })
      if (supirPayload) {
        extras.supir = supirPayload.payload
        effectiveSampling = {
          sampler: supirPayload.samplerInfo.native_sampler,
          scheduler: supirPayload.samplerInfo.native_scheduler,
          supirLabel: supirPayload.samplerInfo.label,
        }
      } else {
        effectiveSampling = await resolveEffectiveSampling()
      }
    } catch (error) {
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
      return
    }
    const summary = buildRunSummary(p, guidanceMode, effectiveSampling, engineOverrideForRequest)
    const paramsSnapshot = buildParamsSnapshot(p, tabType, effectiveSampling, engineOverrideForRequest)

    if (loraNames.length > 0) {
      const loraShas: string[] = []
      const seenShas = new Set<string>()
      for (const loraName of loraNames) {
        const sha = quicksettings.resolveLoraSha(loraName)
        if (!sha) {
          state.value.status = 'error'
          state.value.errorMessage = `LoRA SHA not found for '${loraName}'. Refresh inventory and retry.`
          return
        }
        if (seenShas.has(sha)) continue
        seenShas.add(sha)
        loraShas.push(sha)
      }
      if (loraShas.length > 0) {
        extras.lora_sha = loraShas.length === 1 ? loraShas[0] : loraShas
      }
    }

    const device = (quicksettings.currentDevice || 'cpu') as any

    try {
      let taskId = ''
      if (useInitImage) {
        const payload = buildImg2ImgPayload({
          params: p,
          supportsNegativePrompt: supportsNegative,
          supportsHires: Boolean(engineSurface.supports_hires),
          guidanceMode,
          batchCount: requestBatchCount,
          batchSize: requestBatchSize,
          device,
          settingsRevision,
          engineId: engineOverrideForRequest,
          modelOverride: modelRef,
          hiresFallbackOnOom: Boolean(upscalersStore.fallbackOnOom),
          hiresMinTile: Number(upscalersStore.minTile),
          extras,
          samplerOverride: effectiveSampling?.sampler,
          schedulerOverride: effectiveSampling?.scheduler,
        })
        if (automationRun) {
          if (usesInitFolderSource) {
            delete payload.img2img_init_image
          }
          const { task_id } = await startImageAutomation(buildImageAutomationRequest({
            mode: 'img2img',
            template: payload,
            params: p,
          }))
          taskId = task_id
        } else {
          const { task_id } = await startImg2Img(payload)
          taskId = task_id
        }
      } else {
        let payload: Txt2ImgRequest
        try {
          const nestedSelectorPayloads = buildNestedSelectorPayloads()
          payload = buildTxt2ImgPayload({
            prompt: p.prompt,
            negativePrompt: supportsNegative ? p.negativePrompt : '',
            width: p.width,
            height: p.height,
            steps: p.steps,
            guidanceScale: p.cfgScale,
            sampler: effectiveSampling?.sampler ?? p.sampler,
            scheduler: effectiveSampling?.scheduler ?? p.scheduler,
            seed: p.seed,
            clipSkip: p.clipSkip,
            batchSize: requestBatchSize,
            batchCount: requestBatchCount,
            styles: [],
            device,
            settingsRevision,
            engine: engineOverrideForRequest,
            model: modelRef,
            guidanceMode,
            swapModel: p.swapModel,
            hires: p.hires,
            refiner: p.refiner,
            extras,
          }, {
            hiresFallbackOnOom: Boolean(upscalersStore.fallbackOnOom),
            hiresMinTile: Number(upscalersStore.minTile),
            nestedSelectorPayloads,
          })
        } catch (error) {
          state.value.status = 'error'
          state.value.errorMessage = error instanceof Error ? error.message : String(error)
          return
        }
        if (automationRun) {
          const { task_id } = await startImageAutomation(buildImageAutomationRequest({
            mode: 'txt2img',
            template: payload as unknown as Record<string, unknown>,
            params: p,
          }))
          taskId = task_id
        } else {
          const { task_id } = await startTxt2Img(payload)
          taskId = task_id
        }
      }

      state.value.taskId = taskId
      state.value.currentRun = {
        taskId,
        mode: useInitImage ? 'img2img' : 'txt2img',
        createdAtMs,
        status: 'running',
        summary,
        promptPreview,
        paramsSnapshot,
        thumbnail: null,
      }

      state.value.progress.stage = 'submitted'

      taskLifecycle.saveResume({
        taskId,
        lastEventId: 0,
        createdAtMs,
        paramsSnapshot,
        mode: useInitImage ? 'img2img' : 'txt2img',
        promptPreview,
        summary,
        finishedAtMs: null,
        terminalStatus: null,
      })
      taskLifecycle.attachStream(state.value.taskId, {
        onEventId: (eventId) => {
          if (!Number.isFinite(eventId)) return
          currentEventId = Math.max(0, Math.trunc(eventId))
        },
      })
    } catch (error) {
      const conflictRevision = resolveSettingsRevisionConflict(error)
      if (conflictRevision !== null) {
        try {
          await quicksettings.refreshSettingsRevision(conflictRevision)
        } catch {
          // Ignore refresh failures; fallback revision is already applied.
        }
        state.value.status = 'error'
        state.value.errorMessage = formatSettingsRevisionConflictMessage(quicksettings.getSettingsRevision())
        return
      }
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
    }
  }
  
  function handleTaskEvent(event: TaskEvent): void {
    switch (event.type) {
      case 'status':
        state.value.progress.stage = event.stage
        break
      case 'progress':
        state.value.progress = buildProgressStateFromPayload(event, { fallbackStage: state.value.progress.stage })
        if (event.preview_image) {
          state.value.previewImage = event.preview_image
          state.value.previewStep = event.preview_step ?? null
          if (state.value.currentRun?.taskId) {
            state.value.currentRun.thumbnail = event.preview_image
          }
        }
        break
      case 'automation_iteration': {
        const iterationImages = Array.isArray(event.images) ? event.images : []
        const shouldSkipReplayImages = currentEventId > 0 && currentEventId <= skipAutomationReplayThroughEventId
        if (!shouldSkipReplayImages && iterationImages.length > 0) {
          state.value.gallery = [...state.value.gallery, ...iterationImages]
          if (state.value.currentRun?.taskId) {
            state.value.currentRun.thumbnail = iterationImages[0]
          }
        }
        if (event.info !== undefined) {
          state.value.info = event.info ?? null
        }
        if (typeof event.seed === 'number' && Number.isFinite(event.seed)) {
          state.value.lastSeed = event.seed
        }
        break
      }
      case 'result':
        let parsedInfo: Record<string, unknown> | null = null
        try {
          const candidate = typeof event.info === 'string' ? JSON.parse(event.info) : event.info
          parsedInfo = candidate && typeof candidate === 'object' && !Array.isArray(candidate)
            ? candidate as Record<string, unknown>
            : null
        } catch {
          parsedInfo = null
        }
        const automationSummary = parsedInfo && typeof parsedInfo.automation_summary === 'object'
          ? parsedInfo.automation_summary
          : null
        if (automationSummary && state.value.gallery.length > 0) {
          state.value.gallery = [...state.value.gallery]
        } else {
          state.value.gallery = event.images || []
        }
        state.value.info = event.info ?? null
        const previewBeforeReset = state.value.previewImage
        const firstResultImage = Array.isArray(event.images) && event.images.length > 0 ? event.images[0] : null
        state.value.previewImage = null
        state.value.previewStep = null
        const finishedAtMs = Date.now()
        if (state.value.currentRun?.taskId) {
          state.value.currentRun.status = 'completed'
          if (firstResultImage) {
            state.value.currentRun.thumbnail = firstResultImage
          } else if (previewBeforeReset && !state.value.currentRun.thumbnail) {
            state.value.currentRun.thumbnail = previewBeforeReset
          }
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        try {
          const infoObj = (parsedInfo ?? (typeof event.info === 'string' ? JSON.parse(event.info) : event.info)) as any
          const rawSeed = infoObj?.seed ?? infoObj?.all_seeds?.[0]
            ?? infoObj?.automation_summary?.seed
          const resolvedSeed = typeof rawSeed === 'number' ? rawSeed : Number(rawSeed)
          if (Number.isFinite(resolvedSeed)) {
            state.value.lastSeed = resolvedSeed
          }
        } catch {
          // ignore seed parsing; keep lastSeed as-is
        }
        state.value.finishedAtMs = finishedAtMs
        patchResumeState(resumeKey(tabId), { finishedAtMs, terminalStatus: 'completed' })
        state.value.status = 'done'
        break
      case 'gap':
        // History truncated while disconnected; refresh snapshot and keep streaming.
        if (state.value.taskId) void refreshTaskSnapshot(state.value.taskId)
        break
      case 'error':
        state.value.status = 'error'
        state.value.errorMessage = event.message
        const terminalStatus = resolveErrorRunStatus(event.code, event.message)
        const finishedAtMsOnError = Date.now()
        state.value.finishedAtMs = finishedAtMsOnError
        const previewBeforeError = state.value.previewImage
        state.value.previewImage = null
        state.value.previewStep = null
        if (state.value.currentRun?.taskId) {
          state.value.currentRun.status = terminalStatus
          if (previewBeforeError && !state.value.currentRun.thumbnail) {
            state.value.currentRun.thumbnail = previewBeforeError
          }
          state.value.currentRun.errorMessage = event.message
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        patchResumeState(resumeKey(tabId), { finishedAtMs: finishedAtMsOnError, terminalStatus })
        clearResumeState(resumeKey(tabId))
        stopStream()
        break
      case 'end':
        clearResumeState(resumeKey(tabId))
        if (state.value.status !== 'error') {
          state.value.status = 'done'
        }
        if (state.value.finishedAtMs === null) {
          state.value.finishedAtMs = Date.now()
        }
        state.value.previewImage = null
        state.value.previewStep = null
        stopStream()
        break
    }
  }

  function applyAutomationRecoveryGallery(images: GeneratedImage[] | undefined): void {
    if (!Array.isArray(images) || images.length === 0) return
    state.value.gallery = [...images]
  }

  async function refreshTaskSnapshot(taskId: string): Promise<void> {
    try {
      const res = await fetchTaskResult(taskId)
      if (res.status === 'running') {
        if (typeof res.started_at_ms === 'number' && Number.isFinite(res.started_at_ms)) {
          state.value.startedAtMs = Math.trunc(res.started_at_ms)
        }
        if (typeof res.stage === 'string' && res.stage.trim()) state.value.progress.stage = res.stage
        const p = res.progress
        if (p && typeof p === 'object') {
          state.value.progress = buildProgressStateFromPayload(p, { fallbackStage: state.value.progress.stage })
        }
        if (res.preview_image) state.value.previewImage = res.preview_image
        if (res.preview_step !== undefined) state.value.previewStep = res.preview_step ?? null
        applyAutomationRecoveryGallery(res.automation_gallery_images)
        if (state.value.currentRun) {
          state.value.currentRun.thumbnail = res.preview_image
            ?? (Array.isArray(res.automation_gallery_images) ? res.automation_gallery_images[0] ?? null : null)
            ?? state.value.currentRun.thumbnail
        }
        markAutomationRecoveryWatermark(res.buffer_newest_event_id ?? res.last_event_id)
        return
      }
      if (res.status === 'completed' && res.result) {
        applyAutomationRecoveryGallery(res.automation_gallery_images)
        if (Array.isArray(res.automation_gallery_images) && res.automation_gallery_images.length > 0) {
          const bufferedNewestEventId = typeof res.last_event_id === 'number' && Number.isFinite(res.last_event_id)
            ? Math.max(0, Math.trunc(res.last_event_id) - 2)
            : 0
          markAutomationRecoveryWatermark(bufferedNewestEventId)
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
    state.value.selectedTaskId = ''
    state.value.finishedAtMs = null
    state.value.startedAtMs = typeof res.started_at_ms === 'number' && Number.isFinite(res.started_at_ms)
      ? Math.trunc(res.started_at_ms)
      : (saved.createdAtMs > 0 ? saved.createdAtMs : null)
    if (typeof res.stage === 'string' && res.stage.trim()) state.value.progress.stage = res.stage
    const progressPayload = res.progress
    if (progressPayload && typeof progressPayload === 'object') {
      state.value.progress = buildProgressStateFromPayload(progressPayload, { fallbackStage: state.value.progress.stage })
    }
    if (res.preview_image) state.value.previewImage = res.preview_image
    if (res.preview_step !== undefined) state.value.previewStep = res.preview_step ?? null
    applyAutomationRecoveryGallery(res.automation_gallery_images)
    state.value.currentRun = buildRunItemFromResumeState(saved, {
      status: 'running',
      thumbnail: res.preview_image
        ?? (Array.isArray(res.automation_gallery_images) ? res.automation_gallery_images[0] ?? null : null)
        ?? null,
    })
    markAutomationRecoveryWatermark(res.buffer_newest_event_id ?? res.last_event_id)
  }

  function handleResumeTerminalSnapshot(saved: ResumeState, res: Awaited<ReturnType<typeof fetchTaskResult>>): void {
    if (res.status === 'completed' && res.result) {
      const completedImages = Array.isArray(res.automation_gallery_images) && res.automation_gallery_images.length > 0
        ? [...res.automation_gallery_images]
        : (res.result.images || [])
      state.value.gallery = completedImages
      state.value.info = res.result.info ?? null
      state.value.errorMessage = ''
      state.value.taskId = saved.taskId
      state.value.status = 'done'
      state.value.previewImage = null
      state.value.previewStep = null
      state.value.startedAtMs = saved.createdAtMs > 0 ? saved.createdAtMs : null
      state.value.finishedAtMs = saved.finishedAtMs
      pushHistory(buildRunItemFromResumeState(saved, {
        status: 'completed',
        thumbnail: completedImages[0] ?? null,
      }))
      state.value.selectedTaskId = saved.taskId
      state.value.currentRun = null
      return
    }
    if (res.status === 'error') {
      const terminalStatus = res.error_code
        ? resolveErrorRunStatus(res.error_code, String(res.error || 'Task failed.'))
        : (saved.terminalStatus ?? resolveErrorRunStatus(undefined, String(res.error || 'Task failed.')))
      state.value.status = 'error'
      state.value.errorMessage = String(res.error || 'Task failed.')
      state.value.taskId = saved.taskId
      state.value.info = null
      state.value.gallery = []
      state.value.startedAtMs = saved.createdAtMs > 0 ? saved.createdAtMs : null
      state.value.finishedAtMs = saved.finishedAtMs
      state.value.previewImage = null
      state.value.previewStep = null
      pushHistory(buildRunItemFromResumeState(saved, {
        status: terminalStatus,
        errorMessage: state.value.errorMessage,
      }))
      state.value.selectedTaskId = saved.taskId
      state.value.currentRun = null
    }
  }

  function handleHistorySnapshot(taskId: string, result: Awaited<ReturnType<typeof fetchTaskResult>>): void {
    if (result.status === 'error') {
      state.value.gallery = []
      state.value.info = null
      state.value.previewImage = null
      state.value.previewStep = null
      state.value.lastSeed = null
      state.value.startedAtMs = null
      state.value.finishedAtMs = null
      state.value.taskId = taskId
      state.value.selectedTaskId = taskId
      state.value.currentRun = null
      state.value.status = 'error'
      state.value.errorMessage = result.error || 'Task failed.'
      return
    }
    if (result.status === 'completed' && result.result) {
      if (Array.isArray(result.automation_gallery_images) && result.automation_gallery_images.length > 0) {
        state.value.gallery = [...result.automation_gallery_images]
      } else {
        state.value.gallery = result.result.images || []
      }
      state.value.info = result.result.info ?? null
      state.value.errorMessage = ''
      state.value.status = 'done'
      state.value.taskId = taskId
      state.value.startedAtMs = null
      state.value.finishedAtMs = null
      state.value.previewImage = null
      state.value.previewStep = null
      state.value.selectedTaskId = taskId
      state.value.currentRun = null
      return
    }
    state.value.status = 'error'
    state.value.errorMessage = 'Task is still running.'
  }

  // Attempt to resume an in-flight task after a browser reload/crash.
  void taskLifecycle.tryAutoResume()
  const loadHistory = taskLifecycle.loadHistory

  async function cancel(mode: 'immediate' | 'after_current' = 'immediate'): Promise<void> {
    const taskId = state.value.taskId
    if (!taskId || state.value.status !== 'running') return
    await cancelTask(taskId, mode)
  }

  function clearHistory(): void {
    taskLifecycle.clearHistory()
  }
  
  // Expose reactive state and methods
  return {
    // State
    status: computed(() => state.value.status),
    progress: computed(() => state.value.progress),
    previewImage: computed(() => state.value.previewImage),
    previewStep: computed(() => state.value.previewStep),
    gallery: computed(() => state.value.gallery),
    info: computed(() => state.value.info),
    errorMessage: computed(() => state.value.errorMessage),
    taskId: computed(() => state.value.taskId),
    lastSeed: computed(() => state.value.lastSeed),
    history: computed(() => state.value.history),
    selectedTaskId: computed(() => state.value.selectedTaskId),
    historyLoadingTaskId: computed(() => state.value.historyLoadingTaskId),
    gentimeMs: computed(() => {
      if (state.value.startedAtMs === null || state.value.finishedAtMs === null) return null
      return Math.max(0, state.value.finishedAtMs - state.value.startedAtMs)
    }),
    isRunning: computed(() => state.value.status === 'running'),
    
    // Tab info
    tab,
    params,
    engineType,
    
    // Actions
    generate,
    cancel,
    stopStream,
    loadHistory,
    clearHistory,
    resumeNotice,
  }
}
