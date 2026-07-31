/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model Tabs store (tab definitions + per-tab params + ordering) for the WebUI.
Owns the list of engine tabs, persists tab CRUD/reorder via `/api/ui/tabs`, normalizes/validates tab payloads from the backend, and provides
default parameter shapes per tab type (image vs WAN/LTX video) using engine defaults and form-state schemas. Image-tab second-pass model state lives
under typed `swapModel` owners (global `swapModel`, hires `hires.swapModel`), native SDXL img2img/inpaint SUPIR state lives under the single nested
owner `supir`, and stale legacy `hires.checkpoint` / refiner-embedded `vae` snapshot
fields plus selector-only top-level `swapModel` snapshots are dropped during hydration instead of being preserved as rename glue. Hires upscaler values are stable ids
(`latent:*` / `spandrel:*`) for hires-fix wiring, and img2img UI keeps an explicit resize/upscaler layout state (`img2imgResizeMode`,
`img2imgUpscaler`) decoupled from backend hires dispatch. Image automation, SUPIR mode, and IP-Adapter UI state now stay under canonical owners
(`runAction`, `initSource`, `supir`, `ipAdapter`) instead of drifting into flat helper fields. The nested `supir` owner now also carries the public restore
window control (`restoreCfgSTmin`) as normalized UI state. WAN 2.2 tab state is now split by exact lane: `wan22_14b` keeps the high/low two-stage
owners with sampler/scheduler backfill, while `wan22_5b` owns a single-stage `stage` selector plus top-level prompt/sampler/seed fields and the
shared no-stretch img2vid guide controls (`img2vidImageScale`, `img2vidCropOffsetX`, `img2vidCropOffsetY`). FLUX.2 tabs keep the truthful Klein 4B / base-4B slice contract by capping `textEncoders` to one
`flux2/*` Qwen selector without overriding shared img2img denoise state. LTX normalization treats `mode` as the canonical owner of txt2vid/img2vid,
persists explicit `executionProfile` state, and leaves stale/blank profile values visible until the active checkpoint metadata or user choice resolves
them truthfully without silently rewriting stored raw profile ids.
Image-tab sampler/scheduler defaults are consumed only from backend capabilities; when those defaults are unavailable the store leaves fields blank for request-boundary validation instead of inventing frontend fallback values.
Qwen Image tabs are capability-derived like Anima/LTX2, use the single canonical `qwen_image` image-tab type, normalize persisted state to the
Edit-2511 img2img-only contract, and reject text-encoder labels that are not `qwen_image/<path>` selections. Z-Image L2P tabs are capability-derived exact txt2img tabs, keep fixed 1024 defaults, and reject persisted text-encoder labels
that are not shared `zimage/<path>` selections.

Symbols (top-level; keep in sync; no ghosts):
- `BaseTabType` (type): API tab type discriminator (from backend `ApiTab['type']`).
- `ImageTabType` (type): Image-only tab type discriminator (`BaseTabType` without video tab types).
- `BaseTabMeta` (interface): Tab metadata timestamps (created/updated) tracked client-side.
- `ModelTabsErrorCode` (type): Error code taxonomy for model-tabs store failures.
- `ModelTabsStoreError` (class): Typed store error thrown for tab lookup/API/contract/reorder/serialization failures.
- `WanStageLoraParams` (interface): UI WAN stage LoRA entry (`sha` + optional `weight`) for ordered stage `loras[]` payload wiring.
- `WanStageParams` (interface): UI WAN stage params (high/low), including sampler/scheduler, stage prompt/negative prompt, ordered `loras[]`, and optional explicit `flowShift`, used by video tabs and payload builders.
- `WanImg2VidMode` (type): WAN img2vid temporal mode discriminator (`solo|sliding|svi2|svi2_pro`).
- `WanChunkSeedMode` (type): WAN sliding/SVI per-window seed strategy (`fixed|increment|random`).
- `WanVideoParams` (interface): UI WAN video params (dims/fps/frames + optional init image + img2vid temporal controls + no-stretch guide controls (`img2vidImageScale` + crop offsets) + output/interpolation + SeedVR2 upscaling controls).
- `WanAssetsParams` (interface): WAN asset selectors (metadata/text encoder/VAE) used by WAN requests.
- `LtxGenerationMode` (type): LTX video mode discriminator (`txt2vid|img2vid`).
- `LtxTabParams` (interface): UI LTX video params, including checkpoint-owned `executionProfile` state plus prompt/init-image/video controls.
- `BaseTab` (interface): Generic tab record persisted in the store (id/type/label + params + meta).
- `ImageBaseParams` (interface): Common image-tab params (prompt, seed, steps, CFG, dims, etc.) shared across SD/Flux.1/Flux.2/Chroma/Qwen Image/ZImage/L2P
  (includes optional family-specific fields like `zimageTurbo`, img2img layout state `img2imgResizeMode`/`img2imgUpscaler`,
  inpaint mask controls (`maskRegionSplit` and related toggles), image automation owners (`runAction`, `initSource`, `supir`, `ipAdapter`), and advanced guidance policy controls).
- `ImageRunAction` (type): Run CTA mode discriminator (`generate|infinite`) persisted per image tab.
- `ImageFolderSelectionMode` (type): Folder traversal amount discriminator (`all|count`) used by init/IP-Adapter directory sources.
- `ImageFolderOrderMode` (type): Folder traversal order discriminator (`random|sorted`) used by init/IP-Adapter directory sources.
- `ImageFolderSortBy` (type): Sort key discriminator for ordered directory traversal (`name|size|created_at|modified_at`).
- `InitSourceFormState` (interface): Img2img initial-image source owner (`img|dir` plus directory traversal settings and crop toggle).
- `SupirModeFormState` (interface): Native SDXL img2img/inpaint SUPIR owner (`enabled`, variant/sampler selectors, and tranche-1 restore controls).
- `createDefaultSupirModeFormState` (function): Canonical default factory for the nested SUPIR owner state.
- `IpAdapterSourceFormState` (interface): IP-Adapter source owner (`img|dir`, uploaded reference image, same-as-init shortcut, and directory traversal settings).
- `IpAdapterFormState` (interface): IP-Adapter card state owner (enable flag, asset selectors, source owner, and strength range controls).
- `GuidanceAdvancedParams` (interface): Per-tab advanced guidance policy state (APG/rescale/trunc/renorm).
- `DEFAULT_GUIDANCE_ADVANCED_PARAMS` (constant): Canonical defaults for `ImageBaseParams.guidanceAdvanced`.
- `TabParamsByType` (type): Canonical params map by tab type.
- `TabByType` (type): Typed tab shape (`type` + matching params payload).
- `ModelTabsStorageState` (type): LocalStorage payload contract for light model-tabs state (`activeId` + tab refs).
- `MODEL_TABS_STORAGE_KEY` (const): LocalStorage key used for persisted tabs state (bump when schema changes).
- `buildStoragePayload` (function): Builds a small localStorage payload without heavy per-tab params blobs.
- `isQuotaExceededStorageError` (function): Detects storage quota-exceeded failures across browser variants.
- `nowIso` (function): Returns current time in ISO string form for metadata timestamps.
- `defaultParams` (function): Returns default params for a given tab type (image vs WAN video), merging engine defaults where applicable.
- `defaultImageParamsForType` (function): Returns canonical image-tab defaults for a specific image tab type.
- `normalizeTabType` (function): Validates/coerces raw type values into `BaseTabType`.
- `BASE_REQUIRED_TYPES` (const): Baseline tab types always auto-created by the UI store.
- `requiredTypesFromCapabilities` (function): Derives required tab types from backend capability map (adds capability-exposed `qwen_image`/`zimage_l2p`/`ltx2`/`anima` tabs).
- `asRecordObject` (function): Narrowing helper that normalizes unknown values into plain records for merge-safe processing.
- `isPlainRecord` (function): Validates object values as plain record payloads (no arrays/class instances) for patch serialization safety.
- `PersistSerializationPhase` (type): Serialization boundary phases used by params persistence snapshots and rollback.
- `serializationFailure` (function): Factory for typed fail-loud params serialization errors with contextual details.
- `normalizeSerializableForPersist` (function): Recursively unwraps reactive/proxy branches into plain clone-safe structures for persistence.
- `asParamsRecord` (function): Explicit boundary cast helper from typed tab params to persisted `Record<string, unknown>`.
- `normalizeWanFrameCount` (function): Clamps/snap-normalizes WAN frame counts to the `4n+1` domain.
- `normalizeWanVideoParams` (function): Sanitizes WAN video nested params (frames/window/attention controls) with `img2vidMode` as source of truth.
- `normalizeWan14bParams` (function): Applies exact WAN 14B nested merge normalization for `high/low/video/assets`, enforcing canonical stage scheduler `simple`.
- `normalizeWan5bParams` (function): Applies exact WAN 5B single-stage normalization for `stage/video/assets` plus top-level prompt/sampler/seed owners.
- `shouldPersistWan14bStageSamplingBackfill` (function): Detects persisted WAN 14B params requiring High/Low stage sampler/scheduler migration (`sampler='uni-pc bh2'`, `scheduler='simple'`).
- `buildImageTopLevelBackfillPatch` (function): Builds a missing-top-level-only image-tab backfill patch from the normalized owner shape so hydration can persist absent canonical keys without widening into unrelated nested drift.
- `normalizeQwenImageTextEncoders` (function): Validates Qwen Image persisted text-encoder labels as one `qwen_image/<path>` selector.
- `normalizeZImageL2PTextEncoders` (function): Validates Z-Image L2P persisted text-encoder labels as one shared `zimage/<path>` selector.
- `normalizeImageParams` (function): Applies image-tab nested merge normalization (`hires/refiner`) with sampler/scheduler and strict inpaint-mode reset.
- `normalizeParamsForType` (function): Normalizes raw params payload based on tab type (shape checking; discards invalid fields).
- `normalizeTab` (function): Normalizes a raw tab record (id/type/params/meta) into the store shape.
- `syncImageSparsePersistHints` (function): Syncs sparse image-owner missing-key hints from normalized in-memory params.
- `snapshotImageSparsePersistHints` (function): Captures sparse image-owner missing-key hints before a queued persist mutates them.
- `restoreImageSparsePersistHints` (function): Restores sparse image-owner missing-key hints during rollback.
- `markExplicitImagePersistKeys` (function): Marks image-owner keys explicitly touched by one local patch so sparse pruning does not erase them.
- `pruneSparseImagePersistDefaults` (function): Removes server-missing sparse image-owner defaults from one persisted payload unless explicitly touched.
- `applyPersistedImageSparsePersistHints` (function): Advances sparse image-owner missing-key hints only for the exact keys that just persisted successfully.
- `cloneParamsForPersist` (function): Proxy-safe `structuredClone` boundary for params snapshots/payloads; throws typed serialization failures.
- `restorePendingParamsSnapshot` (function): Restores tab params/meta from pending snapshot after failed persistence attempts.
- `scheduleParamsPersist` (function): Schedules debounced `/api/ui/tabs/:id` params PATCH calls.
- `flushParamsPersist` (function): Flushes pending params PATCH for a tab, resolves queued promises, and rolls back on API failure.
- `useModelTabsStore` (store): Pinia store for tabs; loads/syncs with backend, provides CRUD/reorder actions, and exposes computed helpers.
*/

import { defineStore } from 'pinia'
import { ref, computed, toRaw } from 'vue'
import { fetchTabs, createTabApi, updateTabApi, reorderTabsApi, deleteTabApi } from '../api/client'
import type { ApiTab } from '../api/types'
import type { HiresFormState, RefinerFormState, SwapModelFormState, SwapStageFormState } from '../api/payloads'
import { type EngineType, getEngineConfig, getEngineDefaults } from './engine_config'
import { useEngineCapabilitiesStore } from './engine_capabilities'
import { isWanTabFamily, normalizeTabFamily, resolveImageRequestEngineId } from '../utils/engine_taxonomy'
import { DEFAULT_IMG2IMG_RESIZE_MODE, normalizeImg2ImgResizeMode, type Img2ImgResizeMode } from '../utils/img2img_resize'
import { parseInpaintMode } from '../utils/image_params'
import {
  normalizeWanChunkOverlap,
  normalizeWanWindowCommit,
  normalizeWanWindowStride,
  type WanImg2VidMode as WanImg2VidModeInternal,
} from '../utils/wan_img2vid_temporal'
import { normalizeWanImg2VidImageScale } from '../utils/wan_img2vid_frame_projection'

export type BaseTabType = ApiTab['type']
export type ImageTabType = Exclude<BaseTabType, 'wan22_14b' | 'wan22_5b' | 'ltx2'>
export type WanTabType = Extract<BaseTabType, 'wan22_14b' | 'wan22_5b'>
export type Wan14bTabType = Extract<BaseTabType, 'wan22_14b'>
export type Wan5bTabType = Extract<BaseTabType, 'wan22_5b'>

export interface BaseTabMeta {
  createdAt: string
  updatedAt: string
}

export type ModelTabsErrorCode =
  | 'tab_not_found'
  | 'api_failure'
  | 'invalid_response'
  | 'invalid_reorder'
  | 'serialization_failure'

export class ModelTabsStoreError extends Error {
  readonly code: ModelTabsErrorCode
  readonly cause: unknown
  readonly details: Record<string, unknown> | null

  constructor(
    code: ModelTabsErrorCode,
    message: string,
    options?: { cause?: unknown; details?: Record<string, unknown> | null },
  ) {
    super(message)
    this.name = 'ModelTabsStoreError'
    this.code = code
    this.cause = options?.cause
    this.details = options?.details ?? null
  }
}

export interface WanStageLoraParams {
  sha: string
  weight?: number
}

export interface WanStageParams {
  modelDir: string
  prompt: string
  negativePrompt: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  loras: WanStageLoraParams[]
  flowShift?: number
}

export type WanImg2VidMode = WanImg2VidModeInternal
export type WanChunkSeedMode = 'fixed' | 'increment' | 'random'

export interface WanVideoParams {
  // Core generation fields (txt2vid/img2vid shared)
  width: number
  height: number
  fps: number
  frames: number
  attentionMode: 'global' | 'sliding'
  // Optional initial image (img2vid)
  useInitImage: boolean
  initImageData: string
  initImageName: string
  img2vidMode: WanImg2VidMode
  img2vidChunkFrames: number
  img2vidOverlapFrames: number
  img2vidAnchorAlpha: number
  img2vidResetAnchorToBase: boolean
  img2vidChunkSeedMode: WanChunkSeedMode
  img2vidWindowFrames: number
  img2vidWindowStride: number
  img2vidWindowCommitFrames: number
  img2vidImageScale: number
  img2vidCropOffsetX: number
  img2vidCropOffsetY: number
  // Export options
  format: string
  pixFmt: string
  crf: number
  loopCount: number
  pingpong: boolean
  returnFrames: boolean
  // Interpolation (RIFE target FPS; 0 disables interpolation)
  interpolationFps: number
  // Optional SeedVR2 upscaling (global post-process)
  upscalingEnabled: boolean
  upscalingModel: string
  upscalingResolution: number
  upscalingMaxResolution: number
  upscalingBatchSize: number
  upscalingUniformBatchSize: boolean
  upscalingTemporalOverlap: number
  upscalingPrependFrames: number
  upscalingColorCorrection: 'lab' | 'wavelet' | 'wavelet_adaptive' | 'hsv' | 'adain' | 'none'
  upscalingInputNoiseScale: number
  upscalingLatentNoiseScale: number
}

export interface WanAssetsParams {
  metadata: string
  textEncoder: string
  vae: string
}

export type LtxGenerationMode = 'txt2vid' | 'img2vid'

export interface LtxTabParams {
  schemaVersion: number
  mode: LtxGenerationMode
  prompt: string
  negativePrompt: string
  width: number
  height: number
  fps: number
  frames: number
  steps: number
  cfgScale: number
  executionProfile: string
  seed: number
  checkpoint: string
  vae: string
  textEncoder: string
  initImageData: string
  initImageName: string
  videoReturnFrames: boolean
}

export type Wan14bTabParams = Record<string, unknown> & {
  schemaVersion: number
  high: WanStageParams
  low: WanStageParams
  video: WanVideoParams
  assets: WanAssetsParams
  lightx2v: boolean
  lowFollowsHigh: boolean
}

export type Wan5bStageParams = Record<string, unknown> & {
  modelDir: string
  loras: WanStageLoraParams[]
  flowShift?: number
}

export type Wan5bTabParams = Record<string, unknown> & {
  schemaVersion: number
  prompt: string
  negativePrompt: string
  stage: Wan5bStageParams
  video: WanVideoParams
  assets: WanAssetsParams
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
}

export interface BaseTab {
  id: string
  type: BaseTabType
  title: string
  order: number
  enabled: boolean
  params: Record<string, unknown>
  meta: BaseTabMeta
}

export interface ImageBaseParams {
  schemaVersion: number
  prompt: string
  negativePrompt: string
  width: number
  height: number
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  clipSkip: number
  batchSize: number
  batchCount: number
  runAction: ImageRunAction
  img2imgResizeMode: Img2ImgResizeMode
  img2imgUpscaler: string
  guidanceAdvanced: GuidanceAdvancedParams
  hires: HiresFormState
  swapModel: SwapStageFormState
  refiner: RefinerFormState
  checkpoint: string
  textEncoders: string[]
  useInitImage: boolean
  initSource: InitSourceFormState
  initImageData: string
  initImageName: string
  denoiseStrength: number
  useMask: boolean
  maskImageData: string
  maskImageName: string
  inpaintMode: 'per_step_blend' | 'post_sample_blend' | 'fooocus_inpaint' | 'brushnet'
  perStepBlendStrength: number
  perStepBlendSteps: number
  inpaintFullResPadding: number
  inpaintingFill: number
  maskInvert: boolean
  maskBlur: number
  maskRound: boolean
  maskRegionSplit: boolean
  supir: SupirModeFormState
  ipAdapter: IpAdapterFormState
  zimageTurbo?: boolean
}

export type ImageRunAction = 'generate' | 'infinite'
export type ImageFolderSelectionMode = 'all' | 'count'
export type ImageFolderOrderMode = 'random' | 'sorted'
export type ImageFolderSortBy = 'name' | 'size' | 'created_at' | 'modified_at'

export interface InitSourceFormState {
  mode: 'img' | 'dir'
  folderPath: string
  selectionMode: ImageFolderSelectionMode
  count: number
  order: ImageFolderOrderMode
  sortBy: ImageFolderSortBy
  useCrop: boolean
}

export type SupirVariant = 'v0F' | 'v0Q'
export type SupirColorFixMode = 'None' | 'AdaIN' | 'Wavelet'

export interface SupirModeFormState {
  enabled: boolean
  variant: SupirVariant
  sampler: string
  controlScale: number
  restorationScale: number
  restoreCfgSTmin: number
  colorFix: SupirColorFixMode
}

export function createDefaultSupirModeFormState(): SupirModeFormState {
  return {
    enabled: false,
    variant: 'v0Q',
    sampler: 'restore_euler_edm_stable',
    controlScale: 0.8,
    restorationScale: 4,
    restoreCfgSTmin: 0.05,
    colorFix: 'None',
  }
}

export interface IpAdapterSourceFormState {
  mode: 'img' | 'dir'
  sameAsInit: boolean
  referenceImageData: string
  referenceImageName: string
  folderPath: string
  selectionMode: ImageFolderSelectionMode
  count: number
  order: ImageFolderOrderMode
  sortBy: ImageFolderSortBy
}

export interface IpAdapterFormState {
  enabled: boolean
  model: string
  imageEncoder: string
  source: IpAdapterSourceFormState
  weight: number
  startAt: number
  endAt: number
}

export interface GuidanceAdvancedParams {
  enabled: boolean
  apgEnabled: boolean
  apgStartStep: number
  apgEta: number
  apgMomentum: number
  apgNormThreshold: number
  apgRescale: number
  guidanceRescale: number
  cfgTruncEnabled: boolean
  cfgTruncRatio: number
  renormCfg: number
}

export const DEFAULT_GUIDANCE_ADVANCED_PARAMS: GuidanceAdvancedParams = {
  enabled: false,
  apgEnabled: false,
  apgStartStep: 0,
  apgEta: 0,
  apgMomentum: 0,
  apgNormThreshold: 15,
  apgRescale: 0,
  guidanceRescale: 0,
  cfgTruncEnabled: false,
  cfgTruncRatio: 0.8,
  renormCfg: 0,
}

export type TabParamsByType = {
  sd15: ImageBaseParams
  sdxl: ImageBaseParams
  flux1: ImageBaseParams
  flux2: ImageBaseParams
  qwen_image: ImageBaseParams
  zimage: ImageBaseParams
  zimage_l2p: ImageBaseParams
  chroma: ImageBaseParams
  anima: ImageBaseParams
  ltx2: LtxTabParams
  wan22_14b: Wan14bTabParams
  wan22_5b: Wan5bTabParams
}

export type TabByType<T extends BaseTabType = BaseTabType> = Omit<BaseTab, 'type' | 'params'> & {
  type: T
  params: TabParamsByType[T]
}

type ModelTabsStorageTabRef = Pick<BaseTab, 'id' | 'type'>

type ModelTabsStorageState = {
  tabs: ModelTabsStorageTabRef[]
  activeId: string
}

export const MODEL_TABS_STORAGE_KEY = 'codex:model-tabs:v2'
const STORAGE_KEY = MODEL_TABS_STORAGE_KEY
const TAB_PARAMS_SCHEMA_VERSION = 4

const IMAGE_PARAM_TOP_LEVEL_KEYS = new Set<string>([
  'schemaVersion',
  'prompt',
  'negativePrompt',
  'width',
  'height',
  'sampler',
  'scheduler',
  'steps',
  'cfgScale',
  'seed',
  'clipSkip',
  'batchSize',
  'batchCount',
  'runAction',
  'img2imgResizeMode',
  'img2imgUpscaler',
  'guidanceAdvanced',
  'hires',
  'swapModel',
  'refiner',
  'checkpoint',
  'textEncoders',
  'useInitImage',
  'initSource',
  'initImageData',
  'initImageName',
  'denoiseStrength',
  'useMask',
  'maskImageData',
  'maskImageName',
  'inpaintMode',
  'perStepBlendStrength',
  'perStepBlendSteps',
  'inpaintFullResPadding',
  'inpaintingFill',
  'maskInvert',
  'maskBlur',
  'maskRound',
  'maskRegionSplit',
  'supir',
  'ipAdapter',
  'zimageTurbo',
])
const IMAGE_PARAM_NO_AUTOBACKFILL_KEYS = new Set<string>([
  'inpaintMode',
])
const IMAGE_PARAM_SPARSE_PERSIST_DEFAULT_KEYS = new Set<string>([
  'inpaintMode',
])

const WAN14B_PARAM_TOP_LEVEL_KEYS = new Set<string>([
  'schemaVersion',
  'high',
  'low',
  'video',
  'assets',
  'lightx2v',
  'lowFollowsHigh',
])

const WAN5B_PARAM_TOP_LEVEL_KEYS = new Set<string>([
  'schemaVersion',
  'prompt',
  'negativePrompt',
  'stage',
  'video',
  'assets',
  'sampler',
  'scheduler',
  'steps',
  'cfgScale',
  'seed',
])

const LTX_PARAM_TOP_LEVEL_KEYS = new Set<string>([
  'schemaVersion',
  'mode',
  'prompt',
  'negativePrompt',
  'width',
  'height',
  'fps',
  'frames',
  'steps',
  'cfgScale',
  'executionProfile',
  'seed',
  'checkpoint',
  'vae',
  'textEncoder',
  'initImageData',
  'initImageName',
  'videoReturnFrames',
])

function buildStoragePayload(tabList: BaseTab[], currentActiveId: string): ModelTabsStorageState {
  const tabRefs: ModelTabsStorageTabRef[] = tabList.map((tab) => ({
    id: tab.id,
    type: tab.type,
  }))
  return {
    tabs: tabRefs,
    activeId: currentActiveId,
  }
}

function isQuotaExceededStorageError(error: unknown): boolean {
  if (!(error instanceof DOMException)) return false
  if (error.name === 'QuotaExceededError' || error.name === 'NS_ERROR_DOM_QUOTA_REACHED') return true
  const domError = error as DOMException & { code?: number }
  return domError.code === 22 || domError.code === 1014
}

function nowIso(): string {
  return new Date().toISOString()
}

function requirePersistedTabId(value: unknown, context: string): string {
  const id = typeof value === 'string' ? value.trim() : ''
  if (!id) {
    throw new ModelTabsStoreError(
      'invalid_response',
      `Invalid '/api/ui/tabs' contract: ${context} returned an empty 'id'.`,
    )
  }
  return id
}

function defaultParams<T extends BaseTabType>(
  type: T,
  opts?: { sampler?: string; scheduler?: string },
): TabParamsByType[T] {
  if (type === 'wan22_14b' || type === 'wan22_5b') {
    const stage = (): WanStageParams => ({
      modelDir: '',
      prompt: '',
      negativePrompt: '',
      sampler: 'uni-pc bh2',
      scheduler: 'simple',
      steps: 30,
      cfgScale: 7,
      seed: -1,
      loras: [],
    })
    const video: WanVideoParams = {
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
      upscalingEnabled: false,
      upscalingModel: 'seedvr2_ema_3b_fp16.safetensors',
      upscalingResolution: 1080,
      upscalingMaxResolution: 0,
      upscalingBatchSize: 5,
      upscalingUniformBatchSize: false,
      upscalingTemporalOverlap: 0,
      upscalingPrependFrames: 0,
      upscalingColorCorrection: 'lab',
      upscalingInputNoiseScale: 0,
      upscalingLatentNoiseScale: 0,
    }
    const assets: WanAssetsParams = { metadata: '', textEncoder: '', vae: '' }
    if (type === 'wan22_14b') {
      const wanDefaults: Wan14bTabParams = {
        schemaVersion: TAB_PARAMS_SCHEMA_VERSION,
        high: stage(),
        low: stage(),
        video,
        assets,
        lightx2v: false,
        lowFollowsHigh: false,
      }
      return wanDefaults as TabParamsByType[T]
    }
    const wan5bDefaults: Wan5bTabParams = {
      schemaVersion: TAB_PARAMS_SCHEMA_VERSION,
      prompt: '',
      negativePrompt: '',
      stage: {
        modelDir: '',
        loras: [],
      },
      video,
      assets,
      sampler: 'uni-pc bh2',
      scheduler: 'simple',
      steps: 30,
      cfgScale: 7,
      seed: -1,
    }
    return wan5bDefaults as TabParamsByType[T]
  }

  if (type === 'ltx2') {
    const capsStore = useEngineCapabilitiesStore()
    const ltxExecutionSurface = capsStore.getLtxExecutionSurface('ltx2')
    const defaultProfile = String(ltxExecutionSurface?.default_execution_profile || '').trim()
    const defaultSteps = defaultProfile
      ? ltxExecutionSurface?.default_steps_by_profile[defaultProfile] ?? 30
      : 30
    const defaultGuidance = defaultProfile
      ? ltxExecutionSurface?.default_guidance_scale_by_profile[defaultProfile] ?? 4
      : 4
    const ltxDefaults: TabParamsByType['ltx2'] = {
      schemaVersion: TAB_PARAMS_SCHEMA_VERSION,
      mode: 'txt2vid',
      prompt: '',
      negativePrompt: '',
      width: 768,
      height: 512,
      fps: 24,
      frames: 121,
      steps: defaultSteps,
      cfgScale: defaultGuidance,
      executionProfile: '',
      seed: -1,
      checkpoint: '',
      vae: '',
      textEncoder: '',
      initImageData: '',
      initImageName: '',
      videoReturnFrames: false,
    }
    return ltxDefaults as TabParamsByType[T]
  }

  const config = getEngineConfig(type as EngineType)
  const defaults = getEngineDefaults(type as EngineType)
  const guidance = (!config.capabilities.usesCfg && defaults.distilledCfg !== undefined) ? defaults.distilledCfg : defaults.cfg
  const resolvedSampler = String(opts?.sampler || '').trim()
  const resolvedScheduler = String(opts?.scheduler || '').trim()
  const refinerDefaults: RefinerFormState = {
    enabled: false,
    swapAtStep: 1,
    cfg: 3.5,
    seed: -1,
    model: undefined,
  }
  const swapStageDefaults: SwapStageFormState = {
    enabled: false,
    swapAtStep: 1,
    cfg: guidance,
    seed: -1,
    model: undefined,
  }
  const hiresDefaults: HiresFormState = {
    enabled: false,
    denoise: 0.4,
    scale: 2,
    resizeX: 0,
    resizeY: 0,
    steps: 0,
    upscaler: 'latent:bicubic-aa',
    tile: { tile: 256, overlap: 16 },
    swapModel: undefined,
    sampler: undefined,
    scheduler: undefined,
    prompt: undefined,
    negativePrompt: undefined,
    cfg: undefined,
    distilledCfg: undefined,
    refiner: { ...refinerDefaults },
  }
  const initSourceDefaults: InitSourceFormState = {
    mode: 'img',
    folderPath: '',
    selectionMode: 'all',
    count: 1,
    order: 'sorted',
    sortBy: 'name',
    useCrop: false,
  }
  const ipAdapterSourceDefaults: IpAdapterSourceFormState = {
    mode: 'img',
    sameAsInit: false,
    referenceImageData: '',
    referenceImageName: '',
    folderPath: '',
    selectionMode: 'all',
    count: 1,
    order: 'sorted',
    sortBy: 'name',
  }
  const ipAdapterDefaults: IpAdapterFormState = {
    enabled: false,
    model: '',
    imageEncoder: '',
    source: { ...ipAdapterSourceDefaults },
    weight: 1,
    startAt: 0,
    endAt: 1,
  }
  const supirDefaults = createDefaultSupirModeFormState()
  const imageDefaults: ImageBaseParams = {
    schemaVersion: TAB_PARAMS_SCHEMA_VERSION,
    prompt: '',
    negativePrompt: config.capabilities.usesNegativePrompt ? '' : '',
    width: defaults.width,
    height: defaults.height,
    sampler: resolvedSampler,
    scheduler: resolvedScheduler,
    steps: defaults.steps,
    cfgScale: guidance,
    seed: -1,
    clipSkip: 0,
    batchSize: 1,
    batchCount: 1,
    runAction: 'generate',
    img2imgResizeMode: DEFAULT_IMG2IMG_RESIZE_MODE,
    img2imgUpscaler: 'latent:bicubic-aa',
    guidanceAdvanced: { ...DEFAULT_GUIDANCE_ADVANCED_PARAMS },
    hires: { ...hiresDefaults },
    swapModel: { ...swapStageDefaults },
    refiner: { ...refinerDefaults },
    checkpoint: '',
    textEncoders: [],
    useInitImage: false,
    initSource: { ...initSourceDefaults },
    initImageData: '',
    initImageName: '',
    denoiseStrength: 0.75,
    useMask: false,
    maskImageData: '',
    maskImageName: '',
    inpaintMode: 'per_step_blend',
    perStepBlendStrength: 1,
    perStepBlendSteps: 0,
    inpaintFullResPadding: 32,
    inpaintingFill: 1,
    maskInvert: false,
    maskBlur: 4,
    maskRound: true,
    maskRegionSplit: true,
    supir: { ...supirDefaults },
    ipAdapter: { ...ipAdapterDefaults, source: { ...ipAdapterDefaults.source } },
  }
  if (type === 'zimage') {
    imageDefaults.zimageTurbo = true
  }
  if (type === 'qwen_image') {
    imageDefaults.useInitImage = true
    imageDefaults.denoiseStrength = 1
    imageDefaults.steps = Math.max(2, Math.trunc(Number(imageDefaults.steps)))
  }
  return imageDefaults as TabParamsByType[T]
}

export function defaultImageParamsForType(
  type: ImageTabType,
  opts?: { sampler?: string; scheduler?: string },
): ImageBaseParams
export function defaultImageParamsForType(
  type: BaseTabType,
  opts?: { sampler?: string; scheduler?: string },
): ImageBaseParams {
  if (isWanTabFamily(type) || type === 'ltx2') {
    const msg = `defaultImageParamsForType received '${type}'; expected an image tab type.`
    console.error(`[model_tabs] ${msg}`, { type })
    throw new Error(msg)
  }
  return defaultParams(type, opts)
}

export function normalizeTabType(type: unknown): BaseTabType {
  const raw = String(type || '').trim()
  if (!raw) {
    const msg = 'Model tab type is required, got empty value.'
    console.error(`[model_tabs] ${msg}`, { type })
    throw new Error(msg)
  }
  const normalized = normalizeTabFamily(raw)
  if (normalized) return normalized
  const msg = `Unsupported model tab type '${raw}'.`
  console.error(`[model_tabs] ${msg}`, { type })
  throw new Error(msg)
}

function asRecordObject(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>
  return {}
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(toRaw(value))
  return prototype === Object.prototype || prototype === null
}

function asParamsRecord(params: TabParamsByType[BaseTabType]): Record<string, unknown> {
  return params as unknown as Record<string, unknown>
}

function normalizePositiveInt(rawValue: unknown, fallback: number): number {
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) return Math.max(1, Math.trunc(fallback))
  return Math.max(1, Math.trunc(numeric))
}

function normalizeLtxMode(rawValue: unknown, fallback: LtxGenerationMode): LtxGenerationMode {
  const normalized = String(rawValue || '').trim().toLowerCase()
  if (normalized === 'txt2vid' || normalized === 'img2vid') return normalized
  return fallback
}

function normalizeLtxParams(raw: unknown, defaults: LtxTabParams): LtxTabParams {
  const patch = asRecordObject(raw)
  const merged: LtxTabParams = { ...defaults }
  for (const [key, value] of Object.entries(patch)) {
    if (!LTX_PARAM_TOP_LEVEL_KEYS.has(key)) continue
    ;(merged as unknown as Record<string, unknown>)[key] = value
  }
  merged.mode = normalizeLtxMode(patch.mode, defaults.mode)
  merged.prompt = String(merged.prompt || '')
  merged.negativePrompt = String(merged.negativePrompt || '')
  merged.width = normalizePositiveInt(merged.width, defaults.width)
  merged.height = normalizePositiveInt(merged.height, defaults.height)
  merged.fps = normalizePositiveInt(merged.fps, defaults.fps)
  merged.frames = normalizePositiveInt(merged.frames, defaults.frames)
  merged.steps = normalizePositiveInt(merged.steps, defaults.steps)
  merged.cfgScale = Number.isFinite(Number(merged.cfgScale)) ? Number(merged.cfgScale) : defaults.cfgScale
  merged.executionProfile = String(merged.executionProfile || '').trim()
  merged.seed = Number.isFinite(Number(merged.seed)) ? Math.trunc(Number(merged.seed)) : defaults.seed
  merged.checkpoint = String(merged.checkpoint || '').trim()
  merged.vae = String(merged.vae || '').trim()
  merged.textEncoder = String(merged.textEncoder || '').trim()
  merged.initImageData = String(merged.initImageData || '')
  merged.initImageName = String(merged.initImageName || '')
  merged.videoReturnFrames = normalizeBoolean(merged.videoReturnFrames, defaults.videoReturnFrames)
  merged.schemaVersion = TAB_PARAMS_SCHEMA_VERSION
  return merged
}

function parseParamsSchemaVersion(rawValue: unknown): number | null {
  if (typeof rawValue === 'number' && Number.isFinite(rawValue)) {
    return Math.max(0, Math.trunc(rawValue))
  }
  if (typeof rawValue === 'string') {
    const trimmed = rawValue.trim()
    if (/^-?\d+$/.test(trimmed)) return Math.max(0, Math.trunc(Number(trimmed)))
  }
  return null
}

function migrateImageParamsPatch(rawPatch: Record<string, unknown>): {
  patch: Partial<ImageBaseParams>
  droppedUnknownKeys: string[]
  fromVersion: number | null
} {
  const patch: Record<string, unknown> = {}
  const droppedUnknownKeys: string[] = []
  for (const [key, value] of Object.entries(rawPatch)) {
    if (!IMAGE_PARAM_TOP_LEVEL_KEYS.has(key)) {
      droppedUnknownKeys.push(key)
      continue
    }
    patch[key] = value
  }
  if (isPlainRecord(patch.swapModel)) {
    const rawSwapModel = patch.swapModel as Record<string, unknown>
    const hasSwapStageControlKeys = (
      Object.prototype.hasOwnProperty.call(rawSwapModel, 'enabled')
      || Object.prototype.hasOwnProperty.call(rawSwapModel, 'swapAtStep')
      || Object.prototype.hasOwnProperty.call(rawSwapModel, 'cfg')
      || Object.prototype.hasOwnProperty.call(rawSwapModel, 'seed')
    )
    if (!hasSwapStageControlKeys) {
      delete patch.swapModel
      droppedUnknownKeys.push('swapModel')
    }
  }
  const fromVersion = parseParamsSchemaVersion(rawPatch.schemaVersion)
  patch.schemaVersion = TAB_PARAMS_SCHEMA_VERSION
  return {
    patch: patch as Partial<ImageBaseParams>,
    droppedUnknownKeys,
    fromVersion,
  }
}

function migrateWan14bParamsPatch(rawPatch: Record<string, unknown>): {
  patch: Partial<Wan14bTabParams>
  droppedUnknownKeys: string[]
  fromVersion: number | null
} {
  const patch: Record<string, unknown> = {}
  const droppedUnknownKeys: string[] = []
  for (const [key, value] of Object.entries(rawPatch)) {
    if (!WAN14B_PARAM_TOP_LEVEL_KEYS.has(key)) {
      droppedUnknownKeys.push(key)
      continue
    }
    patch[key] = value
  }
  const fromVersion = parseParamsSchemaVersion(rawPatch.schemaVersion)
  patch.schemaVersion = TAB_PARAMS_SCHEMA_VERSION
  return {
    patch: patch as Partial<Wan14bTabParams>,
    droppedUnknownKeys,
    fromVersion,
  }
}

function migrateWan5bParamsPatch(rawPatch: Record<string, unknown>): {
  patch: Partial<Wan5bTabParams>
  droppedUnknownKeys: string[]
  fromVersion: number | null
} {
  const patch: Record<string, unknown> = {}
  const droppedUnknownKeys: string[] = []
  for (const [key, value] of Object.entries(rawPatch)) {
    if (!WAN5B_PARAM_TOP_LEVEL_KEYS.has(key)) {
      droppedUnknownKeys.push(key)
      continue
    }
    patch[key] = value
  }
  const fromVersion = parseParamsSchemaVersion(rawPatch.schemaVersion)
  patch.schemaVersion = TAB_PARAMS_SCHEMA_VERSION
  return {
    patch: patch as Partial<Wan5bTabParams>,
    droppedUnknownKeys,
    fromVersion,
  }
}

function normalizeWanFrameCount(rawValue: number, min = 9, max = 401): number {
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : min
  const clamped = Math.min(max, Math.max(min, numeric))
  if ((clamped - 1) % 4 === 0) return clamped

  const down = clamped - (((clamped - 1) % 4 + 4) % 4)
  const up = down + 4
  const downInRange = down >= min
  const upInRange = up <= max
  if (downInRange && upInRange) {
    const downDistance = Math.abs(clamped - down)
    const upDistance = Math.abs(up - clamped)
    return downDistance <= upDistance ? down : up
  }
  if (downInRange) return down
  if (upInRange) return up
  return min
}

function normalizeInterpolationTargetFps(rawValue: unknown, fallback: number): number {
  const maxFps = 240
  const fallbackNumeric = Number.isFinite(Number(fallback)) ? Math.trunc(Number(fallback)) : 0
  const fallbackNormalized = Math.max(0, Math.min(maxFps, fallbackNumeric))
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) return fallbackNormalized
  return Math.max(0, Math.min(maxFps, Math.trunc(numeric)))
}

function normalizeUpscalingBatchSize(rawValue: unknown, fallback: number): number {
  const fallbackInt = Number.isFinite(Number(fallback)) ? Math.max(1, Math.trunc(Number(fallback))) : 5
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) return fallbackInt
  const intValue = Math.max(1, Math.trunc(numeric))
  const remainder = (intValue - 1) % 4
  if (remainder === 0) return intValue
  const down = intValue - remainder
  const up = down + 4
  const downValid = down >= 1
  if (downValid) {
    const downDistance = Math.abs(intValue - down)
    const upDistance = Math.abs(up - intValue)
    return downDistance <= upDistance ? down : up
  }
  return up
}

function normalizeUpscalingColorCorrection(rawValue: unknown, fallback: WanVideoParams['upscalingColorCorrection']): WanVideoParams['upscalingColorCorrection'] {
  const value = String(rawValue || '').trim().toLowerCase()
  if (
    value === 'lab'
    || value === 'wavelet'
    || value === 'wavelet_adaptive'
    || value === 'hsv'
    || value === 'adain'
    || value === 'none'
  ) {
    return value
  }
  return fallback
}

function normalizeUnitInterval(rawValue: unknown, fallback: number): number {
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) return Math.min(1, Math.max(0, Number(fallback) || 0))
  return Math.min(1, Math.max(0, numeric))
}

function normalizeBoolean(rawValue: unknown, fallback: boolean): boolean {
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

function normalizeImageRunAction(rawValue: unknown, fallback: ImageRunAction): ImageRunAction {
  return rawValue === 'infinite' ? 'infinite' : fallback
}

function normalizeImageFolderSelectionMode(rawValue: unknown, fallback: ImageFolderSelectionMode): ImageFolderSelectionMode {
  return rawValue === 'count' ? 'count' : fallback
}

function normalizeImageFolderOrderMode(rawValue: unknown, fallback: ImageFolderOrderMode): ImageFolderOrderMode {
  return rawValue === 'random' || rawValue === 'sorted' ? rawValue : fallback
}

function normalizeImageFolderSortBy(rawValue: unknown, fallback: ImageFolderSortBy): ImageFolderSortBy {
  return rawValue === 'size' || rawValue === 'created_at' || rawValue === 'modified_at' || rawValue === 'name'
    ? rawValue
    : fallback
}

function normalizeInitSourceFormState(rawValue: unknown, defaults: InitSourceFormState): InitSourceFormState {
  const patch = asRecordObject(rawValue)
  return {
    mode: patch.mode === 'dir' ? 'dir' : defaults.mode,
    folderPath: String(patch.folderPath || '').trim(),
    selectionMode: normalizeImageFolderSelectionMode(patch.selectionMode, defaults.selectionMode),
    count: Math.max(1, Math.trunc(clampFiniteNumber(patch.count, defaults.count, 1))),
    order: normalizeImageFolderOrderMode(patch.order, defaults.order),
    sortBy: normalizeImageFolderSortBy(patch.sortBy, defaults.sortBy),
    useCrop: normalizeBoolean(patch.useCrop, defaults.useCrop),
  }
}

function normalizeSupirVariant(rawValue: unknown, fallback: SupirVariant): SupirVariant {
  return rawValue === 'v0F' || rawValue === 'v0Q' ? rawValue : fallback
}

function normalizeSupirColorFixMode(rawValue: unknown, fallback: SupirColorFixMode): SupirColorFixMode {
  return rawValue === 'None' || rawValue === 'AdaIN' || rawValue === 'Wavelet' ? rawValue : fallback
}

function normalizePositiveSupirNumber(rawValue: unknown, fallback: number, maxValue: number): number {
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric) || numeric <= 0) return fallback
  return Math.min(maxValue, numeric)
}

function normalizeNonNegativeSupirNumber(rawValue: unknown, fallback: number, maxValue: number): number {
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric) || numeric < 0) return fallback
  return Math.min(maxValue, numeric)
}

export function normalizeSupirSamplerSelection(rawValue: unknown, fallback: string): string {
  const normalized = String(rawValue || '').trim()
  if (!normalized) return fallback
  return normalized
}

function normalizeSupirModeFormState(rawValue: unknown, defaults: SupirModeFormState): SupirModeFormState {
  const patch = asRecordObject(rawValue)
  const sampler = normalizeSupirSamplerSelection(patch.sampler, defaults.sampler)
  return {
    enabled: normalizeBoolean(patch.enabled, defaults.enabled),
    variant: normalizeSupirVariant(patch.variant, defaults.variant),
    sampler,
    controlScale: normalizePositiveSupirNumber(patch.controlScale, defaults.controlScale, 2),
    restorationScale: normalizePositiveSupirNumber(patch.restorationScale, defaults.restorationScale, 6),
    restoreCfgSTmin: normalizeNonNegativeSupirNumber(patch.restoreCfgSTmin, defaults.restoreCfgSTmin, 5),
    colorFix: normalizeSupirColorFixMode(patch.colorFix, defaults.colorFix),
  }
}

function normalizeIpAdapterSourceFormState(rawValue: unknown, defaults: IpAdapterSourceFormState): IpAdapterSourceFormState {
  const patch = asRecordObject(rawValue)
  const mode = patch.mode === 'dir' ? 'dir' : defaults.mode
  return {
    mode,
    sameAsInit: mode === 'img' ? normalizeBoolean(patch.sameAsInit, defaults.sameAsInit) : false,
    referenceImageData: String(patch.referenceImageData || ''),
    referenceImageName: String(patch.referenceImageName || '').trim(),
    folderPath: String(patch.folderPath || '').trim(),
    selectionMode: normalizeImageFolderSelectionMode(patch.selectionMode, defaults.selectionMode),
    count: Math.max(1, Math.trunc(clampFiniteNumber(patch.count, defaults.count, 1))),
    order: normalizeImageFolderOrderMode(patch.order, defaults.order),
    sortBy: normalizeImageFolderSortBy(patch.sortBy, defaults.sortBy),
  }
}

function normalizeIpAdapterFormState(rawValue: unknown, defaults: IpAdapterFormState): IpAdapterFormState {
  const patch = asRecordObject(rawValue)
  return {
    enabled: normalizeBoolean(patch.enabled, defaults.enabled),
    model: String(patch.model || '').trim(),
    imageEncoder: String(patch.imageEncoder || '').trim(),
    source: normalizeIpAdapterSourceFormState(patch.source, defaults.source),
    weight: clampFiniteNumber(patch.weight, defaults.weight, 0, 2),
    startAt: normalizeUnitInterval(patch.startAt, defaults.startAt),
    endAt: normalizeUnitInterval(patch.endAt, defaults.endAt),
  }
}

function normalizeWanVideoParams(raw: Partial<WanVideoParams>, defaults: WanVideoParams): WanVideoParams {
  const merged: WanVideoParams = { ...defaults, ...raw }
  merged.frames = normalizeWanFrameCount(Number(merged.frames))

  const attnMode = String(merged.attentionMode || '').trim().toLowerCase()
  merged.attentionMode = attnMode === 'sliding' ? 'sliding' : 'global'

  const rawRecord = raw as Record<string, unknown>
  const hasExplicitMode = Object.prototype.hasOwnProperty.call(rawRecord, 'img2vidMode')
  const explicitMode = String(rawRecord.img2vidMode || '').trim().toLowerCase()
  if (hasExplicitMode && explicitMode === 'chunk') {
    throw new ModelTabsStoreError(
      'invalid_response',
      "img2vidMode='chunk' was removed. Set mode to 'solo', 'sliding', 'svi2', or 'svi2_pro'.",
      { details: { field: 'img2vidMode', value: rawRecord.img2vidMode } },
    )
  }
  if (hasExplicitMode && explicitMode && explicitMode !== 'solo' && explicitMode !== 'sliding' && explicitMode !== 'svi2' && explicitMode !== 'svi2_pro') {
    throw new ModelTabsStoreError(
      'invalid_response',
      `Unsupported img2vidMode '${explicitMode}'. Expected 'solo', 'sliding', 'svi2', or 'svi2_pro'.`,
      { details: { field: 'img2vidMode', value: rawRecord.img2vidMode } },
    )
  }
  if (explicitMode === 'solo' || explicitMode === 'sliding' || explicitMode === 'svi2' || explicitMode === 'svi2_pro') {
    merged.img2vidMode = explicitMode
  } else {
    const legacyEnabled = rawRecord.img2vidChunkingEnabled
    const hasLegacyChunkToggle = typeof legacyEnabled === 'boolean'
    const legacyChunkEnabled = hasLegacyChunkToggle ? Boolean(legacyEnabled) : false
    const hasSlidingWindow = Number.isFinite(Number(rawRecord.img2vidWindowFrames)) && Number(rawRecord.img2vidWindowFrames) > 0
    const hasChunkFrames = Number.isFinite(Number(rawRecord.img2vidChunkFrames)) && Number(rawRecord.img2vidChunkFrames) > 0
    if (legacyChunkEnabled || hasChunkFrames) {
      throw new ModelTabsStoreError(
        'invalid_response',
        "Legacy chunk-mode persistence is no longer supported. Set mode to 'solo', 'sliding', 'svi2', or 'svi2_pro'.",
        {
          details: {
            img2vidChunkingEnabled: rawRecord.img2vidChunkingEnabled,
            img2vidChunkFrames: rawRecord.img2vidChunkFrames,
          },
        },
      )
    }
    merged.img2vidMode = hasSlidingWindow ? 'sliding' : 'solo'
  }

  const chunkRaw = Number(merged.img2vidChunkFrames)
  if (!Number.isFinite(chunkRaw) || chunkRaw <= 0) {
    merged.img2vidChunkFrames = defaults.img2vidChunkFrames
  } else {
    merged.img2vidChunkFrames = normalizeWanFrameCount(chunkRaw, 9, 401)
  }

  const anchorRaw = Number(merged.img2vidAnchorAlpha)
  merged.img2vidAnchorAlpha = Number.isFinite(anchorRaw) ? Math.min(1, Math.max(0, anchorRaw)) : defaults.img2vidAnchorAlpha

  const modeDefaultResetAnchor = false
  const hasExplicitResetAnchor = Object.prototype.hasOwnProperty.call(rawRecord, 'img2vidResetAnchorToBase')
  if (merged.img2vidMode === 'svi2' || merged.img2vidMode === 'svi2_pro') {
    merged.img2vidResetAnchorToBase = false
  } else if (hasExplicitResetAnchor) {
    merged.img2vidResetAnchorToBase = Boolean(rawRecord.img2vidResetAnchorToBase)
  } else {
    merged.img2vidResetAnchorToBase = modeDefaultResetAnchor
  }

  const seedMode = String(merged.img2vidChunkSeedMode || '').trim().toLowerCase()
  const modeDefaultSeed = merged.img2vidMode === 'sliding' ? 'fixed' : 'increment'
  if (seedMode !== 'fixed' && seedMode !== 'increment' && seedMode !== 'random') {
    merged.img2vidChunkSeedMode = modeDefaultSeed
  } else {
    merged.img2vidChunkSeedMode = seedMode
  }

  const windowRaw = Number(merged.img2vidWindowFrames)
  if (!Number.isFinite(windowRaw) || windowRaw <= 0) {
    merged.img2vidWindowFrames = defaults.img2vidWindowFrames
  } else {
    merged.img2vidWindowFrames = normalizeWanFrameCount(windowRaw, 9, 401)
  }

  const temporalUpperBound = normalizeWanFrameCount(Math.max(9, merged.frames - 4), 9, 401)
  if (temporalUpperBound < merged.frames) {
    if (merged.img2vidChunkFrames >= merged.frames) {
      merged.img2vidChunkFrames = temporalUpperBound
    }
    if (merged.img2vidWindowFrames >= merged.frames) {
      merged.img2vidWindowFrames = temporalUpperBound
    }
  }

  const overlapRaw = Number(merged.img2vidOverlapFrames)
  merged.img2vidOverlapFrames = normalizeWanChunkOverlap(
    overlapRaw,
    merged.img2vidChunkFrames,
    defaults.img2vidOverlapFrames,
  )

  const strideRaw = Number(merged.img2vidWindowStride)
  merged.img2vidWindowStride = normalizeWanWindowStride(
    strideRaw,
    merged.img2vidWindowFrames,
    defaults.img2vidWindowStride,
  )

  const commitRaw = Number(merged.img2vidWindowCommitFrames)
  merged.img2vidWindowCommitFrames = normalizeWanWindowCommit(
    commitRaw,
    merged.img2vidWindowFrames,
    merged.img2vidWindowStride,
    defaults.img2vidWindowCommitFrames,
  )
  merged.img2vidImageScale = normalizeWanImg2VidImageScale(
    merged.img2vidImageScale,
    defaults.img2vidImageScale,
  )
  merged.img2vidCropOffsetX = normalizeUnitInterval(merged.img2vidCropOffsetX, defaults.img2vidCropOffsetX)
  merged.img2vidCropOffsetY = normalizeUnitInterval(merged.img2vidCropOffsetY, defaults.img2vidCropOffsetY)
  merged.returnFrames = normalizeBoolean(merged.returnFrames, defaults.returnFrames ?? false)

  merged.interpolationFps = normalizeInterpolationTargetFps(
    merged.interpolationFps,
    defaults.interpolationFps,
  )
  merged.upscalingEnabled = Boolean(merged.upscalingEnabled)
  merged.upscalingUniformBatchSize = Boolean(merged.upscalingUniformBatchSize)
  const upscalingModel = String(merged.upscalingModel || '').trim()
  merged.upscalingModel = upscalingModel || defaults.upscalingModel
  const resolution = Number(merged.upscalingResolution)
  merged.upscalingResolution = Number.isFinite(resolution)
    ? Math.max(16, Math.trunc(resolution))
    : defaults.upscalingResolution
  const maxResolution = Number(merged.upscalingMaxResolution)
  merged.upscalingMaxResolution = Number.isFinite(maxResolution)
    ? Math.max(0, Math.trunc(maxResolution))
    : defaults.upscalingMaxResolution
  merged.upscalingBatchSize = normalizeUpscalingBatchSize(merged.upscalingBatchSize, defaults.upscalingBatchSize)
  const overlap = Number(merged.upscalingTemporalOverlap)
  merged.upscalingTemporalOverlap = Number.isFinite(overlap)
    ? Math.max(0, Math.trunc(overlap))
    : defaults.upscalingTemporalOverlap
  const prepend = Number(merged.upscalingPrependFrames)
  merged.upscalingPrependFrames = Number.isFinite(prepend)
    ? Math.max(0, Math.trunc(prepend))
    : defaults.upscalingPrependFrames
  merged.upscalingColorCorrection = normalizeUpscalingColorCorrection(
    merged.upscalingColorCorrection,
    defaults.upscalingColorCorrection,
  )
  merged.upscalingInputNoiseScale = normalizeUnitInterval(merged.upscalingInputNoiseScale, defaults.upscalingInputNoiseScale)
  merged.upscalingLatentNoiseScale = normalizeUnitInterval(merged.upscalingLatentNoiseScale, defaults.upscalingLatentNoiseScale)

  return {
    width: merged.width,
    height: merged.height,
    fps: merged.fps,
    frames: merged.frames,
    attentionMode: merged.attentionMode,
    useInitImage: merged.useInitImage,
    initImageData: merged.initImageData,
    initImageName: merged.initImageName,
    img2vidMode: merged.img2vidMode,
    img2vidChunkFrames: merged.img2vidChunkFrames,
    img2vidOverlapFrames: merged.img2vidOverlapFrames,
    img2vidAnchorAlpha: merged.img2vidAnchorAlpha,
    img2vidResetAnchorToBase: merged.img2vidResetAnchorToBase,
    img2vidChunkSeedMode: merged.img2vidChunkSeedMode,
    img2vidWindowFrames: merged.img2vidWindowFrames,
    img2vidWindowStride: merged.img2vidWindowStride,
    img2vidWindowCommitFrames: merged.img2vidWindowCommitFrames,
    img2vidImageScale: merged.img2vidImageScale,
    img2vidCropOffsetX: merged.img2vidCropOffsetX,
    img2vidCropOffsetY: merged.img2vidCropOffsetY,
    format: merged.format,
    pixFmt: merged.pixFmt,
    crf: merged.crf,
    loopCount: merged.loopCount,
    pingpong: merged.pingpong,
    returnFrames: merged.returnFrames,
    interpolationFps: merged.interpolationFps,
    upscalingEnabled: merged.upscalingEnabled,
    upscalingModel: merged.upscalingModel,
    upscalingResolution: merged.upscalingResolution,
    upscalingMaxResolution: merged.upscalingMaxResolution,
    upscalingBatchSize: merged.upscalingBatchSize,
    upscalingUniformBatchSize: merged.upscalingUniformBatchSize,
    upscalingTemporalOverlap: merged.upscalingTemporalOverlap,
    upscalingPrependFrames: merged.upscalingPrependFrames,
    upscalingColorCorrection: merged.upscalingColorCorrection,
    upscalingInputNoiseScale: merged.upscalingInputNoiseScale,
    upscalingLatentNoiseScale: merged.upscalingLatentNoiseScale,
  }
}

function normalizeWan14bStageParams(stagePatch: unknown, stageDefaults: WanStageParams): WanStageParams {
  const stagePatchRecord = asRecordObject(stagePatch)
  const merged: WanStageParams = {
    ...stageDefaults,
    ...(stagePatchRecord as Partial<WanStageParams>),
  }
  const sampler = typeof merged.sampler === 'string' ? merged.sampler.trim() : ''
  merged.sampler = sampler || stageDefaults.sampler
  const scheduler = typeof merged.scheduler === 'string' ? merged.scheduler.trim().toLowerCase() : ''
  merged.scheduler = scheduler === 'simple' ? scheduler : 'simple'
  const normalizedStage: WanStageParams = {
    modelDir: merged.modelDir,
    prompt: merged.prompt,
    negativePrompt: merged.negativePrompt,
    sampler: merged.sampler,
    scheduler: merged.scheduler,
    steps: merged.steps,
    cfgScale: merged.cfgScale,
    seed: merged.seed,
    loras: Array.isArray(merged.loras) ? merged.loras : stageDefaults.loras,
  }
  if (typeof merged.flowShift === 'number' && Number.isFinite(merged.flowShift)) {
    normalizedStage.flowShift = merged.flowShift
  }
  return normalizedStage
}

function normalizeWan14bParams(raw: unknown, defaults: Wan14bTabParams): Wan14bTabParams {
  const rawPatch = asRecordObject(raw)
  const migration = migrateWan14bParamsPatch(rawPatch)
  if (migration.droppedUnknownKeys.length > 0) {
    console.warn(
      `[model_tabs] Dropping stale WAN 14B params key(s) during migration: ${migration.droppedUnknownKeys.join(', ')}`,
      { fromVersion: migration.fromVersion, toVersion: TAB_PARAMS_SCHEMA_VERSION },
    )
  }
  const patch = migration.patch
  const videoPatch = asRecordObject(patch.video)
  const assetsPatch = asRecordObject(patch.assets)
  const normalizedVideo = normalizeWanVideoParams(videoPatch as Partial<WanVideoParams>, defaults.video)
  return {
    ...defaults,
    high: normalizeWan14bStageParams(patch.high, defaults.high),
    low: normalizeWan14bStageParams(patch.low, defaults.low),
    video: normalizedVideo,
    assets: { ...defaults.assets, ...(assetsPatch as Partial<WanAssetsParams>) },
    lightx2v: normalizeBoolean(patch.lightx2v, defaults.lightx2v),
    lowFollowsHigh: normalizeBoolean(patch.lowFollowsHigh, defaults.lowFollowsHigh),
    schemaVersion: TAB_PARAMS_SCHEMA_VERSION,
  }
}

function normalizeWan5bStageParams(stagePatch: unknown, stageDefaults: Wan5bStageParams): Wan5bStageParams {
  const stagePatchRecord = asRecordObject(stagePatch)
  const merged: Wan5bStageParams = {
    ...stageDefaults,
    ...(stagePatchRecord as Partial<Wan5bStageParams>),
  }
  const normalizedStage: Wan5bStageParams = {
    modelDir: String(merged.modelDir || '').trim(),
    loras: Array.isArray(merged.loras) ? merged.loras : stageDefaults.loras,
  }
  if (typeof merged.flowShift === 'number' && Number.isFinite(merged.flowShift)) {
    normalizedStage.flowShift = merged.flowShift
  }
  return normalizedStage
}

function normalizeWan5bParams(raw: unknown, defaults: Wan5bTabParams): Wan5bTabParams {
  const rawPatch = asRecordObject(raw)
  const migration = migrateWan5bParamsPatch(rawPatch)
  if (migration.droppedUnknownKeys.length > 0) {
    console.warn(
      `[model_tabs] Dropping stale WAN 5B params key(s) during migration: ${migration.droppedUnknownKeys.join(', ')}`,
      { fromVersion: migration.fromVersion, toVersion: TAB_PARAMS_SCHEMA_VERSION },
    )
  }
  const patch = migration.patch
  const videoPatch = asRecordObject(patch.video)
  const assetsPatch = asRecordObject(patch.assets)
  const sampler = typeof patch.sampler === 'string' ? patch.sampler.trim() : ''
  const scheduler = typeof patch.scheduler === 'string' ? patch.scheduler.trim().toLowerCase() : ''
  const steps = Number(patch.steps)
  const seed = Number(patch.seed)
  return {
    ...defaults,
    prompt: typeof patch.prompt === 'string' ? patch.prompt : defaults.prompt,
    negativePrompt: typeof patch.negativePrompt === 'string' ? patch.negativePrompt : defaults.negativePrompt,
    stage: normalizeWan5bStageParams(patch.stage, defaults.stage),
    video: normalizeWanVideoParams(videoPatch as Partial<WanVideoParams>, defaults.video),
    assets: { ...defaults.assets, ...(assetsPatch as Partial<WanAssetsParams>) },
    sampler: sampler || defaults.sampler,
    scheduler: scheduler === 'simple' ? scheduler : defaults.scheduler,
    steps: Number.isFinite(steps) ? Math.max(1, Math.trunc(steps)) : defaults.steps,
    cfgScale: clampFiniteNumber(patch.cfgScale, defaults.cfgScale, 0, Number.POSITIVE_INFINITY),
    seed: Number.isFinite(seed) ? Math.trunc(seed) : defaults.seed,
    schemaVersion: TAB_PARAMS_SCHEMA_VERSION,
  }
}

function shouldPersistWan14bStageSamplingBackfill(raw: unknown): boolean {
  const patch = asRecordObject(raw)
  const high = asRecordObject(patch.high)
  const low = asRecordObject(patch.low)
  const highSampler = typeof high.sampler === 'string' ? high.sampler.trim() : ''
  const lowSampler = typeof low.sampler === 'string' ? low.sampler.trim() : ''
  const highScheduler = typeof high.scheduler === 'string' ? high.scheduler.trim().toLowerCase() : ''
  const lowScheduler = typeof low.scheduler === 'string' ? low.scheduler.trim().toLowerCase() : ''
  const needsSamplerBackfill = highSampler.length === 0 || lowSampler.length === 0
  const needsSchedulerBackfill = highScheduler !== 'simple' || lowScheduler !== 'simple'
  return needsSamplerBackfill || needsSchedulerBackfill
}

function buildImageTopLevelBackfillPatch(
  raw: unknown,
  normalized: ImageBaseParams,
): Partial<ImageBaseParams> | null {
  const patch = asRecordObject(raw)
  const backfillPatch: Partial<ImageBaseParams> = {}
  let needsBackfill = false
  for (const key of IMAGE_PARAM_TOP_LEVEL_KEYS) {
    if (IMAGE_PARAM_NO_AUTOBACKFILL_KEYS.has(key)) continue
    if (Object.prototype.hasOwnProperty.call(patch, key)) continue
    const normalizedValue = normalized[key as keyof ImageBaseParams]
    if (normalizedValue === undefined) continue
    ;(backfillPatch as Record<string, unknown>)[key] = normalizedValue
    needsBackfill = true
  }
  return needsBackfill ? backfillPatch : null
}

function normalizeQwenImageTextEncoders(labels: string[]): string[] {
  if (labels.length === 0) return []
  if (labels.length !== 1) {
    throw new ModelTabsStoreError(
      'invalid_response',
      'Qwen Image requires exactly one qwen_image/<path> text encoder selection.',
    )
  }
  const normalized = labels[0].replace(/\\+/g, '/').trim()
  if (!normalized.startsWith('qwen_image/') || normalized.length <= 'qwen_image/'.length) {
    throw new ModelTabsStoreError(
      'invalid_response',
      'Qwen Image text encoder selections must use qwen_image/<path> labels from qwen_image_tenc roots.',
    )
  }
  return [normalized]
}

function normalizeZImageL2PTextEncoders(labels: string[]): string[] {
  if (labels.length === 0) return []
  if (labels.length !== 1) {
    throw new ModelTabsStoreError(
      'invalid_response',
      'Z-Image L2P requires exactly one zimage/<path> text encoder selection.',
    )
  }
  const normalized = labels[0].replace(/\\+/g, '/').trim()
  if (!normalized.startsWith('zimage/') || normalized.length <= 'zimage/'.length) {
    throw new ModelTabsStoreError(
      'invalid_response',
      'Z-Image L2P text encoder selections must use zimage/<path> labels from zimage_tenc roots.',
    )
  }
  return [normalized]
}

function normalizeGuidanceAdvancedParams(raw: unknown, defaults: GuidanceAdvancedParams): GuidanceAdvancedParams {
  const patch = asRecordObject(raw)
  const toFiniteNumber = (value: unknown, fallback: number): number => {
    const numeric = Number(value)
    return Number.isFinite(numeric) ? numeric : fallback
  }
  const clampNumber = (value: unknown, fallback: number, min?: number, max?: number): number => {
    const numeric = toFiniteNumber(value, fallback)
    if (min !== undefined && numeric < min) return min
    if (max !== undefined && numeric > max) return max
    return numeric
  }
  const clampInteger = (value: unknown, fallback: number, min?: number, max?: number): number => {
    const numeric = Math.trunc(clampNumber(value, fallback, min, max))
    if (min !== undefined && numeric < min) return min
    if (max !== undefined && numeric > max) return max
    return numeric
  }
  return {
    enabled: typeof patch.enabled === 'boolean' ? patch.enabled : defaults.enabled,
    apgEnabled: typeof patch.apgEnabled === 'boolean' ? patch.apgEnabled : defaults.apgEnabled,
    apgStartStep: clampInteger(patch.apgStartStep, defaults.apgStartStep, 0),
    apgEta: clampNumber(patch.apgEta, defaults.apgEta),
    apgMomentum: clampNumber(patch.apgMomentum, defaults.apgMomentum, 0, 0.999999),
    apgNormThreshold: clampNumber(patch.apgNormThreshold, defaults.apgNormThreshold, 0),
    apgRescale: clampNumber(patch.apgRescale, defaults.apgRescale, 0, 1),
    guidanceRescale: clampNumber(patch.guidanceRescale, defaults.guidanceRescale, 0, 1),
    cfgTruncEnabled: typeof patch.cfgTruncEnabled === 'boolean' ? patch.cfgTruncEnabled : defaults.cfgTruncEnabled,
    cfgTruncRatio: clampNumber(patch.cfgTruncRatio, defaults.cfgTruncRatio, 0, 1),
    renormCfg: clampNumber(patch.renormCfg, defaults.renormCfg, 0),
  }
}

function clampFiniteNumber(value: unknown, fallback: number, min?: number, max?: number): number {
  const numeric = Number(value)
  const finiteValue = Number.isFinite(numeric) ? numeric : fallback
  if (min !== undefined && finiteValue < min) return min
  if (max !== undefined && finiteValue > max) return max
  return finiteValue
}

function normalizeImageParams(raw: unknown, defaults: ImageBaseParams): ImageBaseParams {
  const rawPatch = asRecordObject(raw)
  const migration = migrateImageParamsPatch(rawPatch)
  if (migration.droppedUnknownKeys.length > 0) {
    console.warn(
      `[model_tabs] Dropping stale image params key(s) during migration: ${migration.droppedUnknownKeys.join(', ')}`,
      { fromVersion: migration.fromVersion, toVersion: TAB_PARAMS_SCHEMA_VERSION },
    )
  }
  const patch = migration.patch
  const hiresPatch = asRecordObject(patch.hires)
  const hiresSwapModelPatch = asRecordObject(hiresPatch.swapModel)
  const hiresRefinerPatch = asRecordObject(hiresPatch.refiner)
  const hiresTilePatch = asRecordObject(hiresPatch.tile)
  const swapModelPatch = asRecordObject(patch.swapModel)
  const refinerPatch = asRecordObject(patch.refiner)
  const initSourcePatch = asRecordObject(patch.initSource)
  const supirPatch = asRecordObject(patch.supir)
  const ipAdapterPatch = asRecordObject(patch.ipAdapter)
  const ipAdapterSourcePatch = asRecordObject(ipAdapterPatch.source)

  const mergedHires: HiresFormState = {
    ...defaults.hires,
    ...(hiresPatch as Partial<HiresFormState>),
    swapModel: Object.keys(hiresSwapModelPatch).length > 0
      ? (hiresSwapModelPatch as Partial<SwapModelFormState>) as SwapModelFormState
      : defaults.hires.swapModel,
    refiner: {
      ...(asRecordObject(defaults.hires.refiner) as Partial<RefinerFormState>),
      ...(hiresRefinerPatch as Partial<RefinerFormState>),
    } as RefinerFormState,
    tile: {
      ...defaults.hires.tile,
      ...(hiresTilePatch as Partial<HiresFormState['tile']>),
    },
  }
  delete (mergedHires as unknown as Record<string, unknown>).modules

  const merged: ImageBaseParams = {
    ...defaults,
    ...patch,
    swapModel: Object.keys(swapModelPatch).length > 0
      ? {
          ...defaults.swapModel,
          ...(swapModelPatch as Partial<SwapStageFormState>),
        }
      : defaults.swapModel,
    hires: mergedHires,
    refiner: {
      ...defaults.refiner,
      ...(refinerPatch as Partial<RefinerFormState>),
    },
    initSource: {
      ...defaults.initSource,
      ...(initSourcePatch as Partial<InitSourceFormState>),
    },
    supir: {
      ...defaults.supir,
      ...(supirPatch as Partial<SupirModeFormState>),
    },
    ipAdapter: {
      ...defaults.ipAdapter,
      ...(ipAdapterPatch as Partial<IpAdapterFormState>),
      source: {
        ...defaults.ipAdapter.source,
        ...(ipAdapterSourcePatch as Partial<IpAdapterSourceFormState>),
      },
    },
  }

  delete (merged.hires as unknown as Record<string, unknown>).checkpoint
  delete (merged.refiner as unknown as Record<string, unknown>).vae
  delete (merged.hires.refiner as unknown as Record<string, unknown>).vae

  merged.useInitImage = normalizeBoolean(merged.useInitImage, defaults.useInitImage)
  merged.useMask = normalizeBoolean(merged.useMask, defaults.useMask)
  merged.maskInvert = normalizeBoolean(merged.maskInvert, defaults.maskInvert)
  merged.maskRound = normalizeBoolean(merged.maskRound, defaults.maskRound)
  merged.maskRegionSplit = normalizeBoolean(merged.maskRegionSplit, defaults.maskRegionSplit)
  if (typeof merged.zimageTurbo !== 'undefined') {
    merged.zimageTurbo = normalizeBoolean(merged.zimageTurbo, Boolean(defaults.zimageTurbo))
  }

  const globalSwapAtStep = Number(merged.refiner.swapAtStep)
  merged.refiner.swapAtStep = Number.isFinite(globalSwapAtStep) && globalSwapAtStep >= 1
    ? Math.trunc(globalSwapAtStep)
    : 1
  merged.swapModel.enabled = normalizeBoolean(merged.swapModel.enabled, defaults.swapModel.enabled)
  const globalModelSwapAtStep = Number(merged.swapModel.swapAtStep)
  merged.swapModel.swapAtStep = Number.isFinite(globalModelSwapAtStep) && globalModelSwapAtStep >= 1
    ? Math.trunc(globalModelSwapAtStep)
    : 1
  merged.swapModel.cfg = clampFiniteNumber(
    merged.swapModel.cfg,
    merged.cfgScale,
    0,
    Number.POSITIVE_INFINITY,
  )
  merged.swapModel.seed = Number.isFinite(Number(merged.swapModel.seed))
    ? Math.trunc(Number(merged.swapModel.seed))
    : defaults.swapModel.seed
  merged.swapModel.model = String(merged.swapModel.model || '').trim() || undefined
  if (merged.hires.refiner) {
    const hiresSwapAtStep = Number(merged.hires.refiner.swapAtStep)
    merged.hires.refiner.swapAtStep = Number.isFinite(hiresSwapAtStep) && hiresSwapAtStep >= 1
      ? Math.trunc(hiresSwapAtStep)
      : 1
  }

  if (typeof merged.sampler !== 'string' || !merged.sampler.trim()) {
    merged.sampler = defaults.sampler
  }
  if (typeof merged.scheduler !== 'string' || !merged.scheduler.trim()) {
    merged.scheduler = defaults.scheduler
  }
  merged.inpaintMode = parseInpaintMode(
    typeof merged.inpaintMode === 'string' ? merged.inpaintMode : null,
  ) ?? defaults.inpaintMode
  merged.perStepBlendStrength = clampFiniteNumber(merged.perStepBlendStrength, defaults.perStepBlendStrength, 0, 1)
  merged.perStepBlendSteps = Math.trunc(
    clampFiniteNumber(merged.perStepBlendSteps, defaults.perStepBlendSteps, 0),
  )
  merged.img2imgResizeMode = normalizeImg2ImgResizeMode(merged.img2imgResizeMode)
  merged.img2imgUpscaler = String(merged.img2imgUpscaler || '').trim() || defaults.img2imgUpscaler
  merged.textEncoders = Array.isArray(merged.textEncoders)
    ? merged.textEncoders
        .map((entry) => String(entry || '').trim())
        .filter((entry, index, array) => entry.length > 0 && array.indexOf(entry) === index)
    : defaults.textEncoders.slice()
  merged.guidanceAdvanced = normalizeGuidanceAdvancedParams(
    patch.guidanceAdvanced,
    defaults.guidanceAdvanced ?? DEFAULT_GUIDANCE_ADVANCED_PARAMS,
  )
  merged.runAction = normalizeImageRunAction(merged.runAction, defaults.runAction)
  merged.initSource = normalizeInitSourceFormState(merged.initSource, defaults.initSource)
  merged.supir = normalizeSupirModeFormState(merged.supir, defaults.supir)
  merged.ipAdapter = normalizeIpAdapterFormState(merged.ipAdapter, defaults.ipAdapter)
  if (merged.ipAdapter.endAt < merged.ipAdapter.startAt) {
    merged.ipAdapter.endAt = merged.ipAdapter.startAt
  }
  merged.schemaVersion = TAB_PARAMS_SCHEMA_VERSION
  return merged
}

function normalizeParamsForType<T extends BaseTabType>(
  type: T,
  raw: unknown,
  defaultsOverride?: TabParamsByType[T],
): TabParamsByType[T] {
  const defaults = defaultsOverride ?? defaultParams(type)
  if (type === 'wan22_14b') {
    return normalizeWan14bParams(raw, defaults as Wan14bTabParams) as TabParamsByType[T]
  }
  if (type === 'wan22_5b') {
    return normalizeWan5bParams(raw, defaults as Wan5bTabParams) as TabParamsByType[T]
  }
  if (type === 'ltx2') {
    return normalizeLtxParams(raw, defaults as TabParamsByType['ltx2']) as TabParamsByType[T]
  }
  const normalized = normalizeImageParams(raw, defaults as ImageBaseParams)
  if (type === 'flux2') {
    normalized.textEncoders = normalized.textEncoders
      .filter((label) => label.startsWith('flux2/'))
      .slice(0, 1)
  }
  if (type === 'qwen_image') {
    normalized.textEncoders = normalizeQwenImageTextEncoders(normalized.textEncoders)
    normalized.useInitImage = true
    normalized.useMask = false
    normalized.maskImageData = ''
    normalized.maskImageName = ''
    normalized.clipSkip = 0
    normalized.batchCount = 1
    normalized.batchSize = 1
    normalized.runAction = 'generate'
    normalized.denoiseStrength = 1
    const qwenSteps = Number(normalized.steps)
    normalized.steps = Number.isFinite(qwenSteps)
      ? Math.max(2, Math.trunc(qwenSteps))
      : Math.max(2, Math.trunc(Number((defaults as ImageBaseParams).steps)))
    normalized.initSource = {
      ...normalized.initSource,
      mode: 'img',
      folderPath: '',
      selectionMode: 'all',
      count: 1,
      order: 'sorted',
      sortBy: 'name',
      useCrop: false,
    }
    normalized.hires = {
      ...normalized.hires,
      enabled: false,
      swapModel: undefined,
      refiner: normalized.hires.refiner
        ? { ...normalized.hires.refiner, enabled: false, model: undefined }
        : normalized.hires.refiner,
    }
    normalized.swapModel = { ...normalized.swapModel, enabled: false, model: undefined }
    normalized.refiner = { ...normalized.refiner, enabled: false, model: undefined }
    normalized.supir = { ...normalized.supir, enabled: false }
    normalized.ipAdapter = {
      ...normalized.ipAdapter,
      enabled: false,
      model: '',
      imageEncoder: '',
      source: {
        ...normalized.ipAdapter.source,
        mode: 'img',
        sameAsInit: false,
        referenceImageData: '',
        referenceImageName: '',
        folderPath: '',
        selectionMode: 'all',
        count: 1,
        order: 'sorted',
        sortBy: 'name',
      },
    }
    normalized.guidanceAdvanced = {
      ...normalized.guidanceAdvanced,
      enabled: false,
      apgEnabled: false,
      cfgTruncEnabled: false,
    }
  }
  if (type === 'zimage_l2p') {
    normalized.textEncoders = normalizeZImageL2PTextEncoders(normalized.textEncoders)
    normalized.useInitImage = false
    normalized.useMask = false
    normalized.maskImageData = ''
    normalized.maskImageName = ''
    normalized.clipSkip = 0
    normalized.batchCount = 1
    normalized.batchSize = 1
    normalized.runAction = 'generate'
    normalized.width = 1024
    normalized.height = 1024
    normalized.hires = {
      ...normalized.hires,
      enabled: false,
      swapModel: undefined,
      refiner: normalized.hires.refiner
        ? { ...normalized.hires.refiner, enabled: false }
        : normalized.hires.refiner,
    }
    normalized.swapModel = { ...normalized.swapModel, enabled: false }
    normalized.refiner = { ...normalized.refiner, enabled: false }
    normalized.supir = { ...normalized.supir, enabled: false }
    normalized.ipAdapter = {
      ...normalized.ipAdapter,
      enabled: false,
      source: {
        ...normalized.ipAdapter.source,
        mode: 'img',
        sameAsInit: false,
        count: 1,
      },
    }
    normalized.guidanceAdvanced = {
      ...normalized.guidanceAdvanced,
      enabled: false,
      apgEnabled: false,
      cfgTruncEnabled: false,
    }
    delete (normalized as unknown as Record<string, unknown>).zimageTurbo
  }
  return normalized as TabParamsByType[T]
}

type RawTab = Omit<BaseTab, 'type' | 'params'> & {
  type: unknown
  params?: unknown
}

function normalizeTab(
  tab: RawTab,
  resolveDefaults?: <T extends BaseTabType>(type: T) => TabParamsByType[T],
): BaseTab {
  const type = normalizeTabType(tab.type)
  const defaults = resolveDefaults ? resolveDefaults(type) : undefined
  return {
    ...tab,
    type,
    params: asParamsRecord(normalizeParamsForType(type, tab.params, defaults)),
  }
}

const BASE_REQUIRED_TYPES: BaseTabType[] = ['sd15', 'sdxl', 'flux1', 'flux2', 'chroma', 'zimage', 'wan22_14b', 'wan22_5b']

export function requiredTypesFromCapabilities(engines: Record<string, unknown>): BaseTabType[] {
  const types: BaseTabType[] = [...BASE_REQUIRED_TYPES]
  if (Object.prototype.hasOwnProperty.call(engines, 'ltx2')) {
    types.push('ltx2')
  }
  if (Object.prototype.hasOwnProperty.call(engines, 'qwen_image')) {
    types.push('qwen_image')
  }
  if (Object.prototype.hasOwnProperty.call(engines, 'zimage_l2p')) {
    types.push('zimage_l2p')
  }
  if (Object.prototype.hasOwnProperty.call(engines, 'anima')) {
    types.push('anima')
  }
  return types
}

export const useModelTabsStore = defineStore('modelTabs', () => {
  const tabs = ref<BaseTab[]>([])
  const activeId = ref<string>('')
  const pendingParamsPersists = new Map<string, PendingParamsPersist>()
  const imageSparsePersistMissingKeysByTabId = new Map<string, Set<string>>()
  let loadPromise: Promise<void> | null = null

  const PARAMS_PERSIST_DEBOUNCE_MS = 220

  type PersistDeferred = {
    version: number
    resolve: () => void
    reject: (reason?: unknown) => void
  }

  type PendingParamsPersist = {
    timer: ReturnType<typeof setTimeout> | null
    inFlight: boolean
    version: number
    persistedVersion: number
    deferreds: PersistDeferred[]
    snapshotParams: Record<string, unknown> | null
    snapshotUpdatedAt: string
    snapshotSparseMissingKeys: string[] | null
    explicitImagePersistKeys: Set<string>
  }

  type PersistSerializationPhase = 'snapshot' | 'patch' | 'persist' | 'rollback'

  function syncImageSparsePersistHints(tabId: string, tabType: BaseTabType, rawParams: unknown): void {
    if (isWanTabFamily(tabType) || tabType === 'ltx2') {
      imageSparsePersistMissingKeysByTabId.delete(tabId)
      return
    }
    const rawPatch = asRecordObject(rawParams)
    const missingKeys = new Set<string>()
    for (const key of IMAGE_PARAM_SPARSE_PERSIST_DEFAULT_KEYS) {
      if (!Object.prototype.hasOwnProperty.call(rawPatch, key)) {
        missingKeys.add(key)
      }
    }
    if (missingKeys.size === 0) {
      imageSparsePersistMissingKeysByTabId.delete(tabId)
      return
    }
    imageSparsePersistMissingKeysByTabId.set(tabId, missingKeys)
  }

  function snapshotImageSparsePersistHints(tabId: string): string[] | null {
    const missingKeys = imageSparsePersistMissingKeysByTabId.get(tabId)
    if (!missingKeys || missingKeys.size === 0) return null
    return Array.from(missingKeys).sort()
  }

  function restoreImageSparsePersistHints(tabId: string, snapshot: string[] | null): void {
    if (!snapshot || snapshot.length === 0) {
      imageSparsePersistMissingKeysByTabId.delete(tabId)
      return
    }
    imageSparsePersistMissingKeysByTabId.set(tabId, new Set(snapshot))
  }

  function markExplicitImagePersistKeys(
    pending: PendingParamsPersist,
    tabType: BaseTabType,
    patch: Record<string, unknown>,
  ): void {
    if (isWanTabFamily(tabType) || tabType === 'ltx2') return
    for (const key of IMAGE_PARAM_SPARSE_PERSIST_DEFAULT_KEYS) {
      if (Object.prototype.hasOwnProperty.call(patch, key)) {
        pending.explicitImagePersistKeys.add(key)
      }
    }
  }

  function pruneSparseImagePersistDefaults(
    tabId: string,
    tabType: BaseTabType,
    params: Record<string, unknown>,
    explicitKeys: ReadonlySet<string>,
  ): Record<string, unknown> {
    if (isWanTabFamily(tabType) || tabType === 'ltx2') return params
    const missingKeys = imageSparsePersistMissingKeysByTabId.get(tabId)
    if (!missingKeys || missingKeys.size === 0) return params
    let prunedParams = params
    const defaults = defaultImageParamsForType(tabType)
    if (
      missingKeys.has('inpaintMode') &&
      !explicitKeys.has('inpaintMode') &&
      Object.prototype.hasOwnProperty.call(prunedParams, 'inpaintMode') &&
      prunedParams.inpaintMode === defaults.inpaintMode
    ) {
      prunedParams = { ...prunedParams }
      delete prunedParams.inpaintMode
    }
    return prunedParams
  }

  function applyPersistedImageSparsePersistHints(
    tabId: string,
    tabType: BaseTabType,
    explicitKeys: ReadonlySet<string>,
  ): string[] | null {
    if (isWanTabFamily(tabType) || tabType === 'ltx2') {
      imageSparsePersistMissingKeysByTabId.delete(tabId)
      return null
    }
    const nextMissingKeys = new Set(imageSparsePersistMissingKeysByTabId.get(tabId) ?? [])
    for (const key of explicitKeys) {
      nextMissingKeys.delete(key)
    }
    if (nextMissingKeys.size === 0) {
      imageSparsePersistMissingKeysByTabId.delete(tabId)
      return null
    }
    const snapshot = Array.from(nextMissingKeys).sort()
    imageSparsePersistMissingKeysByTabId.set(tabId, new Set(snapshot))
    return snapshot
  }

  async function resolveRequiredTypesFromCapabilities(): Promise<BaseTabType[]> {
    const capsStore = useEngineCapabilitiesStore()
    await capsStore.init()
    const engines = capsStore.engines
    return requiredTypesFromCapabilities(engines as Record<string, unknown>)
  }

  function save(): void {
    const payload = buildStoragePayload(tabs.value, activeId.value)
    const serializedPayload = JSON.stringify(payload)
    try {
      localStorage.setItem(STORAGE_KEY, serializedPayload)
      return
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (!isQuotaExceededStorageError(error)) {
        console.warn(`[model_tabs] Failed to persist local state '${STORAGE_KEY}': ${message}`)
        return
      }
      console.warn(`[model_tabs] LocalStorage quota exceeded for '${STORAGE_KEY}'. Writing minimal fallback state.`)
    }

    try {
      localStorage.removeItem(STORAGE_KEY)
      localStorage.setItem(STORAGE_KEY, serializedPayload)
      return
    } catch (fallbackError) {
      const message = fallbackError instanceof Error ? fallbackError.message : String(fallbackError)
      console.warn(`[model_tabs] Failed to persist lightweight state retry for '${STORAGE_KEY}': ${message}`)
    }

    try {
      const fallbackPayload: ModelTabsStorageState = { tabs: [], activeId: activeId.value }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(fallbackPayload))
    } catch (fallbackError) {
      const message = fallbackError instanceof Error ? fallbackError.message : String(fallbackError)
      console.warn(`[model_tabs] Failed to persist minimal fallback state for '${STORAGE_KEY}': ${message}`)
    }
  }

  function ensureTabOrThrow(id: string): BaseTab {
    const tab = tabs.value.find((entry) => entry.id === id)
    if (!tab) {
      throw new ModelTabsStoreError('tab_not_found', `Tab not found: '${id}'.`, { details: { id } })
    }
    return tab
  }

  function mapApiError(operation: string, error: unknown, details?: Record<string, unknown>): ModelTabsStoreError {
    const message = error instanceof Error ? error.message : String(error)
    return new ModelTabsStoreError(
      'api_failure',
      `${operation}: ${message}`,
      { cause: error, details: details ?? null },
    )
  }

  function serializationFailure(
    tabId: string,
    phase: PersistSerializationPhase,
    message: string,
    options?: { cause?: unknown; details?: Record<string, unknown> },
  ): ModelTabsStoreError {
    return new ModelTabsStoreError(
      'serialization_failure',
      `Failed to serialize params for tab '${tabId}' (${phase}): ${message}`,
      {
        cause: options?.cause,
        details: {
          tabId,
          phase,
          ...(options?.details ?? {}),
        },
      },
    )
  }

  function normalizeSerializableForPersist(
    tabId: string,
    phase: PersistSerializationPhase,
    value: unknown,
    path: string,
    seen: WeakSet<object>,
  ): unknown {
    if (value === null) return null
    const kind = typeof value
    if (kind === 'string' || kind === 'number' || kind === 'boolean' || kind === 'bigint') return value
    if (kind === 'undefined') return undefined
    if (kind === 'function' || kind === 'symbol') {
      throw serializationFailure(tabId, phase, `Unsupported value type '${kind}' at '${path}'.`, { details: { path, kind } })
    }
    if (kind !== 'object') return value

    const raw = toRaw(value as object)
    if (Array.isArray(raw)) {
      if (seen.has(raw)) {
        throw serializationFailure(tabId, phase, `Circular reference found at '${path}'.`, { details: { path, kind: 'array' } })
      }
      seen.add(raw)
      const normalizedArray = raw.map((entry, index) =>
        normalizeSerializableForPersist(tabId, phase, entry, `${path}[${index}]`, seen),
      )
      seen.delete(raw)
      return normalizedArray
    }

    const prototype = Object.getPrototypeOf(raw)
    if (prototype !== Object.prototype && prototype !== null) {
      const ctorName = (raw as { constructor?: { name?: string } }).constructor?.name ?? 'unknown'
      throw serializationFailure(tabId, phase, `Unsupported object type '${ctorName}' at '${path}'.`, {
        details: { path, ctorName },
      })
    }

    if (seen.has(raw as object)) {
      throw serializationFailure(tabId, phase, `Circular reference found at '${path}'.`, { details: { path, kind: 'object' } })
    }
    seen.add(raw as object)
    const normalizedRecord: Record<string, unknown> = {}
    for (const [key, entry] of Object.entries(raw as Record<string, unknown>)) {
      normalizedRecord[key] = normalizeSerializableForPersist(tabId, phase, entry, `${path}.${key}`, seen)
    }
    seen.delete(raw as object)
    return normalizedRecord
  }

  function cloneParamsForPersist(
    tabId: string,
    phase: PersistSerializationPhase,
    value: Record<string, unknown>,
  ): Record<string, unknown> {
    if (typeof structuredClone !== 'function') {
      throw serializationFailure(tabId, phase, 'structuredClone is unavailable.')
    }

    let normalizedValue: unknown
    try {
      normalizedValue = normalizeSerializableForPersist(tabId, phase, value, '$', new WeakSet<object>())
    } catch (error) {
      if (error instanceof ModelTabsStoreError) throw error
      const message = error instanceof Error ? error.message : String(error)
      throw serializationFailure(tabId, phase, message, { cause: error })
    }
    if (!isPlainRecord(normalizedValue)) {
      throw serializationFailure(tabId, phase, "Root params payload must be a plain object.", { details: { path: '$' } })
    }

    try {
      return structuredClone(normalizedValue)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      throw serializationFailure(tabId, phase, message, { cause: error })
    }
  }

  function restorePendingParamsSnapshot(tabId: string, tab: BaseTab, pending: PendingParamsPersist): void {
    if (pending.snapshotParams) {
      const rolledBackParams = cloneParamsForPersist(tabId, 'rollback', pending.snapshotParams)
      tab.params = asParamsRecord(normalizeParamsForType(tab.type, rolledBackParams, defaultParamsForType(tab.type)))
    }
    restoreImageSparsePersistHints(tabId, pending.snapshotSparseMissingKeys)
    tab.meta.updatedAt = pending.snapshotUpdatedAt
    pending.version = pending.persistedVersion
    pending.snapshotParams = null
    pending.snapshotSparseMissingKeys = null
    pending.explicitImagePersistKeys.clear()
  }

  function getPendingParamsPersist(tabId: string, tab: BaseTab): PendingParamsPersist {
    const existing = pendingParamsPersists.get(tabId)
    if (existing) return existing
    const created: PendingParamsPersist = {
      timer: null,
      inFlight: false,
      version: 0,
      persistedVersion: 0,
      deferreds: [],
      snapshotParams: null,
      snapshotUpdatedAt: tab.meta.updatedAt,
      snapshotSparseMissingKeys: null,
      explicitImagePersistKeys: new Set<string>(),
    }
    pendingParamsPersists.set(tabId, created)
    return created
  }

  function clearPendingParamsPersist(tabId: string, reason: unknown): void {
    const pending = pendingParamsPersists.get(tabId)
    if (!pending) return
    if (pending.timer !== null) {
      clearTimeout(pending.timer)
      pending.timer = null
    }
    const rejectList = pending.deferreds
    pending.deferreds = []
    rejectList.forEach((entry) => entry.reject(reason))
    pendingParamsPersists.delete(tabId)
  }

  function scheduleParamsPersist(tabId: string): void {
    const pending = pendingParamsPersists.get(tabId)
    if (!pending) return
    if (pending.timer !== null) {
      clearTimeout(pending.timer)
    }
    pending.timer = setTimeout(() => {
      pending.timer = null
      void flushParamsPersist(tabId)
    }, PARAMS_PERSIST_DEBOUNCE_MS)
  }

  async function flushParamsPersist(tabId: string): Promise<void> {
    const pending = pendingParamsPersists.get(tabId)
    if (!pending) return
    if (pending.inFlight) {
      scheduleParamsPersist(tabId)
      return
    }
    if (pending.version <= pending.persistedVersion) {
      if (pending.deferreds.length === 0 && pending.timer === null) {
        pendingParamsPersists.delete(tabId)
      }
      return
    }

    let tab: BaseTab
    try {
      tab = ensureTabOrThrow(tabId)
    } catch (error) {
      clearPendingParamsPersist(tabId, error)
      return
    }
    let paramsToPersist: Record<string, unknown>
    let normalizedParamsToPersist: Record<string, unknown>
    const explicitImagePersistKeysToPersist = new Set(pending.explicitImagePersistKeys)
    const updatedAtSnapshot = tab.meta.updatedAt
    try {
      paramsToPersist = cloneParamsForPersist(tabId, 'persist', tab.params as Record<string, unknown>)
      const migratedParams = asParamsRecord(normalizeParamsForType(tab.type, paramsToPersist, defaultParamsForType(tab.type)))
      normalizedParamsToPersist = cloneParamsForPersist(tabId, 'persist', migratedParams)
      tab.params = cloneParamsForPersist(tabId, 'persist', migratedParams)
      paramsToPersist = pruneSparseImagePersistDefaults(
        tabId,
        tab.type,
        normalizedParamsToPersist,
        explicitImagePersistKeysToPersist,
      )
    } catch (error) {
      const mapped = error instanceof ModelTabsStoreError
        ? error
        : mapApiError(`Failed to serialize params for tab '${tabId}' before persistence`, error, { id: tabId })
      try {
        restorePendingParamsSnapshot(tabId, tab, pending)
        save()
      } catch (rollbackError) {
        clearPendingParamsPersist(tabId, rollbackError)
        return
      }
      clearPendingParamsPersist(tabId, mapped)
      return
    }
    const versionToPersist = pending.version

    pending.inFlight = true
    try {
      await updateTabApi(tabId, { params: paramsToPersist })
      pending.persistedVersion = versionToPersist
      const persistedSparseMissingKeys = applyPersistedImageSparsePersistHints(
        tabId,
        tab.type,
        explicitImagePersistKeysToPersist,
      )
      for (const key of explicitImagePersistKeysToPersist) {
        pending.explicitImagePersistKeys.delete(key)
      }

      const resolveList = pending.deferreds.filter((entry) => entry.version <= versionToPersist)
      pending.deferreds = pending.deferreds.filter((entry) => entry.version > versionToPersist)
      resolveList.forEach((entry) => entry.resolve())

      if (pending.version > pending.persistedVersion) {
        pending.snapshotParams = normalizedParamsToPersist
        pending.snapshotUpdatedAt = updatedAtSnapshot
        pending.snapshotSparseMissingKeys = persistedSparseMissingKeys
      } else {
        pending.snapshotParams = null
        pending.snapshotSparseMissingKeys = null
        pending.snapshotUpdatedAt = tab.meta.updatedAt
      }
      save()
    } catch (error) {
      try {
        restorePendingParamsSnapshot(tabId, tab, pending)
      } catch (rollbackError) {
        clearPendingParamsPersist(tabId, rollbackError)
        return
      }

      const mapped = error instanceof ModelTabsStoreError
        ? error
        : mapApiError(`Failed to update params for tab '${tabId}'`, error, { id: tabId })
      const rejectList = pending.deferreds
      pending.deferreds = []
      rejectList.forEach((entry) => entry.reject(mapped))
      save()
    } finally {
      pending.inFlight = false
      if (pending.version > pending.persistedVersion) {
        scheduleParamsPersist(tabId)
      } else if (pending.deferreds.length === 0 && pending.timer === null) {
        pendingParamsPersists.delete(tabId)
      }
    }
  }

  function preferredSamplingDefaultsForType(type: BaseTabType): { sampler: string; scheduler: string } | null {
    if (isWanTabFamily(type) || type === 'ltx2') return null
    const capsStore = useEngineCapabilitiesStore()
    return capsStore.resolveSamplingDefaults(resolveImageRequestEngineId(type, false))
  }

  function defaultParamsForType<T extends BaseTabType>(type: T): TabParamsByType[T] {
    const preferredSampling = preferredSamplingDefaultsForType(type)
    return defaultParams(type, preferredSampling ?? undefined)
  }

  async function ensureRequiredTabs(requiredTypes: BaseTabType[]): Promise<void> {
    const existing = new Set<BaseTabType>(tabs.value.map(t => t.type))
    let nextOrder = tabs.value.length ? (Math.max(...tabs.value.map(t => t.order)) + 1) : 0
    for (const type of requiredTypes) {
      if (existing.has(type)) continue
      const title = getEngineConfig(type as EngineType).label
      const params = asParamsRecord(defaultParamsForType(type))
      let createdId = ''
      try {
        const created = await createTabApi({ type, title, params })
        createdId = requirePersistedTabId(created?.id, `create required tab '${type}'`)
      } catch (error) {
        throw mapApiError(`Failed to ensure required tab '${type}'`, error, { type, title })
      }
      const createdAt = nowIso()
      tabs.value.push({
        id: createdId,
        type,
        title,
        order: nextOrder++,
        enabled: true,
        params,
        meta: { createdAt, updatedAt: createdAt },
      })
      imageSparsePersistMissingKeysByTabId.delete(createdId)
      existing.add(type)
    }
  }

  async function load(): Promise<void> {
    if (loadPromise) return loadPromise

    loadPromise = (async () => {
      if (pendingParamsPersists.size > 0) {
        for (const tabId of pendingParamsPersists.keys()) {
          clearPendingParamsPersist(
            tabId,
            new ModelTabsStoreError('invalid_response', 'Tabs were reloaded while param updates were pending.'),
          )
        }
      }
      const requiredTypes = await resolveRequiredTypesFromCapabilities()
      const preferredActiveId = activeId.value || (() => {
        try {
          const raw = localStorage.getItem(STORAGE_KEY)
          if (!raw) return ''
          const parsed = JSON.parse(raw) as { activeId?: unknown }
          return typeof parsed.activeId === 'string' ? parsed.activeId : ''
        } catch {
          return ''
        }
      })()

      const res = await fetchTabs()
      if (!res || !Array.isArray(res.tabs)) {
        const msg = "Invalid '/api/ui/tabs' response: missing 'tabs' array."
        console.error(`[model_tabs] ${msg}`, res)
        throw new ModelTabsStoreError('invalid_response', msg, { details: { response: res as unknown as Record<string, unknown> } })
      }

      const rawTabs = res.tabs as unknown[]
      imageSparsePersistMissingKeysByTabId.clear()
      tabs.value = rawTabs.map((tab) => normalizeTab(tab as BaseTab, defaultParamsForType))
      for (let index = 0; index < rawTabs.length; index += 1) {
        const tab = tabs.value[index]
        const rawTab = asRecordObject(rawTabs[index])
        if (!tab) continue
        syncImageSparsePersistHints(tab.id, tab.type, rawTab.params)
        if (tab.type === 'wan22_14b') {
          if (!shouldPersistWan14bStageSamplingBackfill(rawTab.params)) continue
          const params = tab.params as Wan14bTabParams
          void updateParams<Record<string, unknown>>(tab.id, {
            high: params.high,
            low: params.low,
          }).catch((error) => {
            console.warn('[model_tabs] Failed to persist WAN stage sampler/scheduler migration backfill; continuing load.', {
              tabId: tab.id,
              error,
            })
          })
          continue
        }
        if (tab.type === 'ltx2') continue
        const normalizedParams = tab.params as unknown as ImageBaseParams
        const imageBackfillPatch = buildImageTopLevelBackfillPatch(rawTab.params, normalizedParams)
        if (!imageBackfillPatch) continue
        void updateParams<Record<string, unknown>>(tab.id, imageBackfillPatch as unknown as Record<string, unknown>).catch((error) => {
          console.warn('[model_tabs] Failed to queue image-tab top-level params backfill; continuing load.', {
            tabId: tab.id,
            error,
          })
        })
      }
      tabs.value.sort((a, b) => a.order - b.order)
      activeId.value = (preferredActiveId && tabs.value.some(t => t.id === preferredActiveId)) ? preferredActiveId : (tabs.value[0]?.id ?? '')
      await ensureRequiredTabs(requiredTypes)
      tabs.value.sort((a, b) => a.order - b.order)
      if (activeId.value && !tabs.value.some(t => t.id === activeId.value)) activeId.value = tabs.value[0]?.id ?? ''
      save()
    })()

    try {
      await loadPromise
    } finally {
      loadPromise = null
    }
  }

  async function create(type: BaseTabType, title?: string): Promise<string> {
    const resolvedTitle = title?.trim() || getEngineConfig(type as EngineType).label
    const params = asParamsRecord(defaultParamsForType(type))
    let createdId = ''
    try {
      const created = await createTabApi({ type, title: resolvedTitle, params })
      createdId = requirePersistedTabId(created?.id, `create tab '${resolvedTitle}'`)
    } catch (error) {
      throw mapApiError(`Failed to create tab '${resolvedTitle}'`, error, { type, title: resolvedTitle })
    }
    const createdAt = nowIso()
    const nextOrder = tabs.value.length ? Math.max(...tabs.value.map(t => t.order)) + 1 : 0
    tabs.value.push({
      id: createdId,
      type,
      title: resolvedTitle,
      order: nextOrder,
      enabled: true,
      params,
      meta: { createdAt, updatedAt: createdAt },
    })
    imageSparsePersistMissingKeysByTabId.delete(createdId)
    save()
    return createdId
  }

  async function duplicate(id: string): Promise<string> {
    const src = ensureTabOrThrow(id)
    const copy: BaseTab = JSON.parse(JSON.stringify(src))
    copy.title = src.title + ' (copy)'
    let createdId = ''
    try {
      const created = await createTabApi({ type: copy.type as BaseTabType, title: copy.title, params: copy.params })
      createdId = requirePersistedTabId(created?.id, `duplicate tab '${id}'`)
    } catch (error) {
      throw mapApiError(`Failed to duplicate tab '${id}'`, error, { id, sourceType: src.type })
    }
    copy.id = createdId
    copy.order = (Math.max(...tabs.value.map(t => t.order)) || 0) + 1
    copy.meta.createdAt = nowIso()
    copy.meta.updatedAt = copy.meta.createdAt
    tabs.value.push(copy)
    imageSparsePersistMissingKeysByTabId.delete(copy.id)
    save()
    return copy.id
  }

  async function remove(id: string): Promise<void> {
    ensureTabOrThrow(id)
    try {
      await deleteTabApi(id)
    } catch (error) {
      throw mapApiError(`Failed to remove tab '${id}'`, error, { id })
    }
    clearPendingParamsPersist(
      id,
      new ModelTabsStoreError('tab_not_found', `Tab not found: '${id}'.`, { details: { id } }),
    )
    imageSparsePersistMissingKeysByTabId.delete(id)
    tabs.value = tabs.value.filter(t => t.id !== id)
    if (activeId.value === id) activeId.value = tabs.value[0]?.id ?? ''
    normalizeOrder()
    save()
  }

  async function rename(id: string, title: string): Promise<void> {
    const t = ensureTabOrThrow(id)
    try {
      await updateTabApi(id, { title })
    } catch (error) {
      throw mapApiError(`Failed to rename tab '${id}'`, error, { id, title })
    }
    t.title = title
    t.meta.updatedAt = nowIso()
    save()
  }

  async function setEnabled(id: string, value: boolean): Promise<void> {
    const t = ensureTabOrThrow(id)
    try {
      await updateTabApi(id, { enabled: value })
    } catch (error) {
      throw mapApiError(`Failed to update enabled flag for tab '${id}'`, error, { id, enabled: value })
    }
    t.enabled = value
    t.meta.updatedAt = nowIso()
    save()
  }

  async function reorder(ids: string[]): Promise<void> {
    const expectedIds = tabs.value.map(t => t.id)
    if (ids.length !== expectedIds.length) {
      throw new ModelTabsStoreError(
        'invalid_reorder',
        'Invalid reorder payload: id list length does not match tabs length.',
        { details: { expected: expectedIds.length, received: ids.length } },
      )
    }
    if (new Set(ids).size !== ids.length) {
      throw new ModelTabsStoreError(
        'invalid_reorder',
        'Invalid reorder payload: duplicate tab ids are not allowed.',
        { details: { ids } },
      )
    }
    const expectedSet = new Set(expectedIds)
    for (const id of ids) {
      if (!expectedSet.has(id)) {
        throw new ModelTabsStoreError(
          'invalid_reorder',
          `Invalid reorder payload: unknown tab id '${id}'.`,
          { details: { id, expectedIds } },
        )
      }
    }
    try {
      await reorderTabsApi(ids)
    } catch (error) {
      throw mapApiError('Failed to reorder tabs', error, { ids })
    }
    const map = new Map<string, number>()
    ids.forEach((id, idx) => map.set(id, idx))
    tabs.value.forEach(t => { t.order = map.get(t.id) ?? t.order })
    tabs.value.sort((a, b) => a.order - b.order)
    save()
  }

  function setActive(id: string): void { activeId.value = id; save() }

  async function updateParams<T extends Record<string, unknown>>(id: string, patch: Partial<T>): Promise<void> {
    const t = ensureTabOrThrow(id)
    const current = (t.params && typeof t.params === 'object' && !Array.isArray(t.params)) ? (t.params as T) : ({} as T)
    if (current !== t.params) t.params = current

    if (!patch || !isPlainRecord(patch)) {
      throw new ModelTabsStoreError(
        'serialization_failure',
        `Failed to serialize params for tab '${id}' (patch): patch must be a plain object record.`,
        { details: { tabId: id, phase: 'patch' } },
      )
    }
    const patchSnapshot = cloneParamsForPersist(id, 'patch', patch as Record<string, unknown>)

    const pending = getPendingParamsPersist(id, t)
    if (pending.snapshotParams === null) {
      pending.snapshotParams = cloneParamsForPersist(id, 'snapshot', current as unknown as Record<string, unknown>)
      pending.snapshotUpdatedAt = t.meta.updatedAt
      pending.snapshotSparseMissingKeys = snapshotImageSparsePersistHints(id)
    }
    markExplicitImagePersistKeys(pending, t.type, patchSnapshot)

    Object.assign(current, patchSnapshot)
    t.meta.updatedAt = nowIso()

    pending.version += 1
    const targetVersion = pending.version
    const persistPromise = new Promise<void>((resolve, reject) => {
      pending.deferreds.push({ version: targetVersion, resolve, reject })
    })
    scheduleParamsPersist(id)
    save()
    return persistPromise
  }

  function normalizeOrder(): void {
    tabs.value.sort((a, b) => a.order - b.order)
    tabs.value.forEach((t, idx) => { t.order = idx })
  }

  const orderedTabs = computed(() => [...tabs.value].sort((a, b) => a.order - b.order))
  const activeTab = computed(() => tabs.value.find(t => t.id === activeId.value) || null)

  return {
    tabs,
    orderedTabs,
    activeId,
    activeTab,
    load,
    save,
    create,
    duplicate,
    remove,
    rename,
    reorder,
    setEnabled,
    setActive,
    updateParams,
  }
})
