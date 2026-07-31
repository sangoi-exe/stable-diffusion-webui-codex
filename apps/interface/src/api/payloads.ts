/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Zod request schemas + payload builders for image generation (txt2img/img2img).
Defines the canonical `Txt2ImgRequestSchema`, UI form-state types, and helpers to build request payloads (including hires/refiner) and to
apply engine-agnostic request normalization/validation (including required `settings_revision`). FLUX.2 guidance mode is checkpoint-resolved
by callers (`cfg` for base-4B, `distilled_cfg` for distilled 4B) instead of being hard-coded by engine id, and hires normalization feeds the
nested hires owners used by txt2img and img2img payload builders. The canonical txt2img schema now requires the explicit
image selectors carried in `extras` (`model_sha`, `checkpoint_core_only`, `model_format`, and VAE selectors only for VAE-owning engines) so local validation matches the
backend image contract. Qwen Image Edit-2511 is rejected here because its only public image mode is the dedicated img2img builder owned by
`useGeneration.ts`. Z-Image L2P txt2img uses its exact no-VAE backend preflight contract: fixed 1024px dimensions,
Euler/simple sampling, no top-level clip-skip, no generic batch/hires/refiner/swap extras, and only denoiser + Qwen3-4B TEnc selectors in `extras`.

Symbols (top-level; keep in sync; no ghosts):
- `DISTILLED_CFG_ENGINES` (const): Engine ids treated as distilled-guidance engines (use `distilled_cfg`; CFG/negative prompt omitted).
- `DEVICE_VALUES` (const): Allowed device tokens for requests.
- `DeviceEnum` (const): Zod enum built from `DEVICE_VALUES`.
- `TextEncoderOverrideSchema` (const): Zod schema for server-owned text encoder override selectors.
- `SwapModelOptionsSchema` (const): Zod schema for selector-authoritative stage-local model swaps.
- `SwapStageOptionsSchema` (const): Zod schema for the top-level first-pass `swap_model` stage.
- `RefinerOptionsSchema` (const): Zod schema for refiner options.
- `UpscalerTileSchema` (const): Zod schema for upscaler tiling config (tile/overlap + OOM fallback + min tile).
- `HiresOptionsSchema` (const): Zod schema for hires options (including nested refiner).
- `PromptSchema` (const): Zod schema for prompt validation/normalization.
- `Txt2ImgRequestSchema` (const): Zod schema for txt2img/img2img request payloads.
- `Txt2ImgRequest` (type): Inferred request type from `Txt2ImgRequestSchema`.
- `UpscalerTileFormState` (interface): UI form state for tile config used by upscaler-driven stages.
- `HiresFormState` (interface): UI form state for hires options.
- `SwapStageFormState` (interface): UI form state for the global first-pass `swap_model` stage.
- `RefinerFormState` (interface): UI form state for refiner options.
 - `Txt2ImgFormState` (interface): UI form state for txt2img/img2img payload building.
 - `NestedStageSelectorPayloads` (interface): Optional resolved selector payloads for nested stage-local `swap_model` / `refiner` seams.
- `BuildImagePayloadOptions` (interface): Shared payload-builder options for hires normalization and nested selector injection.
- `normalizeDevice` (function): Normalizes and validates a device token.
- `buildNormalizedHiresOptions` (function): Normalizes shared hires form state into the canonical nested hires payload shape used by txt2img/img2img builders.
- `buildTxt2ImgPayload` (function): Builds and validates a `Txt2ImgRequest` from UI form state (supports hires tile prefs: fallback + min_tile).
- `formatZodError` (function): Converts Zod errors (or unknown errors) into a readable message.
*/

import { z, ZodError } from 'zod'
import { resolveTextOverride } from '../utils/image_params'

type Txt2ImgGuidanceMode = 'cfg' | 'distilled_cfg'

// Engines that always use distilled guidance (single-branch conditioning) and therefore use distilled_cfg.
const DISTILLED_CFG_ENGINES = ['flux1', 'flux1_kontext', 'flux1_chroma'] as const
const DEVICE_VALUES = ['cuda', 'cpu', 'mps', 'xpu', 'directml'] as const
const DeviceEnum = z.enum(DEVICE_VALUES)
const FLUX2_BASE_VARIANT_MARKERS = ['flux.2-klein-base-4b', 'flux2-klein-base-4b', 'base-4b', 'base_4b', '/base/'] as const
const QWEN_IMAGE_ENGINE_ID = 'qwen_image'
const ZIMAGE_L2P_ENGINE_ID = 'zimage_l2p'
const ZIMAGE_L2P_DIMENSION = 1024
const ZIMAGE_L2P_ASSET_EXTRA_KEYS = new Set([
  'checkpoint_core_only',
  'model_format',
  'model_sha',
  'tenc_sha',
])
const ZIMAGE_L2P_UNSUPPORTED_EXTRA_KEYS = new Set([
  'batch_count',
  'batch_size',
  'er_sde',
  'guidance',
  'hires',
  'ip_adapter',
  'lora_sha',
  'qwen_image_variant',
  'refiner',
  'swap_model',
  'text_encoder_override',
  'tenc1_sha',
  'tenc2_sha',
  'vae_sha',
  'vae_source',
  'zimage_variant',
])

const TextEncoderOverrideSchema = z
  .object({
    family: z.string().min(1),
    label: z.string().min(1),
    components: z.array(z.string().min(1)).optional(),
  })
  .strict()

const GenericSwapModelOptionsBaseSchema = z
  .object({
    model: z.string().min(1).optional(),
    model_sha: z.string().min(1).optional(),
    checkpoint_core_only: z.boolean().optional(),
    model_format: z.enum(['checkpoint', 'diffusers', 'gguf']).optional(),
    vae_source: z.enum(['built_in', 'external']).optional(),
    vae_sha: z.string().min(1).optional(),
    tenc_sha: z.union([z.string().min(1), z.array(z.string().min(1))]).optional(),
    tenc1_sha: z.string().min(1).optional(),
    tenc2_sha: z.string().min(1).optional(),
    zimage_variant: z.enum(['turbo', 'base']).optional(),
    text_encoder_override: TextEncoderOverrideSchema.optional(),
  })
  .strict()

const RefinerModelOptionsBaseSchema = z
  .object({
    model: z.string().min(1).optional(),
    model_sha: z.string().min(1).optional(),
    checkpoint_core_only: z.boolean().optional(),
    model_format: z.enum(['checkpoint', 'diffusers', 'gguf']).optional(),
    vae_source: z.enum(['built_in', 'external']).optional(),
    vae_sha: z.string().min(1).optional(),
    tenc_sha: z.union([z.string().min(1), z.array(z.string().min(1))]).optional(),
    tenc1_sha: z.string().min(1).optional(),
    tenc2_sha: z.string().min(1).optional(),
    text_encoder_override: TextEncoderOverrideSchema.optional(),
  })
  .strict()

const SwapModelOptionsSchema = GenericSwapModelOptionsBaseSchema
  .superRefine((value, ctx) => {
    const model = String(value.model ?? '').trim()
    const modelSha = String(value.model_sha ?? '').trim()
    if (!model && !modelSha) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "swap_model requires 'model' or 'model_sha'" })
    }
  })

const SwapStageOptionsSchema = GenericSwapModelOptionsBaseSchema
  .extend({
    enable: z.literal(true),
    switch_at_step: z.number().int().min(1),
    cfg: z.number(),
    seed: z.number().int(),
  })
  .strict()
  .superRefine((value, ctx) => {
    const model = String(value.model ?? '').trim()
    const modelSha = String(value.model_sha ?? '').trim()
    if (!model && !modelSha) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "swap_model requires 'model' or 'model_sha'" })
    }
  })

const RefinerOptionsSchema = RefinerModelOptionsBaseSchema
  .extend({
    enable: z.literal(true),
    switch_at_step: z.number().int().min(1),
    cfg: z.number(),
    seed: z.number().int(),
  })
  .strict()
  .superRefine((value, ctx) => {
    const model = String(value.model ?? '').trim()
    const modelSha = String(value.model_sha ?? '').trim()
    if (!model && !modelSha) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "refiner requires 'model' or 'model_sha'" })
    }
  })

const UpscalerTileSchema = z
  .object({
    tile: z.number().int().min(1),
    overlap: z.number().int().min(0),
    fallback_on_oom: z.boolean(),
    min_tile: z.number().int().min(1),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (value.overlap >= value.tile) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "'hires.tile.overlap' must be < tile" })
    }
    if (value.fallback_on_oom && value.min_tile > value.tile) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "'hires.tile.min_tile' must be <= tile when fallback_on_oom is enabled" })
    }
  })

const HiresOptionsSchema = z
  .object({
    enable: z.literal(true),
    denoise: z.number().min(0).max(1),
    scale: z.number().min(1),
    resize_x: z.number().int().min(0),
    resize_y: z.number().int().min(0),
    steps: z.number().int().min(0),
    upscaler: z
      .string()
      .min(1)
      .refine((value) => value.startsWith('latent:') || value.startsWith('spandrel:'), {
        message: "hires.upscaler must be an id like 'latent:*' or 'spandrel:*'",
      }),
    tile: UpscalerTileSchema.optional(),
    swap_model: SwapModelOptionsSchema.optional(),
    sampler: z.string().min(1).optional(),
    scheduler: z.string().min(1).optional(),
    prompt: z.string().optional(),
    negative_prompt: z.string().optional(),
    cfg: z.number().optional(),
    distilled_cfg: z.number().optional(),
    refiner: RefinerOptionsSchema.optional(),
  })
  .strict()

const PromptSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.length > 0, { message: 'Prompt must not be empty' })

export const Txt2ImgRequestSchema = z
  .object({
    device: DeviceEnum,
    settings_revision: z.number().int().min(0),
    prompt: PromptSchema,
    negative_prompt: z.string().optional().default(''),
    width: z.number().int().min(8).max(8192),
    height: z.number().int().min(8).max(8192),
    steps: z.number().int().min(1),
    cfg: z.number().optional(),  // Classic CFG models (SD, SDXL, FLUX.2 base-4B)
    distilled_cfg: z.number().optional(),  // Distilled-guidance models (Flux.1/Chroma + FLUX.2 distilled 4B)
    sampler: z.string().min(1),
    scheduler: z.string().min(1),
    seed: z.number().int(),
    clip_skip: z.number().int().min(0).max(12).optional(),
    styles: z.array(z.string().min(1)).optional(),
    metadata: z.record(z.any()).optional(),
    engine: z.string().min(1).optional(),
    model: z.string().min(1).optional(),
    extras: z
      .object({
        hires: HiresOptionsSchema.optional(),
        swap_model: SwapStageOptionsSchema.optional(),
        refiner: RefinerOptionsSchema.optional(),
        text_encoder_override: z
          .lazy(() => TextEncoderOverrideSchema)
          .optional(),
        // Batch params
        batch_size: z.number().int().min(1).optional(),
        batch_count: z.number().int().min(1).optional(),
        // SHA-based model selection
        tenc_sha: z.union([z.string(), z.array(z.string())]).optional(),
        tenc1_sha: z.string().optional(),
        tenc2_sha: z.string().optional(),
        vae_sha: z.string().optional(),
        vae_source: z.enum(['built_in', 'external']).optional(),
        lora_sha: z.union([z.string(), z.array(z.string())]).optional(),
        model_sha: z.string().min(1),
        checkpoint_core_only: z.boolean(),
        model_format: z.enum(['checkpoint', 'diffusers', 'gguf']),
        // Z-Image variant selection
        zimage_variant: z.enum(['turbo', 'base']).optional(),
      })
      .passthrough(),  // Allow additional dynamic keys for engine-specific extras
  })
  .strict()

export type Txt2ImgRequest = z.infer<typeof Txt2ImgRequestSchema>

export interface UpscalerTileFormState {
  tile: number
  overlap: number
}

export interface HiresFormState {
  enabled: boolean
  denoise: number
  scale: number
  resizeX: number
  resizeY: number
  steps: number
  upscaler: string
  tile: UpscalerTileFormState
  swapModel?: SwapModelFormState
  sampler?: string
  scheduler?: string
  prompt?: string
  negativePrompt?: string
  cfg?: number
  distilledCfg?: number
  refiner?: RefinerFormState
}

export interface SwapModelFormState {
  model?: string
}

export interface SwapStageFormState {
  enabled: boolean
  swapAtStep: number
  cfg: number
  seed: number
  model?: string
}

export interface RefinerFormState {
  enabled: boolean
  swapAtStep: number
  cfg: number
  seed: number
  model?: string
}

export interface Txt2ImgFormState {
  prompt: string
  negativePrompt: string
  width: number
  height: number
  steps: number
  guidanceScale: number
  sampler: string
  scheduler: string
  seed: number
  clipSkip: number
  batchSize: number
  batchCount: number
  styles?: string[]
  device: Txt2ImgRequest['device']
  settingsRevision: number
  engine?: string
  model?: string
  guidanceMode?: Txt2ImgGuidanceMode
  swapModel: SwapStageFormState
  hires?: HiresFormState
  refiner?: RefinerFormState
  extras?: Record<string, unknown>
}

export interface NestedStageSelectorPayloads {
  swapModel?: z.infer<typeof SwapModelOptionsSchema>
  refiner?: z.infer<typeof RefinerOptionsSchema>
  hiresSwapModel?: z.infer<typeof SwapModelOptionsSchema>
  hiresRefiner?: z.infer<typeof RefinerOptionsSchema>
}

export interface BuildImagePayloadOptions {
  hiresFallbackOnOom?: boolean
  hiresMinTile?: number
  nestedSelectorPayloads?: NestedStageSelectorPayloads
}

function buildMinimalSwapModelPayload(
  state: SwapModelFormState | null | undefined,
): z.infer<typeof SwapModelOptionsSchema> | undefined {
  const model = String(state?.model ?? '').trim()
  if (!model) return undefined
  return { model }
}

function buildMinimalSwapStagePayload(
  state: SwapStageFormState | null | undefined,
): z.infer<typeof SwapStageOptionsSchema> | undefined {
  if (!state?.enabled) return undefined
  const payload: z.infer<typeof SwapStageOptionsSchema> = {
    enable: true,
    switch_at_step: state.swapAtStep,
    cfg: state.cfg,
    seed: state.seed,
  }
  const model = String(state.model ?? '').trim()
  if (model) payload.model = model
  return payload
}

function buildMinimalRefinerPayload(
  state: RefinerFormState | null | undefined,
): z.infer<typeof RefinerOptionsSchema> | undefined {
  if (!state?.enabled) return undefined
  const payload: z.infer<typeof RefinerOptionsSchema> = {
    enable: true,
    switch_at_step: state.swapAtStep,
    cfg: state.cfg,
    seed: state.seed,
  }
  const model = String(state.model ?? '').trim()
  if (model) payload.model = model
  return payload
}

function resolveStageSwapModelPayload(
  state: SwapModelFormState | null | undefined,
  resolved: z.infer<typeof SwapModelOptionsSchema> | undefined,
): z.infer<typeof SwapModelOptionsSchema> | undefined {
  const payload = resolved ?? buildMinimalSwapModelPayload(state)
  if (!payload) return undefined
  return SwapModelOptionsSchema.parse(payload)
}

function resolveSwapStagePayload(
  state: SwapStageFormState | null | undefined,
  resolved: z.infer<typeof SwapModelOptionsSchema> | undefined,
): z.infer<typeof SwapStageOptionsSchema> | undefined {
  if (!state?.enabled) return undefined
  const basePayload = buildMinimalSwapStagePayload(state)
  if (!basePayload) return undefined
  return SwapStageOptionsSchema.parse({
    ...basePayload,
    ...(resolved ?? {}),
  })
}

function resolveStageRefinerPayload(
  state: RefinerFormState | null | undefined,
  resolved: z.infer<typeof RefinerOptionsSchema> | undefined,
): z.infer<typeof RefinerOptionsSchema> | undefined {
  if (!state?.enabled) return undefined
  const payload = resolved ?? buildMinimalRefinerPayload(state)
  if (!payload) return undefined
  return RefinerOptionsSchema.parse(payload)
}

function normalizeDevice(device: string): Txt2ImgRequest['device'] {
  const normalized = device.trim().toLowerCase()
  if (DEVICE_VALUES.includes(normalized as (typeof DEVICE_VALUES)[number])) {
    return normalized as Txt2ImgRequest['device']
  }
  throw new Error(`Unsupported device '${device}'`)
}

function normalizeSettingsRevision(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.max(0, Math.trunc(value))
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (/^-?\d+$/.test(trimmed)) return Math.max(0, Math.trunc(Number(trimmed)))
  }
  return 0
}

function inferFlux2GuidanceMode(modelRef: unknown): Txt2ImgGuidanceMode | null {
  const normalized = String(modelRef || '').trim().toLowerCase().replace(/\\+/g, '/')
  if (!normalized) return null
  if (FLUX2_BASE_VARIANT_MARKERS.some((marker) => normalized.includes(marker))) return 'cfg'
  return 'distilled_cfg'
}

function resolveGuidanceMode(state: Txt2ImgFormState): Txt2ImgGuidanceMode {
  if (state.guidanceMode === 'cfg' || state.guidanceMode === 'distilled_cfg') {
    return state.guidanceMode
  }
  if (DISTILLED_CFG_ENGINES.includes(state.engine as typeof DISTILLED_CFG_ENGINES[number])) {
    return 'distilled_cfg'
  }
  if (state.engine === 'flux2') {
    return inferFlux2GuidanceMode(state.model) ?? 'distilled_cfg'
  }
  return 'cfg'
}

function isZImageL2PTxt2ImgState(state: Txt2ImgFormState): boolean {
  return String(state.engine || '').trim().toLowerCase() === ZIMAGE_L2P_ENGINE_ID
}

function hasEnabledHires(state: Txt2ImgFormState): boolean {
  return Boolean(state.hires?.enabled)
}

function assertZImageL2PTxt2ImgState(state: Txt2ImgFormState, opts: BuildImagePayloadOptions): void {
  const width = Math.trunc(Number(state.width))
  const height = Math.trunc(Number(state.height))
  if (width !== ZIMAGE_L2P_DIMENSION || height !== ZIMAGE_L2P_DIMENSION) {
    throw new Error(`Z-Image L2P txt2img requires ${ZIMAGE_L2P_DIMENSION}x${ZIMAGE_L2P_DIMENSION} output.`)
  }
  if (String(state.sampler || '').trim().toLowerCase() !== 'euler') {
    throw new Error("Z-Image L2P txt2img requires sampler 'euler'.")
  }
  if (String(state.scheduler || '').trim().toLowerCase() !== 'simple') {
    throw new Error("Z-Image L2P txt2img requires scheduler 'simple'.")
  }
  const clipSkip = Number(state.clipSkip)
  if (Number.isFinite(clipSkip) && Math.trunc(clipSkip) !== 0) {
    throw new Error('Z-Image L2P txt2img does not support CLIP Skip. Reset CLIP Skip to 0.')
  }
  if (Math.trunc(Number(state.batchSize)) !== 1 || Math.trunc(Number(state.batchCount)) !== 1) {
    throw new Error('Z-Image L2P txt2img requires batch size = 1 and batch count = 1.')
  }
  if (hasEnabledHires(state)) {
    throw new Error('Z-Image L2P txt2img does not support Hires Fix. Disable Hires before generating.')
  }
  if (state.swapModel?.enabled) {
    throw new Error('Z-Image L2P txt2img does not support first-pass model swap. Disable model swap before generating.')
  }
  if (state.refiner?.enabled) {
    throw new Error('Z-Image L2P txt2img does not support refiner. Disable refiner before generating.')
  }
  if (
    opts.nestedSelectorPayloads?.swapModel
    || opts.nestedSelectorPayloads?.refiner
    || opts.nestedSelectorPayloads?.hiresSwapModel
    || opts.nestedSelectorPayloads?.hiresRefiner
  ) {
    throw new Error('Z-Image L2P txt2img does not support nested swap/refiner selector payloads.')
  }
}

function buildZImageL2PAssetExtras(rawExtras: Record<string, unknown> | undefined): Record<string, unknown> {
  if (!rawExtras || typeof rawExtras !== 'object') {
    throw new Error('Z-Image L2P txt2img requires resolved asset selectors in extras.')
  }
  const extras: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(rawExtras)) {
    if (value === undefined) continue
    if (ZIMAGE_L2P_UNSUPPORTED_EXTRA_KEYS.has(key) || !ZIMAGE_L2P_ASSET_EXTRA_KEYS.has(key)) {
      throw new Error(`Z-Image L2P txt2img does not support extras.${key}.`)
    }
    extras[key] = value
  }

  const missing: string[] = []
  for (const key of ZIMAGE_L2P_ASSET_EXTRA_KEYS) {
    if (extras[key] === undefined || extras[key] === null || extras[key] === '') {
      missing.push(key)
    }
  }
  if (missing.length > 0) {
    throw new Error(`Z-Image L2P txt2img requires extras.${missing.join(', extras.')}.`)
  }
  if (extras.checkpoint_core_only !== true) {
    throw new Error('Z-Image L2P txt2img requires extras.checkpoint_core_only=true.')
  }
  if (extras.model_format !== 'checkpoint' && extras.model_format !== 'gguf') {
    throw new Error("Z-Image L2P txt2img requires extras.model_format='checkpoint' or 'gguf'.")
  }
  if (typeof extras.tenc_sha !== 'string' || !extras.tenc_sha.trim()) {
    throw new Error('Z-Image L2P txt2img requires exactly one extras.tenc_sha.')
  }
  return extras
}

function buildZImageL2PTxt2ImgPayload(
  state: Txt2ImgFormState,
  opts: BuildImagePayloadOptions,
): Txt2ImgRequest {
  assertZImageL2PTxt2ImgState(state, opts)
  const extras = buildZImageL2PAssetExtras(state.extras)
  const payload: Record<string, unknown> = {
    device: normalizeDevice(state.device),
    settings_revision: normalizeSettingsRevision(state.settingsRevision),
    prompt: state.prompt.trim(),
    negative_prompt: state.negativePrompt ?? '',
    width: ZIMAGE_L2P_DIMENSION,
    height: ZIMAGE_L2P_DIMENSION,
    steps: state.steps,
    cfg: state.guidanceScale,
    sampler: 'euler',
    scheduler: 'simple',
    seed: state.seed,
    engine: ZIMAGE_L2P_ENGINE_ID,
    model: state.model,
    extras,
  }
  if (!String(state.model || '').trim()) {
    delete payload.model
  }
  return Txt2ImgRequestSchema.parse(payload)
}

export function buildNormalizedHiresOptions(
  state: Pick<Txt2ImgFormState, 'prompt' | 'negativePrompt' | 'hires'> & Partial<Pick<Txt2ImgFormState, 'steps'>>,
  guidanceMode: Txt2ImgGuidanceMode,
  opts: BuildImagePayloadOptions = {},
): z.infer<typeof HiresOptionsSchema> | null {
  if (!state.hires?.enabled) return null

  const hiresFallbackOnOom = opts.hiresFallbackOnOom ?? true
  const hiresMinTilePrefRaw = opts.hiresMinTile
  const hiresMinTilePref = (typeof hiresMinTilePrefRaw === 'number' && Number.isFinite(hiresMinTilePrefRaw))
    ? Math.max(1, Math.trunc(hiresMinTilePrefRaw))
    : 128
  const hiresPrompt = resolveTextOverride(state.prompt, state.hires.prompt)
  const hiresNegativePrompt = resolveTextOverride(state.negativePrompt, state.hires.negativePrompt)
  const tile = state.hires.tile ?? { tile: 256, overlap: 16 }
  const tileSize = Math.max(1, Math.trunc(Number(tile.tile)))
  const overlap = Math.max(0, Math.trunc(Number(tile.overlap)))
  const minTile = Math.max(1, Math.min(tileSize, hiresMinTilePref))
  const sampler = String(state.hires.sampler ?? '').trim()
  const scheduler = String(state.hires.scheduler ?? '').trim()
  const guidanceValue = guidanceMode === 'distilled_cfg'
    ? state.hires.distilledCfg
    : state.hires.cfg
  const baseSteps = Math.max(0, Math.trunc(Number(state.steps ?? 0)))
  const secondPassSteps = Math.max(0, Math.trunc(Number(state.hires.steps)))
  const resolvedSecondPassSteps = secondPassSteps > 0 ? secondPassSteps : baseSteps
  if (state.hires.refiner?.enabled) {
    const swapAtStep = Math.max(1, Math.trunc(Number(state.hires.refiner.swapAtStep)))
    if (resolvedSecondPassSteps < 2 || swapAtStep >= resolvedSecondPassSteps) {
      throw new Error(
        `Hires refiner requires 'Swap At Step' in [1, ${Math.max(1, resolvedSecondPassSteps - 1)}].`,
      )
    }
  }

  return HiresOptionsSchema.parse({
    enable: true,
    denoise: state.hires.denoise,
    scale: state.hires.scale,
    resize_x: state.hires.resizeX,
    resize_y: state.hires.resizeY,
    steps: state.hires.steps,
    upscaler: state.hires.upscaler,
    tile: {
      tile: tileSize,
      overlap,
      fallback_on_oom: Boolean(hiresFallbackOnOom),
      min_tile: minTile,
    },
    swap_model: resolveStageSwapModelPayload(
      state.hires.swapModel,
      opts.nestedSelectorPayloads?.hiresSwapModel,
    ),
    sampler: sampler || undefined,
    scheduler: scheduler || undefined,
    prompt: hiresPrompt,
    negative_prompt: hiresNegativePrompt,
    ...(Number.isFinite(Number(guidanceValue))
      ? (guidanceMode === 'distilled_cfg'
          ? { distilled_cfg: Number(guidanceValue) }
          : { cfg: Number(guidanceValue) })
      : {}),
    refiner: resolveStageRefinerPayload(
      state.hires.refiner,
      opts.nestedSelectorPayloads?.hiresRefiner,
    ),
  })
}

export function buildTxt2ImgPayload(
  state: Txt2ImgFormState,
  opts: BuildImagePayloadOptions = {},
): Txt2ImgRequest {
  if (String(state.engine || '').trim().toLowerCase() === QWEN_IMAGE_ENGINE_ID) {
    throw new Error('Qwen Image Edit-2511 supports img2img only.')
  }
  if (isZImageL2PTxt2ImgState(state)) {
    return buildZImageL2PTxt2ImgPayload(state, opts)
  }

  const guidanceMode = resolveGuidanceMode(state)
  const isDistilledCfgModel = guidanceMode === 'distilled_cfg'
  const totalSteps = Math.max(0, Math.trunc(Number(state.steps)))
  if (state.swapModel.enabled) {
    const swapAtStep = Math.max(1, Math.trunc(Number(state.swapModel.swapAtStep)))
    if (totalSteps < 2 || swapAtStep >= totalSteps) {
      throw new Error(`First-pass swap_model requires 'Swap At Step' in [1, ${Math.max(1, totalSteps - 1)}].`)
    }
  }
  if (state.refiner?.enabled) {
    const swapAtStep = Math.max(1, Math.trunc(Number(state.refiner.swapAtStep)))
    if (totalSteps < 2 || swapAtStep >= totalSteps) {
      throw new Error(`Refiner requires 'Swap At Step' in [1, ${Math.max(1, totalSteps - 1)}].`)
    }
  }

  const payload: Record<string, unknown> = {
    device: normalizeDevice(state.device),
    settings_revision: normalizeSettingsRevision(state.settingsRevision),
    prompt: state.prompt.trim(),
    width: state.width,
    height: state.height,
    steps: state.steps,
    sampler: state.sampler,
    scheduler: state.scheduler,
    seed: state.seed,
  }

  if (Number.isFinite(state.clipSkip) && state.clipSkip >= 0) {
    payload.clip_skip = Math.trunc(state.clipSkip)
  }
  
  // Distilled-guidance models: use distilled_cfg, no CFG/negative prompt (single-branch conditioning)
  // CFG models: use cfg with negative prompt
  if (isDistilledCfgModel) {
    payload.distilled_cfg = state.guidanceScale
  } else {
    payload.cfg = state.guidanceScale
    if (state.negativePrompt?.trim()) {
      payload.negative_prompt = state.negativePrompt
    }
  }

  const styles = state.styles?.filter((entry) => entry.trim().length > 0) ?? []
  if (styles.length > 0) {
    payload.styles = styles
  }

  if (state.engine) {
    payload.engine = state.engine
  }
  if (state.model) {
    payload.model = state.model
  }

  const extras: Record<string, unknown> = {
    batch_size: state.batchSize,
    batch_count: state.batchCount,
  }
  const hires = buildNormalizedHiresOptions(state, guidanceMode, opts)
  if (hires) {
    extras.hires = hires
  }
  const swapModelPayload = resolveSwapStagePayload(state.swapModel, opts.nestedSelectorPayloads?.swapModel)
  if (swapModelPayload) {
    extras.swap_model = swapModelPayload
  }
  const refinerPayload = resolveStageRefinerPayload(state.refiner, opts.nestedSelectorPayloads?.refiner)
  if (refinerPayload) {
    extras.refiner = refinerPayload
  }
  // Merge engine-specific extras from state (e.g., tenc_sha for Z Image)
  if (state.extras) {
    for (const [key, value] of Object.entries(state.extras)) {
      if (value !== undefined) {
        extras[key] = value
      }
    }
  }
  payload.extras = extras

  return Txt2ImgRequestSchema.parse(payload)
}

export function formatZodError(err: unknown): string {
  if (err instanceof ZodError) {
    return err.errors.map((issue) => issue.message).join('; ')
  }
  return err instanceof Error ? err.message : String(err)
}
