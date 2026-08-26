/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Zod-validated payload schemas + builders for WAN video endpoints (txt2vid/img2vid).
Defines the strict API payload schemas and provides helpers that normalize UI inputs (device, stage params, assets, output settings),
handling unset sentinels and producing backend-ready payloads for `/api/*` requests with canonical WAN video keys (including `device` and `settings_revision`).
Img2vid payload builders emit no-stretch guide controls (optional `img2vid_image_scale` + crop offsets) with fail-loud validation, and WAN scheduler fields are enforced as exact canonical `simple`.

Symbols (top-level; keep in sync; no ghosts):
- `WanTxt2VidPayloadSchema` (const): Zod schema for WAN `/api/txt2vid` payload.
- `WanImg2VidPayloadSchema` (const): Zod schema for WAN `/api/img2vid` payload.
- `WanTxt2VidPayload` (type): Zod-inferred payload type for WAN `/api/txt2vid`.
- `WanImg2VidPayload` (type): Zod-inferred payload type for WAN `/api/img2vid`.
- `WanStageLoraInput` (interface): UI-friendly stage LoRA entry (`sha` + optional `weight`) mapped to stage `loras[]`.
- `WanHighStageInput` (interface): UI-friendly WAN high-stage override input (model/Lora/flow only) mapped to `wan_high.*`.
- `WanStageInput` (interface): UI-friendly low-stage params that map to WAN stage overrides in payload.
- `WanVideoOutputInput` (interface): Output options (format, pix_fmt, CRF, loop, pingpong, return-frames) mapped into payload; save flags are backend-owned defaults.
- `WanInterpolationInput` (interface): Interpolation target FPS input (`0`=off, values above base FPS enable backend interpolation).
- `WanAssetsInput` (interface): WAN asset selection (metadata/text encoder/VAE) used to fill payload fields.
- `WanVideoCommonInput` (interface): Shared input fields for txt2vid/img2vid (dims, top-level WAN core owners, stage params, assets).
- `WanImg2VidInput` (interface): Img2vid-specific input extending common WAN fields with temporal-mode controls (`solo|sliding|svi2|svi2_pro`) and no-stretch guide controls (`imageScale` + crop offsets).
- `normalizeDevice` (function): Validates/normalizes device input into the backend enum.
- `snapWanDim` (function): Snaps WAN width/height to a multiple of 16 (rounded up; Diffusers parity).
- `normalizeWanFrameCount` (function): Clamps/snap-normalizes WAN frame counts into the `4n+1` domain.
- `normalizeAttentionMode` (function): Normalizes attention mode input to `global|sliding`.
- `requireCanonicalWanScheduler` (function): Enforces exact canonical WAN scheduler value `simple` for WAN scheduler payload fields.
- `stageToPayload` (function): Converts a `WanStageInput` or `WanHighStageInput` into the backend stage override object, optionally omitting high-stage core fields now owned by the top-level WAN request.
- `isUnsetSentinel` (function): Detects UI sentinel values (e.g., “Automatic”/“Built-in”) that must not be sent as real asset paths.
- `addWanAssets` (function): Injects selected WAN assets into the payload (skips unset/empty values).
- `addWanOutput` (function): Injects output-related fields into the payload.
- `addWanInterpolation` (function): Injects interpolation config into the payload.
- `buildWanTxt2VidPayload` (function): Builds a validated txt2vid payload from UI common input.
- `buildWanImg2VidPayload` (function): Builds a validated img2vid payload from UI input plus init image data.
*/

import { z } from 'zod'
import {
  isWanWindowedImg2VidMode,
  normalizeWanImg2VidMode,
  normalizeWanWindowCommit,
  normalizeWanWindowStride,
  WAN_WINDOW_COMMIT_OVERLAP_MIN,
  WAN_WINDOW_STRIDE_ALIGNMENT,
  type WanImg2VidMode,
} from '../utils/wan_img2vid_temporal'
import { normalizeWanImg2VidImageScale } from '../utils/wan_img2vid_frame_projection'

const DEVICE_VALUES = ['cuda', 'cpu'] as const
const DeviceEnum = z.enum(DEVICE_VALUES)

const PromptSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.length > 0, { message: 'Prompt must not be empty' })

const WanFormatEnum = z.enum(['auto', 'diffusers', 'gguf'])
const WanAttentionModeEnum = z.enum(['global', 'sliding'])
const Img2VidModeEnum = z.enum(['solo', 'sliding', 'svi2', 'svi2_pro'])
const Img2VidChunkSeedModeEnum = z.enum(['fixed', 'increment', 'random'])
const WAN_CANONICAL_SCHEDULER = 'simple' as const

const WAN_DIM_STEP = 16
const WAN_FRAMES_MIN = 9
const WAN_FRAMES_MAX = 401
const WAN_INTERPOLATION_MODEL = 'rife47.pth'
const Img2VidWindowStrideSchema = z.number().int().min(1).max(WAN_FRAMES_MAX - 1)
const Img2VidWindowCommitSchema = z.number().int().min(1).max(WAN_FRAMES_MAX)

const WanFrameCountSchema = z
  .number()
  .int()
  .min(WAN_FRAMES_MIN)
  .max(WAN_FRAMES_MAX)
  .refine((value) => (value - 1) % 4 === 0, { message: `Expected 4n+1 frame count in [${WAN_FRAMES_MIN}, ${WAN_FRAMES_MAX}]` })

const Sha256Schema = z
  .string()
  .transform((value) => value.trim().toLowerCase())
  .refine((value) => /^[0-9a-f]{64}$/.test(value), { message: 'Expected sha256 (64 lowercase hex chars)' })

const RepoIdSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.includes('/') && !value.startsWith('/') && !value.endsWith('/'), { message: "Expected repo id like 'Org/Repo'" })

const VideoInterpolationSchema = z
  .object({
    enabled: z.literal(true),
    model: z.string().min(1),
    times: z.number().int().min(2),
  })
  .strict()

const WanHighStageSchema = z
  .object({
    model_sha: Sha256Schema,
    loras: z
      .array(
        z
          .object({
            sha: Sha256Schema,
            weight: z.number().finite().optional(),
          })
          .strict(),
      )
      .default([]),
    flow_shift: z.number().optional(),
  })
  .strict()

const WanStageSchema = z
  .object({
    model_sha: Sha256Schema,
    prompt: z.string().min(1),
    negative_prompt: z.string().optional(),
    sampler: z.string().min(1).optional(),
    scheduler: z.literal(WAN_CANONICAL_SCHEDULER),
    steps: z.number().int().min(1),
    cfg_scale: z.number(),
    seed: z.number().int().optional(),
    lightning: z.boolean().optional(),
    loras: z
      .array(
        z
          .object({
            sha: Sha256Schema,
            weight: z.number().finite().optional(),
          })
          .strict(),
      )
      .default([]),
    flow_shift: z.number().optional(),
  })
  .strict()

const CommonWanVideoPayloadSchema = z
  .object({
    device: DeviceEnum,
    settings_revision: z.number().int().min(0),

    video_save_output: z.literal(true),
    video_save_metadata: z.literal(true),
    video_return_frames: z.boolean(),
    video_format: z.string().min(1).optional(),
    video_pix_fmt: z.string().min(1).optional(),
    video_crf: z.number().int().min(0).max(51).optional(),
    video_loop_count: z.number().int().min(0).optional(),
    video_pingpong: z.boolean(),
    video_interpolation: VideoInterpolationSchema.optional(),

    wan_high: WanHighStageSchema.optional(),
    wan_low: WanStageSchema.optional(),
    wan_format: WanFormatEnum.optional(),
    wan_metadata_repo: RepoIdSchema,
    wan_vae_sha: Sha256Schema,
    wan_tenc_sha: Sha256Schema,
    gguf_attention_mode: WanAttentionModeEnum.optional(),
  })
  .strict()

export const WanTxt2VidPayloadSchema = CommonWanVideoPayloadSchema.extend({
  txt2vid_prompt: PromptSchema,
  txt2vid_neg_prompt: z.string().optional().default(''),
  txt2vid_width: z.number().int().min(8).max(8192),
  txt2vid_height: z.number().int().min(8).max(8192),
  txt2vid_steps: z.number().int().min(1),
  txt2vid_fps: z.number().int().min(1).max(240),
  txt2vid_num_frames: WanFrameCountSchema,
  txt2vid_sampler: z.string().min(1).optional(),
  txt2vid_scheduler: z.literal(WAN_CANONICAL_SCHEDULER),
  txt2vid_seed: z.number().int().optional(),
  txt2vid_cfg_scale: z.number().optional(),
}).strict()

export type WanTxt2VidPayload = z.infer<typeof WanTxt2VidPayloadSchema>

export const WanImg2VidPayloadSchema = CommonWanVideoPayloadSchema.extend({
  img2vid_prompt: PromptSchema,
  img2vid_neg_prompt: z.string().optional().default(''),
  img2vid_width: z.number().int().min(8).max(8192),
  img2vid_height: z.number().int().min(8).max(8192),
  img2vid_steps: z.number().int().min(1),
  img2vid_fps: z.number().int().min(1).max(240),
  img2vid_num_frames: WanFrameCountSchema,
  img2vid_sampler: z.string().min(1).optional(),
  img2vid_scheduler: z.literal(WAN_CANONICAL_SCHEDULER),
  img2vid_seed: z.number().int().optional(),
  img2vid_cfg_scale: z.number().optional(),
  img2vid_init_image: z.string().min(1),
  img2vid_mode: Img2VidModeEnum,
  img2vid_chunk_frames: WanFrameCountSchema.optional(),
  img2vid_overlap_frames: z.number().int().min(0).max(WAN_FRAMES_MAX - 1).optional(),
  img2vid_anchor_alpha: z.number().min(0).max(1).optional(),
  img2vid_reset_anchor_to_base: z.boolean().optional(),
  img2vid_chunk_seed_mode: Img2VidChunkSeedModeEnum.optional(),
  img2vid_window_frames: WanFrameCountSchema.optional(),
  img2vid_window_stride: Img2VidWindowStrideSchema.optional(),
  img2vid_window_commit_frames: Img2VidWindowCommitSchema.optional(),
  img2vid_image_scale: z.number().positive().finite().optional(),
  img2vid_crop_offset_x: z.number().min(0).max(1).optional(),
  img2vid_crop_offset_y: z.number().min(0).max(1).optional(),
})
  .strict()
  .superRefine((payload, ctx) => {
    const mode = payload.img2vid_mode
    const chunkFrames = payload.img2vid_chunk_frames
    const overlapFrames = payload.img2vid_overlap_frames
    const anchorAlpha = payload.img2vid_anchor_alpha
    const resetAnchorToBase = payload.img2vid_reset_anchor_to_base
    const chunkSeedMode = payload.img2vid_chunk_seed_mode
    const windowFrames = payload.img2vid_window_frames
    const windowStride = payload.img2vid_window_stride
    const windowCommitFrames = payload.img2vid_window_commit_frames

    if (mode === 'solo') {
      if (
        chunkFrames !== undefined
        || overlapFrames !== undefined
        || anchorAlpha !== undefined
        || resetAnchorToBase !== undefined
        || chunkSeedMode !== undefined
      ) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "img2vid_mode='solo' does not allow temporal fields",
          path: ['img2vid_mode'],
        })
      }
      if (windowFrames !== undefined || windowStride !== undefined || windowCommitFrames !== undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: "img2vid_mode='solo' does not allow sliding-window fields",
          path: ['img2vid_mode'],
        })
      }
      return
    }

    if (isWanWindowedImg2VidMode(mode)) {
      const modeLabel = String(mode)
      if (windowFrames === undefined || windowStride === undefined || windowCommitFrames === undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message:
            `img2vid_mode='${modeLabel}' requires img2vid_window_frames, img2vid_window_stride, and img2vid_window_commit_frames`,
          path: ['img2vid_mode'],
        })
        return
      }
      if (windowFrames >= payload.img2vid_num_frames) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'img2vid_window_frames must be smaller than img2vid_num_frames',
          path: ['img2vid_window_frames'],
        })
      }
      if (windowStride >= windowFrames) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'img2vid_window_stride must be smaller than img2vid_window_frames',
          path: ['img2vid_window_stride'],
        })
      }
      if (windowStride % WAN_WINDOW_STRIDE_ALIGNMENT !== 0) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `img2vid_window_stride must be aligned to temporal scale=${WAN_WINDOW_STRIDE_ALIGNMENT}`,
          path: ['img2vid_window_stride'],
        })
      }
      if (windowCommitFrames < windowStride || windowCommitFrames > windowFrames) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: 'img2vid_window_commit_frames must be within [img2vid_window_stride, img2vid_window_frames]',
          path: ['img2vid_window_commit_frames'],
        })
      }
      if ((windowCommitFrames - windowStride) < WAN_WINDOW_COMMIT_OVERLAP_MIN) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `img2vid_window_commit_frames must keep at least ${WAN_WINDOW_COMMIT_OVERLAP_MIN} committed overlap frames beyond stride`,
          path: ['img2vid_window_commit_frames'],
        })
      }
      if (chunkFrames !== undefined || overlapFrames !== undefined) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `img2vid_mode='${modeLabel}' does not allow img2vid_chunk_frames/img2vid_overlap_frames`,
          path: ['img2vid_mode'],
        })
      }
      if ((mode === 'svi2' || mode === 'svi2_pro') && resetAnchorToBase === true) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          message: `img2vid_mode='${modeLabel}' requires img2vid_reset_anchor_to_base=false`,
          path: ['img2vid_reset_anchor_to_base'],
        })
      }
      return
    }
  })

export type WanImg2VidPayload = z.infer<typeof WanImg2VidPayloadSchema>


export interface WanHighStageInput {
  loras?: WanStageLoraInput[]
  modelSha: string
  flowShift?: number
}

export interface WanStageInput extends WanHighStageInput {
  prompt: string
  negativePrompt: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
}

export interface WanStageLoraInput {
  sha: string
  weight?: number
}

export interface WanVideoOutputInput {
  format: string
  pixFmt: string
  crf: number
  loopCount: number
  pingpong: boolean
  returnFrames?: boolean
}

export interface WanInterpolationInput {
  targetFps: number
}

export interface WanAssetsInput {
  metadataRepo: string
  textEncoderSha: string
  vaeSha: string
}

export interface WanVideoCommonInput {
  device: string
  settingsRevision: number
  width: number
  height: number
  fps: number
  frames: number
  prompt: string
  negativePrompt: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  high: WanHighStageInput
  low: WanStageInput
  attentionMode: 'global' | 'sliding'
  format: 'auto' | 'diffusers' | 'gguf'
  assets: WanAssetsInput
  output: WanVideoOutputInput
  interpolation: WanInterpolationInput
}

export interface WanImg2VidInput extends WanVideoCommonInput {
  initImageData: string
  img2vidMode: 'solo' | 'sliding' | 'svi2' | 'svi2_pro'
  imageScale?: number
  cropOffsetX?: number
  cropOffsetY?: number
  anchorAlpha?: number
  resetAnchorToBase?: boolean
  chunkSeedMode?: 'fixed' | 'increment' | 'random'
  windowFrames?: number
  windowStride?: number
  windowCommitFrames?: number
}

function normalizeDevice(device: string): WanTxt2VidPayload['device'] {
  const normalized = device.trim().toLowerCase()
  if (DEVICE_VALUES.includes(normalized as (typeof DEVICE_VALUES)[number])) {
    return normalized as WanTxt2VidPayload['device']
  }
  throw new Error(`Unsupported device '${device}'`)
}

function requireSettingsRevision(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value) && Number.isInteger(value) && value >= 0) {
    return value
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (/^\d+$/.test(trimmed)) return Number(trimmed)
  }
  throw new Error(`settings_revision must be a non-negative integer, got ${String(value)}.`)
}

function snapWanDim(value: number): number {
  if (!Number.isFinite(value)) return value
  const v = Math.trunc(value)
  return Math.ceil(v / WAN_DIM_STEP) * WAN_DIM_STEP
}

function normalizeWanFrameCount(rawValue: number): number {
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

function normalizeAttentionMode(value: 'global' | 'sliding' | string): 'global' | 'sliding' {
  return String(value || '').trim().toLowerCase() === 'sliding' ? 'sliding' : 'global'
}

function normalizeImg2VidMode(value: unknown): WanImg2VidMode {
  return normalizeWanImg2VidMode(value)
}

function normalizeImg2VidImageScale(value: unknown): number | undefined {
  if (value === undefined || value === null || value === '') return undefined
  return normalizeWanImg2VidImageScale(value, 1)
}

function normalizeGuideOffset(
  value: unknown,
  options: {
    fieldName: string
    fallback?: number
  },
): number {
  const fieldName = options.fieldName
  const fallback = options.fallback ?? 0.5
  if (value === undefined || value === null || value === '') return fallback
  if (typeof value === 'boolean') {
    throw new Error(`WAN img2vid ${fieldName} must be a finite number in [0,1]`)
  }
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    throw new Error(`WAN img2vid ${fieldName} must be a finite number in [0,1]`)
  }
  if (numeric < 0 || numeric > 1) {
    throw new Error(`WAN img2vid ${fieldName} must be in [0,1]`)
  }
  return numeric
}

function requireCanonicalWanScheduler(rawValue: unknown, fieldName: string): typeof WAN_CANONICAL_SCHEDULER {
  const value = String(rawValue || '').trim()
  if (!value) {
    throw new Error(`${fieldName} must not be empty.`)
  }
  if (value !== value.toLowerCase()) {
    throw new Error(`${fieldName} must be canonical lowercase, got '${value}'`)
  }
  if (value !== WAN_CANONICAL_SCHEDULER) {
    throw new Error(`${fieldName} must be '${WAN_CANONICAL_SCHEDULER}', got '${value}'`)
  }
  return WAN_CANONICAL_SCHEDULER
}

function stageToPayload(
  stage: WanStageInput | WanHighStageInput,
  options?: { includeCoreFields?: boolean },
): Record<string, unknown> {
  const modelSha = String(stage.modelSha || '').trim().toLowerCase()
  if (!modelSha) {
    throw new Error('WAN stage requires model_sha (sha256)')
  }
  if (!/^[0-9a-f]{64}$/.test(modelSha)) {
    throw new Error(`WAN stage model_sha must be sha256 (64 lowercase hex), got '${stage.modelSha}'`)
  }
  const payload: Record<string, unknown> = { model_sha: modelSha }
  if (options?.includeCoreFields === false) {
    const rawLoras = Array.isArray(stage.loras) ? stage.loras : []
    const normalizedLoras = rawLoras.map((lora, index) => {
      const loraSha = String(lora?.sha || '').trim().toLowerCase()
      if (!loraSha) {
        throw new Error(`WAN stage loras[${index}].sha is required`)
      }
      if (!/^[0-9a-f]{64}$/.test(loraSha)) {
        throw new Error(`WAN stage loras[${index}].sha must be sha256 (64 lowercase hex), got '${lora?.sha}'`)
      }
      const normalized: { sha: string; weight?: number } = { sha: loraSha }
      if (lora?.weight !== undefined) {
        if (typeof lora.weight !== 'number' || !Number.isFinite(lora.weight)) {
          throw new Error(`WAN stage loras[${index}].weight must be a finite number`)
        }
        normalized.weight = lora.weight
      }
      return normalized
    })
    payload.loras = normalizedLoras
    if (typeof stage.flowShift === 'number') payload.flow_shift = stage.flowShift
    return payload
  }
  const fullStage = stage as WanStageInput
  payload.steps = fullStage.steps
  payload.cfg_scale = fullStage.cfgScale
  payload.seed = fullStage.seed
  const prompt = String(fullStage.prompt || '').trim()
  if (!prompt) {
    throw new Error('WAN stage prompt must not be empty.')
  }
  payload.prompt = prompt
  const negativePrompt = String(fullStage.negativePrompt || '').trim()
  payload.negative_prompt = negativePrompt
  const sampler = String(fullStage.sampler || '').trim()
  if (sampler) {
    if (sampler !== sampler.toLowerCase()) {
      throw new Error(`WAN sampler must be canonical lowercase, got '${sampler}'`)
    }
    payload.sampler = sampler
  }
  payload.scheduler = requireCanonicalWanScheduler(fullStage.scheduler, 'WAN stage scheduler')
  const rawLoras = Array.isArray(fullStage.loras) ? fullStage.loras : []
  const normalizedLoras = rawLoras.map((lora, index) => {
    const loraSha = String(lora?.sha || '').trim().toLowerCase()
    if (!loraSha) {
      throw new Error(`WAN stage loras[${index}].sha is required`)
    }
    if (!/^[0-9a-f]{64}$/.test(loraSha)) {
      throw new Error(`WAN stage loras[${index}].sha must be sha256 (64 lowercase hex), got '${lora?.sha}'`)
    }
    const normalized: { sha: string; weight?: number } = { sha: loraSha }
    if (lora?.weight !== undefined) {
      if (typeof lora.weight !== 'number' || !Number.isFinite(lora.weight)) {
        throw new Error(`WAN stage loras[${index}].weight must be a finite number`)
      }
      normalized.weight = lora.weight
    }
    return normalized
  })
  payload.loras = normalizedLoras
  if (typeof stage.flowShift === 'number') payload.flow_shift = stage.flowShift

  return payload
}

function resolveTopLevelPrompts(input: WanVideoCommonInput): { prompt: string; negativePrompt: string } {
  const prompt = String(input.prompt || '').trim()
  if (!prompt) {
    throw new Error('WAN top-level prompt must not be empty.')
  }
  const negativePrompt = String(input.negativePrompt || '').trim()
  return { prompt, negativePrompt }
}

function isUnsetSentinel(raw: string): boolean {
  const v = String(raw || '').trim().toLowerCase()
  if (!v) return true
  return v === 'automatic' || v === 'built in' || v === 'built-in' || v === 'none'
}

function addWanAssets(payload: Record<string, unknown>, assets: WanAssetsInput): void {
  const repo = String(assets.metadataRepo || '').trim()
  if (repo && !isUnsetSentinel(repo)) payload.wan_metadata_repo = repo

  const vaeSha = String(assets.vaeSha || '').trim().toLowerCase()
  if (vaeSha) payload.wan_vae_sha = vaeSha

  const tencSha = String(assets.textEncoderSha || '').trim().toLowerCase()
  if (tencSha) payload.wan_tenc_sha = tencSha
}

function addWanOutput(payload: Record<string, unknown>, out: WanVideoOutputInput): void {
  const resolveToggle = (value: unknown, field: string, fallback: boolean): boolean => {
    if (value === undefined) return fallback
    if (typeof value !== 'boolean') {
      throw new Error(`WAN output.${field} must be boolean when provided`)
    }
    return value
  }
  const format = String(out.format || '').trim()
  if (format) payload.video_format = format
  const pixFmt = String(out.pixFmt || '').trim()
  if (pixFmt) payload.video_pix_fmt = pixFmt
  if (Number.isFinite(out.crf)) payload.video_crf = out.crf
  if (Number.isFinite(out.loopCount)) payload.video_loop_count = out.loopCount
  payload.video_save_output = true
  payload.video_save_metadata = true
  payload.video_pingpong = resolveToggle(out.pingpong, 'pingpong', false)
  payload.video_return_frames = resolveToggle(out.returnFrames, 'returnFrames', false)
}

function addWanInterpolation(
  payload: Record<string, unknown>,
  interpolation: WanInterpolationInput,
  baseFpsValue: number,
): void {
  const numericTargetFps = Number(interpolation.targetFps)
  if (!Number.isFinite(numericTargetFps)) return
  const targetFps = Math.trunc(numericTargetFps)
  if (targetFps <= 0) return
  const numericBaseFps = Number(baseFpsValue)
  if (!Number.isFinite(numericBaseFps)) return
  const baseFps = Math.trunc(numericBaseFps)
  if (baseFps <= 0) return
  if (targetFps <= baseFps) return
  const times = Math.max(2, Math.ceil(targetFps / baseFps))
  payload.video_interpolation = {
    enabled: true,
    model: WAN_INTERPOLATION_MODEL,
    times,
  }
}

export function buildWanTxt2VidPayload(input: WanVideoCommonInput): WanTxt2VidPayload {
  const totalSteps = input.steps + input.low.steps
  const width = snapWanDim(input.width)
  const height = snapWanDim(input.height)
  const frames = normalizeWanFrameCount(input.frames)
  const { prompt, negativePrompt } = resolveTopLevelPrompts(input)
  const payload: Record<string, unknown> = {
    device: normalizeDevice(input.device),
    settings_revision: requireSettingsRevision(input.settingsRevision),
    txt2vid_prompt: prompt,
    txt2vid_neg_prompt: negativePrompt,
    txt2vid_width: width,
    txt2vid_height: height,
    txt2vid_fps: input.fps,
    txt2vid_num_frames: frames,
    // Use total steps to keep WAN stage schedules continuous (GGUF runtime) and to avoid inconsistent payloads when
    // high/low stage steps differ.
    txt2vid_steps: totalSteps,
    txt2vid_cfg_scale: input.cfgScale,
    txt2vid_seed: input.seed,
  }

  const sampler = String(input.sampler || '').trim()
  if (sampler) payload.txt2vid_sampler = sampler
  payload.txt2vid_scheduler = requireCanonicalWanScheduler(input.scheduler, 'WAN txt2vid scheduler')
  addWanOutput(payload, input.output)
  addWanInterpolation(payload, input.interpolation, input.fps)

  payload.wan_high = stageToPayload(input.high, { includeCoreFields: false })
  payload.wan_low = stageToPayload(input.low)
  payload.gguf_attention_mode = normalizeAttentionMode(input.attentionMode)
  if (input.format !== 'auto') payload.wan_format = input.format
  addWanAssets(payload, input.assets)

  return WanTxt2VidPayloadSchema.parse(payload)
}

export function buildWanImg2VidPayload(input: WanImg2VidInput): WanImg2VidPayload {
  const totalSteps = input.steps + input.low.steps
  const width = snapWanDim(input.width)
  const height = snapWanDim(input.height)
  const frames = normalizeWanFrameCount(input.frames)
  const { prompt, negativePrompt } = resolveTopLevelPrompts(input)
  const payload: Record<string, unknown> = {
    device: normalizeDevice(input.device),
    settings_revision: requireSettingsRevision(input.settingsRevision),
    img2vid_prompt: prompt,
    img2vid_neg_prompt: negativePrompt,
    img2vid_width: width,
    img2vid_height: height,
    img2vid_fps: input.fps,
    img2vid_num_frames: frames,
    // Use total steps to keep WAN stage schedules continuous (GGUF runtime) and to avoid inconsistent payloads when
    // high/low stage steps differ.
    img2vid_steps: totalSteps,
    img2vid_cfg_scale: input.cfgScale,
    img2vid_seed: input.seed,
    img2vid_init_image: input.initImageData,
    img2vid_mode: normalizeImg2VidMode(input.img2vidMode),
    img2vid_crop_offset_x: normalizeGuideOffset(input.cropOffsetX, {
      fieldName: 'cropOffsetX',
      fallback: 0.5,
    }),
    img2vid_crop_offset_y: normalizeGuideOffset(input.cropOffsetY, {
      fieldName: 'cropOffsetY',
      fallback: 0.5,
    }),
  }

  const normalizedImageScale = normalizeImg2VidImageScale(input.imageScale)
  if (normalizedImageScale !== undefined) {
    payload.img2vid_image_scale = normalizedImageScale
  }

  const sampler = String(input.sampler || '').trim()
  if (sampler) payload.img2vid_sampler = sampler
  payload.img2vid_scheduler = requireCanonicalWanScheduler(input.scheduler, 'WAN img2vid scheduler')
  const img2vidMode = normalizeImg2VidMode(input.img2vidMode)
  if (typeof input.anchorAlpha === 'number' && Number.isFinite(input.anchorAlpha)) {
    payload.img2vid_anchor_alpha = Math.min(1, Math.max(0, input.anchorAlpha))
  }
  if (typeof input.resetAnchorToBase === 'boolean') {
    payload.img2vid_reset_anchor_to_base = input.resetAnchorToBase
  }
  if (typeof input.chunkSeedMode === 'string') {
    const chunkSeedMode = String(input.chunkSeedMode || '').trim().toLowerCase()
    if (chunkSeedMode === 'fixed' || chunkSeedMode === 'increment' || chunkSeedMode === 'random') {
      payload.img2vid_chunk_seed_mode = chunkSeedMode
    }
  }
  if (isWanWindowedImg2VidMode(img2vidMode)) {
    const rawWindowFrames = Number(input.windowFrames)
    if (Number.isFinite(rawWindowFrames) && rawWindowFrames > 0) {
      payload.img2vid_window_frames = normalizeWanFrameCount(rawWindowFrames)
    }
    const effectiveWindowFrames = Number.isFinite(Number(payload.img2vid_window_frames))
      ? Math.trunc(Number(payload.img2vid_window_frames))
      : WAN_FRAMES_MIN
    const strideRaw = Number(input.windowStride)
    const fallbackStrideRaw = Number(input.windowFrames)
    const normalizedStride = normalizeWanWindowStride(
      strideRaw,
      effectiveWindowFrames,
      Number.isFinite(fallbackStrideRaw) ? fallbackStrideRaw : effectiveWindowFrames - WAN_WINDOW_COMMIT_OVERLAP_MIN,
    )
    payload.img2vid_window_stride = normalizedStride
    const commitRaw = Number(input.windowCommitFrames)
    const fallbackCommitRaw = normalizedStride + WAN_WINDOW_COMMIT_OVERLAP_MIN
    payload.img2vid_window_commit_frames = normalizeWanWindowCommit(
      commitRaw,
      effectiveWindowFrames,
      normalizedStride,
      fallbackCommitRaw,
    )
  }
  addWanOutput(payload, input.output)
  addWanInterpolation(payload, input.interpolation, input.fps)

  payload.wan_high = stageToPayload(input.high, { includeCoreFields: false })
  payload.wan_low = stageToPayload(input.low)
  payload.gguf_attention_mode = normalizeAttentionMode(input.attentionMode)
  if (input.format !== 'auto') payload.wan_format = input.format
  addWanAssets(payload, input.assets)

  return WanImg2VidPayloadSchema.parse(payload)
}
