/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Zod-validated payload schemas + builders for WAN 2.2 5B video endpoints (txt2vid/img2vid).
Defines the strict 5B single-stage API payload schemas and helpers that normalize UI inputs into backend-ready payloads.
The 5B contract keeps prompt/sampler/steps/cfg/seed on the top-level request owner and uses `wan_single` only for the
single-stage selector payload (`model_sha`, `loras`, `flow_shift`).

Symbols (top-level; keep in sync; no ghosts):
- `Wan22_5bTxt2VidPayloadSchema` (const): Zod schema for WAN 2.2 5B `/api/txt2vid` payloads.
- `Wan22_5bImg2VidPayloadSchema` (const): Zod schema for WAN 2.2 5B `/api/img2vid` payloads.
- `Wan22_5bStageLoraInput` (interface): UI-friendly single-stage LoRA entry (`sha` + optional `weight`).
- `Wan22_5bStageInput` (interface): UI-friendly 5B single-stage selector input mapped to `wan_single`.
- `Wan22_5bVideoCommonInput` (interface): Shared 5B input fields for txt2vid/img2vid.
- `Wan22_5bImg2VidInput` (interface): 5B img2vid-specific input fields.
- `buildWan22_5bTxt2VidPayload` (function): Builds a validated 5B txt2vid payload.
- `buildWan22_5bImg2VidPayload` (function): Builds a validated 5B img2vid payload.
*/

import { z } from 'zod'

import {
  normalizeWanImg2VidMode,
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
const Img2VidModeEnum = z.literal('solo')
const WAN_CANONICAL_SCHEDULER = 'simple' as const
const WAN_DIM_STEP = 16
const WAN_FRAMES_MIN = 9
const WAN_FRAMES_MAX = 401
const WAN_INTERPOLATION_MODEL = 'rife47.pth'

const ShaSchema = z
  .string()
  .transform((value) => value.trim().toLowerCase())
  .refine((value) => /^[0-9a-f]{64}$/.test(value), { message: 'Expected sha256 (64 lowercase hex chars)' })
const RepoIdSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.length > 0, { message: 'Repo id must not be empty' })
const LoraSchema = z.object({
  sha: ShaSchema,
  weight: z.number().finite().optional(),
})
const WanSingleStageSchema = z.object({
  model_sha: ShaSchema,
  loras: z.array(LoraSchema).optional(),
  flow_shift: z.number().finite().optional(),
})
const VideoInterpolationSchema = z.object({
  enabled: z.boolean(),
  times: z.number().int().min(2),
  model: z.literal(WAN_INTERPOLATION_MODEL),
})
const CommonVideoExportShape = {
  video_format: z.string().trim().min(1).optional(),
  video_pix_fmt: z.string().trim().min(1).optional(),
  video_crf: z.number().int().min(0).optional(),
  video_loop_count: z.number().int().min(0).optional(),
  video_pingpong: z.boolean().optional(),
  video_return_frames: z.boolean().optional(),
  video_interpolation: VideoInterpolationSchema.optional(),
}

export const Wan22_5bTxt2VidPayloadSchema = z.object({
  device: DeviceEnum,
  settings_revision: z.number().int().min(0),
  txt2vid_prompt: PromptSchema,
  txt2vid_neg_prompt: z.string().optional().default(''),
  txt2vid_width: z.number().int().min(WAN_DIM_STEP),
  txt2vid_height: z.number().int().min(WAN_DIM_STEP),
  txt2vid_fps: z.number().int().min(1),
  txt2vid_num_frames: z.number().int().min(WAN_FRAMES_MIN).max(WAN_FRAMES_MAX),
  txt2vid_steps: z.number().int().min(1),
  txt2vid_cfg_scale: z.number().finite(),
  txt2vid_seed: z.number().int(),
  txt2vid_sampler: z.string().trim().min(1).optional(),
  txt2vid_scheduler: z.literal(WAN_CANONICAL_SCHEDULER),
  wan_single: WanSingleStageSchema,
  gguf_attention_mode: WanAttentionModeEnum,
  wan_format: WanFormatEnum.optional(),
  wan_metadata_repo: RepoIdSchema,
  wan_vae_sha: ShaSchema,
  wan_tenc_sha: ShaSchema,
  ...CommonVideoExportShape,
})

export const Wan22_5bImg2VidPayloadSchema = z.object({
  device: DeviceEnum,
  settings_revision: z.number().int().min(0),
  img2vid_prompt: PromptSchema,
  img2vid_neg_prompt: z.string().optional().default(''),
  img2vid_width: z.number().int().min(WAN_DIM_STEP),
  img2vid_height: z.number().int().min(WAN_DIM_STEP),
  img2vid_fps: z.number().int().min(1),
  img2vid_num_frames: z.number().int().min(WAN_FRAMES_MIN).max(WAN_FRAMES_MAX),
  img2vid_steps: z.number().int().min(1),
  img2vid_cfg_scale: z.number().finite(),
  img2vid_seed: z.number().int(),
  img2vid_sampler: z.string().trim().min(1).optional(),
  img2vid_scheduler: z.literal(WAN_CANONICAL_SCHEDULER),
  img2vid_init_image: z.string().min(1),
  img2vid_mode: Img2VidModeEnum,
  img2vid_image_scale: z.number().positive().optional(),
  img2vid_crop_offset_x: z.number().min(0).max(1),
  img2vid_crop_offset_y: z.number().min(0).max(1),
  wan_single: WanSingleStageSchema,
  gguf_attention_mode: WanAttentionModeEnum,
  wan_format: WanFormatEnum.optional(),
  wan_metadata_repo: RepoIdSchema,
  wan_vae_sha: ShaSchema,
  wan_tenc_sha: ShaSchema,
  ...CommonVideoExportShape,
})

export type Wan22_5bTxt2VidPayload = z.infer<typeof Wan22_5bTxt2VidPayloadSchema>
export type Wan22_5bImg2VidPayload = z.infer<typeof Wan22_5bImg2VidPayloadSchema>

export interface Wan22_5bStageLoraInput {
  sha: string
  weight?: number
}

export interface Wan22_5bStageInput {
  modelSha: string
  loras?: Wan22_5bStageLoraInput[]
  flowShift?: number
}

export interface Wan22_5bVideoOutputInput {
  format: string
  pixFmt: string
  crf: number
  loopCount: number
  pingpong: boolean
  returnFrames?: boolean
}

export interface Wan22_5bInterpolationInput {
  targetFps: number
}

export interface Wan22_5bAssetsInput {
  metadataRepo: string
  textEncoderSha: string
  vaeSha: string
}

export interface Wan22_5bVideoCommonInput {
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
  stage: Wan22_5bStageInput
  attentionMode: 'global' | 'sliding'
  format: 'auto' | 'diffusers' | 'gguf'
  assets: Wan22_5bAssetsInput
  output: Wan22_5bVideoOutputInput
  interpolation: Wan22_5bInterpolationInput
}

export interface Wan22_5bImg2VidInput extends Wan22_5bVideoCommonInput {
  initImageData: string
  img2vidMode: WanImg2VidMode
  imageScale?: number
  cropOffsetX?: number
  cropOffsetY?: number
}

function normalizeDevice(device: string): Wan22_5bTxt2VidPayload['device'] {
  const normalized = String(device || '').trim().toLowerCase()
  if (DEVICE_VALUES.includes(normalized as (typeof DEVICE_VALUES)[number])) {
    return normalized as Wan22_5bTxt2VidPayload['device']
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
  const truncated = Math.trunc(value)
  return Math.ceil(truncated / WAN_DIM_STEP) * WAN_DIM_STEP
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

function requireCanonicalWanScheduler(value: string, fieldName: string): typeof WAN_CANONICAL_SCHEDULER {
  const normalized = String(value || '').trim().toLowerCase()
  if (!normalized) {
    throw new Error(`${fieldName} must not be empty.`)
  }
  if (normalized !== WAN_CANONICAL_SCHEDULER) {
    throw new Error(`${fieldName} must be '${WAN_CANONICAL_SCHEDULER}', got '${value}'`)
  }
  return WAN_CANONICAL_SCHEDULER
}

function stageToPayload(stage: Wan22_5bStageInput): Record<string, unknown> {
  const modelSha = String(stage.modelSha || '').trim().toLowerCase()
  if (!modelSha) {
    throw new Error('WAN 2.2 5B stage requires model_sha (sha256).')
  }
  if (!/^[0-9a-f]{64}$/.test(modelSha)) {
    throw new Error(`WAN 2.2 5B stage model_sha must be sha256 (64 lowercase hex), got '${stage.modelSha}'`)
  }
  const payload: Record<string, unknown> = { model_sha: modelSha }
  const rawLoras = Array.isArray(stage.loras) ? stage.loras : []
  payload.loras = rawLoras.map((lora, index) => {
    const loraSha = String(lora?.sha || '').trim().toLowerCase()
    if (!loraSha) {
      throw new Error(`WAN 2.2 5B stage loras[${index}].sha is required`)
    }
    if (!/^[0-9a-f]{64}$/.test(loraSha)) {
      throw new Error(`WAN 2.2 5B stage loras[${index}].sha must be sha256 (64 lowercase hex), got '${lora?.sha}'`)
    }
    const normalized: { sha: string; weight?: number } = { sha: loraSha }
    if (lora?.weight !== undefined) {
      if (typeof lora.weight !== 'number' || !Number.isFinite(lora.weight)) {
        throw new Error(`WAN 2.2 5B stage loras[${index}].weight must be a finite number`)
      }
      normalized.weight = lora.weight
    }
    return normalized
  })
  if (typeof stage.flowShift === 'number' && Number.isFinite(stage.flowShift)) {
    payload.flow_shift = stage.flowShift
  }
  return payload
}

function resolveTopLevelPrompts(input: Wan22_5bVideoCommonInput): { prompt: string; negativePrompt: string } {
  const prompt = String(input.prompt || '').trim()
  if (!prompt) throw new Error('WAN 2.2 5B prompt must not be empty.')
  return {
    prompt,
    negativePrompt: String(input.negativePrompt || '').trim(),
  }
}

function addWanAssets(payload: Record<string, unknown>, assets: Wan22_5bAssetsInput): void {
  payload.wan_metadata_repo = String(assets.metadataRepo || '').trim()
  payload.wan_vae_sha = String(assets.vaeSha || '').trim().toLowerCase()
  payload.wan_tenc_sha = String(assets.textEncoderSha || '').trim().toLowerCase()
}

function addWanOutput(payload: Record<string, unknown>, output: Wan22_5bVideoOutputInput): void {
  const format = String(output.format || '').trim()
  const pixFmt = String(output.pixFmt || '').trim()
  if (format) payload.video_format = format
  if (pixFmt) payload.video_pix_fmt = pixFmt
  if (Number.isFinite(output.crf)) payload.video_crf = Math.max(0, Math.trunc(output.crf))
  if (Number.isFinite(output.loopCount)) payload.video_loop_count = Math.max(0, Math.trunc(output.loopCount))
  if (typeof output.pingpong === 'boolean') payload.video_pingpong = output.pingpong
  if (typeof output.returnFrames === 'boolean') payload.video_return_frames = output.returnFrames
}

function addWanInterpolation(payload: Record<string, unknown>, interpolation: Wan22_5bInterpolationInput, baseFps: number): void {
  const targetFps = Math.trunc(Number(interpolation.targetFps))
  if (!Number.isFinite(targetFps) || targetFps <= 0 || targetFps <= baseFps) return
  const ratio = targetFps / baseFps
  const times = Math.trunc(Math.round(ratio))
  if (!Number.isFinite(times) || times < 2) return
  payload.video_interpolation = {
    enabled: true,
    times,
    model: WAN_INTERPOLATION_MODEL,
  }
}

function normalizeGuideOffset(rawValue: unknown, fieldName: string, fallback: number): number {
  if (rawValue === undefined || rawValue === null || rawValue === '') return fallback
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) {
    throw new Error(`${fieldName} must be a finite number in [0,1].`)
  }
  if (numeric < 0 || numeric > 1) {
    throw new Error(`${fieldName} must be within [0,1].`)
  }
  return numeric
}

export function buildWan22_5bTxt2VidPayload(input: Wan22_5bVideoCommonInput): Wan22_5bTxt2VidPayload {
  const { prompt, negativePrompt } = resolveTopLevelPrompts(input)
  const payload: Record<string, unknown> = {
    device: normalizeDevice(input.device),
    settings_revision: requireSettingsRevision(input.settingsRevision),
    txt2vid_prompt: prompt,
    txt2vid_neg_prompt: negativePrompt,
    txt2vid_width: snapWanDim(input.width),
    txt2vid_height: snapWanDim(input.height),
    txt2vid_fps: input.fps,
    txt2vid_num_frames: normalizeWanFrameCount(input.frames),
    txt2vid_steps: Math.max(1, Math.trunc(input.steps)),
    txt2vid_cfg_scale: input.cfgScale,
    txt2vid_seed: Math.trunc(input.seed),
    txt2vid_scheduler: requireCanonicalWanScheduler(input.scheduler, 'WAN 2.2 5B txt2vid scheduler'),
    wan_single: stageToPayload(input.stage),
    gguf_attention_mode: input.attentionMode,
  }
  const sampler = String(input.sampler || '').trim()
  if (sampler) payload.txt2vid_sampler = sampler
  if (input.format !== 'auto') payload.wan_format = input.format
  addWanAssets(payload, input.assets)
  addWanOutput(payload, input.output)
  addWanInterpolation(payload, input.interpolation, input.fps)
  return Wan22_5bTxt2VidPayloadSchema.parse(payload)
}

export function buildWan22_5bImg2VidPayload(input: Wan22_5bImg2VidInput): Wan22_5bImg2VidPayload {
  const { prompt, negativePrompt } = resolveTopLevelPrompts(input)
  const img2vidMode = normalizeWanImg2VidMode(input.img2vidMode)
  if (img2vidMode !== 'solo') {
    throw new Error(`WAN 2.2 5B img2vid currently supports only solo mode, got '${img2vidMode}'.`)
  }
  const payload: Record<string, unknown> = {
    device: normalizeDevice(input.device),
    settings_revision: requireSettingsRevision(input.settingsRevision),
    img2vid_prompt: prompt,
    img2vid_neg_prompt: negativePrompt,
    img2vid_width: snapWanDim(input.width),
    img2vid_height: snapWanDim(input.height),
    img2vid_fps: input.fps,
    img2vid_num_frames: normalizeWanFrameCount(input.frames),
    img2vid_steps: Math.max(1, Math.trunc(input.steps)),
    img2vid_cfg_scale: input.cfgScale,
    img2vid_seed: Math.trunc(input.seed),
    img2vid_scheduler: requireCanonicalWanScheduler(input.scheduler, 'WAN 2.2 5B img2vid scheduler'),
    img2vid_init_image: input.initImageData,
    img2vid_mode: img2vidMode,
    img2vid_crop_offset_x: normalizeGuideOffset(input.cropOffsetX, 'cropOffsetX', 0.5),
    img2vid_crop_offset_y: normalizeGuideOffset(input.cropOffsetY, 'cropOffsetY', 0.5),
    wan_single: stageToPayload(input.stage),
    gguf_attention_mode: input.attentionMode,
  }
  const sampler = String(input.sampler || '').trim()
  if (sampler) payload.img2vid_sampler = sampler
  const normalizedImageScale = normalizeWanImg2VidImageScale(input.imageScale)
  if (normalizedImageScale !== undefined) payload.img2vid_image_scale = normalizedImageScale
  if (input.format !== 'auto') payload.wan_format = input.format
  addWanAssets(payload, input.assets)
  addWanOutput(payload, input.output)
  addWanInterpolation(payload, input.interpolation, input.fps)
  return Wan22_5bImg2VidPayloadSchema.parse(payload)
}
