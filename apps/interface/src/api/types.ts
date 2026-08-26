/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Frontend API DTOs and response/payload types.
Defines TypeScript interfaces/types for backend responses (models/options/samplers/tasks/events/inventory) and UI-driven schemas (settings schema, UI blocks/presets, tabs/workflows), including options revision/apply metadata fields used by strict generation contracts.
Inventory DTOs now include first-class IP-Adapter model/image-encoder collections from `/api/models/inventory`, SUPIR diagnostics DTOs from `/api/supir/models`, add-path contracts expose explicit nullable `size_bytes`
metadata (`number | null`) for byte-progress UX and fail-loud validation in sequential library adds. SUPIR diagnostics now include structured
stable sampler rows (`SupirSamplerInfo`) with backend-owned native sampler/scheduler metadata, and engine capabilities include explicit masked-img2img
  support, vid2vid discoverability, SUPIR-mode discoverability, exact-engine img2img inpaint-mode maps, parked exact-engine statuses, VAE usage vs external VAE requirement contracts, plus the optional nested LTX execution-profile surface used by the current checkpoint-aware LTX defaults lane.

Symbols (top-level; keep in sync; no ghosts):
- `ModelInfo` (interface): Model list entry returned by `/api/models`, including explicit `format` and `core_only` checkpoint selectors.
- `RawSamplerInfo` (interface): Raw sampler metadata entry returned by `/api/samplers` (unsupported rows may omit executable defaults).
- `SamplerInfo` (interface): Executable sampler metadata entry used by frontend selector surfaces after client-side filtering.
- `SchedulerInfo` (interface): Scheduler metadata entry returned by `/api/schedulers`.
- `ModelsResponse` (interface): `/api/models` response shape.
- `SamplersResponse` (interface): Raw `/api/samplers` response shape.
- `SupportedSamplersResponse` (interface): Executable sampler response shape returned by `client.ts::fetchSamplers()`.
- `SchedulersResponse` (interface): `/api/schedulers` response shape.
- `OptionsResponse` (interface): `/api/options` response shape.
- `OptionsUpdateResponse` (interface): `/api/options` update response shape.
- `TaskStartResponse` (interface): Start-task response shape (`task_id`) used by multiple endpoints.
- `SeedVR2DitModel` / `SeedVR2ColorCorrection` (types): Curated dedicated SeedVR2 video-upscale option domains.
- `VideoUpscaleRequest` (interface): JSON request shape for `POST /api/video-upscale`.
- `Txt2ImgStartResponse` (interface): Start-task response shape (`task_id`).
- `UpscalerKind` (type): Allowed upscaler kind values (`latent`/`spandrel`).
- `UpscalerDefinition` (interface): Upscaler entry returned by `/api/upscalers`.
- `UpscalersResponse` (interface): `/api/upscalers` response shape.
- `UpscalersHfManifestV1` (interface): Canonical schema for `upscalers/manifest.json` (HF curated metadata).
- `UpscalersHfManifestV1Weight` (interface): One HF manifest weight entry.
- `RemoteUpscalerWeight` (type): Remote HF weight entry (either raw listing or curated + metadata).
- `RemoteUpscalersResponse` (interface): `/api/upscalers/remote` response shape (manifest + raw weights fallback).
- `GeneratedImage` (interface): Base64-encoded image payload used in task results and previews.
- `TaskErrorCode` (type): Machine-readable terminal task error classification used by task snapshots/SSE.
- `TaskEvent` (type): Task SSE event union emitted by `/api/tasks/:id/events` (supports replay via `id:` / `after`, emits `gap` on truncation, and carries optional progress metadata via `message`/`data`).
- `TaskResult` (interface): Polled task result shape returned by `/api/tasks/:id`, including last progress snapshot metadata (`message`/`data`) when available.
- `MemoryResponse` (interface): `/api/memory` response shape.
- `ObliterateVramProcessInfo` (interface): One external GPU process row returned by `/api/obliterate-vram`.
- `ObliterateVramFailure` (interface): One failed external process termination row from `/api/obliterate-vram`.
- `ObliterateVramSkippedProcess` (interface): One skipped external process row from `/api/obliterate-vram`.
- `ObliterateVramExternalKillMode` (type): External process termination mode for `/api/obliterate-vram`.
- `ObliterateVramRequest` (interface): Request payload for `/api/obliterate-vram`.
- `ObliterateVramResponse` (interface): `/api/obliterate-vram` response shape.
- `VersionResponse` (interface): `/api/version` response shape.
- `LtxExecutionSurface` (interface): Optional nested LTX execution-profile/default surface returned under `/api/engines/capabilities`.
- `SupirModelsResponse` (interface): `/api/supir/models` diagnostics payload for installed variants/readiness and sampler inventory.
- `EngineCapabilities` (interface): Per-engine capability flags used to gate UI features, including masked-img2img support, vid2vid discoverability, SUPIR-mode discoverability, recommended sampler/scheduler hint lists, and optional LTX execution-profile metadata.
- `GuidanceAdvancedCapabilities` (interface): Per-engine support map for advanced CFG/APG controls.
- `FamilyCapabilities` (interface): Per-family capability flags from backend (`families`) used to gate prompt/clip controls and optional sampler/scheduler support/exclusion lists.
- `EngineDependencyCheckRow` (interface): One dependency-check row returned by backend readiness contract.
- `EngineDependencyStatus` (interface): Aggregated dependency status (`ready + checks`) for one semantic engine.
- `ParkedExactEngineStatus` (interface): Public placeholder state for an exact parked engine id.
- `EngineCapabilitiesResponse` (interface): `/api/engines/capabilities` response shape.
- `WorkflowItem` (interface): One persisted workflow snapshot row from `/api/ui/workflows`.
- `WorkflowCreateResponse` / `WorkflowUpdateResponse` / `WorkflowDeleteResponse` (interfaces): Workflow mutation receipts from `/api/ui/workflows`.
- `PromptTokenCountRequest` (interface): Request payload for `/api/models/prompt-token-count`.
- `PromptTokenCountResponse` (interface): Response payload for `/api/models/prompt-token-count`.
- `EngineAssetContract` (interface): Per-engine asset requirements contract exposed by the backend (VAE usage, external VAE requirement, text encoders).
- `EngineAssetContractVariants` (interface): Base vs core-only contract variants for one engine id.
- `EmbeddingsResponse` (interface): `/api/embeddings` response shape.
- `PathsResponse` (interface): `/api/paths` response shape.
- `PathsUpdateResponse` (interface): `/api/paths` update response shape.
- `ModelPathLibraryKind` (type): Model library kind for add-path scan/add endpoints (`checkpoint|vae|text_encoder`).
- `ModelPathSizeBytes` (type): Explicit nullable file-size contract used by add-path scan/add responses (`number | null`).
- `ModelPathScanRequest` (interface): Request payload for `/api/models/path-scan`.
- `ModelPathScanItem` (interface): Candidate file row returned by `/api/models/path-scan` with explicit size/already-in-library metadata.
- `ModelPathScanResponse` (interface): `/api/models/path-scan` response shape.
- `ModelPathAddRequest` (interface): Request payload for `/api/models/path-add` and `/api/models/path-add-all`.
- `ModelPathAddItem` (interface): Added-file payload including SHA/hash metadata.
- `ModelPathAddResponse` (interface): `/api/models/path-add` response shape.
- `ModelPathAddAllErrorItem` (interface): Per-file fallback payload returned by `/api/models/path-add-all` when add fails.
- `ModelPathAddAllResult` (interface): Per-item sequential result row returned by `/api/models/path-add-all`.
- `ModelPathAddAllResponse` (interface): `/api/models/path-add-all` response shape.
- `SettingsCategory` (interface): Settings category entry in settings schema responses.
- `SettingsSection` (interface): Settings section entry in settings schema responses.
- `SettingsFieldType` (type): Allowed field types in settings schema definitions.
- `SettingsField` (interface): Settings field entry in settings schema responses.
- `SettingsSchemaResponse` (interface): `/api/settings/schema` response shape.
- `UiFieldType` (type): Allowed field types for server-driven UI blocks.
- `UiFieldBind` (interface): Optional binds mapping UI fields to payload keys.
- `UiField` (interface): UI block field definition.
- `UiBlockWhen` (interface): Conditional activation for a UI block.
- `UiBlockLayout` (interface): Layout metadata for a UI block.
- `UiBlock` (interface): Server-driven UI block definition.
- `UiBlocksResponse` (interface): `/api/ui/blocks` response shape.
- `UiPreset` (interface): Checkpoint-only UI preset definition used by the frontend.
- `UiPresetsResponse` (interface): `/api/ui/presets` response shape.
- `UiPresetApplyResponse` (interface): `/api/ui/presets/apply` response shape returning only the resolved checkpoint owner.
- `ApiTabMeta` (interface): Per-tab metadata timestamps.
- `ApiTab` (interface): Persisted model tab definition (`sd15|sdxl|flux1|flux2|qwen_image|zimage|zimage_l2p|chroma|wan22_14b|wan22_5b|ltx2|anima`).
- `TabsResponse` (interface): `/api/ui/tabs` response shape.
- `WorkflowsResponse` (interface): `/api/ui/workflows` response shape.
- `InventoryResponse` (interface): `/api/models/inventory` response shape.
- `PngInfoAnalyzeResponse` (interface): `/api/tools/pnginfo/analyze` response shape.
*/

export interface ModelInfo {
  title: string
  name: string
  model_name: string
  hash: string | null
  filename: string
  format: 'checkpoint' | 'diffusers' | 'gguf'
  metadata: Record<string, unknown>
  core_only: boolean
  core_only_reason?: string | null
  family_hint?: string | null
}

export interface RawSamplerInfo {
  name: string
  label?: string
  supported?: boolean
  default_scheduler: string | null
  allowed_schedulers: string[]
}

export interface SamplerInfo {
  name: string
  label?: string
  supported?: boolean
  default_scheduler: string
  allowed_schedulers: string[]
}

export interface SchedulerInfo {
  name: string
  label?: string
  supported?: boolean
}

export interface ModelsResponse {
  models: ModelInfo[]
  current: string | null
}

export interface FileMetadataResponse {
  path: string
  kind: 'gguf' | 'safetensors'
  flat: Record<string, unknown>
  nested: Record<string, unknown>
  summary: Record<string, unknown>
}

export interface PngInfoAnalyzeResponse {
  width: number
  height: number
  metadata: Record<string, string>
}

export interface CheckpointMetadataResponse {
  hash: string | null
  'file.name': string
  'file.path': string
  'file.size.bytes': number
  'file.size.megabytes': number
  'file.size.gigabytes': number
  metadata: { raw: Record<string, unknown>; nested: Record<string, unknown> }
}

export interface SamplersResponse {
  samplers: RawSamplerInfo[]
}

export interface SupportedSamplersResponse {
  samplers: SamplerInfo[]
}

export interface SchedulersResponse {
  schedulers: SchedulerInfo[]
}

export interface OptionsResponse {
  values: Record<string, unknown>
  revision?: number | null
}

export interface OptionsUpdateResponse {
  updated: string[]
  revision?: number | null
  applied_now?: string[] | null
  restart_required?: string[] | null
}

export interface TaskStartResponse {
  task_id: string
}

export interface Txt2ImgStartResponse extends TaskStartResponse {}

export type SeedVR2DitModel =
  | 'seedvr2_ema_3b_fp16.safetensors'
  | 'seedvr2_ema_7b_fp16.safetensors'
  | 'seedvr2_ema_7b_sharp_fp16.safetensors'

export type SeedVR2ColorCorrection = 'lab' | 'wavelet' | 'wavelet_adaptive' | 'hsv' | 'adain' | 'none'

export interface VideoUpscaleRequest {
  video_path: string
  device: string
  dit_model: SeedVR2DitModel
  resolution: number
  max_resolution: number
  batch_size: number
  uniform_batch_size: boolean
  temporal_overlap: number
  prepend_frames: number
  color_correction: SeedVR2ColorCorrection
  input_noise_scale: number
  latent_noise_scale: number
}

export interface ImageAutomationLoopRequest {
  mode: 'count' | 'until_cancelled'
  count?: number | null
  delay_ms: number
  stop_on_error: boolean
}

export interface ImageAutomationSeedPolicyRequest {
  mode: 'fixed' | 'increment' | 'random'
  increment_step: number
}

export interface ImageAutomationPromptSourceRequest {
  kind: 'current' | 'list'
  text?: string
  insert_position: 'replace' | 'prepend' | 'append'
  wildcard_root?: string
  wildcard_mode: 'disabled' | 'expand'
}

export interface ImageAutomationFolderSourceRequest {
  kind: 'uploaded_current' | 'server_folder'
  folder_path?: string
  selection_mode?: 'all' | 'count'
  count?: number | null
  order: 'random' | 'sorted'
  sort_by?: 'name' | 'size' | 'created_at' | 'modified_at'
  use_crop: boolean
}

export interface ImageAutomationRequest {
  mode: 'txt2img' | 'img2img'
  template: Record<string, unknown>
  loop: ImageAutomationLoopRequest
  seed_policy: ImageAutomationSeedPolicyRequest
  prompt_source: ImageAutomationPromptSourceRequest
  init_source?: ImageAutomationFolderSourceRequest
}

export type UpscalerKind = 'latent' | 'spandrel'

export interface UpscalerDefinition {
  id: string
  label: string
  kind: UpscalerKind
  meta: Record<string, unknown>
}

export interface UpscalersResponse {
  upscalers: UpscalerDefinition[]
}

export interface UpscalersHfManifestV1Weight {
  id: string
  hf_path: string
  label: string
  arch: string
  scale: number
  license_name: string
  license_url: string
  license_spdx: string | null
  sha256: string
  tags: string[]
  notes: string | null
}

export interface UpscalersHfManifestV1 {
  schema_version: 1
  weights: UpscalersHfManifestV1Weight[]
}

export interface RemoteUpscalerWeightMeta {
  id: string
  arch: string
  scale: number
  license_name: string
  license_url: string
  license_spdx: string | null
  sha256: string
  tags: string[]
  notes: string | null
}

export type RemoteUpscalerWeight =
  | {
      hf_path: string
      label: string
      curated: false
      meta: null
    }
  | {
      hf_path: string
      label: string
      curated: true
      meta: RemoteUpscalerWeightMeta
    }

export interface RemoteUpscalersResponse {
  repo_id: string
  revision: string | null
  manifest_path: string
  manifest_found: boolean
  manifest_error: string | null
  manifest_errors: string[]
  manifest: UpscalersHfManifestV1 | null
  weights: RemoteUpscalerWeight[]
  safeweights_enabled: boolean
  allowed_weight_suffixes: string[]
}

export interface GeneratedImage {
  format: string
  data: string
}

export type TaskErrorCode =
  | 'cancelled'
  | 'out_of_memory'
  | 'integrity_mismatch'
  | 'engine_error'
  | 'internal_error'

export interface AutomationIterationEvent {
  type: 'automation_iteration'
  iteration_index: number
  images: GeneratedImage[]
  info: unknown
  seed: number | null
  prompt_preview: string
  source_label: string | null
}

export type TaskEvent =
  | { type: 'status'; stage: string }
  | {
      type: 'progress'
      stage: string
      percent?: number | null
      step?: number | null
      total_steps?: number | null
      eta_seconds?: number | null
      message?: string | null
      data?: Record<string, unknown> | null
      preview_image?: GeneratedImage
      preview_step?: number | null
    }
  | AutomationIterationEvent
  | { type: 'gap'; oldest_event_id: number; newest_event_id: number; last_event_id: number }
  | { type: 'result'; images?: GeneratedImage[]; info: unknown; video?: { rel_path?: string | null; mime?: string | null } }
  | { type: 'error'; message: string; code?: TaskErrorCode; error_id?: string | null }
  | { type: 'end' }

export interface TaskResult {
  status: 'running' | 'completed' | 'error'
  task_id?: string
  stage?: string
  progress?: {
    stage?: string
    percent?: number | null
    step?: number | null
    total_steps?: number | null
    eta_seconds?: number | null
    message?: string | null
    data?: Record<string, unknown> | null
  } | null
  preview_image?: GeneratedImage
  preview_step?: number | null
  automation_gallery_images?: GeneratedImage[]
  last_event_id?: number
  buffer_oldest_event_id?: number
  buffer_newest_event_id?: number
  started_at_ms?: number | null
  error?: string
  error_code?: TaskErrorCode
  error_id?: string | null
  result?: {
    images?: GeneratedImage[]
    info: unknown
    video?: { rel_path?: string | null; mime?: string | null }
  }
}

export interface MemoryResponse {
  total_vram_mb: number
  attention?: {
    backend?: string
    sdpa_policy?: string
    force_upcast?: boolean
    enable_flash?: boolean
    enable_mem_efficient?: boolean
    pytorch_sdp_enabled?: boolean
  }
}

export interface ObliterateVramProcessInfo {
  pid: number
  process_name: string
  used_gpu_memory_mb: number | null
  gpu_uuid: string
}

export interface ObliterateVramFailure {
  pid: number
  error: string
}

export interface ObliterateVramSkippedProcess {
  pid: number
  reason: string
}

export type ObliterateVramExternalKillMode = 'disabled' | 'all'

export interface ObliterateVramRequest {
  external_kill_mode?: ObliterateVramExternalKillMode
}

export interface ObliterateVramResponse {
  ok: boolean
  message: string
  internal: {
    runtime_unload_models: boolean
    runtime_soft_empty_cache: boolean
    gguf_cache_cleared: boolean
    gc_collect_ran: boolean
    torch_cuda_cache_cleared: boolean
  }
  internal_failures: string[]
  external: {
    kill_mode: ObliterateVramExternalKillMode
    nvidia_smi_available: boolean
    detected_processes: ObliterateVramProcessInfo[]
    terminated_pids: number[]
    skipped: ObliterateVramSkippedProcess[]
    failures: ObliterateVramFailure[]
  }
  warnings: string[]
}

export interface VersionResponse {
  app_version: string
  git_commit: string | null
  python_version: string
  torch_version: string | null
  cuda_version: string | null
}

export interface LtxExecutionSurface {
  allowed_execution_profiles: string[]
  default_execution_profile: string
  default_steps_by_profile: Record<string, number>
  default_guidance_scale_by_profile: Record<string, number>
}

export interface EngineCapabilities {
  supports_txt2img: boolean
  supports_img2img: boolean
  supports_img2img_masking: boolean
  supports_txt2vid: boolean
  supports_img2vid: boolean
  supports_vid2vid: boolean
  supports_hires: boolean
  supports_refiner: boolean
  supports_lora: boolean
  supports_controlnet: boolean
  supports_ip_adapter: boolean
  supports_supir_mode?: boolean
  // Optional: backend recommendation lists for UI hinting.
  recommended_samplers?: string[] | null
  recommended_schedulers?: string[] | null
  default_sampler?: string | null
  default_scheduler?: string | null
  guidance_advanced?: GuidanceAdvancedCapabilities | null
  ltx_execution_surface?: LtxExecutionSurface | null
}

export interface SupirVariantInfo {
  key: string
  label: string
}

export interface SupirExpectedVariantDiagnostics {
  expected_filenames: string[]
  present: boolean
  path: string | null
}

export interface SupirFoundFileDiagnostics {
  name: string
  path: string
  bytes?: number
  mtime?: number
}

export interface SupirWeightsDiagnostics {
  roots: string[]
  expected: Record<string, SupirExpectedVariantDiagnostics>
  found_files: SupirFoundFileDiagnostics[]
}

export interface SupirSamplerInfo {
  id: string
  label: string
  stability: 'stable' | 'dev'
  native_sampler: string
  native_scheduler: string
}

export interface SupirModelsResponse {
  supir_models: SupirWeightsDiagnostics
  variants: SupirVariantInfo[]
  samplers: SupirSamplerInfo[]
  note: string
}

export interface GuidanceAdvancedCapabilities {
  apg_enabled: boolean
  apg_start_step: boolean
  apg_eta: boolean
  apg_momentum: boolean
  apg_norm_threshold: boolean
  apg_rescale: boolean
  guidance_rescale: boolean
  cfg_trunc_ratio: boolean
  renorm_cfg: boolean
}

export interface FamilyCapabilities {
  supports_negative_prompt: boolean
  shows_clip_skip: boolean
  resolution_step: number
  supported_samplers?: string[] | null
  supported_schedulers?: string[] | null
  excluded_samplers?: string[] | null
  excluded_schedulers?: string[] | null
}

export interface EngineDependencyCheckRow {
  id: string
  label: string
  ok: boolean
  message: string
  inpaint_modes?: string[]
}

export interface EngineDependencyStatus {
  ready: boolean
  checks: EngineDependencyCheckRow[]
}

export interface ParkedExactEngineStatus {
  status: 'not_implemented'
  detail: string
}

export interface EngineCapabilitiesResponse {
  engines: Record<string, EngineCapabilities>
  families?: Record<string, FamilyCapabilities>
  smart_cache?: Record<string, { hits: number; misses: number }>
  asset_contracts?: Record<string, EngineAssetContractVariants>
  engine_id_to_semantic_engine: Record<string, string>
  exact_engine_inpaint_modes: Record<string, string[]>
  parked_exact_engines: Record<string, ParkedExactEngineStatus>
  dependency_checks: Record<string, EngineDependencyStatus>
}

export interface PromptTokenCountRequest {
  engine: string
  prompt: string
}

export interface PromptTokenCountResponse {
  engine: string
  prompt_len: number
  count: number
}

export interface EngineAssetContract {
  requires_vae: boolean
  uses_vae: boolean
  tenc_count: number
  tenc_slots?: string[]
  tenc_slot_labels?: string[]
  tenc_kind: string
  tenc_kind_label?: string
  sha_only: boolean
  notes: string
}

export interface EngineAssetContractVariants {
  base: EngineAssetContract
  core_only: EngineAssetContract
}

export interface EmbeddingsResponse {
  loaded: Record<string, { step?: number | null; vectors?: number; shape?: number[] | null; sd_checkpoint?: string | null; sd_checkpoint_name?: string | null }>
  skipped: Record<string, { step?: number | null; vectors?: number; shape?: number[] | null; sd_checkpoint?: string | null; sd_checkpoint_name?: string | null }>
}

export interface PathsResponse { paths: Record<string, string[]> }
export interface PathsUpdateResponse { ok: boolean }

export type ModelPathLibraryKind = 'checkpoint' | 'vae' | 'text_encoder'
export type ModelPathSizeBytes = number | null

export interface ModelPathScanRequest {
  path: string
  key?: string | null
  kind?: ModelPathLibraryKind | null
}

export interface ModelPathScanItem {
  name: string
  path: string
  ext: string
  size_bytes: ModelPathSizeBytes
  already_in_library: boolean
}

export interface ModelPathScanResponse {
  kind: ModelPathLibraryKind
  key?: string | null
  root: string
  items: ModelPathScanItem[]
}

export interface ModelPathAddRequest {
  key: string
  path: string
  kind?: ModelPathLibraryKind | null
}

export interface ModelPathAddItem extends ModelPathScanItem {
  type: ModelPathLibraryKind
  library_key: string
  added: boolean
  sha256: string
  short_hash: string | null
}

export interface ModelPathAddResponse {
  key: string
  kind: ModelPathLibraryKind
  item: ModelPathAddItem
}

export interface ModelPathAddAllErrorItem {
  name: string
  path: string
  ext: string
  size_bytes: ModelPathSizeBytes
  type: ModelPathLibraryKind
  library_key: string
}

export interface ModelPathAddAllResult {
  index: number
  total: number
  ok: boolean
  item: ModelPathAddItem | ModelPathAddAllErrorItem
  detail?: string
}

export interface ModelPathAddAllResponse {
  key: string
  kind: ModelPathLibraryKind
  root: string
  total: number
  added_count: number
  error_count: number
  results: ModelPathAddAllResult[]
}

// Settings schema (extracted from legacy)
export interface SettingsCategory { id: string; label: string }
export interface SettingsSection { key: string; label: string; category_id?: string | null }
export type SettingsFieldType = 'checkbox' | 'slider' | 'radio' | 'dropdown' | 'number' | 'text' | 'color' | 'html'
export interface SettingsField {
  key: string
  label: string
  type: SettingsFieldType
  section: string
  default?: unknown
  min?: number | null
  max?: number | null
  step?: number | null
  choices?: unknown[] | null
  choices_source?: string | null
}
export interface SettingsSchemaResponse {
  categories: SettingsCategory[]
  sections: SettingsSection[]
  fields: SettingsField[]
  source?: string
  version?: number
}

// UI Blocks (server-driven parameter panels)
export type UiFieldType = 'text' | 'number' | 'checkbox' | 'select' | 'slider' | 'textarea'
export interface UiFieldBind { txt2vid?: string; img2vid?: string }
export interface UiField {
  key: string
  label: string
  type: UiFieldType
  default?: unknown
  help?: string
  min?: number
  max?: number
  step?: number
  options?: (string | number)[]
  bind?: UiFieldBind
  visibleIf?: Record<string, unknown>
}
export interface UiBlockWhen { engines?: string[]; tabs?: string[] }
export interface UiBlockLayout { columns?: number }
export interface UiBlock { id: string; when?: UiBlockWhen; layout?: UiBlockLayout; fields: UiField[] }
export interface UiBlocksResponse { version: number; blocks: UiBlock[]; semantic_engine?: string }

// UI Presets (Model UI)
export interface UiPreset { id: string; title: string; tabs?: string[]; model_select: { type: 'exact' | 'pattern'; value: string } }
export interface UiPresetsResponse { version: number; presets: UiPreset[] }
export interface UiPresetApplyResponse { applied: boolean; model: string; checkpoint: string }

// Tabs/workflows persistence
export interface ApiTabMeta { createdAt: string; updatedAt: string }
export interface ApiTab { id: string; type: 'sd15' | 'sdxl' | 'flux1' | 'flux2' | 'qwen_image' | 'zimage' | 'zimage_l2p' | 'chroma' | 'wan22_14b' | 'wan22_5b' | 'anima' | 'ltx2'; title: string; order: number; enabled: boolean; params: Record<string, unknown>; meta: ApiTabMeta }
export interface TabsResponse { version: number; tabs: ApiTab[] }
export interface WorkflowItem { id: string; name: string; source_tab_id: string; type: ApiTab['type']; created_at: string; params_snapshot: Record<string, unknown> }
export interface WorkflowsResponse { version: number; workflows: WorkflowItem[] }
export interface WorkflowCreateResponse { id: string }
export interface WorkflowUpdateResponse { updated: string }
export interface WorkflowDeleteResponse { deleted: string }

// Model inventory (for populating selects)
export interface InventoryResponse {
  vaes: Array<{ name: string; path: string; sha256?: string; format: string; latent_channels?: number | null; scaling_factor?: number | null }>
  text_encoders: Array<{ name: string; path: string; sha256?: string; slot?: string }>
  loras: Array<{ name: string; path: string; sha256?: string }>
  ip_adapter_models: Array<{ name: string; path: string; sha256?: string }>
  ip_adapter_image_encoders: Array<{ name: string; path: string; sha256?: string }>
  wan22: {
    gguf: Array<{
      name: string
      path: string
      sha256?: string
      stage: 'high' | 'low' | 'unknown'
      variant?: 'wan22_5b' | 'wan22_14b' | 'wan22_14b_animate'
      repo_hint?: string
    }>
  }
  metadata: Array<{ name: string; path: string }>
}
