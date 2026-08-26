/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Frontend API client (typed fetch helpers + endpoint wrappers).
Provides JSON/Form fetch helpers and exports functions for models/options/inventory/tasks, image automation, UI tabs/workflows persistence, and UI schema/preset
endpoints under `VITE_API_BASE` (default `/api`). Also caches `/api/options` revision monotonically, uses that revision for bounded conditional
`POST /api/options` writes where callers request CAS semantics, and preserves structured HTTP error metadata (`status/detail/body`)
for conflict-aware generation UX. SUPIR diagnostics parsing now validates the structured stable sampler rows from `/api/supir/models`
(`id` / `label` / `stability` / `native_sampler` / `native_scheduler`) before the shared frontend diagnostics owner consumes them, and exposes explicit
cache invalidation so Refresh can truthfully refetch that diagnostics surface in-session.
Task SSE subscriptions support resume via `after=<event_id>` and expose the latest `lastEventId` for reconnect/replay persistence, including buffered
`automation_iteration` events from `/api/image-automation`.

Symbols (top-level; keep in sync; no ghosts):
- `API_BASE` (const): Base URL prefix for backend endpoints (from Vite env, default `/api`).
- `requestJson` (function): JSON request helper with consistent error handling.
- `requestForm` (function): Form POST helper for multipart endpoints.
- `readErrorDetail` (function): Extracts structured error detail (`message/detail/body`) from failed backend responses.
- `getApiErrorStatus` (function): Reads an HTTP status code from request errors emitted by this client.
- `getCurrentRevisionFromError` (function): Extracts `current_revision` from backend conflict errors (`409`) when present.
- `getCachedOptionsRevision` (function): Returns the cached `/api/options` revision used by generation payload builders.
- `promoteCachedOptionsRevision` (function): Monotonically promotes the cached `/api/options` revision from one validated value or conflict fallback.
- `ModelsFreshnessMarker` (type): Deterministic model-list freshness marker (`invalidationVersion` + content fingerprint + request id).
- `fetchModelsWithFreshness` (function): Fetches `/models` with a freshness marker used by stores to avoid stale-response ambiguity.
- `invalidateModelCatalogCaches` (function): Centralized invalidation for model/inventory caches and invalidation epoch bumps.
- `invalidateSupirModelsCache` (function): Clears the cached `/api/supir/models` diagnostics payload so Refresh can force a truthful refetch.
- `fetchModels` (function): Fetches the model list (`/models`).
- `refreshModels` (function): Forces a checkpoint rescan (`/models?refresh=1`).
- `fetchModelInventory` (function): Fetches the inventory cache (`/models/inventory`).
- `fetchFreshModelInventory` (function): Reads the current inventory truth uncached (`/models/inventory`) for bounded fail-loud validation paths.
- `getModelCatalogInvalidationVersion` (function): Returns the current model-catalog invalidation epoch so callers can discard stale in-flight catalog responses.
- `fetchFileMetadata` (function): Reads GGUF/SafeTensors file metadata (`/models/file-metadata`).
- `fetchCheckpointMetadata` (function): Fetches the metadata modal payload for a checkpoint selection (`/models/checkpoint-metadata`).
- `refreshModelInventory` (function): Forces an inventory rescan (`/models/inventory/refresh`).
- `startModelInventoryRefreshTask` (function): Starts an async inventory refresh task (`/models/inventory/refresh/async`).
- `startImageAutomation` (function): Starts an image automation task (`POST /image-automation`) with nested template settings revision normalization.
- `refreshModelInventoryAsync` (function): Runs async inventory refresh task (`/models/inventory/refresh/async`), recovers validated inventory from terminal snapshots after SSE gaps or `end` without a live result, and resolves as soon as the recovered task snapshot reports `completed`.
- `cacheModelInventorySnapshot` (function): Writes an inventory snapshot into the local API cache (`/models/inventory`).
- `fetchSamplers` (function): Fetches `/samplers`, preserves raw unsupported rows at the DTO boundary, and returns only supported entries after fail-loud metadata validation.
- `fetchSchedulers` (function): Fetches supported schedulers (`/schedulers`) and filters out unsupported entries.
- `analyzePngInfo` (function): Extracts PNG text metadata for the PNG Info view (`POST /tools/pnginfo/analyze` multipart).
- `fetchOptions` (function): Fetches runtime options (`/options`).
- `updateOptions` (function): Updates runtime options (`POST /options`) with required `X-Codex-Expected-Revision` conditional writes.
- `startTxt2Img` (function): Starts a txt2img task (`POST /txt2img`).
- `startImg2Img` (function): Starts an img2img task (`POST /img2img`).
- `startTxt2Vid` (function): Starts a txt2vid task (`POST /txt2vid`).
- `startImg2Vid` (function): Starts an img2vid task (`POST /img2vid`).
- `fetchUpscalers` (function): Fetches the local upscalers list (`/upscalers`).
- `refreshUpscalers` (function): Forces an upscalers re-fetch (clears cache, then calls `/upscalers`).
- `fetchRemoteUpscalers` (function): Fetches curated HF upscalers list (`/upscalers/remote`).
- `downloadUpscalers` (function): Starts an upscalers download task (`POST /upscalers/download`).
- `startUpscale` (function): Starts a standalone upscale task (`POST /upscale` multipart).
- `startVideoUpscale` (function): Starts a dedicated SeedVR2 video-upscale task (`POST /video-upscale` JSON).
- `fetchTaskResult` (function): Fetches a task result (`/tasks/:id`).
- `cancelTask` (function): Requests task cancellation (`/tasks/:id/cancel`).
- `subscribeTask` (function): Subscribes to task SSE events and returns an unsubscribe closure.
- `fetchMemory` (function): Fetches memory stats (`/memory`).
- `fetchObliterateVram` (function): Triggers VRAM cleanup (`POST /obliterate-vram`); external kill is disabled by default unless explicitly requested.
- `fetchVersion` (function): Fetches backend version (`/version`).
- `fetchEmbeddings` (function): Fetches embeddings list (`/embeddings`).
- `fetchEngineCapabilities` (function): Fetches engine capabilities (`/engines/capabilities`).
- `fetchSupirModels` (function): Fetches cached SUPIR diagnostics/readiness (`/supir/models`) for SDXL img2img/inpaint UI discoverability.
- `fetchPromptTokenCount` (function): Counts prompt tokens via backend tokenizer (`POST /models/prompt-token-count`).
- `fetchPaths` (function): Fetches configured paths (`/paths`).
- `fetchFreshPaths` (function): Reads the current configured paths uncached (`/paths`) for bounded fail-loud validation paths.
- `updatePaths` (function): Updates configured paths (`POST /paths`).
- `scanModelPath` (function): Scans a model path for add-path candidates without hashing (`POST /models/path-scan`).
- `addModelPathItem` (function): Adds one file to a model library key and computes SHA at add-time (`POST /models/path-add`).
- `addModelPathItemsAll` (function): Adds all scanned files sequentially (backend add-all helper; no pre-hash) (`POST /models/path-add-all`).
- `fetchSettingsSchema` (function): Fetches settings schema (`/settings/schema`).
- `fetchUiBlocks` (function): Fetches UI blocks schema (`/ui/blocks`).
- `fetchUiPresets` (function): Fetches UI presets (`/ui/presets`).
- `applyUiPreset` (function): Applies a UI preset (`POST /ui/presets/apply`).
- `fetchTabs` (function): Fetches persisted tabs (`/ui/tabs`).
- `createTabApi` (function): Creates a tab (`POST /ui/tabs`).
- `updateTabApi` (function): Updates a tab (`PATCH /ui/tabs/:id`).
- `reorderTabsApi` (function): Reorders tabs (`POST /ui/tabs/reorder`).
- `deleteTabApi` (function): Deletes a tab (`DELETE /ui/tabs/:id`).
- `fetchWorkflows` (function): Fetches workflows (`/ui/workflows`).
- `createWorkflow` (function): Creates a workflow (`POST /ui/workflows`).
- `updateWorkflow` (function): Updates workflow name/source-tab binding/params snapshot (`PATCH /ui/workflows/:id`).
- `deleteWorkflow` (function): Deletes a workflow (`DELETE /ui/workflows/:id`).
- `loadModelsForTab` (function): Loads models for a tab (`POST /models/load`).
- `unloadModelsForTab` (function): Unloads models for a tab (`POST /models/unload`).
*/

import type {
  ModelsResponse,
  RawSamplerInfo,
  SamplerInfo,
  SamplersResponse,
  SupportedSamplersResponse,
  SchedulersResponse,
  OptionsResponse,
  OptionsUpdateResponse,
  Txt2ImgStartResponse,
  TaskStartResponse,
  TaskResult,
  TaskEvent,
  MemoryResponse,
  ObliterateVramRequest,
  ObliterateVramResponse,
  VersionResponse,
  EmbeddingsResponse,
  PathsResponse,
  PathsUpdateResponse,
  ModelPathScanRequest,
  ModelPathScanResponse,
  ModelPathAddRequest,
  ModelPathAddResponse,
  ModelPathAddAllResponse,
  SettingsSchemaResponse,
  UiBlocksResponse,
  UiPresetsResponse,
  UiPresetApplyResponse,
  ImageAutomationRequest,
  InventoryResponse,
  EngineCapabilitiesResponse,
  SupirModelsResponse,
  SupirSamplerInfo,
  PromptTokenCountRequest,
  PromptTokenCountResponse,
  FileMetadataResponse,
  CheckpointMetadataResponse,
  PngInfoAnalyzeResponse,
  UpscalersResponse,
  RemoteUpscalersResponse,
  VideoUpscaleRequest,
} from './types'
import type { Txt2ImgRequest } from './payloads'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

type JsonCacheEntry = {
  value: unknown
  modelCatalogInvalidationVersion?: number
}

const _jsonCache = new Map<string, JsonCacheEntry>()
const _jsonInflight = new Map<string, Promise<unknown>>()
let _cachedOptionsRevision = 0
let _modelsInvalidationVersion = 0
let _modelsRequestId = 0

export type ModelsFreshnessMarker = {
  requestId: number
  invalidationVersion: number
  refreshed: boolean
  fetchedAtMs: number
  contentFingerprint: string
}

type ModelsWithFreshnessResult = {
  response: ModelsResponse
  freshness: ModelsFreshnessMarker
}

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function normalizeRevision(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.max(0, Math.trunc(value))
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return null
    if (/^-?\d+$/.test(trimmed)) {
      return Math.max(0, Math.trunc(Number(trimmed)))
    }
  }
  return null
}

function stableHash32Hex(text: string): string {
  // FNV-1a 32-bit hash for deterministic, low-cost fingerprints.
  let hash = 0x811c9dc5
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

function computeModelsContentFingerprint(payload: ModelsResponse): string {
  const normalizedModels = (Array.isArray(payload.models) ? payload.models : [])
    .map((model) => ({
      title: String(model?.title || '').trim(),
      filename: String(model?.filename || '').trim().replace(/\\+/g, '/'),
      hash: String(model?.hash || '').trim().toLowerCase(),
      core_only: Boolean((model as any)?.core_only),
      family_hint: String((model as any)?.family_hint || '').trim().toLowerCase(),
    }))
    .sort((left, right) => {
      const byTitle = left.title.localeCompare(right.title)
      if (byTitle !== 0) return byTitle
      const byFile = left.filename.localeCompare(right.filename)
      if (byFile !== 0) return byFile
      return left.hash.localeCompare(right.hash)
    })

  const canonical = JSON.stringify({
    current: String(payload.current || ''),
    models: normalizedModels,
  })
  return stableHash32Hex(canonical)
}

function cacheOptionsRevisionFromPayload(payload: unknown): void {
  if (!isRecordObject(payload)) return
  const direct = normalizeRevision(payload.revision)
  const values = payload.values
  const fromValues = isRecordObject(values) ? normalizeRevision(values.codex_options_revision) : null
  const nextRevision = Math.max(direct ?? 0, fromValues ?? 0)
  promoteCachedOptionsRevision(nextRevision)
}

export function getCachedOptionsRevision(): number {
  return _cachedOptionsRevision
}

export function promoteCachedOptionsRevision(value: unknown): number {
  const nextRevision = normalizeRevision(value)
  if (nextRevision !== null && nextRevision > _cachedOptionsRevision) {
    _cachedOptionsRevision = nextRevision
  }
  return _cachedOptionsRevision
}

function readCurrentRevision(value: unknown, depth = 0): number | null {
  if (depth > 5) return null
  if (value === null || value === undefined) return null

  if (typeof value === 'string') {
    const match = value.match(/current[_\s-]?revision[^0-9-]*(-?\d+)/i)
    if (!match) return null
    return normalizeRevision(match[1])
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      const found = readCurrentRevision(item, depth + 1)
      if (found !== null) return found
    }
    return null
  }

  if (!isRecordObject(value)) return null

  for (const key of ['current_revision', 'currentRevision'] as const) {
    const found = normalizeRevision(value[key])
    if (found !== null) return found
  }

  for (const nested of Object.values(value)) {
    const found = readCurrentRevision(nested, depth + 1)
    if (found !== null) return found
  }
  return null
}

export function getApiErrorStatus(error: unknown): number | null {
  if (!isRecordObject(error)) return null
  return normalizeRevision(error.status)
}

export function getCurrentRevisionFromError(error: unknown): number | null {
  if (error instanceof Error) {
    const fromMessage = readCurrentRevision(error.message)
    if (fromMessage !== null) return fromMessage
  }
  return readCurrentRevision(error)
}

function detailToMessage(detail: unknown): string {
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  if (isRecordObject(detail)) {
    const unknownKeysRaw = detail.unknown_keys ?? detail.unknownKeys
    const unknownKeys = Array.isArray(unknownKeysRaw)
      ? unknownKeysRaw.map((entry) => String(entry || '').trim()).filter((entry) => entry.length > 0)
      : []
    if (unknownKeys.length > 0) {
      const context = String(detail.context || detail.field || detail.tab_type || 'request').trim()
      return `Unexpected ${context} key(s): ${unknownKeys.join(', ')}`
    }
    const message = typeof detail.message === 'string'
      ? detail.message.trim()
      : ''
    if (message) return message
    return JSON.stringify(detail)
  }
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((item) => {
        const msg = (item && typeof item === 'object') ? (item as any).msg : null
        return typeof msg === 'string' ? msg : String(item)
      })
      .filter((s) => String(s || '').trim())
    if (msgs.length) return msgs.join('\n')
  }
  if (detail !== undefined) return JSON.stringify(detail)
  return ''
}

async function readErrorDetail(res: Response): Promise<{ message: string; detail: unknown; body: unknown }> {
  const text = await res.text()
  if (!text) return { message: '', detail: null, body: null }
  try {
    const data = JSON.parse(text) as unknown
    if (isRecordObject(data)) {
      const detail = data.detail
      return { message: detailToMessage(detail), detail, body: data }
    }
    return { message: text, detail: null, body: data }
  } catch {
    // not JSON; fall through
  }
  return { message: text, detail: null, body: null }
}

function invalidateJsonCache(prefixPath: string): void {
  for (const key of Array.from(_jsonCache.keys())) {
    if (key === prefixPath || key.startsWith(`${prefixPath}?`)) _jsonCache.delete(key)
  }
  for (const key of Array.from(_jsonInflight.keys())) {
    if (key === prefixPath || key.startsWith(`${prefixPath}?`)) _jsonInflight.delete(key)
  }
}

function isModelCatalogCachePath(path: string): boolean {
  return path === '/models'
    || path.startsWith('/models?')
    || path === '/models/inventory'
    || path.startsWith('/models/inventory?')
}

function bumpModelsInvalidationVersion(): number {
  const next = _modelsInvalidationVersion + 1
  _modelsInvalidationVersion = Number.isSafeInteger(next) ? next : 1
  return _modelsInvalidationVersion
}

export function invalidateModelCatalogCaches(): number {
  invalidateJsonCache('/models')
  invalidateJsonCache('/models/inventory')
  return bumpModelsInvalidationVersion()
}

export function getModelCatalogInvalidationVersion(): number {
  return _modelsInvalidationVersion
}

export function invalidateSupirModelsCache(): void {
  invalidateJsonCache('/supir/models')
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })
  if (!res.ok) {
    const detail = await readErrorDetail(res)
    const err = new Error(detail.message || `HTTP ${res.status} ${res.statusText}`) as Error & {
      status?: number
      detail?: unknown
      body?: unknown
    }
    err.status = res.status
    err.detail = detail.detail
    err.body = detail.body
    throw err
  }
  return (await res.json()) as T
}

function requestJsonCached<T>(path: string): Promise<T> {
  const cached = _jsonCache.get(path)
  if (cached !== undefined) {
    if (
      isModelCatalogCachePath(path)
      && cached.modelCatalogInvalidationVersion !== _modelsInvalidationVersion
    ) {
      _jsonCache.delete(path)
    } else {
      return Promise.resolve(cached.value as T)
    }
  }

  const inflight = _jsonInflight.get(path)
  if (inflight) return inflight as Promise<T>

  const requestModelCatalogInvalidationVersion = isModelCatalogCachePath(path)
    ? _modelsInvalidationVersion
    : undefined
  const p = requestJson<T>(path)
    .then((value) => {
      if (
        requestModelCatalogInvalidationVersion !== undefined
        && requestModelCatalogInvalidationVersion !== _modelsInvalidationVersion
      ) {
        return value
      }
      _jsonCache.set(path, {
        value,
        modelCatalogInvalidationVersion: requestModelCatalogInvalidationVersion,
      })
      return value
    })
    .finally(() => {
      _jsonInflight.delete(path)
    })
  _jsonInflight.set(path, p as Promise<unknown>)
  return p
}

async function requestForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const detail = await readErrorDetail(res)
    const err = new Error(detail.message || `HTTP ${res.status} ${res.statusText}`) as Error & {
      status?: number
      detail?: unknown
      body?: unknown
    }
    err.status = res.status
    err.detail = detail.detail
    err.body = detail.body
    throw err
  }
  return (await res.json()) as T
}

function withSettingsRevision(payload: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...payload }
  const existing = normalizeRevision(out.settings_revision)
  out.settings_revision = existing ?? getCachedOptionsRevision()
  return out
}

export async function fetchModelsWithFreshness(
  options: { refresh?: boolean; invalidate?: boolean } = {},
): Promise<ModelsWithFreshnessResult> {
  const refreshed = options.refresh === true
  const shouldInvalidate = options.invalidate === true || refreshed
  const invalidationVersion = shouldInvalidate
    ? invalidateModelCatalogCaches()
    : _modelsInvalidationVersion
  const path = refreshed ? '/models?refresh=1' : '/models'
  const response = await requestJson<ModelsResponse>(path)
  const freshness: ModelsFreshnessMarker = {
    requestId: ++_modelsRequestId,
    invalidationVersion,
    refreshed,
    fetchedAtMs: Date.now(),
    contentFingerprint: computeModelsContentFingerprint(response),
  }
  return { response, freshness }
}

export function fetchModels(): Promise<ModelsResponse> {
  return fetchModelsWithFreshness().then(({ response }) => response)
}

export function refreshModels(): Promise<ModelsResponse> {
  return fetchModelsWithFreshness({ refresh: true, invalidate: true }).then(({ response }) => response)
}

export function fetchModelInventory(): Promise<InventoryResponse> {
  return requestJsonCached<InventoryResponse>('/models/inventory')
}

export function fetchFreshModelInventory(): Promise<InventoryResponse> {
  return requestJson<InventoryResponse>('/models/inventory')
}

export function fetchFileMetadata(path: string): Promise<FileMetadataResponse> {
  return requestJson<FileMetadataResponse>(`/models/file-metadata?path=${encodeURIComponent(path)}`)
}

export function fetchCheckpointMetadata(value: string): Promise<CheckpointMetadataResponse> {
  return requestJson<CheckpointMetadataResponse>(`/models/checkpoint-metadata?value=${encodeURIComponent(value)}`)
}

export async function refreshModelInventory(): Promise<InventoryResponse> {
  invalidateModelCatalogCaches()
  const inv = await requestJson<InventoryResponse>('/models/inventory/refresh', { method: 'POST' })
  _jsonCache.set('/models/inventory', {
    value: inv,
    modelCatalogInvalidationVersion: _modelsInvalidationVersion,
  })
  return inv
}

export function startModelInventoryRefreshTask(): Promise<TaskStartResponse> {
  invalidateModelCatalogCaches()
  return requestJson<TaskStartResponse>('/models/inventory/refresh/async', { method: 'POST' })
}

const MODEL_INVENTORY_REFRESH_CANCELLED = 'Model inventory refresh cancelled'

function toError(error: unknown, fallbackMessage: string): Error {
  if (error instanceof Error) return error
  const message = String(error || '').trim()
  return new Error(message || fallbackMessage)
}

function makeAbortError(): Error {
  const err = new Error(MODEL_INVENTORY_REFRESH_CANCELLED)
  err.name = 'AbortError'
  return err
}

function parseInventoryTaskPayload(payload: unknown): InventoryResponse | null {
  if (!isRecordObject(payload)) return null
  if (
    Array.isArray(payload.vaes)
    && Array.isArray(payload.ip_adapter_models)
    && Array.isArray(payload.ip_adapter_image_encoders)
    && Array.isArray(payload.text_encoders)
    && Array.isArray(payload.loras)
    && Array.isArray(payload.metadata)
    && isRecordObject(payload.wan22)
  ) {
    return payload as unknown as InventoryResponse
  }
  const direct = payload.inventory
  if (isRecordObject(direct)) return direct as unknown as InventoryResponse
  const nestedResult = parseInventoryTaskPayload(payload.result)
  if (nestedResult) return nestedResult
  const info = payload.info
  return parseInventoryTaskPayload(info)
}

function parseInventoryTaskResult(event: TaskEvent): InventoryResponse | null {
  if (event.type !== 'result') return null
  return parseInventoryTaskPayload(event)
}

async function fetchInventoryTaskSnapshot(taskId: string): Promise<{ status: TaskResult['status']; inventory: InventoryResponse | null }> {
  const snapshot = await fetchTaskResult(taskId)
  if (snapshot.status === 'error') {
    throw new Error(String(snapshot.error || 'inventory refresh task failed'))
  }
  const parsed = parseInventoryTaskPayload(snapshot)
  if (!parsed) {
    return { status: snapshot.status, inventory: null }
  }
  return {
    status: snapshot.status,
    inventory: ensureInventoryTaskPayloadShape(parsed),
  }
}

function requireInventoryArrayField(
  payload: Record<string, unknown>,
  key: string,
  errorMessage: string,
): void {
  if (!Array.isArray(payload[key])) {
    throw new Error(errorMessage)
  }
}

function ensureInventoryTaskPayloadShape(inventory: InventoryResponse): InventoryResponse {
  const payload = inventory as unknown as Record<string, unknown>
  requireInventoryArrayField(payload, 'vaes', 'inventory refresh task result payload missing inventory.vaes[]')
  requireInventoryArrayField(
    payload,
    'ip_adapter_models',
    'inventory refresh task result payload missing inventory.ip_adapter_models[]',
  )
  requireInventoryArrayField(
    payload,
    'ip_adapter_image_encoders',
    'inventory refresh task result payload missing inventory.ip_adapter_image_encoders[]',
  )
  requireInventoryArrayField(payload, 'text_encoders', 'inventory refresh task result payload missing inventory.text_encoders[]')
  requireInventoryArrayField(payload, 'loras', 'inventory refresh task result payload missing inventory.loras[]')
  requireInventoryArrayField(payload, 'metadata', 'inventory refresh task result payload missing inventory.metadata[]')
  const wan22 = payload.wan22
  if (!isRecordObject(wan22)) {
    throw new Error('inventory refresh task result payload missing inventory.wan22 object')
  }
  if (!Array.isArray(wan22.gguf)) {
    throw new Error('inventory refresh task result payload missing inventory.wan22.gguf[]')
  }
  return inventory
}

export async function refreshModelInventoryAsync(options: { signal?: AbortSignal } = {}): Promise<InventoryResponse> {
  const { signal } = options
  if (signal?.aborted) {
    throw makeAbortError()
  }

  const started = await startModelInventoryRefreshTask()
  const taskId = String(started.task_id || '').trim()
  if (!taskId) {
    throw new Error('inventory refresh task start response missing task_id')
  }
  if (signal?.aborted) {
    throw makeAbortError()
  }

  const refreshedInventory = await new Promise<InventoryResponse>((resolve, reject) => {
    let settled = false
    let unsubscribe: (() => void) | null = null
    let resolvedInventory: InventoryResponse | null = null

    const settle = (fn: () => void): void => {
      if (settled) return
      settled = true
      if (signal) {
        try { signal.removeEventListener('abort', onAbort) } catch (_) { /* ignore */ }
      }
      try { unsubscribe?.() } catch (_) { /* ignore */ }
      unsubscribe = null
      fn()
    }

    const onAbort = (): void => {
      settle(() => reject(makeAbortError()))
    }

    if (signal) {
      signal.addEventListener('abort', onAbort, { once: true })
    }

    const recoverFromSnapshot = (reason: 'gap' | 'end'): void => {
      void fetchInventoryTaskSnapshot(taskId)
        .then(({ status, inventory }) => {
          if (settled) return
          if (inventory) {
            resolvedInventory = inventory
          }
          if (status === 'completed') {
            if (!resolvedInventory) {
              settle(() => reject(new Error('inventory refresh task completed without inventory payload')))
              return
            }
            const inventoryPayload = resolvedInventory
            settle(() => resolve(inventoryPayload))
            return
          }
          if (reason === 'end' && !resolvedInventory) {
            settle(() => reject(new Error('inventory refresh task completed without inventory payload')))
          }
        })
        .catch((error) => {
          if (settled) return
          if (reason === 'end') {
            settle(() => reject(toError(error, 'inventory refresh task snapshot recovery failed')))
          }
        })
    }

    unsubscribe = subscribeTask(
      taskId,
      (event) => {
        if (event.type === 'error') {
          const message = String(event.message || '').trim() || 'inventory refresh task failed'
          settle(() => reject(new Error(message)))
          return
        }

        if (event.type === 'result') {
          const parsed = parseInventoryTaskResult(event)
          if (!parsed) {
            settle(() => reject(new Error('inventory refresh task result missing inventory payload')))
            return
          }
          try {
            resolvedInventory = ensureInventoryTaskPayloadShape(parsed)
          } catch (error) {
            settle(() => reject(toError(error, 'inventory refresh task returned invalid payload')))
          }
          return
        }

        if (event.type === 'gap') {
          recoverFromSnapshot('gap')
          return
        }

        if (event.type === 'end') {
          if (!resolvedInventory) {
            recoverFromSnapshot('end')
            return
          }
          settle(() => resolve(resolvedInventory as InventoryResponse))
        }
      },
      (error) => {
        settle(() => reject(toError(error, 'inventory refresh task stream error')))
      },
    )
  })

  _jsonCache.set('/models/inventory', {
    value: refreshedInventory,
    modelCatalogInvalidationVersion: _modelsInvalidationVersion,
  })
  return refreshedInventory
}

export function cacheModelInventorySnapshot(inv: InventoryResponse): void {
  _jsonCache.set('/models/inventory', {
    value: inv,
    modelCatalogInvalidationVersion: _modelsInvalidationVersion,
  })
}

export async function fetchSamplers(): Promise<SupportedSamplersResponse> {
  const res = await requestJsonCached<SamplersResponse>('/samplers')
  const supported: SamplerInfo[] = res.samplers.flatMap((sampler) => {
    if (sampler.supported === false) return []
    const defaultScheduler = typeof sampler.default_scheduler === 'string' ? sampler.default_scheduler.trim() : ''
    const allowedSchedulers = Array.isArray(sampler.allowed_schedulers)
      ? sampler.allowed_schedulers.map((value) => String(value || '').trim()).filter((value) => value.length > 0)
      : []
    const name = String(sampler.name || '').trim() || '<unknown>'
    if (!defaultScheduler) {
      throw new Error(`/api/samplers returned supported sampler '${name}' without default_scheduler.`)
    }
    if (!allowedSchedulers.includes(defaultScheduler)) {
      throw new Error(
        `/api/samplers returned supported sampler '${name}' with default_scheduler '${defaultScheduler}' outside allowed_schedulers.`,
      )
    }
    const normalized: SamplerInfo = {
      ...sampler,
      default_scheduler: defaultScheduler,
      allowed_schedulers: allowedSchedulers,
    }
    return [normalized]
  })
  return { samplers: supported }
}

export async function fetchSchedulers(): Promise<SchedulersResponse> {
  const res = await requestJsonCached<SchedulersResponse>('/schedulers')
  const supported = res.schedulers.filter((scheduler) => scheduler.supported !== false)
  return { schedulers: supported }
}

export function analyzePngInfo(file: File): Promise<PngInfoAnalyzeResponse> {
  const form = new FormData()
  form.append('file', file)
  return requestForm<PngInfoAnalyzeResponse>('/tools/pnginfo/analyze', form)
}

export async function fetchOptions(): Promise<OptionsResponse> {
  const res = await requestJson<OptionsResponse>('/options')
  cacheOptionsRevisionFromPayload(res)
  return res
}

export async function updateOptions(
  payload: Record<string, unknown>,
  options: { expectedRevision: number },
): Promise<OptionsUpdateResponse> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  }
  const expectedRevision = normalizeRevision(options.expectedRevision)
  if (expectedRevision === null) {
    throw new Error('updateOptions requires a non-negative expectedRevision.')
  }
  headers['X-Codex-Expected-Revision'] = String(expectedRevision)
  const res = await requestJson<OptionsUpdateResponse>('/options', {
    method: 'POST',
    headers,
    body: JSON.stringify(payload),
  })
  cacheOptionsRevisionFromPayload(res)
  return res
}

export function startTxt2Img(payload: Txt2ImgRequest): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/txt2img', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function startImg2Img(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/img2img', {
    method: 'POST',
    body: JSON.stringify(withSettingsRevision(payload)),
  })
}

export function startImageAutomation(payload: ImageAutomationRequest): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/image-automation', {
    method: 'POST',
    body: JSON.stringify({
      ...payload,
      template: withSettingsRevision(payload.template),
    }),
  })
}

export function startTxt2Vid(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/txt2vid', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function startImg2Vid(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/img2vid', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchUpscalers(): Promise<UpscalersResponse> {
  return requestJsonCached<UpscalersResponse>('/upscalers')
}

export async function refreshUpscalers(): Promise<UpscalersResponse> {
  invalidateJsonCache('/upscalers')
  const res = await requestJson<UpscalersResponse>('/upscalers')
  _jsonCache.set('/upscalers', { value: res })
  return res
}

export function fetchRemoteUpscalers(opts: { repo_id?: string; revision?: string } = {}): Promise<RemoteUpscalersResponse> {
  const params = new URLSearchParams()
  if (typeof opts.repo_id === 'string' && opts.repo_id.trim()) params.set('repo_id', opts.repo_id.trim())
  if (typeof opts.revision === 'string' && opts.revision.trim()) params.set('revision', opts.revision.trim())
  const q = params.toString()
  return requestJson<RemoteUpscalersResponse>(`/upscalers/remote${q ? `?${q}` : ''}`)
}

export function downloadUpscalers(payload: { repo_id?: string; revision?: string | null; files: string[] }): Promise<TaskStartResponse> {
  return requestJson<TaskStartResponse>('/upscalers/download', { method: 'POST', body: JSON.stringify(payload) })
}

export function startUpscale(image: File, payload: Record<string, unknown>): Promise<TaskStartResponse> {
  const form = new FormData()
  form.append('image', image)
  form.append('payload', JSON.stringify(payload))
  return requestForm<TaskStartResponse>('/upscale', form)
}

export function startVideoUpscale(payload: VideoUpscaleRequest): Promise<TaskStartResponse> {
  return requestJson<TaskStartResponse>('/video-upscale', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchTaskResult(taskId: string): Promise<TaskResult> {
  return requestJson<TaskResult>(`/tasks/${taskId}`)
}

export function cancelTask(taskId: string, mode: 'immediate' | 'after_current' = 'immediate'): Promise<{ status: string; mode: string }> {
  return requestJson<{ status: string; mode: string }>(`/tasks/${encodeURIComponent(taskId)}/cancel`, {
    method: 'POST',
    body: JSON.stringify({ mode }),
  })
}

export function subscribeTask(
  taskId: string,
  onEvent: (event: TaskEvent) => void,
  onError?: (err: unknown) => void,
  opts: { after?: number; onMeta?: (meta: { eventId?: number }) => void } = {},
): () => void {
  const params = new URLSearchParams()
  if (typeof opts.after === 'number' && Number.isFinite(opts.after) && opts.after > 0) {
    params.set('after', String(Math.trunc(opts.after)))
  }
  const q = params.toString()
  const es = new EventSource(`${API_BASE}/tasks/${taskId}/events${q ? `?${q}` : ''}`)
  let ended = false
  es.onmessage = (msg: MessageEvent<string>) => {
    try {
      const payload = JSON.parse(msg.data) as TaskEvent
      const idRaw = (msg as any).lastEventId
      const id = typeof idRaw === 'string' && idRaw.trim() ? Number(idRaw) : null
      if (id !== null && Number.isFinite(id)) {
        try { opts.onMeta?.({ eventId: Math.trunc(id) }) } catch (_) { /* ignore */ }
      }
      // Mark graceful end so we don’t log a browser “error” on normal close
      if ((payload as any)?.type === 'end') {
        ended = true
        // Let consumers receive the end event before closing
        onEvent(payload)
        es.close()
        return
      }
      onEvent(payload)
    } catch (error) {
      console.error('[task-events] failed to parse event', error)
    }
  }
  es.onerror = (err) => {
    // EventSource fires onerror on normal close; suppress noisy logs when ended or closed
    if (ended || (es as any).readyState === 2 /* CLOSED */) return
    console.error('[task-events] stream error', err)
    try { onError?.(err) } catch (_) { /* ignore */ }
  }
  return () => es.close()
}

export function fetchMemory(): Promise<MemoryResponse> {
  return requestJson<MemoryResponse>('/memory')
}

export function fetchObliterateVram(payload: ObliterateVramRequest = {}): Promise<ObliterateVramResponse> {
  const body: ObliterateVramRequest = {
    external_kill_mode: payload.external_kill_mode ?? 'disabled',
  }
  return requestJson<ObliterateVramResponse>('/obliterate-vram', {
    method: 'POST',
    body: JSON.stringify(body),
  })
}

export function fetchVersion(): Promise<VersionResponse> {
  return requestJson<VersionResponse>('/version')
}

export function fetchEmbeddings(): Promise<EmbeddingsResponse> {
  return requestJson<EmbeddingsResponse>('/embeddings')
}

export function fetchEngineCapabilities(): Promise<EngineCapabilitiesResponse> {
  return requestJson<EngineCapabilitiesResponse>('/engines/capabilities')
}

export function fetchSupirModels(): Promise<SupirModelsResponse> {
  return requestJsonCached<SupirModelsResponse>('/supir/models').then((payload) => {
    if (!Array.isArray(payload.samplers)) {
      throw new Error("/api/supir/models returned invalid 'samplers' payload.")
    }
    const samplers: SupirSamplerInfo[] = payload.samplers.map((entry, index) => {
      const row = entry as unknown as Record<string, unknown>
      const id = String(row.id || '').trim()
      const label = String(row.label || '').trim()
      const stability = row.stability === 'dev' ? 'dev' : row.stability === 'stable' ? 'stable' : ''
      const nativeSampler = String(row.native_sampler || '').trim()
      const nativeScheduler = String(row.native_scheduler || '').trim()
      if (!id || !label || !stability || !nativeSampler || !nativeScheduler) {
        throw new Error(`/api/supir/models returned invalid sampler row at index ${index}.`)
      }
      return {
        id,
        label,
        stability,
        native_sampler: nativeSampler,
        native_scheduler: nativeScheduler,
      }
    })
    return {
      ...payload,
      samplers,
    }
  })
}

export function fetchPromptTokenCount(payload: PromptTokenCountRequest): Promise<PromptTokenCountResponse> {
  return requestJson<PromptTokenCountResponse>('/models/prompt-token-count', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchPaths(): Promise<PathsResponse> {
  return requestJsonCached<PathsResponse>('/paths')
}

export function fetchFreshPaths(): Promise<PathsResponse> {
  return requestJson<PathsResponse>('/paths')
}

export function updatePaths(paths: Record<string, string[]>): Promise<PathsUpdateResponse> {
  invalidateJsonCache('/paths')
  invalidateModelCatalogCaches()
  return requestJson<PathsUpdateResponse>('/paths', { method: 'POST', body: JSON.stringify({ paths }) })
}

function invalidateModelPathCaches(): void {
  invalidateJsonCache('/paths')
  invalidateModelCatalogCaches()
}

export function scanModelPath(payload: ModelPathScanRequest): Promise<ModelPathScanResponse> {
  return requestJson<ModelPathScanResponse>('/models/path-scan', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function addModelPathItem(payload: ModelPathAddRequest): Promise<ModelPathAddResponse> {
  invalidateModelPathCaches()
  return requestJson<ModelPathAddResponse>('/models/path-add', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function addModelPathItemsAll(payload: ModelPathAddRequest): Promise<ModelPathAddAllResponse> {
  invalidateModelPathCaches()
  return requestJson<ModelPathAddAllResponse>('/models/path-add-all', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchSettingsSchema(): Promise<SettingsSchemaResponse> {
  return requestJson<SettingsSchemaResponse>('/settings/schema')
}

export function fetchUiBlocks(tab?: string): Promise<UiBlocksResponse> {
  const q = tab ? `?tab=${encodeURIComponent(tab)}` : ''
  return requestJsonCached<UiBlocksResponse>(`/ui/blocks${q}`)
}

export function fetchUiPresets(tab?: string): Promise<UiPresetsResponse> {
  const q = tab ? `?tab=${encodeURIComponent(tab)}` : ''
  return requestJsonCached<UiPresetsResponse>(`/ui/presets${q}`)
}

export function applyUiPreset(id: string, tab: string): Promise<UiPresetApplyResponse> {
  return requestJson<UiPresetApplyResponse>('/ui/presets/apply', {
    method: 'POST',
    body: JSON.stringify({ id, tab }),
  })
}

// Tabs/workflows persistence
import type {
  TabsResponse,
  ApiTab,
  WorkflowCreateResponse,
  WorkflowDeleteResponse,
  WorkflowUpdateResponse,
  WorkflowsResponse,
} from './types'

export function fetchTabs(): Promise<TabsResponse> {
  return requestJsonCached<TabsResponse>('/ui/tabs')
}

export function createTabApi(payload: Partial<ApiTab> & { type: ApiTab['type']; title?: string; params?: Record<string, unknown> }): Promise<{ id: string }> {
  invalidateJsonCache('/ui/tabs')
  return requestJson<{ id: string }>('/ui/tabs', { method: 'POST', body: JSON.stringify(payload) })
}

export function updateTabApi(tabId: string, payload: Partial<Pick<ApiTab, 'title' | 'enabled' | 'params'>>): Promise<{ updated: string }> {
  invalidateJsonCache('/ui/tabs')
  return requestJson<{ updated: string }>(`/ui/tabs/${encodeURIComponent(tabId)}`, { method: 'PATCH', body: JSON.stringify(payload) })
}

export function reorderTabsApi(ids: string[]): Promise<{ ok: boolean }> {
  invalidateJsonCache('/ui/tabs')
  return requestJson<{ ok: boolean }>('/ui/tabs/reorder', { method: 'POST', body: JSON.stringify({ ids }) })
}

export function deleteTabApi(tabId: string): Promise<{ deleted: string }> {
  invalidateJsonCache('/ui/tabs')
  return requestJson<{ deleted: string }>(`/ui/tabs/${encodeURIComponent(tabId)}`, { method: 'DELETE' })
}

export function fetchWorkflows(): Promise<WorkflowsResponse> {
  return requestJson<WorkflowsResponse>('/ui/workflows')
}

export function createWorkflow(payload: { name: string; source_tab_id: string; type: ApiTab['type']; params_snapshot: Record<string, unknown> }): Promise<WorkflowCreateResponse> {
  return requestJson<WorkflowCreateResponse>('/ui/workflows', { method: 'POST', body: JSON.stringify(payload) })
}

export function updateWorkflow(id: string, payload: { name?: string; source_tab_id?: string; params_snapshot?: Record<string, unknown> }): Promise<WorkflowUpdateResponse> {
  return requestJson<WorkflowUpdateResponse>(`/ui/workflows/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(payload) })
}

export function deleteWorkflow(id: string): Promise<WorkflowDeleteResponse> {
  return requestJson<WorkflowDeleteResponse>(`/ui/workflows/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export function loadModelsForTab(tabId: string): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>('/models/load', { method: 'POST', body: JSON.stringify({ tab_id: tabId }) })
}

export function unloadModelsForTab(tabId: string): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>('/models/unload', { method: 'POST', body: JSON.stringify({ tab_id: tabId }) })
}
