/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Frontend-driven XYZ sweep store for image tabs.
Builds parameter grid combos, enqueues jobs, starts txt2img tasks (including required `settings_revision`), streams task events, and supports stop modes/cancellation while collecting
per-cell results. Hires upscaler values are stable ids (`latent:*` / `spandrel:*`) for hires-fix wiring; hires tile prefs (fallback/min_tile) are propagated from the shared upscalers store.
Preflight now fails loud when VAE selection is empty before queuing XYZ requests, and queued txt2img payloads reuse the shared image request contract helper so the sweep lane emits the
same explicit checkpoint/VAE selectors (`model_sha`, `checkpoint_core_only`, `model_format`, `vae_source`), FLUX.2 guidance mode, and asset-contract-backed extras as the main image generation lane.
Qwen Image Edit-2511 is excluded from XYZ owner selection and fails before payload construction because it has no txt2img surface.
The standalone `/xyz` route now pins itself to a compatible image-tab owner (active image tab, then most recently updated image tab, else a new `sdxl` tab) instead of baselining from generic active-tab state.
Baseline sampler/scheduler resolution validates current params or backend capability defaults against executable sampler/scheduler catalogs before queuing requests.

Symbols (top-level; keep in sync; no ghosts):
- `Status` (type): XYZ sweep lifecycle status (`idle`/`running`/`stopped`/`error`/`done`).
- `StopMode` (type): Stop behavior for a running sweep (`immediate` vs `after_current`).
- `XyzJob` (interface): Internal job record for each cell (payload/task id/status/result/error).
- `CompatibleImageTab` (type): Image-tab owner shape accepted by the sweep payload prebuilder.
- `buildXyzBaseForm` / `applyXyzAxis` / `buildXyzTxt2ImgPayload` (function): Exported payload prebuild path used by store runs and focused contract probes.
- `ensureBaselineImageTab` (function): Resolves the owner image tab for `/xyz` with deterministic fallback.
- `useXyzStore` (store): Pinia store for XYZ sweeps; builds combos, runs jobs, subscribes to task SSE, and writes results into cells.
- `enabled`/`xEnabled`/`yEnabled`/`zEnabled` (store refs): Master and per-axis toggles used by the embedded XYZ card + Run integration.
- `XyzStore` (type): Convenience return type alias for `useXyzStore`.
*/

// tags: xyz, store, sweeps
import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'

import { cancelTask, fetchSamplers, fetchSchedulers, startTxt2Img, subscribeTask } from '../api/client'
import { buildTxt2ImgPayload } from '../api/payloads'
import type { Txt2ImgFormState, Txt2ImgRequest } from '../api/payloads'
import type { GeneratedImage, TaskEvent } from '../api/types'
import { buildExplicitImageRequestContract } from '../utils/image_request_contract'
import { AXIS_OPTIONS, buildCombos, labelOf, parseAxisValues, type AxisParam, type AxisValue, type XyzCell } from '../utils/xyz'
import { useModelTabsStore, type ImageBaseParams, type ImageTabType, type TabByType } from './model_tabs'
import { normalizeSamplerSchedulerSelection, useEngineCapabilitiesStore, type SamplingDefaults } from './engine_capabilities'
import { useQuicksettingsStore } from './quicksettings'
import { useUpscalersStore } from './upscalers'
import { isWanTabFamily, normalizeTabFamily, resolveImageRequestEngineId } from '../utils/engine_taxonomy'

type Status = 'idle' | 'running' | 'stopped' | 'error' | 'done'
type StopMode = 'immediate' | 'after_current'
const QWEN_IMAGE_ENGINE_ID = 'qwen_image'

interface XyzJob {
  id: string
  combo: { x: AxisValue; y: AxisValue | null; z: AxisValue | null }
  payload: Txt2ImgRequest
  status: 'queued' | 'running' | 'done' | 'error' | 'stopped'
  taskId?: string
  image?: GeneratedImage
  info?: unknown
  error?: string
}

export type CompatibleImageTab = TabByType<ImageTabType>

export interface XyzAxisApplication {
  enabled: boolean
  param: AxisParam
  value: AxisValue | null
}

export interface BuildXyzTxt2ImgPayloadInput {
  tab: CompatibleImageTab
  samplingDefaults: SamplingDefaults
  axes?: readonly XyzAxisApplication[]
  hiresFallbackOnOom?: boolean
  hiresMinTile?: number
}

export function buildXyzBaseForm(tab: CompatibleImageTab, samplingDefaults: SamplingDefaults): Txt2ImgFormState {
  const quick = useQuicksettingsStore()
  const caps = useEngineCapabilitiesStore()
  const params = tab.params as ImageBaseParams
  const tabFamily = tab.type
  const engineKey = resolveImageRequestEngineId(tabFamily, false)
  if (engineKey === QWEN_IMAGE_ENGINE_ID) {
    throw new Error('Qwen Image Edit-2511 does not support XYZ.')
  }
  const checkpoint = String(params.checkpoint || '').trim()
  const modelLabel = checkpoint || quick.currentModel
  const textEncoders = Array.isArray(params.textEncoders)
    ? params.textEncoders
        .map((value: unknown) => String(value || '').trim())
        .filter((value: string) => value.length > 0)
    : []
  const requestContract = buildExplicitImageRequestContract({
    modelLabel,
    engineKey,
    textEncoderLabels: textEncoders,
    selectedVaeLabel: quick.getVaeForFamily(tabFamily),
    zimageTurbo: engineKey === 'zimage'
      ? Boolean(params.zimageTurbo ?? true)
      : false,
    resolvers: {
      requireModelInfo: quick.requireModelInfo,
      resolveFlux2CheckpointVariant: quick.resolveFlux2CheckpointVariant,
      resolveTextEncoderSha: quick.resolveTextEncoderSha,
      resolveTextEncoderSlot: quick.resolveTextEncoderSlot,
      requireVaeSelection: quick.requireVaeSelection,
      resolveVaeSha: quick.resolveVaeSha,
      getAssetContract: caps.getAssetContract,
    },
  })

  return {
    prompt: params.prompt ?? '',
    negativePrompt: params.negativePrompt ?? '',
    width: params.width ?? 1024,
    height: params.height ?? 1024,
    steps: params.steps ?? 30,
    guidanceScale: params.cfgScale ?? 7,
    sampler: samplingDefaults.sampler,
    scheduler: samplingDefaults.scheduler,
    seed: params.seed ?? -1,
    clipSkip: params.clipSkip ?? 0,
    batchSize: 1,
    batchCount: 1,
    styles: [],
    device: quick.currentDevice as Txt2ImgFormState['device'],
    settingsRevision: quick.getSettingsRevision(),
    engine: engineKey,
    model: modelLabel,
    guidanceMode: requestContract.guidanceMode,
    swapModel: { ...params.swapModel },
    hires: params.hires
      ? {
          ...params.hires,
          tile: { ...params.hires.tile },
          swapModel: params.hires.swapModel ? { ...params.hires.swapModel } : undefined,
          refiner: params.hires.refiner ? { ...params.hires.refiner } : params.hires.refiner,
        }
      : undefined,
    refiner: params.refiner ? { ...params.refiner } : undefined,
    extras: { ...requestContract.extras },
  }
}

export function applyXyzAxis(form: Txt2ImgFormState, param: AxisParam, value: AxisValue): void {
  switch (param) {
    case 'prompt':
      form.prompt = String(value)
      break
    case 'negative':
      form.negativePrompt = String(value)
      break
    case 'cfg':
      form.guidanceScale = Number(value)
      break
    case 'steps':
      form.steps = Number(value)
      break
    case 'sampler':
      form.sampler = String(value)
      break
    case 'scheduler':
      form.scheduler = String(value)
      break
    case 'seed':
      form.seed = Number(value)
      break
    case 'width':
      form.width = Number(value)
      break
    case 'height':
      form.height = Number(value)
      break
    case 'hires_scale':
      form.hires = form.hires || { enabled: true, scale: 2.0, denoise: 0.4, steps: 0, resizeX: 0, resizeY: 0, upscaler: 'latent:bicubic-aa', tile: { tile: 256, overlap: 16 } }
      form.hires.enabled = true
      form.hires.scale = Number(value)
      break
    case 'hires_steps':
      form.hires = form.hires || { enabled: true, scale: 2.0, denoise: 0.4, steps: 0, resizeX: 0, resizeY: 0, upscaler: 'latent:bicubic-aa', tile: { tile: 256, overlap: 16 } }
      form.hires.enabled = true
      form.hires.steps = Number(value)
      break
    case 'refiner_model':
      form.refiner = form.refiner || { enabled: true, swapAtStep: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
      form.refiner.enabled = true
      form.refiner.model = String(value)
      break
    case 'refiner_steps':
      form.refiner = form.refiner || { enabled: true, swapAtStep: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
      form.refiner.enabled = true
      form.refiner.swapAtStep = Math.max(1, Math.trunc(Number(value)))
      break
    case 'refiner_cfg':
      form.refiner = form.refiner || { enabled: true, swapAtStep: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
      form.refiner.enabled = true
      form.refiner.cfg = Number(value)
      break
    default:
      break
  }
}

export function buildXyzTxt2ImgPayload(input: BuildXyzTxt2ImgPayloadInput): Txt2ImgRequest {
  const form = buildXyzBaseForm(input.tab, input.samplingDefaults)
  for (const axis of input.axes ?? []) {
    if (!axis.enabled || axis.value === null) continue
    applyXyzAxis(form, axis.param, axis.value)
  }
  return buildTxt2ImgPayload(form, {
    hiresFallbackOnOom: input.hiresFallbackOnOom,
    hiresMinTile: input.hiresMinTile,
  })
}

export const useXyzStore = defineStore('xyz', () => {
  const xParam = ref<AxisParam>('cfg')
  const yParam = ref<AxisParam>('steps')
  const zParam = ref<AxisParam>('sampler')
  const enabled = ref(false)
  const xEnabled = ref(true)
  const yEnabled = ref(true)
  const zEnabled = ref(true)

  const xValuesText = ref('6, 7, 8')
  const yValuesText = ref('20, 28')
  const zValuesText = ref('')

  const status = ref<Status>('idle')
  const errorMessage = ref('')
  const stopRequested = ref(false)
  const stopMode = ref<StopMode>('immediate')

  const progress = reactive({ total: 0, completed: 0, current: '' })
  const cells = ref<XyzCell[]>([])
  const jobs = ref<XyzJob[]>([])
  const activeTaskId = ref<string | null>(null)

  let unsubscribe: (() => void) | null = null

  const axisKind = (param: AxisParam): 'text' | 'number' => {
    return AXIS_OPTIONS.find((o) => o.id === param)?.kind ?? 'text'
  }

  const xParsedValues = computed<AxisValue[]>(() => parseAxisValues(xValuesText.value, axisKind(xParam.value)))
  const yParsedValues = computed<AxisValue[]>(() => parseAxisValues(yValuesText.value, axisKind(yParam.value)))
  const zParsedValues = computed<AxisValue[]>(() => parseAxisValues(zValuesText.value, axisKind(zParam.value)))

  const xValues = computed<AxisValue[]>(() => (xEnabled.value ? xParsedValues.value : ['(base)']))
  const yValues = computed<AxisValue[]>(() => (yEnabled.value ? yParsedValues.value : []))
  const zValues = computed<AxisValue[]>(() => (zEnabled.value ? zParsedValues.value : []))

  const combos = computed(() => buildCombos(xValues.value, yValues.value, zValues.value))

  const groupedByZ = computed(() => {
    const groups = new Map<string, XyzCell[]>()
    for (const cell of cells.value) {
      const key = labelOf(cell.z)
      const arr = groups.get(key) ?? []
      arr.push(cell)
      groups.set(key, arr)
    }
    return Array.from(groups.entries()).map(([label, rows]) => ({ label, rows }))
  })

  async function stop(mode: StopMode = 'immediate'): Promise<void> {
    stopRequested.value = true
    stopMode.value = mode
    if (unsubscribe) {
      unsubscribe()
      unsubscribe = null
    }
    const taskId = activeTaskId.value
    if (taskId && mode === 'immediate') {
      try { await cancelTask(taskId, 'immediate') } catch (err) { console.warn('[xyz] cancel failed', err) }
    }
    if (status.value === 'running') status.value = 'stopped'
  }

  function resetProgress(): void {
    progress.total = 0
    progress.completed = 0
    progress.current = ''
  }

  function resetStopState(): void {
    stopRequested.value = false
    stopMode.value = 'immediate'
  }

  function isCompatibleImageTab(tab: unknown): boolean {
    if (!tab || typeof tab !== 'object') return false
    const candidate = tab as { type?: unknown; params?: unknown }
    const family = normalizeTabFamily(candidate.type)
    if (!family || isWanTabFamily(family) || family === 'ltx2' || family === QWEN_IMAGE_ENGINE_ID) return false
    return Boolean(candidate.params && typeof candidate.params === 'object')
  }

  function updatedAtMs(tab: CompatibleImageTab): number {
    const raw = String(tab.meta?.updatedAt || '')
    const next = Date.parse(raw)
    return Number.isFinite(next) ? next : 0
  }

  async function ensureBaselineImageTab(): Promise<CompatibleImageTab> {
    const tabs = useModelTabsStore()
    await tabs.load()
    const active = tabs.activeTab
    if (active && isCompatibleImageTab(active)) {
      tabs.setActive(active.id)
      return active as unknown as CompatibleImageTab
    }
    const fallback = [...tabs.orderedTabs]
      .filter((tab) => isCompatibleImageTab(tab))
      .sort((left, right) => updatedAtMs(right as unknown as CompatibleImageTab) - updatedAtMs(left as unknown as CompatibleImageTab))[0] ?? null
    if (fallback) {
      tabs.setActive(fallback.id)
      return fallback as unknown as CompatibleImageTab
    }
    const createdId = await tabs.create('sdxl')
    const created = (tabs.tabs.find((tab) => tab.id === createdId && isCompatibleImageTab(tab)) as unknown as CompatibleImageTab | undefined) ?? null
    if (!created) {
      throw new Error(`Failed to create baseline image tab for /xyz: '${createdId}' not found after create.`)
    }
    tabs.setActive(created.id)
    return created
  }

  async function awaitResult(taskId: string): Promise<{ images: GeneratedImage[]; info?: unknown }> {
    return new Promise((resolve, reject) => {
      let result: { images: GeneratedImage[]; info?: unknown } | null = null
      unsubscribe = subscribeTask(
        taskId,
        (event: TaskEvent) => {
          if (event.type === 'result') {
            result = { images: Array.isArray(event.images) ? event.images : [], info: event.info }
          }
          if (event.type === 'error') {
            reject(new Error(event.message ?? 'Task failed'))
          }
          if (event.type === 'end') {
            if (result) resolve(result)
            else reject(new Error('Task ended without result'))
          }
        },
        (err) => reject(err instanceof Error ? err : new Error(String(err)))
      )
    })
  }

  async function run(): Promise<void> {
    const quick = useQuicksettingsStore()
    let baselineTab: CompatibleImageTab
    try {
      baselineTab = await ensureBaselineImageTab()
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : String(error)
      status.value = 'error'
      return
    }
    const params = baselineTab.params
    const caps = useEngineCapabilitiesStore()
    await caps.init()
    const engineKey = resolveImageRequestEngineId(baselineTab.type, false)
    const familyCapabilities = caps.getFamilyForEngine(engineKey)
    if (!familyCapabilities) {
      errorMessage.value = `Family capabilities for '${engineKey}' are not loaded.`
      status.value = 'error'
      return
    }

    if (!enabled.value) {
      errorMessage.value = 'Enable XYZ before running.'
      status.value = 'error'
      return
    }
    if (!xEnabled.value && !yEnabled.value && !zEnabled.value) {
      errorMessage.value = 'Enable at least one axis before running XYZ.'
      status.value = 'error'
      return
    }
    if (xEnabled.value && !xParsedValues.value.length) {
      errorMessage.value = 'X axis needs at least one value while enabled.'
      status.value = 'error'
      return
    }
    if (yEnabled.value && !yParsedValues.value.length) {
      errorMessage.value = 'Y axis needs at least one value while enabled.'
      status.value = 'error'
      return
    }
    if (zEnabled.value && !zParsedValues.value.length) {
      errorMessage.value = 'Z axis needs at least one value while enabled.'
      status.value = 'error'
      return
    }
    const samplerAxisEnabled =
      (xEnabled.value && xParam.value === 'sampler')
      || (yEnabled.value && yParam.value === 'sampler')
      || (zEnabled.value && zParam.value === 'sampler')
    const schedulerAxisEnabled =
      (xEnabled.value && xParam.value === 'scheduler')
      || (yEnabled.value && yParam.value === 'scheduler')
      || (zEnabled.value && zParam.value === 'scheduler')
    if (samplerAxisEnabled && schedulerAxisEnabled) {
      errorMessage.value = 'Sampler and Scheduler axes cannot be varied together in the same XYZ run.'
      status.value = 'error'
      return
    }
    if (!params?.prompt?.trim()) {
      errorMessage.value = 'Prompt must not be empty before running XYZ.'
      status.value = 'error'
      return
    }
    try {
      quick.requireVaeSelection(quick.getVaeForFamily(baselineTab.type))
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : String(error)
      status.value = 'error'
      return
    }
    const backendSamplingDefaults = caps.resolveSamplingDefaults(engineKey)
    let resolvedSampling: SamplingDefaults | null = null
    try {
      const [samplerResponse, schedulerResponse] = await Promise.all([fetchSamplers(), fetchSchedulers()])
      resolvedSampling = normalizeSamplerSchedulerSelection({
        samplers: samplerResponse.samplers,
        schedulers: schedulerResponse.schedulers,
        familyCapabilities,
        sampler: params.sampler,
        scheduler: params.scheduler,
        preferredSamplers: backendSamplingDefaults ? [backendSamplingDefaults.sampler] : [],
        preferredSchedulers: backendSamplingDefaults ? [backendSamplingDefaults.scheduler] : [],
      })
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : String(error)
      status.value = 'error'
      return
    }
    if (!resolvedSampling) {
      errorMessage.value = `XYZ requires a valid sampler and scheduler for '${engineKey}' before queuing requests. Select valid values or refresh backend capabilities.`
      status.value = 'error'
      return
    }

    errorMessage.value = ''
    resetStopState()
    resetProgress()

    const comboList = combos.value
    const upscalers = useUpscalersStore()
    const hiresFallbackOnOom = Boolean(upscalers.fallbackOnOom)
    const hiresMinTile = Number(upscalers.minTile)
    const nextCells: XyzCell[] = comboList.map((combo) => ({ x: combo.x, y: combo.y, z: combo.z, status: 'queued' }))
    const nextJobs: XyzJob[] = []

    // Pre-build job queue with payload snapshots
    for (const [index, combo] of comboList.entries()) {
      try {
        const payload = buildXyzTxt2ImgPayload({
          tab: baselineTab,
          samplingDefaults: resolvedSampling,
          axes: [
            { enabled: xEnabled.value, param: xParam.value, value: combo.x },
            { enabled: yEnabled.value, param: yParam.value, value: combo.y },
            { enabled: zEnabled.value, param: zParam.value, value: combo.z },
          ],
          hiresFallbackOnOom,
          hiresMinTile,
        })
        nextJobs.push({
          id: `job-${index + 1}`,
          combo: { x: combo.x, y: combo.y, z: combo.z },
          payload,
          status: 'queued',
        })
      } catch (err) {
        errorMessage.value = err instanceof Error ? err.message : String(err)
        status.value = 'error'
        jobs.value = []
        cells.value = []
        return
      }
    }

    jobs.value = nextJobs
    cells.value = nextCells
    progress.total = comboList.length
    progress.completed = 0
    status.value = 'running'

    for (let idx = 0; idx < jobs.value.length; idx++) {
      const job = jobs.value[idx]
      const cell = cells.value[idx]
      if (!job || !cell) continue

      if (stopRequested.value && stopMode.value === 'after_current') {
        job.status = 'stopped'
        cell.status = 'stopped'
        continue
      }
      if (stopRequested.value && stopMode.value === 'immediate') {
        job.status = 'stopped'
        cell.status = 'stopped'
        break
      }

      job.status = 'running'
      cell.status = 'running'
      const currentParts: string[] = []
      if (xEnabled.value) currentParts.push(labelOf(job.combo.x))
      if (yEnabled.value) currentParts.push(labelOf(job.combo.y))
      if (zEnabled.value) currentParts.push(labelOf(job.combo.z))
      progress.current = currentParts.join(' / ') || 'base'

      try {
        const { task_id } = await startTxt2Img(job.payload)
        job.taskId = task_id
        activeTaskId.value = task_id
        const result = await awaitResult(task_id)
        job.status = 'done'
        cell.status = 'done'
        job.image = result.images?.[0]
        cell.image = result.images?.[0]
        job.info = result.info
        cell.info = result.info
        progress.completed += 1
      } catch (err) {
        job.status = stopRequested.value ? 'stopped' : 'error'
        cell.status = job.status
        const msg = err instanceof Error ? err.message : String(err)
        job.error = msg
        cell.error = msg
        errorMessage.value = msg
        status.value = stopRequested.value ? 'stopped' : 'error'
        if (!stopRequested.value || stopMode.value === 'immediate') {
          break
        }
      } finally {
        activeTaskId.value = null
        if (unsubscribe) {
          unsubscribe()
          unsubscribe = null
        }
      }
    }

    if (status.value === 'running') {
      status.value = stopRequested.value ? 'stopped' : 'done'
    }
  }

  return {
    xParam,
    yParam,
    zParam,
    enabled,
    xEnabled,
    yEnabled,
    zEnabled,
    xValuesText,
    yValuesText,
    zValuesText,
    xValues,
    yValues,
    zValues,
    combos,
    groupedByZ,
    status,
    errorMessage,
    progress,
    cells,
    stopRequested,
    stopMode,
    run,
    stop,
  }
})

export type XyzStore = ReturnType<typeof useXyzStore>
