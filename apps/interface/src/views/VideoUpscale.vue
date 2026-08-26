<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dedicated SeedVR2 video-upscale utility route.
Lets users select a backend-visible source-video path without uploading video bytes through the browser, configure curated SeedVR2 options,
run the task through the shared task/SSE contract, and play, open, zoom, or download the exported MP4 result with truthful audio status.

Symbols (top-level; keep in sync; no ghosts):
- `VideoUpscale` (component): Dedicated SeedVR2 source-video upscale workspace.
- `start` (function): Starts one `POST /api/video-upscale` task and opens its SSE stream.
- `cancel` (function): Requests immediate cancellation for the current task.
- `handleTaskEvent` (function): Applies the shared task/SSE event contract to route-local state.
- `openBrowser` / `loadBrowserPath` / `openBrowserItem` (functions): Operate the narrow backend file-browser integration for source-video paths.
- `seedvr2DeviceSupported` (computed): Gates the run action to CUDA or MPS, the supported SeedVR2 backends.
- `toOutputUrl` (function): Converts a task output relative path into the existing output-file URL.
-->

<template>
  <section class="panels video-panels cdx-video-upscale">
    <div class="panel-stack">
      <div class="panel cdx-video-upscale__source-panel">
        <div class="panel-header cdx-video-upscale__panel-header">
          <div>
            <p class="cdx-video-upscale__eyebrow">Dedicated video utility</p>
            <h1 class="cdx-video-upscale__title">SeedVR2 Video Upscale</h1>
          </div>
          <span class="cdx-video-upscale__badge">SeedVR2</span>
        </div>
        <div class="panel-body">
          <p class="caption cdx-video-upscale__intro">
            Choose a video path that the backend can read directly. This page never uploads the source video through the browser.
          </p>

          <div class="panel-section cdx-video-upscale__source-field">
            <label class="label-muted" for="video-upscale-source-path">Source video path</label>
            <div class="cdx-video-upscale__path-row">
              <input
                id="video-upscale-source-path"
                v-model="videoPath"
                class="ui-input cdx-video-upscale__path-input"
                type="text"
                :disabled="isRunning"
                placeholder="/media/source-video.mp4"
                @keyup.enter="start"
              />
              <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="openBrowser">Browse server files</button>
            </div>
            <p class="caption">The backend validates the selected file before decoding. Remote URLs and browser uploads are not accepted.</p>
          </div>

          <div class="panel-section">
            <div class="cdx-video-upscale__section-heading">
              <div>
                <p class="label-muted">SeedVR2 profile</p>
                <p class="caption">Curated controls for a predictable upscale pass.</p>
              </div>
            </div>

            <div class="cdx-video-upscale__settings-grid">
              <div class="cdx-video-upscale__field">
                <label class="label-muted" for="video-upscale-model">Model</label>
                <select id="video-upscale-model" v-model="ditModel" class="select-md" :disabled="isRunning">
                  <option value="seedvr2_ema_3b_fp16.safetensors">SeedVR2 EMA 3B FP16</option>
                  <option value="seedvr2_ema_7b_fp16.safetensors">SeedVR2 EMA 7B FP16</option>
                  <option value="seedvr2_ema_7b_sharp_fp16.safetensors">SeedVR2 EMA 7B Sharp FP16</option>
                </select>
              </div>

              <div class="cdx-video-upscale__field">
                <label class="label-muted" for="video-upscale-correction">Color correction</label>
                <select id="video-upscale-correction" v-model="colorCorrection" class="select-md" :disabled="isRunning">
                  <option value="lab">LAB</option>
                  <option value="wavelet">Wavelet</option>
                  <option value="wavelet_adaptive">Wavelet adaptive</option>
                  <option value="hsv">HSV</option>
                  <option value="adain">AdaIN</option>
                  <option value="none">None</option>
                </select>
              </div>

              <SliderField
                label="Target resolution"
                :modelValue="resolution"
                :min="16"
                :max="4096"
                :step="16"
                :inputStep="16"
                :disabled="isRunning"
                inputClass="cdx-input-w-md"
                tooltip="Target long edge in pixels. SeedVR2 keeps the source aspect ratio."
                @update:modelValue="resolution = $event"
              />

              <SliderField
                label="Maximum resolution"
                :modelValue="maxResolution"
                :min="0"
                :max="8192"
                :step="16"
                :inputStep="16"
                :disabled="isRunning"
                inputClass="cdx-input-w-md"
                tooltip="Use 0 to leave the target resolution uncapped."
                @update:modelValue="maxResolution = $event"
              />

              <div class="cdx-video-upscale__field">
                <label class="label-muted" for="video-upscale-batch">Batch size</label>
                <select id="video-upscale-batch" v-model.number="batchSize" class="select-md" :disabled="isRunning">
                  <option :value="1">1 frame</option>
                  <option :value="5">5 frames</option>
                  <option :value="9">9 frames</option>
                  <option :value="13">13 frames</option>
                  <option :value="17">17 frames</option>
                  <option :value="33">33 frames</option>
                  <option :value="65">65 frames</option>
                  <option :value="129">129 frames</option>
                </select>
                <p class="caption">Curated values satisfy SeedVR2’s 4n+1 frame contract.</p>
              </div>

              <label class="cdx-video-upscale__toggle-field">
                <input v-model="uniformBatchSize" type="checkbox" :disabled="isRunning" />
                <span>
                  <strong>Uniform batches</strong>
                  <small>Use one batch size across the complete source.</small>
                </span>
              </label>

              <SliderField
                label="Temporal overlap"
                :modelValue="temporalOverlap"
                :min="0"
                :max="128"
                :step="1"
                :disabled="isRunning"
                inputClass="cdx-input-w-md"
                @update:modelValue="temporalOverlap = $event"
              />

              <SliderField
                label="Prepend frames"
                :modelValue="prependFrames"
                :min="0"
                :max="128"
                :step="1"
                :disabled="isRunning"
                inputClass="cdx-input-w-md"
                @update:modelValue="prependFrames = $event"
              />

              <SliderField
                label="Input noise"
                :modelValue="inputNoiseScale"
                :min="0"
                :max="1"
                :step="0.01"
                :inputStep="0.01"
                :disabled="isRunning"
                inputClass="cdx-input-w-md"
                @update:modelValue="inputNoiseScale = $event"
              />

              <SliderField
                label="Latent noise"
                :modelValue="latentNoiseScale"
                :min="0"
                :max="1"
                :step="0.01"
                :inputStep="0.01"
                :disabled="isRunning"
                inputClass="cdx-input-w-md"
                @update:modelValue="latentNoiseScale = $event"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="panel-stack panel-stack--sticky">
      <RunCard
        title="Run"
        :showBatchControls="false"
        :generateDisabled="!canRun"
        :isRunning="isRunning"
        :disabled="isRunning"
        generateLabel="Upscale video"
        runningLabel="Upscaling video…"
        :generateTitle="generateTitle"
        @generate="start"
        @cancel="cancel"
      >
        <RunProgressStatus
          v-if="errorMessage"
          variant="error"
          title="Video upscale failed"
          :message="errorMessage"
          :show-progress-bar="false"
        />
        <RunProgressStatus
          v-else-if="isRunning"
          :stage="progress?.stage || 'starting'"
          :message="progress?.message || ''"
          :percent="progress?.percent ?? null"
          :step="progress?.step ?? null"
          :total-steps="progress?.totalSteps ?? null"
          :eta-seconds="progress?.etaSeconds ?? null"
        />
        <RunProgressStatus
          v-else-if="outputUrl"
          variant="success"
          title="Video ready"
          :message="audioStatus"
          :show-progress-bar="false"
        />
        <RunProgressStatus
          v-if="notice"
          variant="info"
          title="Notice"
          :message="notice"
          :show-progress-bar="false"
        />
      </RunCard>

      <ResultsCard
        title="Results"
        :showGenerate="false"
        headerClass="three-cols results-sticky"
        headerRightClass="results-actions"
      >
        <template #header-right>
          <button v-if="outputUrl" class="btn btn-sm btn-outline" type="button" @click="videoZoomOpen = true">Zoom</button>
          <a v-if="outputUrl" class="btn btn-sm btn-outline" :href="outputUrl" target="_blank" rel="noreferrer">Open</a>
          <a v-if="outputUrl" class="btn btn-sm btn-secondary" :href="outputUrl" download>Download</a>
        </template>

        <div v-if="outputUrl" class="cdx-video-upscale__result">
          <video class="cdx-video-upscale__result-video" :src="outputUrl" controls @dblclick.prevent.stop />
          <div class="cdx-video-upscale__result-meta">
            <p class="cdx-video-upscale__audio-status">{{ audioStatus }}</p>
            <p v-if="resultDimensions" class="caption">{{ resultDimensions }}</p>
          </div>
        </div>
        <div v-else class="results-empty-state cdx-video-upscale__empty-state">
          <div class="results-empty-title">
            <template v-if="isRunning">Upscaling source video…</template>
            <template v-else>No upscaled video yet</template>
          </div>
          <div class="caption">The exported MP4 will appear here with playback, open, zoom, and download actions.</div>
        </div>

        <details v-if="resultInfo" class="cdx-video-upscale__details">
          <summary>Task details</summary>
          <pre>{{ formatJson(resultInfo) }}</pre>
        </details>
      </ResultsCard>

      <VideoZoomOverlay
        :modelValue="videoZoomOpen"
        :src="outputUrl"
        aria-label="Zoomed SeedVR2 upscaled video"
        @update:modelValue="videoZoomOpen = $event"
      />
    </div>
  </section>

  <Modal v-model="browserOpen" title="Browse backend video files" panelClass="cdx-video-upscale__browser-modal">
    <p class="caption cdx-video-upscale__browser-copy">Select a file that is visible to the backend process. Directories only navigate; clicking a video file selects it.</p>
    <div class="cdx-video-upscale__browser-pathbar">
      <button class="btn btn-sm btn-outline" type="button" :disabled="browserLoading || !browserData.parent" @click="goToParent">Up</button>
      <input v-model="browserPath" class="ui-input" type="text" :disabled="browserLoading" @keyup.enter="loadBrowserPath" />
      <button class="btn btn-sm btn-secondary" type="button" :disabled="browserLoading" @click="loadBrowserPath">
        {{ browserLoading ? 'Loading…' : 'Go' }}
      </button>
    </div>
    <p v-if="browserError" class="cdx-video-upscale__browser-error">{{ browserError }}</p>
    <p v-else-if="!browserData.exists" class="caption">The path does not exist on the backend host.</p>
    <div v-else class="cdx-video-upscale__browser-list">
      <button
        v-for="item in browserData.items"
        :key="`${item.type}:${item.name}`"
        class="cdx-video-upscale__browser-item"
        :data-type="item.type"
        type="button"
        :disabled="browserLoading"
        @click="openBrowserItem(item)"
      >
        <span class="cdx-video-upscale__browser-item-icon" aria-hidden="true">{{ item.type === 'directory' ? '▸' : '•' }}</span>
        <span class="cdx-video-upscale__browser-item-name">{{ item.name }}</span>
        <span v-if="item.type === 'file' && item.size !== undefined" class="cdx-video-upscale__browser-item-size">{{ formatFileSize(item.size) }}</span>
      </button>
      <p v-if="browserData.items.length === 0" class="caption">No matching video files are available in this directory.</p>
    </div>
  </Modal>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { cancelTask, startVideoUpscale, subscribeTask } from '../api/client'
import type { SeedVR2ColorCorrection, SeedVR2DitModel, TaskEvent, VideoUpscaleRequest } from '../api/types'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { formatJson, useResultsCard } from '../composables/useResultsCard'
import Modal from '../components/ui/Modal.vue'
import SliderField from '../components/ui/SliderField.vue'
import VideoZoomOverlay from '../components/ui/VideoZoomOverlay.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunProgressStatus from '../components/results/RunProgressStatus.vue'

type BrowserItem = {
  name: string
  type: 'file' | 'directory'
  size?: number
}

type BrowserData = {
  path: string
  exists: boolean
  parent: string
  items: BrowserItem[]
}

type ProgressState = {
  stage: string
  message: string
  percent: number | null
  step: number | null
  totalSteps: number | null
  etaSeconds: number | null
}

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'
const VIDEO_EXTENSIONS = '.mp4,.mkv,.mov,.avi,.webm,.m4v,.mpeg,.mpg,.ts,.mts'

const quicksettings = useQuicksettingsStore()
const { notice, toast } = useResultsCard()

const videoPath = ref('')
const ditModel = ref<SeedVR2DitModel>('seedvr2_ema_3b_fp16.safetensors')
const resolution = ref(1080)
const maxResolution = ref(0)
const batchSize = ref(5)
const uniformBatchSize = ref(false)
const temporalOverlap = ref(0)
const prependFrames = ref(0)
const colorCorrection = ref<SeedVR2ColorCorrection>('lab')
const inputNoiseScale = ref(0)
const latentNoiseScale = ref(0)

const taskId = ref('')
const isRunning = ref(false)
const errorMessage = ref('')
const progress = ref<ProgressState | null>(null)
const resultInfo = ref<unknown>(null)
const outputUrl = ref('')
const videoZoomOpen = ref(false)

const browserOpen = ref(false)
const browserLoading = ref(false)
const browserPath = ref('')
const browserError = ref('')
const browserData = ref<BrowserData>({ path: '', exists: false, parent: '', items: [] })

let unsubscribeTask: (() => void) | null = null

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function toOutputUrl(relPath: string): string {
  const normalized = String(relPath || '').replace(/\\+/g, '/').replace(/^\/+/, '')
  if (!normalized) throw new Error('Video-upscale task returned an empty output path.')
  const encodedPath = normalized.split('/').map((part) => encodeURIComponent(part)).join('/')
  return `${API_BASE}/output/${encodedPath}`
}

function joinPath(directory: string, name: string): string {
  const normalizedDirectory = String(directory || '').trim()
  if (!normalizedDirectory) return name
  const separator = normalizedDirectory.includes('\\') && !normalizedDirectory.includes('/') ? '\\' : '/'
  return `${normalizedDirectory.replace(/[\\/]+$/, '')}${separator}${name}`
}

function formatFileSize(rawSize: number): string {
  const size = Number(rawSize)
  if (!Number.isFinite(size) || size < 0) return ''
  if (size < 1024) return `${Math.trunc(size)} B`
  if (size < 1024 ** 2) return `${(size / 1024).toFixed(1)} KiB`
  if (size < 1024 ** 3) return `${(size / (1024 ** 2)).toFixed(1)} MiB`
  return `${(size / (1024 ** 3)).toFixed(2)} GiB`
}

function stopTaskStream(): void {
  if (unsubscribeTask) unsubscribeTask()
  unsubscribeTask = null
}

const currentDevice = computed(() => String(quicksettings.currentDevice || '').trim().toLowerCase())
const seedvr2DeviceSupported = computed(() => currentDevice.value === 'cuda' || currentDevice.value === 'mps')
const canRun = computed(() => (
  !isRunning.value
  && Boolean(videoPath.value.trim())
  && Boolean(currentDevice.value)
  && seedvr2DeviceSupported.value
))
const generateTitle = computed(() => {
  if (isRunning.value) return 'A video-upscale task is already running.'
  if (!videoPath.value.trim()) return 'Choose a backend-visible source video path.'
  if (!currentDevice.value) return 'Choose a device in QuickSettings.'
  if (!seedvr2DeviceSupported.value) return 'SeedVR2 video upscaling requires CUDA or MPS.'
  return 'Start SeedVR2 video upscaling.'
})

const audioStatus = computed(() => {
  if (!isRecord(resultInfo.value) || !isRecord(resultInfo.value.audio)) return 'Audio status unavailable.'
  const audio = resultInfo.value.audio
  if (audio.source_has_audio === true && audio.preserved === true) return 'Source audio preserved.'
  if (audio.source_has_audio === false) return 'Source had no audio stream.'
  return 'Audio preservation verification is incomplete.'
})

const resultDimensions = computed(() => {
  if (!isRecord(resultInfo.value) || !isRecord(resultInfo.value.output)) return ''
  const output = resultInfo.value.output
  const width = typeof output.width === 'number' ? output.width : null
  const height = typeof output.height === 'number' ? output.height : null
  const fps = typeof output.fps === 'number' ? output.fps : null
  const frames = typeof output.frames === 'number' ? output.frames : null
  const parts: string[] = []
  if (width !== null && height !== null) parts.push(`${width} × ${height}`)
  if (fps !== null) parts.push(`${fps} fps`)
  if (frames !== null) parts.push(`${frames} frames`)
  return parts.join(' · ')
})

watch(outputUrl, (currentOutputUrl) => {
  if (!currentOutputUrl) videoZoomOpen.value = false
})

function handleTaskEvent(event: TaskEvent): void {
  switch (event.type) {
    case 'status':
      progress.value = {
        stage: event.stage,
        message: '',
        percent: null,
        step: null,
        totalSteps: null,
        etaSeconds: null,
      }
      break
    case 'progress':
      progress.value = {
        stage: event.stage,
        message: String(event.message || ''),
        percent: event.percent ?? null,
        step: event.step ?? null,
        totalSteps: event.total_steps ?? null,
        etaSeconds: event.eta_seconds ?? null,
      }
      break
    case 'result': {
      const relPath = typeof event.video?.rel_path === 'string' ? event.video.rel_path : ''
      if (!relPath) {
        errorMessage.value = 'Video-upscale task completed without a saved video artifact.'
        isRunning.value = false
        stopTaskStream()
        return
      }
      try {
        outputUrl.value = toOutputUrl(relPath)
        resultInfo.value = event.info
      } catch (error) {
        errorMessage.value = error instanceof Error ? error.message : String(error)
        isRunning.value = false
        stopTaskStream()
      }
      break
    }
    case 'error':
      errorMessage.value = event.message
      isRunning.value = false
      stopTaskStream()
      break
    case 'end':
      isRunning.value = false
      stopTaskStream()
      break
    case 'gap':
      toast('Task event replay gap detected. The live stream continues from the current state.')
      break
    case 'automation_iteration':
      break
  }
}

async function start(): Promise<void> {
  if (!canRun.value) return

  stopTaskStream()
  errorMessage.value = ''
  progress.value = null
  resultInfo.value = null
  outputUrl.value = ''
  videoZoomOpen.value = false
  taskId.value = ''

  const payload: VideoUpscaleRequest = {
    video_path: videoPath.value.trim(),
    device: currentDevice.value,
    dit_model: ditModel.value,
    resolution: resolution.value,
    max_resolution: maxResolution.value,
    batch_size: batchSize.value,
    uniform_batch_size: uniformBatchSize.value,
    temporal_overlap: temporalOverlap.value,
    prepend_frames: prependFrames.value,
    color_correction: colorCorrection.value,
    input_noise_scale: inputNoiseScale.value,
    latent_noise_scale: latentNoiseScale.value,
  }

  isRunning.value = true
  try {
    const response = await startVideoUpscale(payload)
    if (!response.task_id.trim()) throw new Error('Video-upscale task did not return a task id.')
    taskId.value = response.task_id
    unsubscribeTask = subscribeTask(taskId.value, handleTaskEvent, (error) => {
      if (error) console.warn('[video-upscale] task stream error', error)
    })
  } catch (error) {
    isRunning.value = false
    errorMessage.value = error instanceof Error ? error.message : String(error)
  }
}

async function cancel(): Promise<void> {
  if (!taskId.value) return
  try {
    await cancelTask(taskId.value, 'immediate')
    toast('Cancellation requested.')
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function openBrowser(): void {
  browserOpen.value = true
  browserError.value = ''
  browserPath.value = videoPath.value.trim()
  void loadBrowserPath()
}

async function loadBrowserPath(): Promise<void> {
  browserLoading.value = true
  browserError.value = ''
  try {
    const query = new URLSearchParams({
      path: browserPath.value.trim(),
      extensions: VIDEO_EXTENSIONS,
    })
    const response = await fetch(`${API_BASE}/tools/browse-files?${query.toString()}`)
    const payload = await response.json().catch(() => ({}))
    if (!response.ok) {
      const detail = isRecord(payload) && typeof payload.detail === 'string' ? payload.detail : `HTTP ${response.status}`
      throw new Error(detail)
    }
    if (!isRecord(payload) || !Array.isArray(payload.items)) {
      throw new Error('Backend file browser returned an invalid response.')
    }
    browserData.value = {
      path: typeof payload.path === 'string' ? payload.path : browserPath.value.trim(),
      exists: payload.exists === true,
      parent: typeof payload.parent === 'string' ? payload.parent : '',
      items: payload.items.flatMap((entry): BrowserItem[] => {
        if (!isRecord(entry)) return []
        const name = typeof entry.name === 'string' ? entry.name : ''
        const type = entry.type === 'file' || entry.type === 'directory' ? entry.type : null
        if (!name || !type) return []
        const size = typeof entry.size === 'number' && Number.isFinite(entry.size) ? entry.size : undefined
        return [{ name, type, size }]
      }),
    }
    browserPath.value = browserData.value.path
  } catch (error) {
    browserData.value = { path: browserPath.value.trim(), exists: false, parent: '', items: [] }
    browserError.value = error instanceof Error ? error.message : String(error)
  } finally {
    browserLoading.value = false
  }
}

function goToParent(): void {
  if (!browserData.value.parent) return
  browserPath.value = browserData.value.parent
  void loadBrowserPath()
}

function openBrowserItem(item: BrowserItem): void {
  if (item.type === 'directory') {
    browserPath.value = joinPath(browserData.value.path, item.name)
    void loadBrowserPath()
    return
  }
  videoPath.value = joinPath(browserData.value.path, item.name)
  browserOpen.value = false
}

onBeforeUnmount(() => {
  stopTaskStream()
})
</script>
