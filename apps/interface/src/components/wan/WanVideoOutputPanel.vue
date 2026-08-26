<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN video output options panel.
Configures export format/pixel format/CRF/loop options, compact output toggles (`pingpong`, `returnFrames`), and RIFE interpolation target FPS (`0=off`) for WAN video tasks.

Symbols (top-level; keep in sync; no ghosts):
- `WanVideoOutputPanel` (component): Video output settings panel used by WANTab.
- `updateVideo` (function): Emits a partial `video` patch (`update:video`).
- `normalizeNonNegativeInteger` (function): Parses/clamps integer inputs to non-negative range with optional max.
- `normalizeInterpolationTargetFps` (function): Normalizes interpolation target FPS (`0` disables interpolation).
- `onLoopCountChange` (function): Applies normalized loop-count updates.
- `onCrfChange` (function): Applies normalized CRF updates.
- `onInterpolationTargetFpsChange` (function): Applies normalized interpolation target FPS updates.
- `interpolationCaption` (computed): User-facing interpolation summary (`Off`/`Target`) with effective backend output FPS.
-->

<template>
  <div :class="['gen-card', { 'gen-card--embedded': embedded }]">
    <div v-if="!embedded" class="row-split">
      <span class="label-muted">Video Output</span>
    </div>

    <div class="gc-row">
      <div class="gc-col">
        <label class="label-muted">Format</label>
        <select class="select-md" :disabled="disabled" :value="video.format" @change="updateVideo({ format: ($event.target as HTMLSelectElement).value })">
          <option value="video/h264-mp4">H.264 MP4</option>
          <option value="video/h265-mp4">H.265 MP4</option>
          <option value="video/webm">WebM</option>
          <option value="image/gif">GIF</option>
        </select>
      </div>
      <div class="gc-col">
        <label class="label-muted">Pixel Format</label>
        <select class="select-md" :disabled="disabled" :value="video.pixFmt" @change="updateVideo({ pixFmt: ($event.target as HTMLSelectElement).value })">
          <option value="yuv420p">yuv420p</option>
          <option value="yuv444p">yuv444p</option>
          <option value="yuv422p">yuv422p</option>
        </select>
      </div>
    </div>

    <div class="gc-row">
      <SliderField
        class="gc-col gc-col--compact"
        label="Loop Count"
        :modelValue="video.loopCount"
        :min="0"
        :max="32"
        :step="1"
        :disabled="disabled"
        inputClass="cdx-input-w-md"
        @update:modelValue="onLoopCountChange"
      />
      <SliderField
        class="gc-col gc-col--compact"
        label="CRF"
        :modelValue="video.crf"
        :min="0"
        :max="51"
        :step="1"
        :disabled="disabled"
        inputClass="cdx-input-w-md"
        @update:modelValue="onCrfChange"
      />
      <SliderField
        class="gc-col gc-col--compact"
        label="Interpolation (RIFE)"
        :modelValue="video.interpolationFps"
        :min="0"
        :max="240"
        :step="1"
        :disabled="disabled"
        inputClass="cdx-input-w-md"
        @update:modelValue="onInterpolationTargetFpsChange"
      >
        <template #below>
          <span class="caption">{{ interpolationCaption }}</span>
        </template>
      </SliderField>
      <div class="gc-col gc-col--presets wan-video-output-toggle-row">
        <button
          :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.pingpong ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :disabled="disabled"
          :aria-pressed="video.pingpong"
          @click="updateVideo({ pingpong: !video.pingpong })"
        >
          Ping-pong
        </button>
        <button
          :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.returnFrames ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :disabled="disabled"
          :aria-pressed="video.returnFrames"
          @click="updateVideo({ returnFrames: !video.returnFrames })"
        >
          Return frames
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import SliderField from '../ui/SliderField.vue'
import type { WanVideoParams } from '../../stores/model_tabs'

const props = withDefaults(defineProps<{
  video: WanVideoParams
  embedded?: boolean
  disabled?: boolean
}>(), {
  embedded: false,
  disabled: false,
})

const emit = defineEmits<{
  (e: 'update:video', patch: Partial<WanVideoParams>): void
}>()

function updateVideo(patch: Partial<WanVideoParams>): void {
  emit('update:video', patch)
}

function normalizeNonNegativeInteger(value: unknown, fallback: number, max?: number): number {
  const fallbackInt = Number.isFinite(fallback) ? Math.max(0, Math.trunc(fallback)) : 0
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return fallbackInt
  const parsed = Math.max(0, Math.trunc(numeric))
  if (typeof max === 'number' && Number.isFinite(max)) {
    return Math.min(Math.max(0, Math.trunc(max)), parsed)
  }
  return parsed
}

const MAX_INTERPOLATION_FPS = 240

function normalizeInterpolationTargetFps(value: unknown, fallback: number): number {
  const fallbackValue = Number.isFinite(fallback) ? Math.trunc(fallback) : 0
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return Math.max(0, Math.min(MAX_INTERPOLATION_FPS, fallbackValue))
  return Math.max(0, Math.min(MAX_INTERPOLATION_FPS, Math.trunc(numeric)))
}

function onLoopCountChange(value: number): void {
  updateVideo({ loopCount: normalizeNonNegativeInteger(value, props.video.loopCount, 32) })
}

function onCrfChange(value: number): void {
  updateVideo({ crf: normalizeNonNegativeInteger(value, props.video.crf, 51) })
}

function onInterpolationTargetFpsChange(value: number): void {
  updateVideo({
    interpolationFps: normalizeInterpolationTargetFps(value, props.video.interpolationFps),
  })
}

const interpolationCaption = computed<string>(() => {
  const targetFps = normalizeInterpolationTargetFps(
    props.video.interpolationFps,
    0,
  )
  const baseFps = normalizeNonNegativeInteger(props.video.fps, 0, MAX_INTERPOLATION_FPS)
  if (targetFps <= 0) {
    return baseFps > 0 ? `Off · Output: ${baseFps} fps` : 'Off'
  }
  if (baseFps <= 0) return `Target: ${targetFps} fps`
  if (targetFps <= baseFps) return `Disabled · Target (${targetFps} fps) <= base (${baseFps} fps)`
  const times = Math.max(2, Math.ceil(targetFps / baseFps))
  const outputFps = baseFps * times
  return `Target: ${targetFps} fps · Output: ${outputFps} fps`
})

</script>
