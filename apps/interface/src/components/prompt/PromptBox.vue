<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt input box with label + optional token count.
Wraps `PromptEditor` with a label and, when a truthful token-engine id is supplied, a backend-backed token counter
(`/api/models/prompt-token-count`) used by generation views.

Symbols (top-level; keep in sync; no ghosts):
- `PromptBox` (component): Prompt editor wrapper with label + token-count badge.
- `refreshTokenCount` (function): Debounced backend token-count refresh for the current prompt text.
-->

<template>
  <div class="prompt-box" :data-variant="variant">
    <div class="prompt-header">
      <span class="label-muted">{{ label }}</span>
    </div>
    <div class="prompt-editor-wrap">
      <span
        v-if="showTokenBadge"
        class="ear-badge"
        :title="tokenError || undefined"
      >
        {{ tokenError ? 'tok !' : `${tokenCount} tok` }}
      </span>
      <PromptEditor v-model="inner" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import { fetchPromptTokenCount } from '../../api/client'
import PromptEditor from './PromptEditor.vue'

const props = defineProps<{ label: string; modelValue: string; tokenEngine?: string; variant?: 'positive'|'negative' }>()
const emit = defineEmits<{ (e:'update:modelValue', v:string): void }>()

const inner = computed({
  get: () => props.modelValue || '',
  set: (v: string) => emit('update:modelValue', v),
})

const tokenCount = ref(0)
const tokenError = ref('')
const requestSeq = ref(0)
let debounceTimer: ReturnType<typeof setTimeout> | null = null

const showTokenBadge = computed(
  () => String(props.tokenEngine || '').trim().length > 0 && inner.value.trim().length > 0,
)

function scheduleTokenCount(engine: string, prompt: string): void {
  if (debounceTimer !== null) {
    clearTimeout(debounceTimer)
  }
  debounceTimer = setTimeout(() => {
    debounceTimer = null
    void refreshTokenCount(engine, prompt)
  }, 220)
}

async function refreshTokenCount(engine: string, prompt: string): Promise<void> {
  const seq = requestSeq.value + 1
  requestSeq.value = seq
  try {
    const response = await fetchPromptTokenCount({ engine, prompt })
    if (seq !== requestSeq.value) return
    tokenCount.value = Math.max(0, Math.trunc(response.count))
    tokenError.value = ''
  } catch (error) {
    if (seq !== requestSeq.value) return
    const message = error instanceof Error ? error.message : String(error)
    tokenError.value = message || 'token count failed'
    console.error('[PromptBox] failed to fetch prompt token count', { engine, message })
  }
}

watch(
  () => [inner.value, props.tokenEngine] as const,
  ([text, engineRaw]) => {
    const prompt = String(text || '')
    const trimmedPrompt = prompt.trim()
    tokenError.value = ''
    if (!trimmedPrompt) {
      tokenCount.value = 0
      requestSeq.value += 1
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer)
        debounceTimer = null
      }
      return
    }

    const engine = String(engineRaw || '').trim()
    if (!engine) {
      tokenCount.value = 0
      requestSeq.value += 1
      if (debounceTimer !== null) {
        clearTimeout(debounceTimer)
        debounceTimer = null
      }
      return
    }

    scheduleTokenCount(engine, prompt)
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  requestSeq.value += 1
  if (debounceTimer !== null) {
    clearTimeout(debounceTimer)
    debounceTimer = null
  }
})
</script>

<!-- styles moved to styles/components/prompt-box.css -->
