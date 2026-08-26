<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generic output/assets card for video workspaces.
Provides the neutral video-card header while leaving the actual output controls owned by the calling view/family runtime.

Symbols (top-level; keep in sync; no ghosts):
- `VideoOutputCard` (component): Generic output/assets wrapper card.
-->

<template>
  <div class="gen-card cdx-video-card">
    <div class="cdx-video-card-header">
      <div class="cdx-video-card-header__left">
        <span class="cdx-video-card-header__title">{{ title }}</span>
      </div>
      <div v-if="$slots['header-actions']" class="cdx-video-card-header__right">
        <slot name="header-actions" />
      </div>
    </div>

    <div class="mt-2 cdx-video-card-body">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
withDefaults(defineProps<{
  title?: string
  disabled?: boolean
}>(), {
  title: 'Video Output',
  disabled: false,
})
</script>
