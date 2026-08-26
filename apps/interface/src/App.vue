<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Root WebUI layout and router shell.
Renders the global header + navigation tabs + router outlet (home + enabled non-chroma model tabs + gallery/workflows + image/video utilities) and computes
`--sticky-offset` from the header height so sticky result headers stay aligned.

Symbols (top-level; keep in sync; no ghosts):
- `App` (component): Root application component and layout shell.
- `setStickyOffset` (function): Computes and sets CSS `--sticky-offset` based on the header height.
- `requestStickyOffsetRecalc` (function): Schedules a sticky-offset recalculation via `requestAnimationFrame`.
- `retryBootstrap` (function): Retries hard-fatal app bootstrap initialization.
- `retryDeferredBootstrap` (function): Retries non-fatal deferred bootstrap initialization.
- `enabledTabs` (const): Computed list of enabled non-chroma model tabs used to render the nav.
-->

<template>
  <section v-if="bootstrap.criticalStatus === 'fatal'" class="bootstrap-screen">
    <div class="panel bootstrap-panel bootstrap-panel--fatal">
      <div class="panel-header">Fatal bootstrap error</div>
      <div class="panel-body">
        <p class="bootstrap-lead">
          Required initialization failed and the interface was not started.
        </p>
        <p class="bootstrap-detail">
          <strong>Context:</strong> {{ bootstrap.fatalContext || 'Bootstrap' }}
        </p>
        <pre class="bootstrap-error">{{ bootstrap.fatalMessage || 'Unknown error' }}</pre>
        <button class="btn btn-primary" type="button" @click="retryBootstrap">Retry</button>
      </div>
    </div>
  </section>

  <section v-else-if="bootstrap.criticalStatus !== 'ready'" class="bootstrap-screen">
    <div class="panel bootstrap-panel">
      <div class="panel-header">Initializing WebUI</div>
      <div class="panel-body">
        <p class="bootstrap-lead">Loading required backend state...</p>
      </div>
    </div>
  </section>

  <div v-else class="layout">
    <div class="main-shell">
      <header class="main-header" ref="headerRef">
        <div class="main-header-qs">
          <QuickSettingsBar />
        </div>
      </header>
      <section
        v-if="bootstrap.deferredStatus === 'loading'"
        class="bootstrap-inline-status"
        role="status"
        aria-live="polite"
      >
        <p class="bootstrap-inline-status-title">Finalizing startup in background…</p>
        <p class="bootstrap-inline-status-detail">Model tabs are loading; core UI is already ready.</p>
      </section>
      <section
        v-else-if="bootstrap.deferredStatus === 'error'"
        class="bootstrap-inline-status bootstrap-inline-status--error"
        role="status"
        aria-live="polite"
      >
        <p class="bootstrap-inline-status-title">Background startup failed.</p>
        <p class="bootstrap-inline-status-detail">
          <strong>Context:</strong> {{ bootstrap.deferredContext || 'Deferred bootstrap' }}
        </p>
        <pre class="bootstrap-error">{{ bootstrap.deferredMessage || 'Unknown error' }}</pre>
        <div class="bootstrap-inline-actions">
          <button class="btn btn-primary" type="button" @click="retryDeferredBootstrap">Retry background init</button>
        </div>
      </section>
      <nav class="tabs-nav">
        <!-- Home workspace (agnostic) -->
        <RouterLink class="tab-link" to="/">home</RouterLink>
        <!-- Model tabs (enabled only) -->
        <RouterLink v-for="t in enabledTabs" :key="t.id" class="tab-link" :to="`/models/${t.id}`">
          {{ t.title }}
        </RouterLink>
        <!-- Model & workflow tools -->
        <RouterLink class="tab-link" to="/gallery">gallery</RouterLink>
        <RouterLink class="tab-link" to="/workflows">workflows</RouterLink>
        <!-- Utilities on the right -->
        <RouterLink class="tab-link" to="/tools">🔧 tools</RouterLink>
        <RouterLink class="tab-link" to="/upscale">upscale</RouterLink>
        <RouterLink class="tab-link" to="/video-upscale">video upscale</RouterLink>
        <RouterLink class="tab-link" to="/pnginfo">png info</RouterLink>
        <RouterLink class="tab-link" to="/extensions">extensions</RouterLink>
      </nav>
      <main class="main-content">
        <RouterView />
      </main>
      <AppFooter />
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: layout, navigation
import QuickSettingsBar from './components/QuickSettingsBar.vue'
import AppFooter from './components/AppFooter.vue'
import { computed, onMounted, onBeforeUnmount, ref, watch } from 'vue'
import { useModelTabsStore } from './stores/model_tabs'
import { useBootstrapStore } from './stores/bootstrap'

const headerRef = ref<HTMLElement | null>(null)
let headerRO: ResizeObserver | null = null
let stickyOffsetRAF: number | null = null

function setStickyOffset(): void {
  const h = headerRef.value?.getBoundingClientRect().height ?? 0
  // Tabs não são sticky: o offset deve ser exatamente a altura do header.
  // Não subtrair padding do conteúdo para evitar subposição.
  document.documentElement.style.setProperty('--sticky-offset', `${Math.max(0, h)}px`)
}

function requestStickyOffsetRecalc(): void {
  if (stickyOffsetRAF !== null) return
  stickyOffsetRAF = window.requestAnimationFrame(() => {
    stickyOffsetRAF = null
    setStickyOffset()
  })
}

const tabs = useModelTabsStore()
const bootstrap = useBootstrapStore()
const enabledTabs = computed(() => tabs.orderedTabs.filter((tab) => tab.enabled && tab.type !== 'chroma'))

async function retryBootstrap(): Promise<void> {
  try {
    await bootstrap.retry()
  } catch {
    // Fatal state is already set by the bootstrap store.
  }
}

async function retryDeferredBootstrap(): Promise<void> {
  await bootstrap.retryDeferred()
}

watch(
  headerRef,
  (next, prev) => {
    if (!headerRO) return
    if (prev) headerRO.unobserve(prev)
    if (next) headerRO.observe(next)
    requestStickyOffsetRecalc()
  },
)

watch(
  () => bootstrap.criticalStatus,
  (status) => {
    if (status !== 'ready') return
    requestStickyOffsetRecalc()
  },
)

onMounted(() => {
  bootstrap.installGlobalErrorHandlers()
  bootstrap.start().catch(() => {
    // Fatal state is already set by bootstrap store.
  })

  if (typeof ResizeObserver !== 'undefined') {
    headerRO = new ResizeObserver(requestStickyOffsetRecalc)
    if (headerRef.value) {
      headerRO.observe(headerRef.value)
      requestStickyOffsetRecalc()
    }
  }
  window.addEventListener('resize', requestStickyOffsetRecalc)
})

onBeforeUnmount(() => {
  if (headerRO) headerRO.disconnect()
  headerRO = null
  window.removeEventListener('resize', requestStickyOffsetRecalc)
  if (stickyOffsetRAF !== null) window.cancelAnimationFrame(stickyOffsetRAF)
  stickyOffsetRAF = null
})
</script>
