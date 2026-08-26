<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical video workspace owner for current video families.
Assembles the shared generic video prompt/init-image/core/stage/output cards inside the route-owned body, mounts only the active family runtime helper,
and keeps the WAN/LTX family branches under one truthful `VideoModelTab.vue` owner instead of separate family workspaces.

Symbols (top-level; keep in sync; no ghosts):
- `VideoModelTab` (component): Canonical route-owned video workspace for current video families.
- `VideoTabType` (type): Supported video tab types rendered by this owner.
- `tab` (computed): Current tab selected by `tabId`.
- `videoTabType` (computed): Normalized current video tab type (`wan22_14b|wan22_5b|ltx2`) for the active branch.
-->

<template>
  <section v-if="videoTabType === 'wan22_14b'">
    <VideoModelTabWanRuntime :tab-id="tabId" v-slot="wan">
      <section v-if="wan.tab" class="panels video-panels">
        <div class="panel-stack">
          <div class="panel">
            <div class="panel-header">Prompt</div>
            <div class="panel-body">
              <div id="wan-guided-high-prompt">
                <VideoPromptStageCard
                  title="High Prompt"
                  :prompt="wan.highPrompt"
                  :negative="wan.highNegative"
                  :hide-negative="wan.hideHighNegativePrompt"
                  token-engine="wan22_14b"
                  collapsible
                  :open="wan.highPromptOpen"
                  @update:prompt="wan.setHighPromptText"
                  @update:negative="wan.setHighNegativeText"
                  @update:open="wan.toggleHighPrompt"
                >
                  <template #header-actions>
                    <button class="btn btn-sm btn-secondary" type="button" @click="wan.setShowHighPromptLoraModal(true)">LoRA</button>
                  </template>
                </VideoPromptStageCard>
                <LoraModal
                  :modelValue="wan.showHighPromptLoraModal"
                  :show-negative-target="!wan.hideHighNegativePrompt"
                  @update:modelValue="wan.setShowHighPromptLoraModal"
                  @insert="wan.onHighPromptLoraInsert"
                />
              </div>

              <div class="mt-3">
                <VideoPromptStageCard
                  title="Low Prompt"
                  :prompt="wan.lowPrompt"
                  :negative="wan.lowNegative"
                  :hide-negative="wan.hideLowNegativePrompt"
                  token-engine="wan22_14b"
                  collapsible
                  :open="wan.lowPromptOpen"
                  @update:prompt="wan.setLowPromptText"
                  @update:negative="wan.setLowNegativeText"
                  @update:open="wan.toggleLowPrompt"
                >
                  <template #header-actions>
                    <button class="btn btn-sm btn-secondary" type="button" @click="wan.setShowLowPromptLoraModal(true)">LoRA</button>
                  </template>
                </VideoPromptStageCard>
                <LoraModal
                  :modelValue="wan.showLowPromptLoraModal"
                  :show-negative-target="!wan.hideLowNegativePrompt"
                  @update:modelValue="wan.setShowLowPromptLoraModal"
                  @insert="wan.onLowPromptLoraInsert"
                />
              </div>

              <div v-if="wan.mode === 'img2vid'" id="wan-guided-init-image" class="mt-3">
                <InitialImageBlock
                  :disabled="wan.isRunning"
                  sectionTitle="Img2Vid Parameters"
                  sectionSubtitle="Initial image"
                  initImageLabel="Image"
                  :showFrameGuideEditor="true"
                  :initImageData="wan.video.initImageData"
                  :initImageName="wan.video.initImageName"
                  :zoomFrameGuide="wan.wanInitImageZoomFrameGuide"
                  @set:initImage="wan.onInitImageFile"
                  @clear:initImage="wan.clearInit"
                  @reject:initImage="wan.onInitImageRejected"
                  @update:zoom-frame-guide="wan.onZoomFrameGuideUpdate"
                />
              </div>
            </div>
          </div>

          <div class="panel">
            <div class="panel-header">Generation Parameters</div>
            <div class="panel-body">
              <VideoCoreParamsCard
                title="Video"
                :width="wan.video.width"
                :height="wan.video.height"
                :width-min="64"
                :width-max="2048"
                :width-step="wan.dimensionInputStep"
                :width-input-step="wan.dimensionInputStep"
                :width-nudge-step="wan.dimensionInputStep"
                :height-min="64"
                :height-max="2048"
                :height-step="wan.dimensionInputStep"
                :height-input-step="wan.dimensionInputStep"
                :height-nudge-step="wan.dimensionInputStep"
                :frames="wan.video.frames"
                :fps="wan.video.fps"
                :min-frames="9"
                :max-frames="401"
                :frame-step="4"
                :frame-nudge-step="4"
                frame-rule-label="4n+1"
                :disabled="wan.isRunning"
                :show-temporal-section="wan.mode === 'img2vid'"
                @update:width="wan.applyWidth"
                @update:height="wan.applyHeight"
                @update:frames="(value) => wan.setVideo({ frames: value })"
                @update:fps="(value) => wan.setVideo({ fps: value })"
              >
                <template #width-right>
                  <NumberStepperInput
                    :modelValue="wan.video.width"
                    :min="64"
                    :max="2048"
                    :step="wan.dimensionInputStep"
                    :nudgeStep="wan.dimensionInputStep"
                    inputClass="cdx-input-w-md"
                    :disabled="wan.isRunning"
                    @update:modelValue="wan.applyWidth"
                  />
                  <select
                    class="ui-input ui-input-sm select-md cdx-input-w-sm"
                    :disabled="wan.isRunning"
                    :value="wan.aspectMode"
                    aria-label="Aspect ratio"
                    title="Aspect ratio"
                    @change="wan.onAspectModeChange"
                  >
                    <option value="free">Free</option>
                    <option value="current">Lock</option>
                    <option value="image" :disabled="wan.initImageAspectRatio === null">Image</option>
                    <option value="16:9">16:9</option>
                    <option value="1:1">1:1</option>
                    <option value="9:16">9:16</option>
                    <option value="4:3">4:3</option>
                    <option value="3:4">3:4</option>
                  </select>
                </template>
                <template #width-below>
                  <span v-if="wan.aspectMode !== 'free'" class="caption">Keeps ratio while editing width/height.</span>
                </template>
                <template #temporal>
                  <div class="cdx-video-inline-section">
                    <div class="cdx-video-card-header">
                      <div class="cdx-video-card-header__left">
                        <span class="cdx-video-card-header__title">Temporal Loom</span>
                      </div>
                      <div class="cdx-video-card-header__right">
                        <span class="cdx-video-badge-experimental">EXPERIMENTAL</span>
                        <button
                          :class="[
                            'btn',
                            'qs-toggle-btn',
                            'qs-toggle-btn--sm',
                            wan.temporalControlsEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off',
                          ]"
                          type="button"
                          :disabled="wan.isRunning"
                          :aria-pressed="wan.temporalControlsEnabled"
                          @click="wan.setImg2VidTemporalEnabled(!wan.temporalControlsEnabled)"
                        >
                          {{ wan.temporalControlsEnabled ? 'Enabled' : 'Disabled' }}
                        </button>
                      </div>
                    </div>
                    <div v-if="wan.temporalControlsEnabled" class="param-blocks cdx-video-temporal-controls mt-2">
                      <div class="param-grid cdx-video-temporal-row" data-cols="3">
                        <div class="field">
                          <label class="label-muted">Mode</label>
                          <select
                            class="select-md"
                            :disabled="wan.isRunning"
                            :value="wan.temporalEnabledMode"
                            @change="wan.setImg2VidTemporalMode(wan.normalizeImg2VidTemporalEnabledMode(($event.target as HTMLSelectElement).value))"
                          >
                            <option value="sliding">Sliding Window</option>
                            <option value="svi2">SVI 2.0</option>
                            <option value="svi2_pro">SVI 2.0 Pro</option>
                          </select>
                        </div>
                        <div class="field">
                          <label class="label-muted">
                            <HoverTooltip
                              class="cdx-slider-field__label-tooltip"
                              title="Attention Mode"
                              :content="[
                                'Global uses full temporal context.',
                                'Sliding limits attention context to reduce memory/cost.',
                              ]"
                            >
                              <span class="cdx-slider-field__label-trigger">
                                <span>Attention Mode</span>
                                <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
                              </span>
                            </HoverTooltip>
                          </label>
                          <select
                            class="select-md"
                            :disabled="wan.isRunning"
                            :value="wan.video.attentionMode"
                            @change="wan.setVideo({ attentionMode: wan.normalizeAttentionMode(($event.target as HTMLSelectElement).value) })"
                          >
                            <option value="global">Global</option>
                            <option value="sliding">Sliding</option>
                          </select>
                        </div>
                        <div class="field">
                          <label class="label-muted">
                            <HoverTooltip
                              class="cdx-slider-field__label-tooltip"
                              title="Temporal Seed Mode"
                              :content="[
                                'Fixed: same seed for every window.',
                                'Increment: adds window index to the base seed.',
                                'Random: independent seed per window.',
                              ]"
                            >
                              <span class="cdx-slider-field__label-trigger">
                                <span>Temporal Seed Mode</span>
                                <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
                              </span>
                            </HoverTooltip>
                          </label>
                          <select
                            class="select-md"
                            :disabled="wan.isRunning"
                            :value="wan.video.img2vidChunkSeedMode"
                            @change="wan.setVideo({ img2vidChunkSeedMode: wan.normalizeChunkSeedMode(($event.target as HTMLSelectElement).value) })"
                          >
                            <option value="increment">Increment</option>
                            <option value="fixed">Fixed</option>
                            <option value="random">Random</option>
                          </select>
                        </div>
                      </div>
                      <div v-if="wan.isWindowedTemporalMode(wan.video.img2vidMode)" class="param-grid cdx-video-temporal-row" data-cols="5">
                        <SliderField
                          class="field"
                          label="Window Frames"
                          :modelValue="wan.video.img2vidWindowFrames"
                          :min="9"
                          :max="401"
                          :step="4"
                          :inputStep="1"
                          :nudgeStep="4"
                          :disabled="wan.isRunning"
                          inputClass="cdx-input-w-sm"
                          tooltipTitle="Window Frames"
                          :tooltip="[
                            'Temporal context size per window.',
                            'Must satisfy 4n+1.',
                          ]"
                          @update:modelValue="(value) => wan.setVideo({ img2vidWindowFrames: value })"
                        />
                        <SliderField
                          class="field"
                          label="Window Stride"
                          :modelValue="wan.video.img2vidWindowStride"
                          :min="wan.WAN_WINDOW_STRIDE_ALIGNMENT"
                          :max="wan.maxAlignedWindowStride(wan.video.img2vidWindowFrames)"
                          :step="wan.WAN_WINDOW_STRIDE_ALIGNMENT"
                          :inputStep="1"
                          :nudgeStep="wan.WAN_WINDOW_STRIDE_ALIGNMENT"
                          :disabled="wan.isRunning"
                          inputClass="cdx-input-w-sm"
                          tooltipTitle="Window Stride"
                          :tooltip="[
                            'How far the window slides each iteration.',
                            'Must be aligned to temporal scale=4.',
                          ]"
                          @update:modelValue="(value) => wan.setVideo({ img2vidWindowStride: value })"
                        />
                        <SliderField
                          class="field"
                          label="Commit Frames"
                          :modelValue="wan.video.img2vidWindowCommitFrames"
                          :min="Math.min(wan.video.img2vidWindowFrames, wan.video.img2vidWindowStride + wan.WAN_WINDOW_COMMIT_OVERLAP_MIN)"
                          :max="wan.video.img2vidWindowFrames"
                          :step="1"
                          :inputStep="1"
                          :nudgeStep="1"
                          :disabled="wan.isRunning"
                          inputClass="cdx-input-w-sm"
                          tooltipTitle="Commit Frames"
                          :tooltip="[
                            'Frames committed from each window before advancing.',
                            'Must stay within [stride + 4, window].',
                          ]"
                          @update:modelValue="(value) => wan.setVideo({ img2vidWindowCommitFrames: value })"
                        />
                        <SliderField
                          class="field"
                          label="Anchor Alpha"
                          :modelValue="wan.video.img2vidAnchorAlpha"
                          :min="0"
                          :max="1"
                          :step="0.05"
                          :inputStep="0.05"
                          :nudgeStep="0.05"
                          :disabled="wan.isRunning"
                          inputClass="cdx-input-w-sm"
                          tooltipTitle="Anchor Alpha"
                          :tooltip="[
                            'Controls base-anchor influence at window handoff.',
                            '0 = continue from previous output only.',
                            '1 = stronger re-anchor to init image.',
                          ]"
                          @update:modelValue="(value) => wan.setVideo({ img2vidAnchorAlpha: value })"
                        />
                        <div class="field cdx-video-temporal-anchor-toggle">
                          <HoverTooltip
                            class="cdx-slider-field__label-tooltip"
                            title="Reset Anchor to Base"
                            :content="[
                              'Enabled: hard reset anchor at every window handoff.',
                              'Disabled: keeps temporal carry-over and only applies soft anchor blend.',
                              'SVI 2.0 / Pro force this option off by design.',
                            ]"
                          >
                            <button
                              :class="[
                                'btn',
                                'qs-toggle-btn',
                                'qs-toggle-btn--sm',
                                'cdx-video-temporal-anchor-toggle__btn',
                                wan.video.img2vidResetAnchorToBase ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off',
                              ]"
                              type="button"
                              :disabled="wan.isRunning || wan.video.img2vidMode === 'svi2' || wan.video.img2vidMode === 'svi2_pro'"
                              :aria-pressed="wan.video.img2vidResetAnchorToBase"
                              @click="wan.setVideo({ img2vidResetAnchorToBase: !wan.video.img2vidResetAnchorToBase })"
                            >
                              Reset Anchor
                            </button>
                          </HoverTooltip>
                        </div>
                      </div>
                    </div>
                    <div v-else class="caption mt-2">Native WAN22 mode runs img2vid without temporal window partitioning.</div>
                  </div>
                </template>
              </VideoCoreParamsCard>

              <div class="mt-3">
                <VideoOutputCard title="Video Output">
                  <div class="gc-row">
                    <div class="gc-col">
                      <label class="label-muted">Format</label>
                      <select class="select-md" :disabled="wan.isRunning" :value="wan.video.format" @change="wan.setVideo({ format: ($event.target as HTMLSelectElement).value })">
                        <option value="video/h264-mp4">H.264 MP4</option>
                        <option value="video/h265-mp4">H.265 MP4</option>
                        <option value="video/webm">WebM</option>
                        <option value="image/gif">GIF</option>
                      </select>
                    </div>
                    <div class="gc-col">
                      <label class="label-muted">Pixel Format</label>
                      <select class="select-md" :disabled="wan.isRunning" :value="wan.video.pixFmt" @change="wan.setVideo({ pixFmt: ($event.target as HTMLSelectElement).value })">
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
                      :modelValue="wan.video.loopCount"
                      :min="0"
                      :max="32"
                      :step="1"
                      :disabled="wan.isRunning"
                      inputClass="cdx-input-w-md"
                      @update:modelValue="wan.onLoopCountChange"
                    />
                    <SliderField
                      class="gc-col gc-col--compact"
                      label="CRF"
                      :modelValue="wan.video.crf"
                      :min="0"
                      :max="51"
                      :step="1"
                      :disabled="wan.isRunning"
                      inputClass="cdx-input-w-md"
                      @update:modelValue="wan.onCrfChange"
                    />
                    <SliderField
                      class="gc-col gc-col--compact"
                      label="Interpolation (RIFE)"
                      :modelValue="wan.video.interpolationFps"
                      :min="0"
                      :max="240"
                      :step="1"
                      :disabled="wan.isRunning"
                      inputClass="cdx-input-w-md"
                      @update:modelValue="wan.onInterpolationTargetFpsChange"
                    >
                      <template #below>
                        <span class="caption">{{ wan.interpolationCaption }}</span>
                      </template>
                    </SliderField>
                    <div class="gc-col gc-col--presets cdx-video-compact-toggle-column">
                      <button
                        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', wan.video.pingpong ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                        type="button"
                        :disabled="wan.isRunning"
                        :aria-pressed="wan.video.pingpong"
                        @click="wan.setVideo({ pingpong: !wan.video.pingpong })"
                      >
                        Ping-pong
                      </button>
                      <button
                        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', wan.video.returnFrames ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                        type="button"
                        :disabled="wan.isRunning"
                        :aria-pressed="wan.video.returnFrames"
                        @click="wan.setVideo({ returnFrames: !wan.video.returnFrames })"
                      >
                        Return frames
                      </button>
                    </div>
                  </div>
                </VideoOutputCard>
              </div>

              <div id="wan-guided-high-stage" class="mt-3">
                <VideoStageBasicParamsCard
                  title="High Noise"
                  :samplers="wan.wanStageSamplers"
                  :schedulers="wan.wanStageSchedulers"
                  :recommended-samplers="wan.wanRecommendedSamplers"
                  :recommended-schedulers="wan.wanRecommendedSchedulers"
                  :sampler="wan.high.sampler"
                  :scheduler="wan.high.scheduler"
                  :steps="wan.high.steps"
                  :cfg-scale="wan.high.cfgScale"
                  :seed="wan.high.seed"
                  :steps-max="150"
                  :cfg-max="30"
                  :cfg-step="0.5"
                  :cfg-input-step="0.5"
                  :cfg-nudge-step="0.5"
                  :can-reuse-seed="wan.canReuseHighSeed"
                  :disabled="wan.isRunning"
                  @update:sampler="(value) => wan.setHigh({ sampler: value })"
                  @update:scheduler="(value) => wan.setHigh({ scheduler: value })"
                  @update:steps="(value) => wan.setHigh({ steps: Math.trunc(value) })"
                  @update:cfgScale="(value) => wan.setHigh({ cfgScale: value })"
                  @update:seed="(value) => wan.setHigh({ seed: Math.trunc(value) })"
                  @randomize-seed="wan.randomizeHighSeed"
                  @reuse-seed="wan.reuseHighSeed"
                />
              </div>

              <div class="mt-3" id="wan-guided-low-stage">
                <VideoStageBasicParamsCard
                  title="Low Noise"
                  :samplers="wan.wanStageSamplers"
                  :schedulers="wan.wanStageSchedulers"
                  :recommended-samplers="wan.wanRecommendedSamplers"
                  :recommended-schedulers="wan.wanRecommendedSchedulers"
                  :sampler="wan.low.sampler"
                  :scheduler="wan.low.scheduler"
                  :steps="wan.low.steps"
                  :cfg-scale="wan.low.cfgScale"
                  :seed="wan.low.seed"
                  :steps-max="150"
                  :cfg-max="30"
                  :cfg-step="0.5"
                  :cfg-input-step="0.5"
                  :cfg-nudge-step="0.5"
                  :can-reuse-seed="wan.canReuseLowSeed"
                  :disabled="wan.isRunning || wan.lowFollowsHigh"
                  collapsible
                  :open="wan.lowNoiseOpen"
                  :caption="wan.lowFollowsHigh ? 'Low stage mirrors High (sampler/scheduler/steps/CFG/seed).' : ''"
                  @update:open="wan.toggleLowNoise"
                  @update:sampler="(value) => wan.setLow({ sampler: value })"
                  @update:scheduler="(value) => wan.setLow({ scheduler: value })"
                  @update:steps="(value) => wan.setLow({ steps: Math.trunc(value) })"
                  @update:cfgScale="(value) => wan.setLow({ cfgScale: value })"
                  @update:seed="(value) => wan.setLow({ seed: Math.trunc(value) })"
                  @randomize-seed="wan.randomizeLowSeed"
                  @reuse-seed="wan.reuseLowSeed"
                >
                  <template #header-actions>
                    <button
                      :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', wan.lowFollowsHigh ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                      type="button"
                      :disabled="wan.isRunning"
                      :aria-pressed="wan.lowFollowsHigh"
                      @click="wan.onLowFollowsHighChange(!wan.lowFollowsHigh)"
                    >
                      Use High settings
                    </button>
                  </template>
                </VideoStageBasicParamsCard>
              </div>
            </div>
          </div>
        </div>

        <div class="panel-stack panel-stack--sticky">
          <RunCard
            :isRunning="wan.isRunning"
            :generateDisabled="wan.isRunning || !wan.canRunGeneration"
            :generateTitle="wan.generateTitle"
            generateId="wan-guided-generate"
            :showBatchControls="false"
            @generate="wan.onGenerateClick"
            @cancel="wan.cancel()"
          >
            <template #header-right>
              <div class="results-header-actions">
                <button v-if="wan.history.length && !wan.isRunning" class="btn btn-sm btn-secondary" type="button" :disabled="wan.isRunning" @click="wan.reuseLast">
                  Reuse last
                </button>
              </div>
            </template>

            <div v-if="wan.copyNotice" class="caption">{{ wan.copyNotice }}</div>
            <RunSummaryChips class="video-results-summary" :text="wan.runSummary" />
            <RunProgressStatus
              v-if="wan.isRunning"
              :stage="wan.progress.stage"
              :percent="wan.progress.percent"
              :step="wan.progress.step"
              :total-steps="wan.progress.totalSteps"
              :eta-seconds="wan.progress.etaSeconds"
              :show-progress-bar="true"
            />
            <RunProgressStatus
              v-else-if="wan.errorMessage"
              variant="error"
              title="Run failed"
              :message="wan.errorMessage"
              :show-progress-bar="false"
            />
          </RunCard>

          <GenerationResultsPanel class="video-results-panel" showHistory :showMedia="Boolean(wan.videoUrl)" :showInfo="Boolean(wan.info)">
            <template #header-right>
              <button class="btn btn-sm btn-outline" type="button" :disabled="wan.workflowBusy" @click="wan.sendToWorkflows">
                {{ wan.workflowBusy ? 'Saving…' : 'Save snapshot' }}
              </button>
              <button class="btn btn-sm btn-outline" type="button" @click="wan.copyCurrentParams">Copy params</button>
            </template>

            <template #history-actions>
              <button class="btn btn-sm btn-ghost" type="button" title="Clear history" :disabled="!wan.history.length || wan.isRunning" @click="wan.clearHistory">
                Clear
              </button>
            </template>

            <template #history>
              <ResultsHistoryStrip
                :items="wan.history"
                :selectedTaskId="wan.selectedTaskId"
                :formatTitle="wan.formatHistoryTitle"
                :toDataUrl="wan.toDataUrl"
                @select="wan.onSelectHistoryStripItem"
              />

              <details v-if="wan.diffText" class="accordion mt-2">
                <summary>Diff vs previous run</summary>
                <div class="accordion-body">
                  <pre class="text-xs break-words">{{ wan.diffText }}</pre>
                </div>
              </details>
            </template>

            <template #media-actions>
              <button v-if="wan.videoUrl" class="btn btn-sm btn-outline" type="button" @click="wan.openResultVideoZoom">Zoom</button>
              <a v-if="wan.videoUrl" class="btn btn-sm btn-outline" :href="wan.videoUrl" target="_blank" rel="noreferrer">Open</a>
            </template>

            <template #media>
              <video class="w-full rounded" :src="wan.videoUrl || ''" controls @dblclick.prevent.stop />
              <p class="caption mt-1">Tip: if playback fails, install ffmpeg and ensure CODEX_ROOT/output is writable.</p>
            </template>

            <template #viewer>
              <ResultViewer mode="video" :frames="wan.framesResult" :toDataUrl="wan.toDataUrl" emptyText="No results yet.">
                <template #empty>
                  <div class="results-empty-state">
                    <div class="results-empty-title">
                      <template v-if="wan.isRunning">Generating…</template>
                      <template v-else-if="wan.videoUrl">Frames not returned</template>
                      <template v-else>No results yet</template>
                    </div>
                    <div v-if="wan.videoUrl" class="caption">Enable “Return frames” in Video Output to include frames in the result payload.</div>
                    <div v-else-if="!wan.isRunning" class="caption">Generate to see results here.</div>
                  </div>
                </template>
              </ResultViewer>
            </template>

            <template #info-actions>
              <button class="btn btn-sm btn-outline" type="button" @click="wan.copyInfo">Copy info</button>
            </template>

            <template #info>
              <pre class="text-xs break-words">{{ wan.formatJson(wan.info) }}</pre>
            </template>

            <template #after-viewer>
              <VideoZoomOverlay :modelValue="wan.videoZoomOpen" :src="wan.videoUrl || ''" aria-label="Zoomed WAN result video" @update:modelValue="wan.setVideoZoomOpen" />
            </template>
          </GenerationResultsPanel>
        </div>

        <RunHistoryDetailsModal
          :modelValue="wan.historyDetailsOpen"
          :title="wan.historyDetailsTitle"
          :preview-url="wan.historyDetailsImageUrl"
          :preview-alt="wan.historyDetailsTitle"
          :mode-label="wan.historyDetailsModeLabel"
          :created-at-label="wan.historyDetailsCreatedAtLabel"
          :status="wan.historyDetailsItem?.status || ''"
          :task-id="wan.historyDetailsItem?.taskId || ''"
          :summary="wan.historyDetailsItem?.summary || ''"
          :error-message="wan.historyDetailsItem?.errorMessage || ''"
          :params-snapshot="wan.historyDetailsItem?.paramsSnapshot"
          :sections="wan.historyDetailsSections"
          :load-disabled="!wan.historyDetailsItem || wan.isRunning || wan.historyLoadingTaskId === wan.historyDetailsItem.taskId"
          :load-label="wan.historyDetailsItem && wan.historyLoadingTaskId === wan.historyDetailsItem.taskId ? 'Loading…' : 'Load'"
          :apply-disabled="!wan.historyDetailsItem || wan.isRunning"
          :copy-disabled="!wan.historyDetailsItem || wan.isRunning"
          @update:modelValue="wan.setHistoryDetailsOpen"
          @load="wan.onLoadHistoryDetails"
          @apply="wan.onApplyHistoryDetails"
          @copy="wan.onCopyHistoryDetails"
        />

        <Teleport to="body">
          <div
            v-if="wan.guidedActive && wan.guidedRect"
            :ref="bindElementRef(wan.setGuidedTooltipEl)"
            class="codex-guided-tooltip"
            :data-placement="wan.guidedTooltipPlacement"
            :style="wan.guidedTooltipStyle"
          >
            <div class="codex-guided-tooltip-title">Guided gen</div>
            <div class="codex-guided-tooltip-body">{{ wan.guidedMessage }}</div>
            <div class="codex-guided-tooltip-actions">
              <button class="btn btn-sm btn-secondary" type="button" @click="wan.stopGuided">Close</button>
            </div>
          </div>
        </Teleport>
      </section>
      <section v-else>
        <div class="panel"><div class="panel-body">Tab not found.</div></div>
      </section>
    </VideoModelTabWanRuntime>
  </section>

  <section v-else-if="videoTabType === 'wan22_5b'">
    <VideoModelTabWan22_5bRuntime :tab-id="tabId" v-slot="wan5b">
      <section v-if="wan5b.tab" class="panels video-panels">
        <div class="panel-stack">
          <div class="panel">
            <div class="panel-header">Prompt</div>
            <div class="panel-body">
              <VideoPromptStageCard
                title="Prompt"
                :prompt="wan5b.prompt"
                :negative="wan5b.negativePrompt"
                :hide-negative="wan5b.hideNegativePrompt"
                token-engine="wan22_5b"
                :corner-label="wan5b.modeLabel"
                collapsible
                :open="wan5b.promptOpen"
                @update:prompt="wan5b.setPromptText"
                @update:negative="wan5b.setNegativeText"
                @update:open="wan5b.togglePrompt"
              >
                <template #header-actions>
                  <button class="btn btn-sm btn-secondary" type="button" @click="wan5b.setShowPromptLoraModal(true)">LoRA</button>
                </template>
              </VideoPromptStageCard>
              <LoraModal
                :modelValue="wan5b.showPromptLoraModal"
                :show-negative-target="!wan5b.hideNegativePrompt"
                @update:modelValue="wan5b.setShowPromptLoraModal"
                @insert="wan5b.onPromptLoraInsert"
              />

              <div v-if="wan5b.mode === 'img2vid'" id="wan-guided-init-image" class="mt-3">
                <InitialImageBlock
                  :disabled="wan5b.isRunning"
                  sectionTitle="Img2Vid Parameters"
                  sectionSubtitle="Initial image"
                  initImageLabel="Image"
                  :showFrameGuideEditor="true"
                  :initImageData="wan5b.video.initImageData"
                  :initImageName="wan5b.video.initImageName"
                  :zoomFrameGuide="wan5b.wanInitImageZoomFrameGuide"
                  @set:initImage="wan5b.onInitImageFile"
                  @clear:initImage="wan5b.clearInit"
                  @reject:initImage="wan5b.onInitImageRejected"
                  @update:zoom-frame-guide="wan5b.onZoomFrameGuideUpdate"
                />
              </div>
            </div>
          </div>

          <div class="panel">
            <div class="panel-header">Generation Parameters</div>
            <div class="panel-body">
              <VideoCoreParamsCard
                title="Video"
                :width="wan5b.video.width"
                :height="wan5b.video.height"
                :width-min="64"
                :width-max="2048"
                :width-step="wan5b.dimensionInputStep"
                :width-input-step="wan5b.dimensionInputStep"
                :width-nudge-step="wan5b.dimensionInputStep"
                :height-min="64"
                :height-max="2048"
                :height-step="wan5b.dimensionInputStep"
                :height-input-step="wan5b.dimensionInputStep"
                :height-nudge-step="wan5b.dimensionInputStep"
                :frames="wan5b.video.frames"
                :fps="wan5b.video.fps"
                :min-frames="9"
                :max-frames="401"
                :frame-step="4"
                :frame-nudge-step="4"
                frame-rule-label="4n+1"
                :disabled="wan5b.isRunning"
                :show-temporal-section="false"
                @update:width="wan5b.applyWidth"
                @update:height="wan5b.applyHeight"
                @update:frames="(value) => wan5b.setVideo({ frames: value })"
                @update:fps="(value) => wan5b.setVideo({ fps: value })"
              >
                <template #width-right>
                  <NumberStepperInput
                    :modelValue="wan5b.video.width"
                    :min="64"
                    :max="2048"
                    :step="wan5b.dimensionInputStep"
                    :nudgeStep="wan5b.dimensionInputStep"
                    inputClass="cdx-input-w-md"
                    :disabled="wan5b.isRunning"
                    @update:modelValue="wan5b.applyWidth"
                  />
                  <select
                    class="ui-input ui-input-sm select-md cdx-input-w-sm"
                    :disabled="wan5b.isRunning"
                    :value="wan5b.aspectMode"
                    aria-label="Aspect ratio"
                    title="Aspect ratio"
                    @change="wan5b.onAspectModeChange"
                  >
                    <option value="free">Free</option>
                    <option value="current">Lock</option>
                    <option value="image" :disabled="wan5b.initImageAspectRatio === null">Image</option>
                    <option value="16:9">16:9</option>
                    <option value="1:1">1:1</option>
                    <option value="9:16">9:16</option>
                    <option value="4:3">4:3</option>
                    <option value="3:4">3:4</option>
                  </select>
                </template>
                <template #width-below>
                  <span v-if="wan5b.aspectMode !== 'free'" class="caption">Keeps ratio while editing width/height.</span>
                </template>
                <p v-if="wan5b.mode === 'img2vid'" class="caption mt-2">WAN 2.2 5B img2vid runs only the single-stage solo path.</p>
              </VideoCoreParamsCard>

              <div class="mt-3">
                <VideoOutputCard title="Video Output">
                  <div class="gc-row">
                    <div class="gc-col">
                      <label class="label-muted">Format</label>
                      <select class="select-md" :disabled="wan5b.isRunning" :value="wan5b.video.format" @change="wan5b.setVideo({ format: ($event.target as HTMLSelectElement).value })">
                        <option value="video/h264-mp4">H.264 MP4</option>
                        <option value="video/h265-mp4">H.265 MP4</option>
                        <option value="video/webm">WebM</option>
                        <option value="image/gif">GIF</option>
                      </select>
                    </div>
                    <div class="gc-col">
                      <label class="label-muted">Pixel Format</label>
                      <select class="select-md" :disabled="wan5b.isRunning" :value="wan5b.video.pixFmt" @change="wan5b.setVideo({ pixFmt: ($event.target as HTMLSelectElement).value })">
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
                      :modelValue="wan5b.video.loopCount"
                      :min="0"
                      :max="32"
                      :step="1"
                      :disabled="wan5b.isRunning"
                      inputClass="cdx-input-w-md"
                      @update:modelValue="wan5b.onLoopCountChange"
                    />
                    <SliderField
                      class="gc-col gc-col--compact"
                      label="CRF"
                      :modelValue="wan5b.video.crf"
                      :min="0"
                      :max="51"
                      :step="1"
                      :disabled="wan5b.isRunning"
                      inputClass="cdx-input-w-md"
                      @update:modelValue="wan5b.onCrfChange"
                    />
                    <SliderField
                      class="gc-col gc-col--compact"
                      label="Interpolation (RIFE)"
                      :modelValue="wan5b.video.interpolationFps"
                      :min="0"
                      :max="240"
                      :step="1"
                      :disabled="wan5b.isRunning"
                      inputClass="cdx-input-w-md"
                      @update:modelValue="wan5b.onInterpolationTargetFpsChange"
                    >
                      <template #below>
                        <span class="caption">{{ wan5b.interpolationCaption }}</span>
                      </template>
                    </SliderField>
                    <div class="gc-col gc-col--presets cdx-video-compact-toggle-column">
                      <button
                        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', wan5b.video.pingpong ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                        type="button"
                        :disabled="wan5b.isRunning"
                        :aria-pressed="wan5b.video.pingpong"
                        @click="wan5b.setVideo({ pingpong: !wan5b.video.pingpong })"
                      >
                        Ping-pong
                      </button>
                      <button
                        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', wan5b.video.returnFrames ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                        type="button"
                        :disabled="wan5b.isRunning"
                        :aria-pressed="wan5b.video.returnFrames"
                        @click="wan5b.setVideo({ returnFrames: !wan5b.video.returnFrames })"
                      >
                        Return frames
                      </button>
                    </div>
                  </div>
                </VideoOutputCard>
              </div>

              <div class="mt-3">
                <VideoStageBasicParamsCard
                  title="Sampling"
                  :samplers="wan5b.wanStageSamplers"
                  :schedulers="wan5b.wanStageSchedulers"
                  :recommended-samplers="wan5b.wanRecommendedSamplers"
                  :recommended-schedulers="wan5b.wanRecommendedSchedulers"
                  :sampler="wan5b.sampler"
                  :scheduler="wan5b.scheduler"
                  :steps="wan5b.steps"
                  :cfg-scale="wan5b.cfgScale"
                  :seed="wan5b.seed"
                  :steps-max="150"
                  :cfg-max="30"
                  :cfg-step="0.5"
                  :cfg-input-step="0.5"
                  :cfg-nudge-step="0.5"
                  :can-reuse-seed="wan5b.canReuseSeed"
                  :disabled="wan5b.isRunning"
                  @update:sampler="wan5b.setSamplerValue"
                  @update:scheduler="wan5b.setSchedulerValue"
                  @update:steps="wan5b.setStepsValue"
                  @update:cfgScale="wan5b.setCfgScaleValue"
                  @update:seed="wan5b.setSeedValue"
                  @randomize-seed="wan5b.randomizeSeed"
                  @reuse-seed="wan5b.reuseSeed"
                />
              </div>
            </div>
          </div>
        </div>

        <div class="panel-stack panel-stack--sticky">
          <RunCard
            :isRunning="wan5b.isRunning"
            :generateDisabled="wan5b.isRunning || !wan5b.canRunGeneration"
            :generateTitle="wan5b.generateTitle"
            generateId="wan22-5b-generate"
            :showBatchControls="false"
            @generate="wan5b.onGenerateClick"
            @cancel="wan5b.cancel()"
          >
            <template #header-right>
              <div class="results-header-actions">
                <button v-if="wan5b.history.length && !wan5b.isRunning" class="btn btn-sm btn-secondary" type="button" :disabled="wan5b.isRunning" @click="wan5b.reuseLast">
                  Reuse last
                </button>
              </div>
            </template>

            <div v-if="wan5b.resumeNotice || wan5b.copyNotice" class="caption">{{ wan5b.resumeNotice || wan5b.copyNotice }}</div>
            <RunSummaryChips class="video-results-summary" :text="wan5b.runSummary" />
            <RunProgressStatus
              v-if="wan5b.isRunning"
              :stage="wan5b.progress.stage"
              :percent="wan5b.progress.percent"
              :step="wan5b.progress.step"
              :total-steps="wan5b.progress.totalSteps"
              :eta-seconds="wan5b.progress.etaSeconds"
              :show-progress-bar="true"
            />
            <RunProgressStatus
              v-else-if="wan5b.errorMessage"
              variant="error"
              title="Run failed"
              :message="wan5b.errorMessage"
              :show-progress-bar="false"
            />
          </RunCard>

          <GenerationResultsPanel class="video-results-panel" showHistory :showMedia="Boolean(wan5b.videoUrl)" :showInfo="Boolean(wan5b.info)">
            <template #header-right>
              <button class="btn btn-sm btn-outline" type="button" :disabled="wan5b.workflowBusy" @click="wan5b.sendToWorkflows">
                {{ wan5b.workflowBusy ? 'Saving…' : 'Save snapshot' }}
              </button>
              <button class="btn btn-sm btn-outline" type="button" @click="wan5b.copyCurrentParams">Copy params</button>
            </template>

            <template #history-actions>
              <button class="btn btn-sm btn-ghost" type="button" title="Clear history" :disabled="!wan5b.history.length || wan5b.isRunning" @click="wan5b.clearHistory">
                Clear
              </button>
            </template>

            <template #history>
              <ResultsHistoryStrip
                :items="wan5b.history"
                :selectedTaskId="wan5b.selectedTaskId"
                :formatTitle="wan5b.formatHistoryTitle"
                :toDataUrl="wan5b.toDataUrl"
                @select="wan5b.onSelectHistoryStripItem"
              />

              <details v-if="wan5b.diffText" class="accordion mt-2">
                <summary>Diff vs previous run</summary>
                <div class="accordion-body">
                  <pre class="text-xs break-words">{{ wan5b.diffText }}</pre>
                </div>
              </details>
            </template>

            <template #media-actions>
              <button v-if="wan5b.videoUrl" class="btn btn-sm btn-outline" type="button" @click="wan5b.openResultVideoZoom">Zoom</button>
              <a v-if="wan5b.videoUrl" class="btn btn-sm btn-outline" :href="wan5b.videoUrl" target="_blank" rel="noreferrer">Open</a>
            </template>

            <template #media>
              <video class="w-full rounded" :src="wan5b.videoUrl || ''" controls @dblclick.prevent.stop />
              <p class="caption mt-1">Tip: if playback fails, install ffmpeg and ensure CODEX_ROOT/output is writable.</p>
            </template>

            <template #viewer>
              <ResultViewer mode="video" :frames="wan5b.framesResult" :toDataUrl="wan5b.toDataUrl" emptyText="No results yet.">
                <template #empty>
                  <div class="results-empty-state">
                    <div class="results-empty-title">
                      <template v-if="wan5b.isRunning">Generating…</template>
                      <template v-else-if="wan5b.videoUrl">Frames not returned</template>
                      <template v-else>No results yet</template>
                    </div>
                    <div v-if="wan5b.videoUrl" class="caption">Enable “Return frames” in Video Output to include frames in the result payload.</div>
                    <div v-else-if="!wan5b.isRunning" class="caption">Generate to see results here.</div>
                  </div>
                </template>
              </ResultViewer>
            </template>

            <template #info-actions>
              <button class="btn btn-sm btn-outline" type="button" @click="wan5b.copyInfo">Copy info</button>
            </template>

            <template #info>
              <pre class="text-xs break-words">{{ wan5b.formatJson(wan5b.info) }}</pre>
            </template>

            <template #after-viewer>
              <VideoZoomOverlay :modelValue="wan5b.videoZoomOpen" :src="wan5b.videoUrl || ''" aria-label="Zoomed WAN 2.2 5B result video" @update:modelValue="wan5b.setVideoZoomOpen" />
            </template>
          </GenerationResultsPanel>
        </div>

        <RunHistoryDetailsModal
          :modelValue="wan5b.historyDetailsOpen"
          :title="wan5b.historyDetailsTitle"
          :preview-url="wan5b.historyDetailsImageUrl"
          :preview-alt="wan5b.historyDetailsTitle"
          :mode-label="wan5b.historyDetailsModeLabel"
          :created-at-label="wan5b.historyDetailsCreatedAtLabel"
          :status="wan5b.historyDetailsItem?.status || ''"
          :task-id="wan5b.historyDetailsItem?.taskId || ''"
          :summary="wan5b.historyDetailsItem?.summary || ''"
          :error-message="wan5b.historyDetailsItem?.errorMessage || ''"
          :params-snapshot="wan5b.historyDetailsItem?.paramsSnapshot"
          :sections="wan5b.historyDetailsSections"
          :load-disabled="!wan5b.historyDetailsItem || wan5b.isRunning || wan5b.historyLoadingTaskId === wan5b.historyDetailsItem.taskId"
          :load-label="wan5b.historyDetailsItem && wan5b.historyLoadingTaskId === wan5b.historyDetailsItem.taskId ? 'Loading…' : 'Load'"
          :apply-disabled="!wan5b.historyDetailsItem || wan5b.isRunning"
          :copy-disabled="!wan5b.historyDetailsItem || wan5b.isRunning"
          @update:modelValue="wan5b.setHistoryDetailsOpen"
          @load="wan5b.onLoadHistoryDetails"
          @apply="wan5b.onApplyHistoryDetails"
          @copy="wan5b.onCopyHistoryDetails"
        />
      </section>
      <section v-else>
        <div class="panel"><div class="panel-body">Tab not found.</div></div>
      </section>
    </VideoModelTabWan22_5bRuntime>
  </section>

  <section v-else-if="videoTabType === 'ltx2'">
    <VideoModelTabLtxRuntime :tab-id="tabId" v-slot="ltx">
      <section v-if="ltx.tab && ltx.params" class="panels video-panels">
        <div class="panel-stack">
          <div class="panel">
            <div class="panel-header">Prompt</div>
            <div class="panel-body">
              <VideoPromptStageCard
                title="Prompt"
                :prompt="ltx.params.prompt"
                :negative="ltx.params.negativePrompt"
                :hide-negative="ltx.hideNegativePrompt"
                token-engine="ltx2"
                :corner-label="ltx.promptModeLabel"
                @update:prompt="(value) => ltx.updateParamsPatch({ prompt: value })"
                @update:negative="(value) => ltx.updateParamsPatch({ negativePrompt: value })"
              />

              <div v-if="ltx.mode === 'img2vid'" class="mt-3">
                <InitialImageBlock
                  :disabled="ltx.isRunning"
                  sectionTitle="Img2Vid Parameters"
                  sectionSubtitle="Initial image"
                  initImageLabel="Image"
                  :initImageData="ltx.params.initImageData"
                  :initImageName="ltx.params.initImageName"
                  @set:initImage="ltx.onInitImageFile"
                  @clear:initImage="ltx.clearInit"
                  @reject:initImage="ltx.onInitImageRejected"
                />
              </div>
            </div>
          </div>

          <div class="panel">
            <div class="panel-header">Generation Parameters</div>
            <div class="panel-body">
              <VideoCoreParamsCard
                title="Video"
                :width="ltx.params.width"
                :height="ltx.params.height"
                :width-min="ltx.ltxDimMin"
                :width-max="ltx.ltxDimMax"
                :width-step="ltx.dimensionAlignment"
                :width-input-step="1"
                :width-nudge-step="ltx.dimensionAlignment"
                :height-min="ltx.ltxDimMin"
                :height-max="ltx.ltxDimMax"
                :height-step="ltx.dimensionAlignment"
                :height-input-step="1"
                :height-nudge-step="ltx.dimensionAlignment"
                :frames="ltx.params.frames"
                :fps="ltx.params.fps"
                :min-frames="ltx.ltxFramesMin"
                :max-frames="ltx.ltxFramesMax"
                :frame-step="ltx.ltxFrameAlignment"
                :frame-nudge-step="ltx.ltxFrameAlignment"
                frame-rule-label="8n+1"
                :min-fps="1"
                :max-fps="60"
                :disabled="ltx.isRunning"
                @update:width="(value) => ltx.updateParamsPatch({ width: ltx.normalizeDimensionInput(value, ltx.params!.width) })"
                @update:height="(value) => ltx.updateParamsPatch({ height: ltx.normalizeDimensionInput(value, ltx.params!.height) })"
                @update:frames="(value) => ltx.updateParamsPatch({ frames: ltx.normalizeFrameInput(value, ltx.params!.frames) })"
                @update:fps="(value) => ltx.updateParamsPatch({ fps: ltx.normalizePositiveInt(value, ltx.params!.fps, 1, 240) })"
              >
                <p v-if="ltx.dimensionWarning" class="panel-status mt-2">{{ ltx.dimensionWarning }}</p>
                <p v-if="ltx.frameWarning" class="panel-status mt-2">{{ ltx.frameWarning }}</p>
              </VideoCoreParamsCard>

              <div class="mt-3">
                <VideoStageBasicParamsCard
                  title="Sampling"
                  :show-sampler="false"
                  :show-scheduler="false"
                  :steps="ltx.params.steps"
                  :cfg-scale="ltx.params.cfgScale"
                  :seed="ltx.params.seed"
                  :steps-max="100"
                  :cfg-max="20"
                  :cfg-step="0.1"
                  :cfg-input-step="0.1"
                  :cfg-nudge-step="0.5"
                  :show-seed-actions="false"
                  :disabled="ltx.isRunning"
                  @update:steps="(value) => ltx.updateParamsPatch({ steps: ltx.normalizePositiveInt(value, ltx.params!.steps, 1) })"
                  @update:cfgScale="(value) => ltx.updateParamsPatch({ cfgScale: ltx.normalizeFiniteNumber(value, ltx.params!.cfgScale, 0) })"
                  @update:seed="(value) => ltx.updateParamsPatch({ seed: Math.trunc(value) })"
                >
                  <div class="grid gap-3 md:grid-cols-2 mt-3">
                    <div class="form-field">
                      <label class="label-muted">Execution Profile</label>
                      <select
                        class="select-md"
                        :value="ltx.params.executionProfile"
                        :disabled="ltx.isRunning"
                        @change="(event) => ltx.updateParamsPatch({ executionProfile: (event.target as HTMLSelectElement).value })"
                      >
                        <option value="">Select profile</option>
                        <option
                          v-for="option in ltx.executionProfileOptions"
                          :key="option.value"
                          :value="option.value"
                          :disabled="!option.supported"
                        >
                          {{ option.label }}
                        </option>
                      </select>
                    </div>
                  </div>
                  <p v-if="ltx.executionProfileWarning" class="panel-status mt-2">{{ ltx.executionProfileWarning }}</p>
                </VideoStageBasicParamsCard>
              </div>

              <div class="mt-3">
                <VideoOutputCard title="Video Output">
                  <div class="row-split mb-2">
                    <span class="label-muted">Return frames</span>
                    <button
                      :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', ltx.params.videoReturnFrames ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                      type="button"
                      :disabled="ltx.isRunning"
                      :aria-pressed="ltx.params.videoReturnFrames"
                      @click="ltx.updateParamsPatch({ videoReturnFrames: !ltx.params.videoReturnFrames })"
                    >
                      {{ ltx.params.videoReturnFrames ? 'Enabled' : 'Disabled' }}
                    </button>
                  </div>

                </VideoOutputCard>
              </div>
            </div>
          </div>
        </div>

        <div class="panel-stack panel-stack--sticky">
          <RunCard
            :isRunning="ltx.isRunning"
            :generateDisabled="ltx.runGenerateDisabled"
            :generateTitle="ltx.runGenerateTitle"
            :showBatchControls="false"
            @generate="ltx.generate()"
            @cancel="ltx.cancel()"
          >
            <div v-if="ltx.resumeNotice || ltx.copyNotice" class="caption">{{ ltx.resumeNotice || ltx.copyNotice }}</div>
            <RunSummaryChips class="video-results-summary" :text="ltx.runSummary" />
            <RunProgressStatus
              v-if="ltx.isRunning"
              :stage="ltx.progress.stage"
              :percent="ltx.progress.percent"
              :step="ltx.progress.step"
              :total-steps="ltx.progress.totalSteps"
              :eta-seconds="ltx.progress.etaSeconds"
              :show-progress-bar="true"
            />
            <RunProgressStatus
              v-else-if="ltx.errorMessage"
              variant="error"
              title="Run failed"
              :message="ltx.errorMessage"
              :show-progress-bar="false"
            />
            <RunProgressStatus
              v-else-if="ltx.status === 'done' && (ltx.videoUrl || ltx.frames.length)"
              variant="success"
              title="Run complete"
              :message="ltx.successMessage"
              :show-progress-bar="false"
            />
          </RunCard>

          <GenerationResultsPanel class="video-results-panel" showHistory :showMedia="Boolean(ltx.videoUrl)" :showInfo="Boolean(ltx.info)">
            <template #header-right>
              <button class="btn btn-sm btn-outline" type="button" :disabled="ltx.workflowBusy" @click="ltx.sendToWorkflows">
                {{ ltx.workflowBusy ? 'Saving…' : 'Save snapshot' }}
              </button>
              <button class="btn btn-sm btn-outline" type="button" @click="ltx.copyCurrentParams">Copy params</button>
            </template>

            <template #history-actions>
              <button class="btn btn-sm btn-ghost" type="button" title="Clear history" :disabled="!ltx.history.length || ltx.isRunning" @click="ltx.clearHistory">
                Clear
              </button>
            </template>

            <template #history>
              <ResultsHistoryStrip
                :items="ltx.history"
                :selectedTaskId="ltx.selectedTaskId"
                :formatTitle="ltx.formatHistoryTitle"
                :toDataUrl="ltx.toDataUrl"
                @select="ltx.onSelectHistoryItem"
              />
            </template>

            <template #media-actions>
              <button v-if="ltx.videoUrl" class="btn btn-sm btn-outline" type="button" @click="ltx.openResultVideoZoom">Zoom</button>
              <a v-if="ltx.videoUrl" class="btn btn-sm btn-outline" :href="ltx.videoUrl" target="_blank" rel="noreferrer">Open</a>
            </template>

            <template #media>
              <video class="rounded" :src="ltx.videoUrl || ''" controls @dblclick.prevent.stop />
            </template>

            <template #viewer>
              <ResultViewer mode="video" :frames="ltx.frames" :toDataUrl="ltx.toDataUrl" emptyText="No frames yet.">
                <template #empty>
                  <div class="results-empty-state">
                    <div class="results-empty-title">
                      <template v-if="ltx.isRunning">Generating…</template>
                      <template v-else-if="ltx.videoUrl">Frames not returned</template>
                      <template v-else>No frames yet</template>
                    </div>
                    <div v-if="ltx.videoUrl" class="caption">
                      <template v-if="ltx.params.videoReturnFrames">The backend completed without returned frames.</template>
                      <template v-else>Enable “Return frames” to include frames in the result payload.</template>
                    </div>
                    <div v-else-if="!ltx.isRunning" class="caption">Generate to see returned frames here.</div>
                  </div>
                </template>
              </ResultViewer>
            </template>

            <template #info-actions>
              <button class="btn btn-sm btn-outline" type="button" @click="ltx.copyJson(ltx.info, 'Copied generation info.')">
                Copy info
              </button>
            </template>

            <template #info>
              <pre class="text-xs break-words">{{ ltx.formatJson(ltx.info) }}</pre>
            </template>

            <template #after-viewer>
              <VideoZoomOverlay
                :modelValue="ltx.videoZoomOpen"
                :src="ltx.videoUrl || ''"
                aria-label="Zoomed LTX result video"
                @update:modelValue="ltx.setVideoZoomOpen"
              />
            </template>
          </GenerationResultsPanel>
        </div>
      </section>
      <section v-else>
        <div class="panel"><div class="panel-body">Tab not found.</div></div>
      </section>
    </VideoModelTabLtxRuntime>
  </section>

  <section v-else-if="tab">
    <div class="panel"><div class="panel-body">Unsupported video tab type: {{ tab.type }}</div></div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { computed, type ComponentPublicInstance } from 'vue'

import ResultViewer from '../components/ResultViewer.vue'
import VideoCoreParamsCard from '../components/video/VideoCoreParamsCard.vue'
import InitialImageBlock from '../components/InitialImageBlock.vue'
import VideoOutputCard from '../components/video/VideoOutputCard.vue'
import VideoPromptStageCard from '../components/video/VideoPromptStageCard.vue'
import VideoStageBasicParamsCard from '../components/video/VideoStageBasicParamsCard.vue'
import LoraModal from '../components/modals/LoraModal.vue'
import RunHistoryDetailsModal from '../components/modals/RunHistoryDetailsModal.vue'
import GenerationResultsPanel from '../components/results/GenerationResultsPanel.vue'
import ResultsHistoryStrip from '../components/results/ResultsHistoryStrip.vue'
import RunCard from '../components/results/RunCard.vue'
import RunProgressStatus from '../components/results/RunProgressStatus.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import HoverTooltip from '../components/ui/HoverTooltip.vue'
import NumberStepperInput from '../components/ui/NumberStepperInput.vue'
import SliderField from '../components/ui/SliderField.vue'
import VideoZoomOverlay from '../components/ui/VideoZoomOverlay.vue'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'
import { isWanTabFamily } from '../utils/engine_taxonomy'
import VideoModelTabLtxRuntime from './video-model/VideoModelTabLtxRuntime.vue'
import VideoModelTabWan22_5bRuntime from './video-model/VideoModelTabWan22_5bRuntime.vue'
import VideoModelTabWanRuntime from './video-model/VideoModelTabWanRuntime.vue'

const props = defineProps<{ tabId: string }>()

const store = useModelTabsStore()

const tabId = computed(() => props.tabId)
const tab = computed(() => store.tabs.find((entry) => entry.id === tabId.value) || null)

type VideoTabType = Extract<BaseTabType, 'wan22_14b' | 'wan22_5b' | 'ltx2'>

const videoTabType = computed<VideoTabType | null>(() => {
  const value = tab.value?.type
  if (value === 'ltx2' || isWanTabFamily(value)) return value
  return null
})

function bindElementRef(setter: (element: Element | null) => void) {
  return (ref: Element | ComponentPublicInstance | null) => {
    if (ref instanceof Element) {
      setter(ref)
      return
    }
    if (ref && '$el' in ref && ref.$el instanceof Element) {
      setter(ref.$el)
      return
    }
    setter(null)
  }
}
</script>
