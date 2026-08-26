/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical runtime CSS topology inventory for the frontend.
Defines the ordered stylesheet inventory shipped by `src/main.ts -> src/styles.css`,
and records the reference-only CSS files that must never become runtime owners.

Symbols (top-level; keep in sync; no ghosts):
- `styleEntry` (function): Freezes one topology entry with its source path, kind, and optional notes.
- `STYLE_TOPOLOGY` (constant): Canonical ordered frontend CSS inventory, including the root entry and reference-only sheets.
- `RUNTIME_CSS_TOPOLOGY` (constant): Ordered runtime stylesheet entries excluding the root entry and reference-only files.
- `REFERENCE_ONLY_CSS_TOPOLOGY` (constant): Explicit non-runtime CSS entries kept for reference only.
*/

function styleEntry(sourcePath, kind, notes = '') {
  return Object.freeze({
    sourcePath,
    kind,
    notes: String(notes || ''),
  })
}

export const STYLE_TOPOLOGY = Object.freeze([
  styleEntry('apps/interface/src/styles.css', 'root-entry', 'Sole runtime CSS entry imported by src/main.ts.'),
  styleEntry('apps/interface/src/styles/components/generation-settings-card.css', 'component'),
  styleEntry('apps/interface/src/styles/components/img2img-basic-parameters-card.css', 'component'),
  styleEntry('apps/interface/src/styles/components/initial-image-block.css', 'component'),
  styleEntry('apps/interface/src/styles/components/hires-settings-card.css', 'component'),
  styleEntry('apps/interface/src/styles/components/refiner-settings-card.css', 'component'),
  styleEntry('apps/interface/src/styles/components/result-viewer.css', 'component'),
  styleEntry('apps/interface/src/styles/components/inpaint-mask-editor.css', 'component'),
  styleEntry('apps/interface/src/styles/components/prompt-fields.css', 'component'),
  styleEntry('apps/interface/src/styles/components/video-settings-card.css', 'component'),
  styleEntry('apps/interface/src/styles/components/video-generation-cards.css', 'component'),
  styleEntry('apps/interface/src/styles/components/quicksettings.css', 'component'),
  styleEntry('apps/interface/src/styles/components/quicksettings-overrides-modal.css', 'component'),
  styleEntry('apps/interface/src/styles/components/asset-metadata-modal.css', 'component'),
  styleEntry('apps/interface/src/styles/components/views-shared.css', 'component'),
  styleEntry('apps/interface/src/styles/components/bootstrap-screen.css', 'component'),
  styleEntry('apps/interface/src/styles/components/dependency-check-panel.css', 'component'),
  styleEntry('apps/interface/src/styles/components/cdx-stepper-input.css', 'component'),
  styleEntry('apps/interface/src/styles/components/cdx-slider-field.css', 'component'),
  styleEntry('apps/interface/src/styles/components/cdx-hover-tooltip.css', 'component'),
  styleEntry('apps/interface/src/styles/components/cdx-dropzone.css', 'component'),
  styleEntry('apps/interface/src/styles/components/cdx-segmented-control.css', 'component'),
  styleEntry('apps/interface/src/styles/components/guided-gen.css', 'component'),
  styleEntry('apps/interface/src/styles/components/image-source-controls.css', 'component'),
  styleEntry('apps/interface/src/styles/components/ip-adapter-card.css', 'component'),
  styleEntry('apps/interface/src/styles/components/prompt-box.css', 'component'),
  styleEntry('apps/interface/src/styles/components/prompt-chip.css', 'component'),
  styleEntry('apps/interface/src/styles/components/run-card.css', 'component'),
  styleEntry('apps/interface/src/styles/components/settings-form.css', 'component'),
  styleEntry('apps/interface/src/styles/components/cdx-list.css', 'component'),
  styleEntry('apps/interface/src/styles/components/param-blocks.css', 'component'),
  styleEntry('apps/interface/src/styles/components/xyz-sweep-card.css', 'component'),
  styleEntry('apps/interface/src/styles/views/settings.css', 'view'),
  styleEntry('apps/interface/src/styles/views/tools.css', 'view'),
  styleEntry('apps/interface/src/styles/views/wan.css', 'view'),
  styleEntry('apps/interface/src/styles/views/video-upscale.css', 'view'),
  styleEntry('apps/interface/src/styles/views/pnginfo.css', 'view'),
  styleEntry('apps/interface/src/styles/EXAMPLE-dashboard-surface-base.css', 'reference-only', 'Reference sheet only; must never be imported by src/styles.css.'),
  styleEntry('apps/interface/src/styles/EXAMPLE-dashboard-surface-theme.css', 'reference-only', 'Reference sheet only; must never be imported by src/styles.css.'),
])

export const RUNTIME_CSS_TOPOLOGY = Object.freeze(
  STYLE_TOPOLOGY.filter((entry) => entry.kind !== 'root-entry' && entry.kind !== 'reference-only'),
)

export const REFERENCE_ONLY_CSS_TOPOLOGY = Object.freeze(
  STYLE_TOPOLOGY.filter((entry) => entry.kind === 'reference-only'),
)
