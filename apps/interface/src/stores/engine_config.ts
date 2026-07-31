/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Static engine configs (defaults + capability flags) for the WebUI.
Defines engine types and per-engine defaults/capabilities used by dynamic tabs and composables to pick UI defaults and gate fields.
Variant-dependent engines (for example Z-Image Turbo/Base and FLUX.2 Klein distilled/base-4B) must be gated by per-tab checkpoint state
in views/payload builders, not by these static flags. Qwen Image is the exact Edit-2511 img2img-only public surface.

Symbols (top-level; keep in sync; no ghosts):
- `EngineType` (type): Known engine identifiers used by the UI config (includes `flux2`, `qwen_image`, `zimage_l2p`, `anima`, and `ltx2`).
- `TaskType` (type): Supported task identifiers (txt2img/img2img/txt2vid/img2vid).
- `EngineCapabilities` (interface): Capability flags used to gate UI controls (CFG/negative prompt/etc.).
- `EngineDefaults` (interface): Default generation parameters (width/height/steps/cfg/etc.).
- `EngineConfig` (interface): Full engine config entry (id/label/capabilities/defaults).
- `getEngineConfig` (function): Returns the config entry for a given engine id.
- `getEngineDefaults` (function): Returns default parameters for a given engine id.
- `getEngineCapabilities` (function): Returns capability flags for a given engine id.
- `getAllEngines` (function): Returns all engine config entries.
- `getImageEngines` (function): Returns image-capable engine configs.
- `getVideoEngines` (function): Returns video-capable engine configs.
*/

export type EngineType = 
  | 'sd15' 
  | 'sdxl' 
  | 'flux1' 
  | 'flux2'
  | 'qwen_image'
  | 'zimage' 
  | 'zimage_l2p'
  | 'chroma'
  | 'anima'
  | 'ltx2'
  | 'wan22_14b' 
  | 'wan22_5b'

export type TaskType = 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid'

export interface EngineCapabilities {
  // Supported tasks
  tasks: TaskType[]
  
  // CFG behavior
  usesCfg: boolean
  usesDistilledCfg: boolean
  usesNegativePrompt: boolean
  
  // Video specific
  isVideoEngine: boolean
}

export interface EngineDefaults {
  width: number
  height: number
  steps: number
  cfg: number           // standard CFG (diffusion)
  distilledCfg?: number // distilled CFG (flow)
}

export interface EngineConfig {
  id: EngineType
  label: string
  capabilities: EngineCapabilities
  defaults: EngineDefaults
}

// =============================================================================
// Engine Configurations
// =============================================================================

const ENGINE_CONFIGS: Record<EngineType, EngineConfig> = {
  sd15: {
    id: 'sd15',
    label: 'SD 1.5',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: false,
    },
    defaults: {
      width: 512,
      height: 512,
      steps: 20,
      cfg: 7,
    },
  },
  
  sdxl: {
    id: 'sdxl',
    label: 'SDXL',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 30,
      cfg: 7,
    },
  },
  
  flux1: {
    id: 'flux1',
    label: 'FLUX.1',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: false,
      usesDistilledCfg: true,
      usesNegativePrompt: false,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 4,
      cfg: 1,
      distilledCfg: 3.5,
    },
  },

  flux2: {
    id: 'flux2',
    label: 'FLUX.2',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: true,
      usesDistilledCfg: true,
      usesNegativePrompt: true,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 20,
      cfg: 4,
      distilledCfg: 4,
    },
  },

  qwen_image: {
    id: 'qwen_image',
    label: 'Qwen Image',
    capabilities: {
      tasks: ['img2img'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: false,
    },
    defaults: {
      width: 1328,
      height: 1328,
      steps: 50,
      cfg: 4,
    },
  },
  
  zimage: {
    id: 'zimage',
    label: 'Z Image',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 9,
      cfg: 1,
    },
  },

  zimage_l2p: {
    id: 'zimage_l2p',
    label: 'Z-Image L2P',
    capabilities: {
      tasks: ['txt2img'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 30,
      cfg: 2,
    },
  },
  
  chroma: {
    id: 'chroma',
    label: 'Chroma',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: false,
      usesDistilledCfg: true,
      usesNegativePrompt: false,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 4,
      cfg: 1,
      distilledCfg: 3.5,
    },
  },

  anima: {
    id: 'anima',
    label: 'Anima',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 30,
      cfg: 4,
    },
  },

  ltx2: {
    id: 'ltx2',
    label: 'LTX 2.3',
    capabilities: {
      tasks: ['txt2vid', 'img2vid'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: true,
    },
    defaults: {
      width: 768,
      height: 512,
      steps: 3,
      cfg: 1,
    },
  },
  
  wan22_14b: {
    id: 'wan22_14b',
    label: 'WAN 2.2 14B',
    capabilities: {
      tasks: ['txt2vid', 'img2vid'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: true,
    },
    defaults: {
      width: 768,
      height: 432,
      steps: 30,
      cfg: 7,
    },
  },
  
  wan22_5b: {
    id: 'wan22_5b',
    label: 'WAN 2.2 5B',
    capabilities: {
      tasks: ['txt2vid', 'img2vid'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      isVideoEngine: true,
    },
    defaults: {
      width: 768,
      height: 432,
      steps: 30,
      cfg: 7,
    },
  },
}

// =============================================================================
// Exports
// =============================================================================

export function getEngineConfig(engine: EngineType): EngineConfig {
  return ENGINE_CONFIGS[engine]
}

export function getEngineDefaults(engine: EngineType): EngineDefaults {
  return ENGINE_CONFIGS[engine].defaults
}

export function getEngineCapabilities(engine: EngineType): EngineCapabilities {
  return ENGINE_CONFIGS[engine].capabilities
}

export function getAllEngines(): EngineConfig[] {
  return Object.values(ENGINE_CONFIGS)
}

export function getImageEngines(): EngineConfig[] {
  return getAllEngines().filter(e => !e.capabilities.isVideoEngine)
}

export function getVideoEngines(): EngineConfig[] {
  return getAllEngines().filter(e => e.capabilities.isVideoEngine)
}
