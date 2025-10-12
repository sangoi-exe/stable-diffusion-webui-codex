// Ambient declarations for legacy global helpers injected by script.js
type GradioRoot = Document | ShadowRoot | HTMLElement;

declare function gradioApp(): GradioRoot;

type UiMutationCallback = (mutations: MutationRecord[]) => void;
type UiCallback = () => void;

declare function onUiLoaded(cb: UiCallback): void;
declare function onUiUpdate(cb: UiMutationCallback): void;
declare function onAfterUiUpdate(cb: UiCallback): void;
declare function onUiTabChange(cb: UiCallback): void;
declare function onOptionsChanged(cb: UiCallback): void;
declare function onOptionsAvailable(cb: UiCallback): void;

declare function updateInput(el: HTMLElement): void;
declare function get_tab_index(tabId: string): number;

interface StableDiffusionOptions extends Record<string, unknown> {
  show_progress_in_title?: boolean;
  show_progressbar?: boolean;
  prevent_screen_sleep_during_generation?: boolean;
  notification_volume?: number;
  return_grid?: boolean;
  live_preview_refresh_period?: number;
  js_modal_lightbox?: boolean;
  js_modal_lightbox_initially_zoomed?: boolean;
  js_modal_lightbox_gamepad?: boolean;
  js_modal_lightbox_gamepad_repeat?: number;
  js_live_preview_in_modal_lightbox?: boolean;
  disable_token_counters?: boolean;
  keyedit_precision_attention?: number;
  keyedit_precision_extra?: number;
  keyedit_move?: string;
  keyedit_delimiters?: string;
  keyedit_delimiters_whitespace?: string;
  show_progress_type?: string;
  use_old_hires_fix_width_height?: boolean;
  extra_networks_add_text_separator?: string;
  lora_filter_disabled?: boolean;
  _categories?: Record<string, string[]>;
  _comments_before?: Record<string, string>;
  _comments_after?: Record<string, string>;
  sd_checkpoint_hash?: string | null;
}

declare var opts: StableDiffusionOptions;

interface LocalizationDictionary extends Record<string, string> {
  rtl?: boolean;
}

interface GradioComponentConfig {
  id: number;
  props: {
    elem_id?: string;
    webui_tooltip?: string;
    placeholder?: string;
    [key: string]: unknown;
  };
}

interface GradioConfig {
  components: GradioComponentConfig[];
}

declare const localization: LocalizationDictionary;

declare global {
  interface Window {
    localization?: LocalizationDictionary;
    gradio_config: GradioConfig;
    _uiUpdQ?: UiMutationCallback[];
    _uiAfterQ?: UiCallback[];
    _uiLoadQ?: UiCallback[];
    opts: StableDiffusionOptions;
    args_to_array?: typeof Array.from;
    inputAccordionChecked?: (id: string, checked: boolean) => void;
  }
}
