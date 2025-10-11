// Ambient declarations for legacy global helpers injected by script.js
declare function gradioApp(): Document | ShadowRoot;
declare function onUiLoaded(cb: () => void): void;
declare function onUiUpdate(cb: (m: MutationRecord[]) => void): void;
declare function updateInput(el: HTMLElement): void;

