# 2025-10-18 — Split Hires VAE/TE selector

Context
- User reports VAE dropdown not opening; observed markup suggests legacy combined control (`VAE / Text Encoder`) with multiselect.
- Goal: remove legacy combined selector and provide explicit controls without breaking backend payloads.

Change
- modules/ui.py: In Hires. fix section, replaced `hr_additional_modules` UI with:
  - `hr_vae` (single-select, label "Hires VAE")
  - `hr_text_encoders` (multiselect, label "Hires Text Encoder(s)")
  - Hidden compatibility dropdown `hr_vae_te` that aggregates the selections to the legacy payload list shape.
- Added refresh wiring and assembly function `_assemble_hr_modules` to keep `hr_vae_te` synchronized.
- Choices for VAE/Text Encoders sourced from `modules_forge.main_entry.refresh_models()` to avoid duplication.

Compatibility
- Kept `elem_id='hr_vae_te'` and maintained `hr_additional_modules` variable in submit inputs, ensuring no backend change required.

Validation
- `rg` for `hr_additional_modules` references confirms it remains in submit list.
- Manual reasoning: Gradio will accept hidden dropdown with values present in its `choices` (we set union of VAE and TE plus sentinel).

Risks
- If extensions expect a visible combined control, they'll now see split controls; payload shape unchanged.
- Requires rebuild/reload of UI; users should hard-refresh once (incognito already bypasses cache).

