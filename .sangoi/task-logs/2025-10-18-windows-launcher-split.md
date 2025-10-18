Date: 2025-10-18
Task: Split Windows launcher — verify-only run-webui.bat and new install-webui.bat

Rationale
- User requested removing installation from run-webui.bat and creating a dedicated installer script. Goal: clear separation between install and launch.

Changes
- run-webui.bat: now verify-only.
  - No venv creation or pip installs.
  - Activates existing venv and runs import verification for core libs.
  - If venv missing or verification fails, prints error advising to run install-webui.bat.
- install-webui.bat: new installer.
  - Creates/activates venv, upgrades pip toolchain (stamped per minor version).
  - Installs Torch CUDA 12.8 pinned: torch==2.7.1, torchvision==0.22.1, torchaudio==2.7.1 via index-url.
  - Installs requirements from requirements_versions.txt with SHA256 stamp.
  - Verifies core libs.

Validation
- Syntax and flow verified; no destructive commands.
- Stamps prevent redundant installs; verification script common to both.

Next Steps
- (Optional) Add more granular verify messages (e.g., mismatched torch cuda build) in run-webui.

