Task: Add 'Open video' button in Txt2Vid/Img2Vid
Date: 2025-10-18

Changes
- modules/ui_common.py: `video_link_from_generation_info()` builds links using `webpath()` for mp4/webm if present.
- modules/ui.py: per tab (txt2vid/img2vid) added a button that consumes `generation_info` and renders link HTML.

Notes
- Safe: works when export is enabled and files exist; otherwise renders a simple message. No changes to payloads.

