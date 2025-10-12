#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
report_dir="$root_dir/codex/reports/pyright"
mkdir -p "$report_dir"

ts="$(date -u +%Y%m%d-%H%M%S)"
txt_out="$report_dir/pyright-$ts.txt"
json_out="$report_dir/pyright-$ts.json"

echo "[pyright] writing reports to: $report_dir"

# Stats/text report
if command -v pyright >/dev/null 2>&1; then
  pyright --stats | tee "$txt_out"
else
  echo "[pyright] global 'pyright' not found; trying npx..." | tee "$txt_out"
  npx -y pyright --stats | tee -a "$txt_out"
fi

# JSON report
if command -v pyright >/dev/null 2>&1; then
  pyright --outputjson > "$json_out"
else
  npx -y pyright --outputjson > "$json_out"
fi

echo "[pyright] done: $txt_out, $json_out"

