from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.request import urlopen, Request


PIN_RE = re.compile(r"^([A-Za-z0-9_.\-]+)==([^#\s]+)\s*(?:#.*)?$")


@dataclasses.dataclass(frozen=True)
class Pin:
    name: str
    version: str
    raw: str


def parse_pins(text: str) -> Tuple[List[str], List[Pin]]:
    lines: List[str] = []
    pins: List[Pin] = []
    for line in text.splitlines():
        m = PIN_RE.match(line.strip())
        if m:
            pins.append(Pin(name=m.group(1), version=m.group(2), raw=line))
        else:
            lines.append(line)
    return lines, pins


def fetch_latest(name: str) -> Optional[str]:
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        req = Request(url, headers={"User-Agent": "req-updater/1.0"})
        with urlopen(req, timeout=15) as r:  # nosec - trusted endpoint
            data = json.load(r)
        ver = data.get("info", {}).get("version")
        if not ver:
            return None
        # Normalize pre-releases away if present in default info and a stable exists
        if any(tag in ver for tag in ("a", "b", "rc")):
            versions = sorted(data.get("releases", {}).keys(), key=lambda s: tuple(int(p) if p.isdigit() else p for p in re.split(r"[.]+", s)))
            stable = [v for v in versions if not any(t in v for t in ("a", "b", "rc"))]
            if stable:
                return stable[-1]
        return ver
    except Exception:
        return None


def update_pins(
    base_text: str,
    exclude: Iterable[str],
    force: Dict[str, str],
    *,
    drop_excluded: bool = False,
    annotate_excluded: bool = True,
    excluded_comment: str = "pinned: bumping often causes resolver conflicts",
    max_workers: int = 16,
) -> str:
    other_lines, pins = parse_pins(base_text)
    name_to_pin = {p.name.lower(): p for p in pins}

    # Prepare worklist
    work: List[str] = []
    for p in pins:
        key = p.name.lower()
        if key in {x.lower() for x in exclude}:
            continue
        if key in {x.lower() for x in force.keys()}:
            continue
        work.append(p.name)

    results: Dict[str, Optional[str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_latest, name): name for name in work}
        for fut in concurrent.futures.as_completed(futs):
            name = futs[fut]
            try:
                results[name.lower()] = fut.result()
            except Exception:
                results[name.lower()] = None

    # Build output
    out_lines: List[str] = []
    for line in other_lines:
        out_lines.append(line)

    for p in pins:
        key = p.name.lower()
        if key in {x.lower() for x in force.keys()}:
            new_ver = force[key]
            out_lines.append(f"{p.name}=={new_ver}")
        elif key in {x.lower() for x in exclude}:
            if not drop_excluded:
                if annotate_excluded:
                    out_lines.append(f"{p.name}=={p.version}  # {excluded_comment}")
                else:
                    out_lines.append(p.raw)
            # else: skip writing this pin entirely
        else:
            latest = results.get(key)
            out_lines.append(f"{p.name}=={latest or p.version}")

    return "\n".join(out_lines) + "\n"


def parse_force(items: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items:
        if "==" not in item:
            raise SystemExit(f"--force entries must look like NAME==VERSION, got: {item}")
        name, version = item.split("==", 1)
        mapping[name.lower()] = version
    return mapping


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("Update pinned requirements to latest PyPI versions")
    ap.add_argument("--file", default="requirements_versions.txt", type=Path, help="Input requirements file")
    ap.add_argument("--write", action="store_true", help="Overwrite the input file in-place")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[
            # Locally compiled or CUDA-coupled
            "torch",
            "torchvision",
            "torchaudio",
            # Frequent conflict cluster
            "fastapi",
            "gradio",
            "pydantic",
            "protobuf",
            "httpx",
            "httpcore",
            "open-clip-torch",
            "onnxruntime-gpu",
        ],
        help="Package(s) to keep as-is (may repeat)",
    )
    ap.add_argument(
        "--drop-excluded",
        action="store_true",
        help="Remove excluded packages from the output instead of preserving their current pins.",
    )
    ap.add_argument(
        "--no-annotate-excluded",
        dest="annotate_excluded",
        action="store_false",
        help="Do not append a conflict warning comment to excluded pins.",
    )
    ap.add_argument(
        "--force",
        action="append",
        default=[],
        help="Force NAME==VERSION for specific packages (may repeat)",
    )
    args = ap.parse_args(argv)

    text = args.file.read_text(encoding="utf-8")
    forced = parse_force(args.force)
    updated = update_pins(
        text,
        exclude=args.exclude,
        force=forced,
        drop_excluded=args.drop_excluded,
        annotate_excluded=args.annotate_excluded,
    )

    if args.write:
        backup = args.file.with_suffix(args.file.suffix + ".bak")
        backup.write_text(text, encoding="utf-8")
        args.file.write_text(updated, encoding="utf-8")
        print(f"Updated {args.file} (backup at {backup})")
    else:
        sys.stdout.write(updated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
