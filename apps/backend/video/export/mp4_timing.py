"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Preserve the verified terminal VFR frame duration in one staged H.264 MP4 artifact.
Parses only the bounded atom path emitted by the local ffmpeg export contract, including its optional leading empty edit for delayed video,
validates every structural field and fixed-offset write, then patches the video track before the exporter verifies and publishes the artifact.
This is not a generic MP4 editing API.

Symbols (top-level; keep in sync; no ghosts):
- `Mp4TimingError` (class): Fail-loud staged-MP4 timing error for the caller-owned export boundary.
- `terminal_frame_duration_seconds` (function): Validates the decoded terminal source-frame duration.
- `expected_video_duration_seconds` (function): Calculates the required normalized video-track duration.
- `patch_terminal_frame_duration` (function): Writes the terminal sample, track, edit-list, and movie durations in the staged MP4.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from apps.backend.video.io.ffmpeg import VideoTiming


class Mp4TimingError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class _Mp4Atom:
    kind: bytes
    payload_offset: int
    end_offset: int


@dataclass(frozen=True, slots=True)
class _Mp4Write:
    offset: int
    width: int
    value: int
    field: str


def _read_mp4_atom(handle: Any, *, offset: int, enclosing_end: int) -> _Mp4Atom:
    handle.seek(offset)
    header = handle.read(8)
    if len(header) != 8:
        raise Mp4TimingError(f"Timestamp-aware MP4 has a truncated atom header at byte {offset}.")
    raw_size, kind = struct.unpack(">I4s", header)
    header_size = 8
    if raw_size == 1:
        extended_size = handle.read(8)
        if len(extended_size) != 8:
            raise Mp4TimingError(f"Timestamp-aware MP4 has a truncated extended atom header at byte {offset}.")
        atom_size = struct.unpack(">Q", extended_size)[0]
        header_size = 16
    elif raw_size == 0:
        atom_size = enclosing_end - offset
    else:
        atom_size = raw_size
    atom_end = offset + atom_size
    if atom_size < header_size or atom_end > enclosing_end:
        raise Mp4TimingError(
            "Timestamp-aware MP4 has an invalid atom boundary: "
            f"atom={kind!r}, offset={offset}, size={atom_size}, enclosing_end={enclosing_end}."
        )
    return _Mp4Atom(kind=kind, payload_offset=offset + header_size, end_offset=atom_end)


def _iter_mp4_atoms(handle: Any, *, start: int, end: int) -> tuple[_Mp4Atom, ...]:
    atoms: list[_Mp4Atom] = []
    offset = start
    while offset < end:
        atom = _read_mp4_atom(handle, offset=offset, enclosing_end=end)
        atoms.append(atom)
        offset = atom.end_offset
    if offset != end:
        raise Mp4TimingError(f"Timestamp-aware MP4 atom scan ended at byte {offset}, expected {end}.")
    return tuple(atoms)


def _find_required_mp4_atom(handle: Any, *, start: int, end: int, kind: bytes, owner: str) -> _Mp4Atom:
    matches = [atom for atom in _iter_mp4_atoms(handle, start=start, end=end) if atom.kind == kind]
    if len(matches) != 1:
        raise Mp4TimingError(
            f"Timestamp-aware MP4 requires exactly one {kind.decode('ascii', errors='replace')} atom under {owner}, "
            f"found {len(matches)}."
        )
    return matches[0]


def _read_mp4_uint(handle: Any, *, offset: int, width: int, field: str) -> int:
    handle.seek(offset)
    raw = handle.read(width)
    if len(raw) != width:
        raise Mp4TimingError(f"Timestamp-aware MP4 has a truncated {field} field at byte {offset}.")
    return int.from_bytes(raw, byteorder="big", signed=False)


def _read_mp4_int(handle: Any, *, offset: int, width: int, field: str) -> int:
    handle.seek(offset)
    raw = handle.read(width)
    if len(raw) != width:
        raise Mp4TimingError(f"Timestamp-aware MP4 has a truncated {field} field at byte {offset}.")
    return int.from_bytes(raw, byteorder="big", signed=True)


def _validate_mp4_atom_field(atom: _Mp4Atom, *, offset: int, width: int, field: str) -> None:
    if width <= 0 or offset < atom.payload_offset or offset + width > atom.end_offset:
        raise Mp4TimingError(
            f"Timestamp-aware MP4 has an invalid {field} field within {atom.kind.decode('ascii', errors='replace')} "
            f"at byte {offset}."
        )


def _validate_mp4_write(write: _Mp4Write, *, file_size: int) -> None:
    if write.width <= 0 or write.offset < 0 or write.offset + write.width > file_size:
        raise Mp4TimingError(f"Timestamp-aware MP4 has an invalid planned {write.field} write at byte {write.offset}.")
    if write.value < 0 or write.value >= 1 << (8 * write.width):
        raise Mp4TimingError(f"Timestamp-aware MP4 {write.field} value is out of range: {write.value}.")


def _write_mp4_uint(handle: Any, *, write: _Mp4Write) -> None:
    handle.seek(write.offset)
    written = handle.write(write.value.to_bytes(write.width, byteorder="big", signed=False))
    if written != write.width:
        raise Mp4TimingError(f"Timestamp-aware MP4 could not write {write.field} at byte {write.offset}.")


def _mp4_version(handle: Any, atom: _Mp4Atom, *, owner: str) -> int:
    _validate_mp4_atom_field(atom, offset=atom.payload_offset, width=1, field=f"{owner} version")
    return _read_mp4_uint(handle, offset=atom.payload_offset, width=1, field=f"{owner} version")


def _mp4_timescale_and_duration_fields(handle: Any, atom: _Mp4Atom, *, owner: str) -> tuple[int, int, int]:
    version = _mp4_version(handle, atom, owner=owner)
    if version == 0:
        timescale_offset = atom.payload_offset + 12
        duration_offset = atom.payload_offset + 16
        duration_width = 4
    elif version == 1:
        timescale_offset = atom.payload_offset + 20
        duration_offset = atom.payload_offset + 24
        duration_width = 8
    else:
        raise Mp4TimingError(f"Timestamp-aware MP4 {owner} atom has unsupported version {version}.")
    _validate_mp4_atom_field(atom, offset=timescale_offset, width=4, field=f"{owner} timescale")
    _validate_mp4_atom_field(atom, offset=duration_offset, width=duration_width, field=f"{owner} duration")
    timescale = _read_mp4_uint(handle, offset=timescale_offset, width=4, field=f"{owner} timescale")
    if timescale <= 0:
        raise Mp4TimingError(f"Timestamp-aware MP4 {owner} atom has invalid timescale {timescale}.")
    return timescale, duration_offset, duration_width


def _mp4_track_duration_field(handle: Any, atom: _Mp4Atom) -> tuple[int, int]:
    version = _mp4_version(handle, atom, owner="tkhd")
    if version == 0:
        duration_offset, duration_width = atom.payload_offset + 20, 4
    elif version == 1:
        duration_offset, duration_width = atom.payload_offset + 28, 8
    else:
        raise Mp4TimingError(f"Timestamp-aware MP4 tkhd atom has unsupported version {version}.")
    _validate_mp4_atom_field(atom, offset=duration_offset, width=duration_width, field="tkhd duration")
    return duration_offset, duration_width


def _mp4_edit_duration_fields(handle: Any, atom: _Mp4Atom) -> tuple[int, int, int]:
    version = _mp4_version(handle, atom, owner="elst")
    entry_count_offset = atom.payload_offset + 4
    _validate_mp4_atom_field(atom, offset=entry_count_offset, width=4, field="elst entry count")
    entry_count = _read_mp4_uint(handle, offset=entry_count_offset, width=4, field="elst entry count")
    if entry_count not in {1, 2}:
        raise Mp4TimingError(
            "Timestamp-aware MP4 requires one media edit with at most one leading empty edit, "
            f"found {entry_count}."
        )
    if version == 0:
        duration_width, entry_width = 4, 12
    elif version == 1:
        duration_width, entry_width = 8, 20
    else:
        raise Mp4TimingError(f"Timestamp-aware MP4 elst atom has unsupported version {version}.")

    entries_offset = atom.payload_offset + 8
    _validate_mp4_atom_field(
        atom,
        offset=entries_offset,
        width=entry_count * entry_width,
        field="elst entries",
    )
    entries: list[tuple[int, int, int]] = []
    for entry_index in range(entry_count):
        duration_offset = entries_offset + entry_index * entry_width
        media_time_offset = duration_offset + duration_width
        rate_offset = media_time_offset + duration_width
        segment_duration = _read_mp4_uint(
            handle,
            offset=duration_offset,
            width=duration_width,
            field=f"elst[{entry_index}] segment duration",
        )
        media_time = _read_mp4_int(
            handle,
            offset=media_time_offset,
            width=duration_width,
            field=f"elst[{entry_index}] media time",
        )
        rate_integer = _read_mp4_int(
            handle,
            offset=rate_offset,
            width=2,
            field=f"elst[{entry_index}] media rate integer",
        )
        rate_fraction = _read_mp4_int(
            handle,
            offset=rate_offset + 2,
            width=2,
            field=f"elst[{entry_index}] media rate fraction",
        )
        if (rate_integer, rate_fraction) != (1, 0):
            raise Mp4TimingError(
                "Timestamp-aware MP4 requires normal-rate video edit-list entries; "
                f"elst[{entry_index}] has rate {rate_integer}+{rate_fraction}/65536."
            )
        entries.append((duration_offset, segment_duration, media_time))

    leading_empty_duration_units = 0
    media_entry_index = 0
    if entry_count == 2:
        _, leading_empty_duration_units, leading_media_time = entries[0]
        if leading_empty_duration_units <= 0 or leading_media_time != -1:
            raise Mp4TimingError(
                "Timestamp-aware MP4 has an invalid leading video edit; expected a positive empty edit."
            )
        media_entry_index = 1

    media_duration_offset, _, media_time = entries[media_entry_index]
    if media_time != 0:
        raise Mp4TimingError(
            "Timestamp-aware MP4 requires the video media edit to start at media time zero; "
            f"found {media_time}."
        )
    return media_duration_offset, duration_width, leading_empty_duration_units


def _find_mp4_video_track(handle: Any, moov_atom: _Mp4Atom) -> _Mp4Atom:
    video_tracks: list[_Mp4Atom] = []
    for track_atom in _iter_mp4_atoms(handle, start=moov_atom.payload_offset, end=moov_atom.end_offset):
        if track_atom.kind != b"trak":
            continue
        mdia_atom = _find_required_mp4_atom(
            handle,
            start=track_atom.payload_offset,
            end=track_atom.end_offset,
            kind=b"mdia",
            owner="trak",
        )
        hdlr_atom = _find_required_mp4_atom(
            handle,
            start=mdia_atom.payload_offset,
            end=mdia_atom.end_offset,
            kind=b"hdlr",
            owner="mdia",
        )
        _validate_mp4_atom_field(hdlr_atom, offset=hdlr_atom.payload_offset + 8, width=4, field="hdlr handler type")
        handle.seek(hdlr_atom.payload_offset + 8)
        if handle.read(4) == b"vide":
            video_tracks.append(track_atom)
    if len(video_tracks) != 1:
        raise Mp4TimingError(f"Timestamp-aware MP4 requires exactly one video track, found {len(video_tracks)}.")
    return video_tracks[0]


def terminal_frame_duration_seconds(timing: VideoTiming) -> float:
    terminal_duration = timing.frames[-1].duration_seconds
    if terminal_duration is None or not math.isfinite(float(terminal_duration)) or float(terminal_duration) <= 0:
        raise Mp4TimingError("Timestamp-aware export requires a positive decoded duration for the terminal source frame.")
    return float(terminal_duration)


def expected_video_duration_seconds(timing: VideoTiming) -> float:
    first_presentation = float(timing.frames[0].presentation_seconds)
    last_presentation = float(timing.frames[-1].presentation_seconds)
    expected_duration = last_presentation - first_presentation + terminal_frame_duration_seconds(timing)
    if not math.isfinite(expected_duration) or expected_duration <= 0:
        raise Mp4TimingError(f"Timestamp-aware export calculated invalid video duration {expected_duration!r}.")
    return expected_duration


def patch_terminal_frame_duration(
    *,
    staged_path: Path,
    timing: VideoTiming,
    timing_tolerance_seconds: float,
) -> None:
    """Patch the staged video track only after ffmpeg has written the canonical MP4."""

    expected_duration_seconds = expected_video_duration_seconds(timing)
    terminal_duration_seconds = terminal_frame_duration_seconds(timing)
    with staged_path.open("r+b") as handle:
        file_size = handle.seek(0, 2)
        moov_atom = _find_required_mp4_atom(handle, start=0, end=file_size, kind=b"moov", owner="MP4 file")
        mvhd_atom = _find_required_mp4_atom(
            handle, start=moov_atom.payload_offset, end=moov_atom.end_offset, kind=b"mvhd", owner="moov"
        )
        track_atom = _find_mp4_video_track(handle, moov_atom)
        tkhd_atom = _find_required_mp4_atom(
            handle, start=track_atom.payload_offset, end=track_atom.end_offset, kind=b"tkhd", owner="video trak"
        )
        edts_atom = _find_required_mp4_atom(
            handle, start=track_atom.payload_offset, end=track_atom.end_offset, kind=b"edts", owner="video trak"
        )
        elst_atom = _find_required_mp4_atom(
            handle, start=edts_atom.payload_offset, end=edts_atom.end_offset, kind=b"elst", owner="edts"
        )
        mdia_atom = _find_required_mp4_atom(
            handle, start=track_atom.payload_offset, end=track_atom.end_offset, kind=b"mdia", owner="video trak"
        )
        mdhd_atom = _find_required_mp4_atom(
            handle, start=mdia_atom.payload_offset, end=mdia_atom.end_offset, kind=b"mdhd", owner="mdia"
        )
        minf_atom = _find_required_mp4_atom(
            handle, start=mdia_atom.payload_offset, end=mdia_atom.end_offset, kind=b"minf", owner="mdia"
        )
        stbl_atom = _find_required_mp4_atom(
            handle, start=minf_atom.payload_offset, end=minf_atom.end_offset, kind=b"stbl", owner="minf"
        )
        stts_atom = _find_required_mp4_atom(
            handle, start=stbl_atom.payload_offset, end=stbl_atom.end_offset, kind=b"stts", owner="stbl"
        )

        movie_timescale, mvhd_duration_offset, mvhd_duration_width = _mp4_timescale_and_duration_fields(
            handle, mvhd_atom, owner="mvhd"
        )
        media_timescale, mdhd_duration_offset, mdhd_duration_width = _mp4_timescale_and_duration_fields(
            handle, mdhd_atom, owner="mdhd"
        )
        entry_count_offset = stts_atom.payload_offset + 4
        _validate_mp4_atom_field(stts_atom, offset=entry_count_offset, width=4, field="stts entry count")
        entry_count = _read_mp4_uint(handle, offset=entry_count_offset, width=4, field="stts entry count")
        if entry_count <= 0:
            raise Mp4TimingError("Timestamp-aware MP4 video track has no stts timing entries.")
        entry_offset = stts_atom.payload_offset + 8
        if entry_offset + entry_count * 8 > stts_atom.end_offset:
            raise Mp4TimingError("Timestamp-aware MP4 stts entries exceed the declared stts atom boundary.")
        sample_count = 0
        accumulated_duration_units = 0
        final_entry_offset: int | None = None
        final_entry_sample_count = 0
        for entry_index in range(entry_count):
            current_offset = entry_offset + entry_index * 8
            count = _read_mp4_uint(handle, offset=current_offset, width=4, field=f"stts[{entry_index}] sample count")
            duration_units = _read_mp4_uint(
                handle, offset=current_offset + 4, width=4, field=f"stts[{entry_index}] sample duration"
            )
            sample_count += count
            accumulated_duration_units += count * duration_units
            final_entry_offset = current_offset
            final_entry_sample_count = count
        if sample_count != timing.frame_count:
            raise Mp4TimingError(
                "Timestamp-aware MP4 sample/timing count mismatch before terminal-duration patch: "
                f"samples={sample_count}, timing={timing.frame_count}."
            )
        if final_entry_offset is None or final_entry_sample_count != 1:
            raise Mp4TimingError("Timestamp-aware MP4 cannot isolate the terminal video sample duration for this encoder output.")

        terminal_duration_units = int(round(terminal_duration_seconds * media_timescale))
        if terminal_duration_units <= 0:
            raise Mp4TimingError("Timestamp-aware MP4 terminal duration rounded to zero media ticks.")
        current_terminal_units = _read_mp4_uint(
            handle, offset=final_entry_offset + 4, width=4, field="terminal stts sample duration"
        )
        prefix_duration_units = accumulated_duration_units - current_terminal_units
        source_last_offset_units = int(
            round((float(timing.frames[-1].presentation_seconds) - float(timing.frames[0].presentation_seconds)) * media_timescale)
        )
        timing_tolerance_units = max(1, int(math.ceil(timing_tolerance_seconds * media_timescale)))
        if abs(prefix_duration_units - source_last_offset_units) > timing_tolerance_units:
            raise Mp4TimingError(
                "Timestamp-aware MP4 did not preserve source presentation offsets before terminal-duration patch: "
                f"source={source_last_offset_units}, encoded={prefix_duration_units}, timescale={media_timescale}."
            )

        video_duration_units = prefix_duration_units + terminal_duration_units
        patched_duration_seconds = float(video_duration_units) / float(media_timescale)
        if abs(patched_duration_seconds - expected_duration_seconds) > timing_tolerance_seconds:
            raise Mp4TimingError(
                "Timestamp-aware MP4 terminal duration cannot represent the decoded source timing within tolerance: "
                f"source={expected_duration_seconds:.9f}, patched={patched_duration_seconds:.9f}."
            )
        media_edit_duration_units = int(round(patched_duration_seconds * movie_timescale))
        if media_edit_duration_units <= 0:
            raise Mp4TimingError("Timestamp-aware MP4 terminal duration rounded to zero movie ticks.")

        tkhd_duration_offset, tkhd_duration_width = _mp4_track_duration_field(handle, tkhd_atom)
        elst_duration_offset, elst_duration_width, leading_empty_duration_units = _mp4_edit_duration_fields(
            handle,
            elst_atom,
        )
        track_duration_units = leading_empty_duration_units + media_edit_duration_units
        current_movie_duration = _read_mp4_uint(
            handle, offset=mvhd_duration_offset, width=mvhd_duration_width, field="mvhd duration"
        )
        writes = (
            _Mp4Write(
                offset=final_entry_offset + 4,
                width=4,
                value=terminal_duration_units,
                field="terminal stts sample duration",
            ),
            _Mp4Write(
                offset=mdhd_duration_offset,
                width=mdhd_duration_width,
                value=video_duration_units,
                field="mdhd duration",
            ),
            _Mp4Write(
                offset=tkhd_duration_offset,
                width=tkhd_duration_width,
                value=track_duration_units,
                field="tkhd duration",
            ),
            _Mp4Write(
                offset=elst_duration_offset,
                width=elst_duration_width,
                value=media_edit_duration_units,
                field="elst media segment duration",
            ),
            _Mp4Write(
                offset=mvhd_duration_offset,
                width=mvhd_duration_width,
                value=max(current_movie_duration, track_duration_units),
                field="mvhd duration",
            ),
        )
        for write in writes:
            _validate_mp4_write(write, file_size=file_size)
            _read_mp4_uint(handle, offset=write.offset, width=write.width, field=f"current {write.field}")
        for write in writes:
            _write_mp4_uint(handle, write=write)
