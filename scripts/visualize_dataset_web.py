"""Lightweight dataset visualization web app.

Browse the breast MRI multimodal manifest in a browser:

* List every (patient, visit) sample with its ``dataset_id`` and ``split``.
* Click into a sample to see its labels (parsed from the manifest column
  values + the on-disk ``labels.json`` if present), plus thumbnails of every
  npy volume grouped by modality.
* For each volume show three orthogonal slices and a slice slider so you can
  scroll through axial / sagittal / coronal views without leaving the page.

The server only needs ``numpy`` and ``Pillow`` — no Flask, no matplotlib. Run
it pointing at any manifest CSV produced by the foundation pipeline:

    PYTHONPATH=. python -m \
        src.breast_mri_ai.breast_dce_moe_pinn_foundation.scripts.visualize_dataset_web \
        --manifest src/breast_mri_ai/breast_dce_moe_pinn_foundation/configs/foundation_manifest_full_v4.csv \
        --port 8765

Then open http://<server>:8765/ in a browser.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import urllib.parse
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


# Resolve a few well-known paths relative to this file so the script keeps
# working when invoked directly (e.g. ``python visualize_dataset_web.py``)
# without setting cwd to the project root.
_SCRIPT_DIR = Path(__file__).resolve().parent
_FOUNDATION_DIR = _SCRIPT_DIR.parent  # .../breast_dce_moe_pinn_foundation
_PROJECT_ROOT = _FOUNDATION_DIR.parents[2]  # .../breast_pcr
_DEFAULT_MANIFEST = _FOUNDATION_DIR / "configs" / "foundation_manifest_full_v4.csv"


# ---------------------------------------------------------------------------
# Manifest indexing
# ---------------------------------------------------------------------------


@dataclass
class VolumeRecord:
    modality: str
    phase_index: int | None
    relative_time: float | None
    path: Path

    def display_name(self) -> str:
        if self.phase_index is None:
            return f"{self.modality}"
        return f"{self.modality} (phase {self.phase_index})"


@dataclass
class SampleRecord:
    sample_id: str
    patient_id: str
    dataset_id: str
    visit_timepoint: str
    split: str
    labels: dict[str, str] = field(default_factory=dict)
    volumes: list[VolumeRecord] = field(default_factory=list)
    manifest_row: dict[str, str] = field(default_factory=dict)


_LABEL_KEYS = (
    "pCR",
    "HER2",
    "ER",
    "PR",
    "HR",
    "molecular_subtype",
    "survival_time",
    "survival_event",
)


def _first(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _resolve_path(value: str, manifest_dir: Path, dataset_root: Path | None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path
    candidates: list[Path] = []
    if not path.is_absolute():
        candidates.append((manifest_dir / path).resolve())
        if dataset_root is not None:
            candidates.append((dataset_root / path).resolve())
    candidates.append(path)
    for cand in candidates:
        if cand.exists():
            return cand
    return path  # may still be missing — caller handles


def index_manifest(manifest_path: Path, dataset_root: Path | None) -> list[SampleRecord]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    samples: dict[tuple[str, str, str, str], SampleRecord] = {}
    manifest_dir = manifest_path.parent

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            patient_id = _first(row, "patient_id", "subject_id", default="unknown")
            sample_id = _first(row, "sample_id", "study_uid", default=f"{patient_id}")
            dataset_id = _first(row, "dataset_id", default="unknown")
            visit = _first(row, "visit_timepoint", "timepoint", "visit", default="unknown")
            split = _first(row, "split", default="")
            key = (dataset_id, patient_id, sample_id, visit)
            sample = samples.get(key)
            if sample is None:
                labels = {k: _first(row, k, default="") for k in _LABEL_KEYS}
                sample = SampleRecord(
                    sample_id=sample_id,
                    patient_id=patient_id,
                    dataset_id=dataset_id,
                    visit_timepoint=visit,
                    split=split,
                    labels=labels,
                    manifest_row=dict(row),
                )
                samples[key] = sample

            path_value = _first(row, "path", "file_path", "npy_path", "volume_path", default="")
            if not path_value:
                continue
            resolved = _resolve_path(path_value, manifest_dir, dataset_root)
            modality = _first(row, "modality", default="unknown")
            phase_value = row.get("phase_index", "")
            try:
                phase_index = int(phase_value) if phase_value not in (None, "", "nan") else None
            except (TypeError, ValueError):
                phase_index = None
            try:
                relative_time = float(row.get("relative_time", "")) if row.get("relative_time") not in (None, "") else None
            except (TypeError, ValueError):
                relative_time = None
            sample.volumes.append(
                VolumeRecord(
                    modality=modality,
                    phase_index=phase_index,
                    relative_time=relative_time,
                    path=resolved,
                )
            )

    ordered: list[SampleRecord] = []
    for sample in samples.values():
        sample.volumes.sort(
            key=lambda v: (
                _modality_order(v.modality),
                999 if v.phase_index is None else v.phase_index,
                str(v.path),
            )
        )
        ordered.append(sample)
    ordered.sort(key=lambda s: (s.dataset_id, s.split, s.patient_id, s.sample_id, s.visit_timepoint))
    return ordered


_MODALITY_ORDER = {"DCE": 0, "T1": 1, "T2": 2, "DWI": 3, "ADC": 4, "derived": 5}


def _modality_order(modality: str) -> int:
    return _MODALITY_ORDER.get(modality, 90)


# ---------------------------------------------------------------------------
# NPY → PNG rendering
# ---------------------------------------------------------------------------


def _load_volume(path: Path) -> np.ndarray:
    arr = np.load(path, mmap_mode="r")
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if arr.ndim == 4:
        # take first channel for visualization
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D/4D volume at {path}, got {arr.shape}")
    return arr


def _normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 1.0))
    hi = float(np.percentile(finite, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(slice_2d, lo, hi)
    out = ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return out


def render_slice_png(
    path: Path, axis: int, index: int | None = None, max_size: int = 384
) -> bytes:
    """Render a single 2D slice from the npy volume into a PNG byte string."""

    volume = _load_volume(path)
    axis = max(0, min(2, axis))
    depth = volume.shape[axis]
    if index is None:
        index = depth // 2
    index = max(0, min(depth - 1, index))
    if axis == 0:
        slice_2d = volume[index, :, :]
    elif axis == 1:
        slice_2d = volume[:, index, :]
    else:
        slice_2d = volume[:, :, index]
    pixels = _normalize_for_display(np.asarray(slice_2d, dtype=np.float32))
    image = Image.fromarray(pixels, mode="L")
    h, w = image.size[1], image.size[0]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1.0:
        image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def get_volume_shape(path: Path) -> tuple[int, int, int]:
    volume = _load_volume(path)
    return tuple(int(d) for d in volume.shape)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_BASE_CSS = """
:root {
  --bg:#0d1117; --panel:#161b22; --border:#30363d; --text:#e6edf3;
  --muted:#8b949e; --accent:#58a6ff;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg); color: var(--text);
}
header {
  padding: 14px 22px; border-bottom: 1px solid var(--border);
  display: flex; align-items: center; gap: 14px;
  background: var(--panel); position: sticky; top: 0; z-index: 5;
}
header h1 { margin: 0; font-size: 18px; font-weight: 600; }
header .meta { color: var(--muted); font-size: 13px; }
.container { padding: 18px 22px; }
input[type=text], select {
  background: #0d1117; color: var(--text); border: 1px solid var(--border);
  padding: 6px 10px; border-radius: 6px; font-size: 13px;
}
table { border-collapse: collapse; width: 100%; }
th, td { padding: 6px 10px; border-bottom: 1px solid var(--border); text-align: left; font-size: 13px; }
th { background: var(--panel); position: sticky; top: 56px; }
tr:hover td { background: rgba(88,166,255,0.06); cursor: pointer; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
.badge { display: inline-block; padding: 2px 7px; border-radius: 999px; font-size: 11px;
  background: #21262d; color: var(--muted); }
.badge.split-train { background: #1f6feb; color: #fff; }
.badge.split-val   { background: #b08800; color: #fff; }
.badge.split-test  { background: #6e40c9; color: #fff; }
.badge.split-inference { background: #1f883d; color: #fff; }
.label-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 8px; margin: 14px 0 22px; }
.label-grid .item { background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 8px 10px; font-size: 13px; }
.label-grid .item .key { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .04em; }
.label-grid .item .val { font-size: 14px; word-break: break-word; }
.modality-block { margin-bottom: 28px; }
.modality-block h3 { margin: 8px 0; font-size: 15px; color: var(--accent); }
.volume-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(380px, 1fr)); gap: 16px; }
.volume-card {
  background: var(--panel); border: 1px solid var(--border); border-radius: 8px;
  padding: 12px; display: flex; flex-direction: column; gap: 8px;
}
.volume-card h4 { margin: 0; font-size: 14px; }
.volume-card .path { font-size: 11px; color: var(--muted); word-break: break-all; }
.slice-row { display: flex; gap: 6px; }
.slice-tile { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 4px;
  background: #0d1117; border: 1px solid var(--border); border-radius: 6px; padding: 6px; }
.slice-tile img { width: 100%; max-width: 220px; border-radius: 4px; image-rendering: pixelated; background: #000; }
.slice-tile .axis { font-size: 11px; color: var(--muted); text-transform: uppercase; }
.slice-tile input[type=range] { width: 100%; }
.controls { display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
.warning { color: #f85149; font-size: 12px; }
.json-block { background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 10px; max-height: 360px; overflow: auto; font-family: ui-monospace, Menlo, monospace; font-size: 12px;
  white-space: pre-wrap; word-break: break-word; }
nav.breadcrumb { font-size: 13px; color: var(--muted); margin-bottom: 12px; }
nav.breadcrumb a { color: var(--accent); }
.summary { color: var(--muted); font-size: 13px; margin-bottom: 14px; }
.toolbar { display: flex; gap: 10px; align-items: center; margin-bottom: 12px; flex-wrap: wrap; }
"""


def _html_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _split_badge(split: str) -> str:
    cls = f"split-{_html_escape(split or 'unknown')}"
    return f'<span class="badge {cls}">{_html_escape(split or "?")}</span>'


def render_index_page(samples: list[SampleRecord], manifest_path: Path) -> str:
    rows_html: list[str] = []
    dataset_options = sorted({s.dataset_id for s in samples})
    split_options = sorted({s.split for s in samples if s.split})

    for idx, sample in enumerate(samples):
        labels_short = ", ".join(
            f"{k}={v}" for k, v in sample.labels.items() if v not in ("", None)
        )
        rows_html.append(
            f"<tr data-dataset='{_html_escape(sample.dataset_id)}' data-split='{_html_escape(sample.split)}' "
            f"onclick=\"window.location='/sample/{idx}'\">"
            f"<td>{idx}</td>"
            f"<td>{_html_escape(sample.dataset_id)}</td>"
            f"<td>{_split_badge(sample.split)}</td>"
            f"<td>{_html_escape(sample.patient_id)}</td>"
            f"<td>{_html_escape(sample.sample_id)}</td>"
            f"<td>{_html_escape(sample.visit_timepoint)}</td>"
            f"<td>{len(sample.volumes)}</td>"
            f"<td style='color:var(--muted)'>{_html_escape(labels_short)}</td>"
            f"</tr>"
        )

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>Breast MRI Dataset Browser</title>
<style>{_BASE_CSS}</style></head><body>
<header>
  <h1>Breast MRI Dataset Browser</h1>
  <span class='meta'>{_html_escape(manifest_path)} · {len(samples)} samples</span>
</header>
<div class='container'>
  <div class='toolbar'>
    <input type='text' id='filter' placeholder='Search patient / sample / dataset...' oninput='applyFilters()' style='min-width:280px;'>
    <select id='datasetFilter' onchange='applyFilters()'>
      <option value=''>All datasets</option>
      {''.join(f"<option>{_html_escape(d)}</option>" for d in dataset_options)}
    </select>
    <select id='splitFilter' onchange='applyFilters()'>
      <option value=''>All splits</option>
      {''.join(f"<option>{_html_escape(s)}</option>" for s in split_options)}
    </select>
    <span class='summary' id='summary'></span>
  </div>
  <table>
    <thead><tr>
      <th>#</th><th>Dataset</th><th>Split</th><th>Patient</th><th>Sample</th><th>Visit</th><th>Volumes</th><th>Labels</th>
    </tr></thead>
    <tbody id='rows'>
      {''.join(rows_html)}
    </tbody>
  </table>
</div>
<script>
function applyFilters() {{
  const q = document.getElementById('filter').value.toLowerCase();
  const ds = document.getElementById('datasetFilter').value;
  const sp = document.getElementById('splitFilter').value;
  let visible = 0;
  document.querySelectorAll('#rows tr').forEach(tr => {{
    const text = tr.innerText.toLowerCase();
    const matches = (!q || text.includes(q)) &&
                    (!ds || tr.dataset.dataset === ds) &&
                    (!sp || tr.dataset.split === sp);
    tr.style.display = matches ? '' : 'none';
    if (matches) visible++;
  }});
  document.getElementById('summary').innerText = visible + ' / {len(samples)} shown';
}}
applyFilters();
</script>
</body></html>"""


def render_sample_page(sample: SampleRecord, sample_idx: int, total: int) -> str:
    label_items: list[str] = []
    for key, value in sample.labels.items():
        if value in ("", None):
            continue
        label_items.append(
            f"<div class='item'><div class='key'>{_html_escape(key)}</div>"
            f"<div class='val'>{_html_escape(value)}</div></div>"
        )
    label_grid = (
        "<div class='label-grid'>" + "".join(label_items) + "</div>"
        if label_items
        else "<div class='summary'>No labels for this sample.</div>"
    )

    by_modality: dict[str, list[VolumeRecord]] = {}
    for vol in sample.volumes:
        by_modality.setdefault(vol.modality, []).append(vol)

    modality_html: list[str] = []
    for modality, volumes in by_modality.items():
        cards: list[str] = []
        for vol_idx, vol in enumerate(volumes):
            cards.append(_render_volume_card(sample_idx, vol_idx, vol))
        modality_html.append(
            f"<section class='modality-block'><h3>{_html_escape(modality)} · {len(volumes)} volume(s)</h3>"
            f"<div class='volume-grid'>{''.join(cards)}</div></section>"
        )

    extra_labels = _read_disk_labels(sample)
    disk_block = (
        f"<h3 style='margin-top:32px;'>labels.json on disk</h3>"
        f"<pre class='json-block'>{_html_escape(json.dumps(extra_labels, ensure_ascii=False, indent=2))}</pre>"
        if extra_labels
        else ""
    )

    prev_link = (
        f"<a href='/sample/{sample_idx - 1}'>&larr; prev</a>" if sample_idx > 0 else "<span style='color:var(--muted)'>&larr; prev</span>"
    )
    next_link = (
        f"<a href='/sample/{sample_idx + 1}'>next &rarr;</a>" if sample_idx + 1 < total else "<span style='color:var(--muted)'>next &rarr;</span>"
    )

    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{_html_escape(sample.sample_id)}</title>
<style>{_BASE_CSS}</style></head><body>
<header>
  <h1>{_html_escape(sample.sample_id)}</h1>
  <span class='meta'>{_html_escape(sample.dataset_id)} · {_split_badge(sample.split)} · visit {_html_escape(sample.visit_timepoint)}</span>
</header>
<div class='container'>
  <nav class='breadcrumb'>
    <a href='/'>&larr; All samples</a> &nbsp; · &nbsp; {prev_link} &nbsp; · &nbsp; {next_link}
  </nav>
  <div class='summary'>Patient: {_html_escape(sample.patient_id)} &middot; {len(sample.volumes)} volumes total</div>
  {label_grid}
  {''.join(modality_html) if modality_html else "<div class='summary'>No volumes attached.</div>"}
  {disk_block}
</div>
</body></html>"""


def _render_volume_card(sample_idx: int, vol_idx: int, vol: VolumeRecord) -> str:
    if not vol.path.exists():
        return (
            f"<div class='volume-card'><h4>{_html_escape(vol.display_name())}</h4>"
            f"<div class='path'>{_html_escape(vol.path)}</div>"
            f"<div class='warning'>npy file not found.</div></div>"
        )

    try:
        shape = get_volume_shape(vol.path)
    except Exception as exc:  # noqa: BLE001
        return (
            f"<div class='volume-card'><h4>{_html_escape(vol.display_name())}</h4>"
            f"<div class='path'>{_html_escape(vol.path)}</div>"
            f"<div class='warning'>Failed to load: {_html_escape(exc)}</div></div>"
        )

    axis_names = ("axial (Z)", "coronal (Y)", "sagittal (X)")
    tiles: list[str] = []
    for axis in range(3):
        depth = shape[axis]
        mid = depth // 2
        img_id = f"img-{sample_idx}-{vol_idx}-{axis}"
        slider_id = f"sl-{sample_idx}-{vol_idx}-{axis}"
        label_id = f"lbl-{sample_idx}-{vol_idx}-{axis}"
        url = f"/render?sample={sample_idx}&vol={vol_idx}&axis={axis}&index={mid}"
        tiles.append(
            f"<div class='slice-tile'>"
            f"<span class='axis'>{axis_names[axis]} · <span id='{label_id}'>{mid}</span> / {depth - 1}</span>"
            f"<img id='{img_id}' src='{url}' loading='lazy'>"
            f"<input id='{slider_id}' type='range' min='0' max='{depth - 1}' value='{mid}' "
            f"oninput=\"updateSlice({sample_idx},{vol_idx},{axis})\">"
            f"</div>"
        )

    rel = vol.relative_time
    rel_text = "" if rel is None else f" · t={rel:g}"
    return (
        f"<div class='volume-card'>"
        f"<h4>{_html_escape(vol.display_name())} <span class='path' style='font-weight:normal'>shape={shape}{_html_escape(rel_text)}</span></h4>"
        f"<div class='path'>{_html_escape(vol.path)}</div>"
        f"<div class='slice-row'>{''.join(tiles)}</div>"
        f"<script>"
        f"function updateSlice(s,v,a){{"
        f"  const sl=document.getElementById('sl-'+s+'-'+v+'-'+a);"
        f"  const img=document.getElementById('img-'+s+'-'+v+'-'+a);"
        f"  const lbl=document.getElementById('lbl-'+s+'-'+v+'-'+a);"
        f"  lbl.innerText=sl.value;"
        f"  img.src='/render?sample='+s+'&vol='+v+'&axis='+a+'&index='+sl.value+'&t='+Date.now();"
        f"}}"
        f"</script>"
        f"</div>"
    )


def _read_disk_labels(sample: SampleRecord) -> dict[str, Any] | None:
    if not sample.volumes:
        return None
    # Walk up from the first volume looking for labels.json next to the visit
    # directory or its patient parent — that is the on-disk convention used by
    # full_v4_*/<dataset>/<patient>/<visit>/.
    parents: list[Path] = []
    seen: set[Path] = set()
    for vol in sample.volumes:
        for ancestor in vol.path.parents:
            if ancestor in seen:
                continue
            seen.add(ancestor)
            parents.append(ancestor)
            if len(parents) > 6:
                break
    for parent in parents:
        for name in ("labels_optimized.json", "labels.json"):
            candidate = parent / name
            if candidate.exists():
                try:
                    return json.loads(candidate.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    return None
    return None


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    samples: list[SampleRecord] = []
    manifest_path: Path = Path()

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.info("%s - - [%s] %s", self.address_string(), self.log_date_time_string(), format % args)

    # The handler dispatches three URL families:
    #   GET /                  -> sample index
    #   GET /sample/<idx>      -> per-sample volume + label page
    #   GET /render?...        -> a single PNG slice
    def do_GET(self) -> None:  # noqa: N802
        try:
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/":
                self._send_html(render_index_page(self.samples, self.manifest_path))
                return
            sample_match = re.match(r"^/sample/(\d+)$", parsed.path)
            if sample_match:
                idx = int(sample_match.group(1))
                if 0 <= idx < len(self.samples):
                    self._send_html(render_sample_page(self.samples[idx], idx, len(self.samples)))
                else:
                    self._send_text(404, "sample index out of range")
                return
            if parsed.path == "/render":
                self._handle_render(urllib.parse.parse_qs(parsed.query))
                return
            self._send_text(404, "not found")
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("error handling %s", self.path)
            self._send_text(500, f"server error: {exc}")

    def _handle_render(self, params: dict[str, list[str]]) -> None:
        try:
            sample_idx = int(params["sample"][0])
            vol_idx = int(params["vol"][0])
            axis = int(params.get("axis", ["0"])[0])
            index_param = params.get("index", [None])[0]
            index = int(index_param) if index_param is not None else None
        except (KeyError, ValueError):
            self._send_text(400, "bad parameters")
            return
        if not 0 <= sample_idx < len(self.samples):
            self._send_text(404, "no such sample")
            return
        sample = self.samples[sample_idx]
        if not 0 <= vol_idx < len(sample.volumes):
            self._send_text(404, "no such volume")
            return
        vol = sample.volumes[vol_idx]
        if not vol.path.exists():
            self._send_text(404, "npy missing on disk")
            return
        try:
            png = render_slice_png(vol.path, axis=axis, index=index)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("render failed for %s", vol.path)
            self._send_text(500, f"render error: {exc}")
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Cache-Control", "max-age=300")
        self.send_header("Content-Length", str(len(png)))
        self.end_headers()
        self.wfile.write(png)

    def _send_html(self, body: str) -> None:
        payload = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_text(self, status: int, message: str) -> None:
        payload = message.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def serve(samples: list[SampleRecord], manifest_path: Path, host: str, port: int) -> None:
    _Handler.samples = samples
    _Handler.manifest_path = manifest_path
    server = ThreadingHTTPServer((host, port), _Handler)
    LOGGER.info("Serving %d samples on http://%s:%d/", len(samples), host, port)
    LOGGER.info("Press Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        LOGGER.info("Shutting down.")
    finally:
        server.server_close()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Web viewer for breast MRI npy + labels.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_DEFAULT_MANIFEST,
        help="Path to the foundation manifest CSV.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=None,
        help="Optional dataset root used to resolve relative paths inside the manifest.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8765, help="Port (default 8765).")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of samples to load.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    manifest = _resolve_manifest_arg(args.manifest)
    samples = index_manifest(manifest, args.dataset_root)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        print(f"No samples loaded from {manifest}", file=sys.stderr)
        return 1
    serve(samples, manifest, args.host, args.port)
    return 0


def _resolve_manifest_arg(path: Path) -> Path:
    """Find ``path`` against cwd / project root / foundation configs.

    The script is convenient to run via ``python path/to/visualize_dataset_web.py``
    from any directory, so we don't want a relative manifest argument to die
    just because the cwd happens to differ from the project root.
    """

    candidates = [path]
    if not path.is_absolute():
        candidates.append(_PROJECT_ROOT / path)
        candidates.append(_FOUNDATION_DIR / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # Falls through with the original — index_manifest will raise with the
    # caller-specified path so the error message matches what they asked for.
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
