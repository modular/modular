# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Compose the split CPU + accelerator warms into one GPU-consumer cache dir.

Takes ``warm_cpu``'s output dir and ``warm_accelerators``'s output dir and
produces the single directory the GPU consumer test adopts, equivalent in
shape to the old unified ``eager_gc_warm`` output, so the consumer is
unchanged:

- every ``*.mef`` from both inputs (CPU slot + accelerator slots 0..N-1);
- the accelerator warm's ``cache/.max_cache/``, which carries the GPU-context
  warm stamp the consumer's ``warm_stamp_matches`` checks
  (``accelerators=N;...``), NOT ``warm_cpu``'s ``accelerators=0`` stamp;
- one merged ``manifest.json``: ``warm_cpu``'s envelope (``host_arch``,
  ``cpu_target``, ``toolchain``) plus ``warm_accelerators``' gpu bits
  (``gpu.arch``, ``device_count``), and the union of both entry lists (cpu +
  gpu:0..N-1).

No MAX runtime on purpose, no device, no recompile: it only moves files and
merges two JSON manifests. This dir is exactly what ``manifest_adoptable`` +
``_adopt_*_from_manifest`` already read.
"""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterable
from pathlib import Path

import click

_MANIFEST_NAME = "manifest.json"
# Mirrors gc_compile._cache_dir(): <MODULAR_DERIVED_PATH>/cache/.max_cache.
_CACHE_SUBDIR = Path("cache") / ".max_cache"


def _copy_mefs(sources: Iterable[Path], out: Path) -> None:
    """Copy every per-slot ``*.mef`` from *sources* into *out* under its bare
    name (the manifest references MEFs by bare name relative to
    MODULAR_DERIVED_PATH, so they must sit alongside it)."""
    for src in sources:
        for mef in sorted(src.glob("*.mef")):
            shutil.copy2(mef, out / mef.name)


def _copy_gpu_stamp(accel_dir: Path, out: Path) -> None:
    """Copy the accelerator warm's ``cache/.max_cache`` into *out*: it holds the
    GPU-context stamp (``accelerators=N;...``) the consumer's
    warm_stamp_matches() must see, not warm_cpu's ``accelerators=0`` stamp."""
    shutil.copytree(
        accel_dir / _CACHE_SUBDIR, out / _CACHE_SUBDIR, dirs_exist_ok=True
    )


def _merge_manifests(
    cpu_dir: Path, accel_dir: Path, out: Path
) -> tuple[dict[str, object], list[dict[str, str]]]:
    """Merge warm_cpu's + warm_accelerators' manifests into the single GPU
    envelope the consumer reads, write it to *out*, and return (envelope,
    entries). json.loads returns Any (the loose manifest shape the consumer
    reads back the same way)."""
    cpu_manifest = json.loads((cpu_dir / _MANIFEST_NAME).read_text())
    accel_manifest = json.loads((accel_dir / _MANIFEST_NAME).read_text())
    cpu_env = cpu_manifest["envelope"]
    accel_env = accel_manifest["envelope"]
    envelope: dict[str, object] = {
        "host_arch": cpu_env["host_arch"],
        "cpu_target": cpu_env["cpu_target"],
        "toolchain": cpu_env["toolchain"],
        "gpu": accel_env["gpu"],
        "device_count": accel_env["device_count"],
    }
    entries = list(cpu_manifest["entries"]) + list(accel_manifest["entries"])
    (out / _MANIFEST_NAME).write_text(
        json.dumps({"envelope": envelope, "entries": entries}, indent=2)
    )
    return envelope, entries


@click.command()
@click.option(
    "--cpu-warm",
    "cpu_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="warm_cpu output dir (CPU MEFs + CPU manifest).",
)
@click.option(
    "--accelerators-warm",
    "accel_dir",
    required=True,
    type=click.Path(path_type=Path),
    help="warm_accelerators output dir (accelerator MEFs + manifest fragment).",
)
def main(cpu_dir: Path, accel_dir: Path) -> None:
    out = Path(os.environ["MODULAR_DERIVED_PATH"])
    _copy_mefs((cpu_dir, accel_dir), out)
    _copy_gpu_stamp(accel_dir, out)
    envelope, entries = _merge_manifests(cpu_dir, accel_dir, out)
    print(
        f"XARCH_COMPOSE: merged {len(entries)} entries "
        f"({sorted(e['device_class'] for e in entries)}); envelope={envelope}",
        flush=True,
    )


if __name__ == "__main__":
    main()
