#!/usr/bin/env python3
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
"""Drops initcheck's vendor-library-only reports, keeps everything else.

compute-sanitizer does not instrument cuDNN's precompiled engine kernels, so
cuDNN's own staging buffers never acquire a shadow: an engine kernel that reads
memory it wrote earlier in the same plan (or that it never wrote, for a padded
staging buffer) reports as an uninitialized read. The report's device stack is
entirely vendor frames -- there is no Mojo kernel to attribute it to.

The rule this implements: a report with no `.mojo` frame AND a vendor library
on its host backtrace is the vendor's, not ours; a report with any Mojo frame
is a finding. The calibration is the four conv targets this lane runs clean
after `VENDOR_CUDNN_ZERO_OUTPUT`/`VENDOR_BLAS_ZERO_OUTPUT` zeroed what the
vendor was about to write: before that zeroing, each of their reports carried a
Mojo frame (`nn_conv_conv_closure_conv_gpu...`, `ref_epilogue_kernel`, the
vendor-matmul epilogue) and this rule would have kept every one of them.

Deliberately a *filter*, not a findings pass of its own: it takes the same
marker regex `run_sanitizer.sh` greps with, emits `grep -n`-shaped output, and
the only lines it can ever remove are report headers it has positively
attributed to a vendor library. Anything it cannot resolve -- a missing host
backtrace, a frame it cannot parse, a marker it has no rule for -- is
reported. A suppression rule must fail toward reporting.

    classify_initcheck.py --markers <regex> <test.log>

Exits 1 when findings survive, 0 when none do. Suppressed reports are
summarized on stderr, never hidden.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The only report class this script has a rule for. Everything else the sweep
# gates on passes through unchanged.
_UNINIT = re.compile(r"Uninitialized __(?:global|shared|local|device)__")

# Libraries whose kernels compute-sanitizer leaves uninstrumented, so writes
# they perform leave no shadow. `libcudnn_engines_precompiled` is where the
# padded staging kernels live; the others cover cuDNN's dispatcher and cuBLAS.
_VENDOR_LIBS = (
    "libcudnn",
    "libcublas",
)

# A Mojo device frame, e.g.  `at nn_conv_...6A6A+0x520 in conv.mojo:6353`.
_MOJO_FRAME = re.compile(r"in [\w./-]+\.mojo:\d+")

# The first line of any report, which is what ends the previous one. The
# sanitizer's own lines are one contiguous `=========` stream, so records
# cannot be split on anything else.
_REPORT_TITLE = re.compile(
    r"^=+\s+(?:Uninitialized|Invalid|Race reported|Barrier error|misaligned|"
    r"MemoryManager detected|CUDA_EXCEPTION|illegal memory access)"
)

_PREFIX = "========="


def _is_device_frame(line: str) -> bool:
    """Whether a report line is the faulting kernel or one of its device frames."""
    body = line[len(_PREFIX) :].strip() if line.startswith(_PREFIX) else line
    return body.startswith(("at ", "Device Frame:"))


def _vendor_only(record: list[str]) -> bool:
    """Whether a report is entirely a vendor library's, with no Mojo code.

    The attribution is the *device* stack: host frames name the MAX call site
    that reached the vendor library, which is not the kernel doing the read.
    """
    device_frames = [line for line in record if _is_device_frame(line)]
    if any(_MOJO_FRAME.search(line) for line in device_frames):
        return False
    return any(lib in line for line in record for lib in _VENDOR_LIBS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markers", required=True, help="ERE the sweep gates on"
    )
    parser.add_argument("log", help="path to a target's test.log")
    args = parser.parse_args()

    markers = re.compile(args.markers)
    lines = Path(args.log).read_text(errors="replace").splitlines()

    suppressed: set[int] = set()
    sites: dict[str, int] = {}

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if not (line.startswith(_PREFIX) and _UNINIT.search(line)):
            idx += 1
            continue
        # The record runs to the next report title.
        end = idx + 1
        while end < len(lines) and not _REPORT_TITLE.match(lines[end]):
            end += 1
        record = lines[idx:end]
        if _vendor_only(record):
            suppressed.update(range(idx + 1, end + 1))
            kernel = record[0].split(_PREFIX, 1)[1].strip()
            key = kernel.split("+0x", 1)[0].strip()[:80]
            sites[key] = sites.get(key, 0) + 1
        idx = end

    survivors = [
        f"{n}:{text}"
        for n, text in enumerate(lines, start=1)
        if markers.search(text)
        and "ERROR SUMMARY: 0 errors" not in text
        and n not in suppressed
    ]

    if sites:
        total = sum(sites.values())
        print(
            f"initcheck: suppressed {total} vendor-library-only report(s) -- "
            f"compute-sanitizer does not instrument cuDNN/cuBLAS kernels, so "
            f"their writes leave no shadow. No Mojo frame in any of them.",
            file=sys.stderr,
        )
        for kernel, count in sorted(sites.items()):
            print(f"  {count} x {kernel}", file=sys.stderr)

    for survivor in survivors:
        print(survivor)

    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
