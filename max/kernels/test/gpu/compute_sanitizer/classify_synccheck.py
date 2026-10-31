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
"""Drops synccheck's named-barrier divergence noise, keeps everything else.

`named_barrier[N](id)` lowers to `bar.sync id, N`, and PTX lets the
participating warps reach it from different instruction addresses provided they
all arrive and all name the same count -- which is exactly what a partial-CTA
rendezvous in a warp-specialized kernel is. synccheck models a barrier as one
instruction address, so it reports every such rendezvous as "Divergent
thread(s) in block": about 48 SM100 attention/MLA targets, all noise. NVIDIA
supports `--suppressions` for memcheck, initcheck and racecheck but not for
synccheck, so the filtering has to happen on our side.

The rule this implements: a divergence attributed to a `named_barrier` call
site is the tool's gap; a divergence attributed to anything else -- a plain
`barrier()` above all -- is a finding. The negative control
`negative_control_synccheck_named_barrier.mojo` calibrates the rule, and
`positive_control_synccheck_barrier.mojo`, the same kernel on a plain
`barrier()`, is the case it must never swallow.

Deliberately a *filter*, not a findings pass of its own: it takes the same
marker regex `run_sanitizer.sh` greps with, emits `grep -n`-shaped output, and
the only lines it can ever remove are divergence headers it has positively
attributed to a `named_barrier`. Anything it cannot resolve -- an unreadable
path, a frame it cannot parse, a marker it has no rule for (`Missing wait`, the
SM100 tcgen05 instrumentation gap) -- is reported. A suppression rule must fail
toward reporting.

    classify_synccheck.py --markers <regex> <test.log>

Exits 1 when findings survive, 0 when none do, so it can stand in for the grep
in the sweep's gate. Suppressed sites are summarized on stderr, never hidden.
"""

from __future__ import annotations

import argparse
import functools
import os
import re
import sys
from pathlib import Path

# The only marker this script has a calibrated rule for.
_DIVERGENCE = re.compile(
    r"Barrier error detected\.\s*Divergent thread\(s\) in block"
)

# A compute-sanitizer frame line, e.g.
#   ==     at 0x2a0 in /proc/self/cwd/max/.../softmax_warp.mojo:3666:kern(...)
# The path is absolute inside the bazel sandbox, so only its tail is usable.
_FRAME = re.compile(r"^=+\s+at\s+\S+\s+in\s+(?P<path>.+?\.mojo):(?P<line>\d+)")

_NAMED_BARRIER = re.compile(r"\bnamed_barrier\s*\[")

# Line tables can attribute a multi-line call to its closing paren rather than
# to the `named_barrier[` token, so the rule matches the whole statement. The
# walk back is bounded by bracket balance, not by a line count: a fixed window
# would also swallow a plain `barrier()` that merely sits near a
# `named_barrier`, and in warp-specialized code the two are often neighbours.
_MAX_CONTINUATION_LINES = 6
_OPENERS = "(["
_CLOSERS = ")]"


@functools.lru_cache(maxsize=1)
def _basename_index(repo_root: Path) -> dict[str, tuple[Path, ...]]:
    """Maps each `.mojo` basename in the checkout to the files bearing it."""
    index: dict[str, list[Path]] = {}
    skip = {".git", ".derived", "__pycache__", "node_modules"}
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            d for d in dirnames if d not in skip and not d.startswith("bazel-")
        ]
        for name in filenames:
            if name.endswith(".mojo"):
                index.setdefault(name, []).append(Path(dirpath) / name)
    return {k: tuple(v) for k, v in index.items()}


def _resolve(repo_root: Path, logged_path: str) -> tuple[Path, ...]:
    """Maps a sanitizer frame's file onto candidates in this checkout.

    Frames carry a BARE BASENAME (`block.mojo:100`), not a repo-relative path,
    so this is a basename lookup rather than a path join. A basename can be
    ambiguous -- `softmax_warp.mojo` exists under both `sm100/` and
    `sm100/mha_depth512/` -- and the frame gives nothing to disambiguate with,
    so every candidate is returned and the caller decides.
    """
    candidate = Path(logged_path)
    if candidate.is_file():
        return (candidate,)
    hits = _basename_index(repo_root).get(candidate.name, ())
    if hits:
        return hits
    # A path-shaped frame from some other toolchain: walk the tail inward.
    parts = candidate.parts
    for i in range(len(parts)):
        suffix = repo_root.joinpath(*parts[i:])
        if suffix.is_file():
            return (suffix,)
    return ()


def _unclosed(text: str) -> int:
    """Excess closers in `text` -- how deep a continuation it is."""
    closers = sum(text.count(c) for c in _CLOSERS)
    openers = sum(text.count(c) for c in _OPENERS)
    return closers - openers


def _statement_at(source: list[str], line_no: int) -> str:
    """The full statement the attributed line belongs to.

    Walks back only while the text gathered so far closes more brackets than it
    opens, i.e. only while the attributed line really is the tail of a call
    that began earlier. A self-contained line pulls in nothing.
    """
    idx = line_no - 1
    gathered = [source[idx]]
    depth = _unclosed(source[idx])
    steps = 0
    while depth > 0 and idx > 0 and steps < _MAX_CONTINUATION_LINES:
        idx -= 1
        steps += 1
        gathered.append(source[idx])
        depth += _unclosed(source[idx])
    return "\n".join(reversed(gathered))


def _is_named_barrier_site(
    repo_root: Path, logged_path: str, line_no: int
) -> tuple[bool, int]:
    """Whether the frame names a `named_barrier`, and how many files it could be.

    With an ambiguous basename this suppresses when ANY candidate has a
    `named_barrier` at that line. Requiring all of them to agree would leave
    the SM100 attention family unsuppressed -- both `softmax_warp.mojo` files
    are warp-specialized kernels dense with named barriers, and their line
    numbering does not correspond -- which is the noise this exists to remove.
    The residual risk is a real divergence in one file colliding with a
    `named_barrier` at the same line number in its same-named twin; the
    candidate count is reported so that stays visible rather than silent.
    """
    candidates = _resolve(repo_root, logged_path)
    for path in candidates:
        try:
            source = path.read_text(errors="replace").splitlines()
        except OSError:
            continue
        if not 1 <= line_no <= len(source):
            continue
        if _NAMED_BARRIER.search(_statement_at(source, line_no)):
            return True, len(candidates)
    return False, len(candidates)


def _suppressed_header_lines(
    repo_root: Path, lines: list[str]
) -> tuple[set[int], dict[str, int]]:
    """Finds the 1-based line numbers of divergence headers that are tool noise.

    A record is its header plus the indented `=========` lines under it; the
    attributing frame is the first `at ... in <file>:<line>` among them.
    """
    suppressed: set[int] = set()
    sites: dict[str, int] = {}

    for idx, line in enumerate(lines):
        if not _DIVERGENCE.search(line):
            continue
        site = None
        for follower in lines[idx + 1 :]:
            if not follower.startswith("========="):
                break
            body = follower[len("=========") :]
            # An unindented body starts the next record: this one had no frame.
            if body and not body.startswith((" ", "\t")):
                break
            match = _FRAME.match(follower)
            if match:
                site = (match.group("path"), int(match.group("line")))
                break
        if site is None:
            continue
        matched, n_candidates = _is_named_barrier_site(
            repo_root, site[0], site[1]
        )
        if matched:
            suppressed.add(idx + 1)
            key = f"{Path(site[0]).name}:{site[1]}"
            if n_candidates > 1:
                key += f" ({n_candidates} same-named files)"
            sites[key] = sites.get(key, 0) + 1

    return suppressed, sites


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--markers", required=True, help="ERE the sweep gates on"
    )
    parser.add_argument("log", help="path to a target's test.log")
    args = parser.parse_args()

    # <repo>/max/kernels/test/gpu/compute_sanitizer/classify_synccheck.py
    repo_root = Path(__file__).resolve().parents[5]
    markers = re.compile(args.markers)

    lines = Path(args.log).read_text(errors="replace").splitlines()
    suppressed, sites = _suppressed_header_lines(repo_root, lines)

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
            f"synccheck: suppressed {total} divergence report(s) across "
            f"{len(sites)} named_barrier site(s) -- PTX permits one barrier "
            f"name to be reached from several instruction addresses, synccheck "
            f"does not model that. Calibrated by "
            f"negative_control_synccheck_named_barrier.mojo.",
            file=sys.stderr,
        )
        for key, count in sorted(sites.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"synccheck:   {count:6d}  {key}", file=sys.stderr)

    for survivor in survivors:
        print(survivor)

    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main())
