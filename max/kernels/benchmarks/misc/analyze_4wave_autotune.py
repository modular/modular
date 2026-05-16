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
"""Per-(N, K, M) winner report for kbench output from the 4-wave matmul
autotune sweep (`tuning_table_mi355_{fp8,bf16}.yaml`).

Reads the CSV(s) kbench writes to `kbench-output/` and prints, for each
(N, K) cell, the M-sweep with the winning kernel + tile choice and the
runner-up gap. Used to inform the closed-form M cutoffs in
`_matmul_gpu`'s AMD MI355X dispatch block.

Usage:
  # After running kbench (which writes per-cell output.csv files):
  python3 analyze_4wave_autotune.py kbench-output/

  # Or a specific consolidated CSV:
  python3 analyze_4wave_autotune.py kbench-output/output.csv

The CSV must include a `spec` column whose value is
`bench_matmul/.../N=NNNN/.../M=MM/.../TUNE_4WAVE_KERNEL=K/TUNE_4WAVE_BM=...`
— the format kbench emits by default.
"""

import csv
import glob
import sys
from collections import defaultdict
from collections.abc import Iterable

KERNEL_LABEL = {1: "split_k(4)", 2: "split_k(2)", 3: "non-split"}

ConfigKey = tuple[int, int, int, int]  # (kernel, BM, BN, BK)
CellKey = tuple[int, int, int]  # (N, K, M)


def parse_spec(spec: str) -> dict[str, str]:
    """`bench/key=val/key=val/...` → dict (strips leading $ on names)."""
    out: dict[str, str] = {}
    for part in spec.split("/"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.lstrip("$")] = v
    return out


def parse(paths: Iterable[str]) -> dict[CellKey, dict[ConfigKey, float]]:
    cells: dict[CellKey, dict[ConfigKey, float]] = defaultdict(dict)
    for path in paths:
        with open(path) as f:
            for row in csv.DictReader(f):
                spec = parse_spec(row.get("spec", ""))
                try:
                    n = int(spec.get("N", 0))
                    k = int(spec.get("K", 0))
                    m = int(spec.get("M", 0))
                    kern = int(spec.get("TUNE_4WAVE_KERNEL", 0))
                    bm = int(spec.get("TUNE_4WAVE_BM", 0))
                    bn = int(spec.get("TUNE_4WAVE_BN", 0))
                    bk = int(spec.get("TUNE_4WAVE_BK", 0))
                    metric = float(row.get("Arithmetic (GFLOPS/s)", 0))
                except (ValueError, TypeError):
                    continue
                if not n or not m or not kern or not metric:
                    continue
                cells[(n, k, m)][(kern, bm, bn, bk)] = metric
    return cells


def main() -> None:
    paths: list[str] = []
    for arg in sys.argv[1:]:
        if "*" in arg or "?" in arg:
            paths.extend(glob.glob(arg))
        elif arg.endswith(".csv"):
            paths.append(arg)
        else:
            paths.extend(glob.glob(f"{arg}/**/*.csv", recursive=True))
    if not paths:
        print(
            "usage: analyze_4wave_autotune.py <kbench-output-dir or *.csv...>",
            file=sys.stderr,
        )
        sys.exit(2)
    cells = parse(paths)
    if not cells:
        print("no rows parsed (check spec column format)", file=sys.stderr)
        sys.exit(1)

    by_nk = defaultdict(list)
    for (n, k, m), configs in cells.items():
        by_nk[(n, k)].append((m, configs))

    for (n, k), rows in sorted(by_nk.items()):
        print(f"\n== N={n} K={k} ==")
        print(
            f"  {'M':>6} | {'best GFLOPS/s':>14} | kernel       | "
            f"BM  BN  BK  | runner-up gap"
        )
        print("  " + "-" * 70)
        for m, configs in sorted(rows):
            ranked = sorted(configs.items(), key=lambda kv: -kv[1])
            if not ranked:
                continue
            (kern, bm, bn, bk), best = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else best
            gap = (best - second) / second * 100 if second > 0 else 0.0
            label = KERNEL_LABEL.get(kern, f"k={kern}")
            print(
                f"  {m:>6} | {best:>14,.0f} | {label:<12} | "
                f"{bm:>2}  {bn:>2}  {bk:>3}  | {gap:+6.1f}%"
            )


if __name__ == "__main__":
    main()
