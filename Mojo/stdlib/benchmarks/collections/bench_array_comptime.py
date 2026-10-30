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

"""Measures cold compilation of compile-time Array accesses (#6975).

Example:
    python bench_array_comptime.py /path/to/mojo --import-path /path/to/stdlib
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def generate_source(case: str, size: int, iterations: int) -> str:
    """Generates the issue's mutation/read loop and verifies its result."""
    container = (
        f"Array[Int, {size}](fill=1)"
        if case == "array"
        else f"SIMD[DType.int64, {size}](0)"
    )
    write = "i * 3 + 1" if case == "array" else "Int64(i * 3 + 1)"
    read = "values[j]" if case == "array" else "Int(values[j])"
    expected = iterations * (3 * size * (size - 1) // 2 + size)
    return f"""def kernel(iters: Int) -> Int:
    var values = {container}
    var i = 0
    while i < {size}:
        values[i] = {write}
        i += 1
    var acc = 0
    var n = 0
    while n < iters:
        var j = 0
        while j < {size}:
            acc += {read}
            j += 1
        n += 1
    return acc


def main():
    comptime result = kernel({iterations})
    comptime assert result == {expected}
    print(result)
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler")
    parser.add_argument("--import-path", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--cases",
        choices=["array", "simd"],
        nargs="+",
        default=["array", "simd"],
    )
    args = parser.parse_args()
    if min(*args.sizes, args.iterations, args.repetitions) < 1:
        parser.error("sizes, iterations, and repetitions must be positive")

    print("case size iterations median_s min_s max_s", flush=True)
    with tempfile.TemporaryDirectory(prefix="mojo-array-comptime-") as tmp:
        root = Path(tmp)
        for case in args.cases:
            for size in args.sizes:
                source = root / f"{case}_{size}.mojo"
                source.write_text(generate_source(case, size, args.iterations))
                command = [
                    args.compiler,
                    "build",
                    str(source),
                    "--emit=object",
                    "-o",
                    str(root / "out.o"),
                ]
                if args.import_path:
                    command.extend(["-I", str(args.import_path)])
                times = []
                for _ in range(args.repetitions):
                    with tempfile.TemporaryDirectory(
                        dir=root, prefix="cache-"
                    ) as cache:
                        env = os.environ | {"MODULAR_CACHE_DIR": cache}
                        start = time.perf_counter()
                        result = subprocess.run(
                            command, env=env, capture_output=True, text=True
                        )
                        elapsed = time.perf_counter() - start
                    if result.returncode:
                        parser.exit(1, result.stderr)
                    times.append(elapsed)
                print(
                    f"{case} {size} {args.iterations} {statistics.median(times):.3f} {min(times):.3f} {max(times):.3f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
