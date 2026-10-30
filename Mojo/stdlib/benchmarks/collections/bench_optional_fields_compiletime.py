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
"""Measures cold compilation of nested Optional payloads (issue #7391).

Example:
    python bench_optional_fields_compiletime.py /path/to/mojo \
        --import-path /path/to/stdlib --fields 16 24 32 36 40
"""

from __future__ import annotations

import argparse
import os
import statistics
import subprocess
import tempfile
import time
from pathlib import Path


def generate_source(count: int) -> str:
    """Generates the conditional assignments and nested return from #7391."""
    fields = "\n".join(f"    var m{i}: Optional[M]" for i in range(count))
    init = "\n".join(f"        self.m{i} = None" for i in range(count))
    branches = "\n".join(
        f"        {'if' if i == 0 else 'elif'} tag == {i}:\n"
        f"            self.m{i} = M()"
        for i in range(count)
    )
    return f"""struct M(Movable):
    var data: List[Byte]

    def __init__(out self):
        self.data = List[Byte]()


struct Many(Movable):
{fields}

    def __init__(out self):
{init}

    def fill(mut self, mut tag: Int):
{branches}


struct Top(Movable):
    var a: Optional[Many]
    var b: Optional[Many]

    def __init__(out self):
        self.a = None
        self.b = None


def decode(n: Int) -> Top:
    var t = Top()
    var tag = n
    var a = Many()
    a.fill(tag)
    t.a = a^
    var b = Many()
    b.fill(tag)
    t.b = b^
    return t^


def main():
    var t = decode(3)
    print(Bool(t.a))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiler")
    parser.add_argument("--import-path", type=Path)
    parser.add_argument(
        "--fields", nargs="+", type=int, default=[16, 24, 32, 36, 40]
    )
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1 or any(n < 1 for n in args.fields):
        parser.error("field counts and repetitions must be positive")

    print("fields min_s median_s max_s", flush=True)
    with tempfile.TemporaryDirectory(prefix="mojo-optional-fields-") as tmp:
        root = Path(tmp)
        for count in args.fields:
            source = root / f"optional_{count}.mojo"
            source.write_text(generate_source(count))
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
            for rep in range(args.repetitions):
                env = os.environ | {
                    "MODULAR_CACHE_DIR": str(root / f"cache-{count}-{rep}")
                }
                start = time.perf_counter()
                result = subprocess.run(
                    command, env=env, capture_output=True, text=True
                )
                if result.returncode:
                    parser.exit(1, result.stderr)
                times.append(time.perf_counter() - start)
            print(
                f"{count} {min(times):.3f} {statistics.median(times):.3f} {max(times):.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
