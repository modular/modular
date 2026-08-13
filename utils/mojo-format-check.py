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
"""mojo-format-check: Check Mojo source file formatting without modifying files.

This script wraps the `mblack` formatter (the same underlying tool used by
`mojo format`) and invokes it in check mode. It is a drop-in solution for CI
and pre-commit workflows until `mojo format --check` is natively supported by
the `mojo` driver binary.

Usage:
    mojo-format-check [--diff] [--quiet] [--line-length N] <sources...>

Exit codes:
    0   All files are already correctly formatted.
    1   One or more files would be reformatted (no files are modified).
    123 Internal formatter error or parse failure.

Example:
    mojo-format-check src/ tests/ examples/
    mojo-format-check --diff stdlib/
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _find_mblack() -> str | None:
    """Locate the mblack binary, mirroring the lookup order used by the mojo driver.

    Search order:
    1. ``MODULAR_MOJO_MAX_MBLACK_PATH`` environment variable (set by the mojo wheel).
    2. ``mblack`` on ``PATH`` (installed via pixi / pip).
    3. Relative to the ``max`` package root derived from this script's location.
    """
    # 1. Honour the env-var that the mojo wheel / SDK sets.
    env_path = os.environ.get("MODULAR_MOJO_MAX_MBLACK_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    # 2. Plain PATH lookup — works for `pixi run` and venv installs.
    which = shutil.which("mblack")
    if which:
        return which

    # 3. Try to find mblack relative to the installed `max` package root.
    #    The wheel layout is: lib/python3.x/site-packages/max/  →  bin/mblack
    try:
        import max  # noqa: PLC0415

        candidate = Path(max.__file__).parent.parent.parent.parent.parent / "bin" / "mblack"
        if candidate.exists():
            return str(candidate)
    except ImportError:
        pass

    return None


def main() -> None:
    mblack = _find_mblack()
    if not mblack:
        print(
            "error: Could not find the `mblack` formatter.\n"
            "Make sure mojo is installed (e.g. `pixi install` or `pip install max`).",
            file=sys.stderr,
        )
        sys.exit(123)

    # Build the mblack argument list.
    # sys.argv[1:] contains the user's arguments (files, --quiet, --line-length, etc.)
    # We prepend --check, and also --diff when the user did not ask for --quiet,
    # so that the output shows exactly which lines would change.
    user_args = sys.argv[1:]

    mblack_args = ["--check"]

    # Add --diff automatically unless the caller passed --quiet
    # (--diff produces verbose output that --quiet would suppress anyway).
    if "--quiet" not in user_args and "-q" not in user_args:
        mblack_args.append("--diff")

    mblack_args.extend(user_args)

    result = subprocess.run([mblack] + mblack_args)

    # Propagate exit code unchanged:
    #   0   → already formatted
    #   1   → would reformat
    #   123 → internal error (mblack / black convention)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
