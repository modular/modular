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
"""Regenerate nix/release-wheels.json from Modular's package index.

The versions are read from bazel/mojo.MODULE.bazel (the same source
nix/versions.nix uses), so the JSON always describes the release this tree is
paired with.  Usage:  python3 nix/update-release-wheels.py [--index URL]
(default index: https://whl.modular.com/nightly/simple; use
https://pypi.org/simple for stable releases)."""

import argparse
import json
import os
import re
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Only the packages this flake rebuilds from source (used for comparison).
MOJO_PKGS = ["mojo", "mojo-compiler", "mojo-compiler-mojo-libs", "mojo-lldb-libs"]
MAX_PKGS = ["mblack"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="https://whl.modular.com/nightly/simple")
    ap.add_argument("--out", default=os.path.join(ROOT, "nix", "release-wheels.json"))
    args = ap.parse_args()

    module = open(os.path.join(ROOT, "bazel", "mojo.MODULE.bazel")).read()
    mojo_v = re.search(r'MOJO_PACKAGE_VERSION = "([^"]+)"', module).group(1)
    max_v = re.search(r'MAX_PACKAGE_VERSION = "([^"]+)"', module).group(1)
    print(f"mojo {mojo_v}, max {max_v}", file=sys.stderr)

    out = {}
    for pkg, version in [(p, mojo_v) for p in MOJO_PKGS] + [(p, max_v) for p in MAX_PKGS]:
        html = urllib.request.urlopen(f"{args.index}/{pkg}/", timeout=60).read().decode()
        for m in re.finditer(r'href="([^"#]+)#sha256=([0-9a-f]+)"', html):
            url, sha = m.groups()
            fn = url.rsplit("/", 1)[1]
            if f"-{version}-" in fn:
                out.setdefault(pkg, {})[fn] = {"url": url, "sha256": sha}
        print(f"{pkg}: {len(out.get(pkg, {}))} files", file=sys.stderr)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
