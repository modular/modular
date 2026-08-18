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
"""Deterministic wheel builder used by the Nix release packaging.

Produces PEP 427 wheels bit-for-bit reproducibly: entries are sorted, all
timestamps are fixed to 1980-01-01, permissions are normalised, and RECORD is
generated last.  Usage:

  mkwheel.py --name mojo-compiler --version 1.2.3 --tag py3-none-manylinux_2_34_x86_64 \
      --metadata METADATA --entry-points entry_points.txt \
      --add SRC=DEST                # DEST relative to the wheel root
      --data platlib:SRC=DEST       # -> <dist>-<ver>.data/platlib/DEST
      --license FILE                # copied into <dist-info>/LICENSE
      --out DIR
"""

import argparse
import base64
import hashlib
import io
import os
import stat
import sys
import zipfile

FIXED_TIME = (1980, 1, 1, 0, 0, 0)


def urlsafe_b64(digest: bytes) -> str:
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def arcname(path: str) -> str:
    """Normalise an in-wheel path (POSIX separators, no leading './')."""
    name = os.path.normpath(path).replace(os.sep, "/")
    return name[2:] if name.startswith("./") else name


def add_tree(entries: dict, src: str, dest: str) -> None:
    if os.path.isdir(src):
        for root, dirs, files in os.walk(src):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, src)
                if "__pycache__" in rel.split(os.sep) or rel.endswith(".pyc"):
                    continue
                entries[arcname(os.path.join(dest, rel))] = p
    else:
        entries[arcname(dest)] = src


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--entry-points")
    ap.add_argument("--license")
    ap.add_argument("--purelib", action="store_true", help="Root-Is-Purelib: true")
    ap.add_argument("--add", action="append", default=[])
    ap.add_argument("--data", action="append", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dist = args.name.replace("-", "_")
    dist_info = f"{dist}-{args.version}.dist-info"
    data_dir = f"{dist}-{args.version}.data"

    entries: dict[str, str] = {}
    for spec in args.add:
        src, dest = spec.split("=", 1)
        add_tree(entries, src, dest)
    for spec in args.data:
        kind, rest = spec.split(":", 1)
        src, dest = rest.split("=", 1)
        add_tree(entries, src, f"{data_dir}/{kind}/{dest}")

    generated: dict[str, bytes] = {}
    with open(args.metadata, "rb") as f:
        generated[f"{dist_info}/METADATA"] = f.read()
    generated[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: modular-nix-mkwheel 1.0\n"
        f"Root-Is-Purelib: {'true' if args.purelib else 'false'}\n"
        f"Tag: {args.tag}\n\n"
    ).encode()
    if args.entry_points:
        with open(args.entry_points, "rb") as f:
            generated[f"{dist_info}/entry_points.txt"] = f.read()
    if args.license:
        with open(args.license, "rb") as f:
            generated[f"{dist_info}/LICENSE"] = f.read()

    os.makedirs(args.out, exist_ok=True)
    wheel_path = os.path.join(args.out, f"{dist}-{args.version}-{args.tag}.whl")
    record_lines = []

    def write(zf: zipfile.ZipFile, arcname: str, data: bytes, executable: bool) -> None:
        zi = zipfile.ZipInfo(arcname, date_time=FIXED_TIME)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.create_system = 3  # unix
        mode = 0o755 if executable else 0o644
        zi.external_attr = (stat.S_IFREG | mode) << 16
        zf.writestr(zi, data)
        digest = hashlib.sha256(data).digest()
        record_lines.append(f"{arcname},sha256={urlsafe_b64(digest)},{len(data)}")

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname in sorted(entries):
            src = entries[arcname]
            with open(src, "rb") as f:
                data = f.read()
            executable = os.access(src, os.X_OK) and not os.path.isdir(src)
            write(zf, arcname, data, executable)
        for arcname in sorted(generated):
            write(zf, arcname, generated[arcname], False)
        record_lines.append(f"{dist_info}/RECORD,,")
        zi = zipfile.ZipInfo(f"{dist_info}/RECORD", date_time=FIXED_TIME)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zi.create_system = 3
        zi.external_attr = (stat.S_IFREG | 0o644) << 16
        zf.writestr(zi, "\n".join(record_lines) + "\n")

    print(wheel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
