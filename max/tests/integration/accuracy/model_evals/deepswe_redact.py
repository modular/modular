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
"""Scrubs the model endpoint's API key from a DeepSWE results directory.

The agent reaches the model from inside its task container, so the key is in
that container's environment, and the model runs arbitrary shell commands
there: an ``env`` lands the key in the trial's trajectory and log. Every file
under the directory is rewritten with each occurrence of the key replaced,
including copies split across lines by terminal wrapping or JSON escaping.

Exits non-zero when any file could not be checked, so callers can withhold
the directory rather than share it half scrubbed. With
``--remove-unscrubbable`` (meant for the last attempt, as root), a file that
still cannot be scrubbed is deleted instead and listed in
``REMOVED_UNSCRUBBABLE_FILES.txt``, so one bad file does not cost the whole
directory; it then exits non-zero only when such a file cannot be deleted
either, or a directory could not be listed.

usage: deepswe_redact.py [--remove-unscrubbable] <results dir>
(the key is read from OPENAI_API_KEY, never from argv, so it stays out of the
process list)
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys

REDACTED = b"<redacted:OPENAI_API_KEY>"
REMOVED_LIST = "REMOVED_UNSCRUBBABLE_FILES.txt"
# A key too short to be a real credential would match ordinary text.
MIN_SECRET_LEN = 12
# What may sit between two characters of a key that a terminal wrapped or a
# JSON encoder escaped: raw whitespace or a literal \n, \r or \t.
_GAP = rb"(?:\s|\\[nrt])*"


def secret_pattern(secret: str) -> re.Pattern[bytes]:
    """Builds a pattern matching ``secret`` with any wrapping inside it."""
    chars = [re.escape(bytes([b])) for b in secret.encode()]
    return re.compile(_GAP.join(chars))


def redact_tree(
    root: pathlib.Path, secret: str
) -> tuple[list[pathlib.Path], list[tuple[pathlib.Path, str]]]:
    """Replaces every occurrence of ``secret`` in the files under ``root``.

    A file or directory that cannot be read or rewritten is recorded and
    skipped, so one root-owned file does not leave the rest unscrubbed. One
    that disappears during the walk is ignored: it cannot leak anything.

    Args:
        root: Directory to scrub in place.
        secret: The value to remove.

    Returns:
        A ``(changed, failed)`` tuple: the files that contained the secret and
        were rewritten, and ``(path, error)`` for every path that could not be
        checked.
    """
    changed: list[pathlib.Path] = []
    failed: list[tuple[pathlib.Path, str]] = []
    if len(secret) < MIN_SECRET_LEN:
        return changed, failed
    pattern = secret_pattern(secret)

    def on_walk_error(exc: OSError) -> None:
        if isinstance(exc, FileNotFoundError):
            return
        failed.append((pathlib.Path(exc.filename or root), str(exc)))

    for dirpath, _, filenames in os.walk(root, onerror=on_walk_error):
        for name in sorted(filenames):
            path = pathlib.Path(dirpath, name)
            if path.is_symlink() or not path.is_file():
                continue
            try:
                scrubbed, count = pattern.subn(REDACTED, path.read_bytes())
                if count:
                    path.write_bytes(scrubbed)
                    changed.append(path)
            except FileNotFoundError:
                continue
            except OSError as exc:
                failed.append((path, str(exc)))
    return changed, failed


def remove_unscrubbable(
    root: pathlib.Path, failed: list[tuple[pathlib.Path, str]]
) -> tuple[list[tuple[pathlib.Path, str]], list[tuple[pathlib.Path, str]]]:
    """Deletes the files ``redact_tree`` could not scrub.

    The deleted files are listed in ``root/REMOVED_UNSCRUBBABLE_FILES.txt``.
    A directory that could not be listed is left alone, since what it holds
    is unknown.

    Args:
        root: The directory that was scrubbed.
        failed: ``(path, error)`` pairs from ``redact_tree``.

    Returns:
        A ``(removed, remaining)`` tuple: the ``(path, error)`` pairs that were
        deleted, and those that are still unscrubbed.
    """
    removed: list[tuple[pathlib.Path, str]] = []
    remaining: list[tuple[pathlib.Path, str]] = []
    for path, error in failed:
        if path.is_dir():
            remaining.append((path, error))
            continue
        try:
            path.unlink(missing_ok=True)
            removed.append((path, error))
        except OSError as exc:
            remaining.append((path, f"{error}; could not delete it: {exc}"))
    if removed:
        try:
            with open(root / REMOVED_LIST, "a") as f:
                for path, error in removed:
                    f.write(f"{path.relative_to(root)}: {error}\n")
        except OSError as exc:
            print(
                f"Could not write {root / REMOVED_LIST}: {exc}", file=sys.stderr
            )
    return removed, remaining


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrubs the API key from a DeepSWE results directory."
    )
    parser.add_argument("root", type=pathlib.Path)
    parser.add_argument("--remove-unscrubbable", action="store_true")
    args = parser.parse_args()
    root: pathlib.Path = args.root
    secret = os.environ.get("OPENAI_API_KEY", "")
    if not root.is_dir():
        return
    changed, failed = redact_tree(root, secret)
    for path in changed:
        print(f"Redacted the API key from {path}")
    if failed and args.remove_unscrubbable:
        removed, failed = remove_unscrubbable(root, failed)
        warn = (
            "::warning::"
            if os.environ.get("GITHUB_ACTIONS") == "true"
            else "WARNING: "
        )
        for path, error in removed:
            print(
                f"{warn}Deleted {path}, which could not be scrubbed: {error}",
                file=sys.stderr,
            )
    for path, error in failed:
        print(f"Could not scrub {path}: {error}", file=sys.stderr)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
