# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Allowlist filtering for binding generation.

Binding every op, attribute and type a dialect declares costs real binary size —
each `.def` instantiates a pile of nanobind templates — and only a fraction of
them is ever named from Python. An allowlist keeps generation to that fraction.

Skipping an entry degrades gracefully rather than breaking: `type_hook` in
`Support/Nanobind/TypeCasters.h` falls back to the base `mlir::Attribute` /
`Type` / `OpState` binding for any TypeID that was never registered.

What an allowlist must not do is leave a *kept* entry referring to a skipped one.
An unregistered type in a signature makes that overload uncallable, and one in a
*default* value is worse: it is cast eagerly at import time. `missing_references`
finds those, and generation fails rather than emitting them.

The check reports rather than repairs, because it can't see far enough to
repair. Generation runs once per TableGen file, each seeing only its own
transitive includes, so a reference is only visible when both ends land in the
same table — and neither end is guaranteed to. Only an allowlist that already
spells out every such name is consistent across every invocation, which is why
the list is checked in rather than inferred. What the check misses still shows up
in the generated stubs, as a raw C++ name where a Python type should be.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from .backend import Backend
from .types import Entry

# A qualified C++ identifier, e.g. `M::MO::AddOp` or `mlir::TypedValue`.
_QUALIFIED_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_:]*")


def read_allowlist(text: str) -> set[str]:
    """Parses an allowlist file into a set of TableGen record names.

    Blank lines and `#` comments are ignored, as is anything after the record
    name on a line, which lets entries carry a trailing comment.

    Args:
        text: Contents of the allowlist file.

    Returns:
        The record names named by the file.
    """
    names = set()
    for line in text.splitlines():
        name = line.split("#", 1)[0].split()
        if name:
            names.add(name[0])
    return names


def _normalize(name: str) -> str:
    return name.removeprefix("::")


def _referenced_names(text: str) -> Iterable[str]:
    """Yields every qualified C++ name, and name prefix, mentioned in `text`.

    Prefixes matter because generated code refers to an entry through its
    members: `M::MO::AddOp::getInputX` has to count as a reference to
    `M::MO::AddOp`.
    """
    for match in _QUALIFIED_NAME.finditer(text):
        parts = _normalize(match.group(0)).split("::")
        for end in range(1, len(parts) + 1):
            yield "::".join(parts[:end])


def _cpp_names(entry: Entry) -> Iterable[str]:
    """Yields the C++ names generated code would refer to `entry` by.

    Every suffix of each name counts, because the generated bindings are
    included into a `using namespace` and refer to their own dialect
    unqualified: `M::KGEN::PogMetadataAttr` is written `PogMetadataAttr`.
    """
    # `enum_type` is the underlying enum for an EnumAttrInfo, whose `type` is
    # the specialized attribute class. Both get bound, so both are references.
    for attr in ("type", "enum_type"):
        try:
            name = getattr(entry, attr, None)
        except Exception:
            continue
        if name is None:
            continue
        parts = _normalize(name.cpp).split("::")
        for start in range(len(parts)):
            yield "::".join(parts[start:])


def missing_references(
    seed: set[str],
    candidates: list[Entry],
    backend: Backend,
) -> dict[str, set[str]]:
    """Finds bindable records the allowlist needs but doesn't name.

    Args:
        seed: Record names the allowlist asked for.
        candidates: Every entry in the table, generated or not.
        backend: Backend whose output decides what counts as a reference.

    Returns:
        Each unlisted record that a listed one refers to, mapped to the listed
        records referring to it. Empty when the allowlist is self-contained.
    """
    # Only entries that emit something can be referenced. Most of the table is
    # TableGen plumbing — type constraints, predicates, anonymous records —
    # that the backend has no generator for.
    text = {}
    # C++ name -> records binding it. One name can come from several records
    # (an enum and its specialized attribute class), so keep them all.
    owners: dict[str, set[str]] = {}
    for entry in candidates:
        try:
            generated = "\n".join(backend.generate(entry))
        except Exception:
            # The table reaches well beyond what this invocation emits, into
            # records that were never expected to generate. Those can't be
            # referenced — but an allowlisted record failing is a real error.
            if entry.name in seed:
                raise
            continue
        if not generated:
            continue
        text[entry.name] = generated
        for cpp in _cpp_names(entry):
            owners.setdefault(cpp, set()).add(entry.name)

    missing: dict[str, set[str]] = {}
    for name in seed & text.keys():
        for referenced in _referenced_names(text[name]):
            for owner in owners.get(referenced, ()):
                if owner not in seed:
                    missing.setdefault(owner, set()).add(name)
    return missing
