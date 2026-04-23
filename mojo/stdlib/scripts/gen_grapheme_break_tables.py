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
"""Generate Mojo Grapheme Break Property lookup tables for UAX #29.

Reads Unicode data files from ``unicode-data/`` next to this script and
generates:
- std/collections/string/_grapheme_break_lookups.mojo  (property tables)

The checked-in data files let the script run offline and give reviewers a
stable, diffable record of the inputs used to produce each regeneration.

Updating to a new Unicode version (released annually, typically September):
    1. Change UNICODE_VERSION below (e.g. "16.0.0" -> "17.0.0")
    2. Refresh the local data files: python gen_grapheme_break_tables.py --refresh
    3. Run tests: ./bazelw test //oss/modular/mojo/stdlib/test/collections/string/...
    4. Commit the regenerated files *and* the refreshed ``unicode-data/*.txt``
"""

import argparse
import sys
import urllib.request
from collections.abc import Callable, Generator
from pathlib import Path

UNICODE_VERSION = "16.0.0"
BASE_URL = f"https://www.unicode.org/Public/{UNICODE_VERSION}/ucd"

# Source URLs, kept as documentation of where the checked-in files come from
# and used by ``--refresh`` to re-download them.
GBP_URL = f"{BASE_URL}/auxiliary/GraphemeBreakProperty.txt"
EMOJI_URL = f"{BASE_URL}/emoji/emoji-data.txt"
INCB_URL = f"{BASE_URL}/DerivedCoreProperties.txt"

# Filenames (within ``unicode-data/``) for each source.  These match the
# upstream basenames so the files are recognizable to anyone who has worked
# with the UCD before.
GBP_FILE = "GraphemeBreakProperty.txt"
EMOJI_FILE = "emoji-data.txt"
INCB_FILE = "DerivedCoreProperties.txt"

DATA_SOURCES: list[tuple[str, str]] = [
    (GBP_FILE, GBP_URL),
    (EMOJI_FILE, EMOJI_URL),
    (INCB_FILE, INCB_URL),
]

# Grapheme Break Property enum values (must match the Mojo constants)
GBP_NAMES = [
    "Other",  # 0
    "CR",  # 1
    "LF",  # 2
    "Control",  # 3
    "Extend",  # 4
    "ZWJ",  # 5
    "Regional_Indicator",  # 6
    "Prepend",  # 7
    "SpacingMark",  # 8
    "L",  # 9
    "V",  # 10
    "T",  # 11
    "LV",  # 12
    "LVT",  # 13
    "Extended_Pictographic",  # 14
]

GBP_MAP = {name: idx for idx, name in enumerate(GBP_NAMES)}

# Indic_Conjunct_Break property values for GB9c
INCB_NAMES = [
    "None",  # 0
    "Consonant",  # 1
    "Extend",  # 2
    "Linker",  # 3
]
INCB_MAP = {name: idx for idx, name in enumerate(INCB_NAMES)}


def refresh_local_data(data_dir: Path) -> None:
    """Download the current UCD files into ``data_dir``, overwriting them."""
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in DATA_SOURCES:
        dest = data_dir / filename
        print(f"# Downloading {url} -> {dest}", file=sys.stderr)
        with urllib.request.urlopen(url) as resp:
            dest.write_bytes(resp.read())


def parse_unicode_ranges(
    text: str, property_filter: str | None = None
) -> Generator[tuple[int, int, ...], None, None]:
    """Parse Unicode data file format into (start, end, properties...) tuples.

    Args:
        text: The file contents.
        property_filter: If set, only return entries where any semicolon-
            separated field matches this value.

    Yields:
        (start_codepoint, end_codepoint, prop1, prop2, ...) tuples.
    """
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line[: line.index("#")]
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue

        if property_filter and property_filter not in parts:
            continue

        cp_range = parts[0]
        if ".." in cp_range:
            start_s, end_s = cp_range.split("..")
            start = int(start_s, 16)
            end = int(end_s, 16)
        else:
            start = int(cp_range, 16)
            end = start

        yield (start, end, *parts[1:])


def build_gbp_ranges(
    gbp_text: str, emoji_text: str
) -> list[tuple[int, int, int]]:
    """Build sorted, non-overlapping GBP range table.

    Returns a list of (start, end, gbp_value) tuples sorted by start.
    """
    ranges = []

    # Parse GraphemeBreakProperty.txt
    for start, end, prop in parse_unicode_ranges(gbp_text):
        if prop in GBP_MAP:
            ranges.append((start, end, GBP_MAP[prop]))

    # Parse Extended_Pictographic from emoji-data.txt
    for start, end, _prop in parse_unicode_ranges(
        emoji_text, "Extended_Pictographic"
    ):
        ranges.append((start, end, GBP_MAP["Extended_Pictographic"]))

    # Sort by start codepoint
    ranges.sort(key=lambda x: x[0])

    return ranges


def build_incb_ranges(dcp_text: str) -> list[tuple[int, int, int]]:
    """Build Indic_Conjunct_Break property ranges from DerivedCoreProperties.txt.

    Required for UAX #29 rule GB9c, which keeps Indic conjunct clusters
    (e.g. Devanagari consonant + virama + consonant) as a single grapheme.
    The InCB property is in a separate file from the main GBP data.

    The format is: `codepoint ; InCB; Value # comment`
    (three semicolon-separated fields).

    Returns a list of (start, end, incb_value) tuples sorted by start.
    """
    ranges = []
    # Filter to "InCB" entries; parse_unicode_ranges yields
    # (start, end, "InCB", value) for these lines.
    for start, end, _, sub_prop in parse_unicode_ranges(dcp_text, "InCB"):
        if sub_prop in INCB_MAP:
            ranges.append((start, end, INCB_MAP[sub_prop]))

    ranges.sort(key=lambda x: x[0])
    return ranges


def compress_to_range_starts(
    ranges: list[tuple[int, int, int]],
) -> tuple[list[int], list[int]]:
    """Convert sorted, non-overlapping (start, end, value) ranges to change-points.

    Requires the input to be sorted by start codepoint with no overlaps
    (which the UCD data files guarantee).

    Returns two parallel lists:
    - starts: sorted list of codepoints where the property value changes
    - values: the new property value at each change-point

    For lookup, binary search starts to find which range a codepoint falls in.
    Between explicit ranges, the value is Other (0).
    """
    starts: list[int] = []
    values: list[int] = []
    prev_val = 0  # implicit starting value is Other
    prev_end = -1

    for start, end, val in ranges:
        assert start > prev_end, f"Overlapping range at U+{start:04X}"

        # Insert an Other gap marker after the previous range if needed
        if start > prev_end + 1 and prev_val != 0:
            starts.append(prev_end + 1)
            values.append(0)
            prev_val = 0

        # Emit a new entry if the value changes (merges adjacent same-value)
        if val != prev_val:
            starts.append(start)
            values.append(val)
            prev_val = val

        prev_end = end

    # Reset to Other after the last range
    if prev_val != 0:
        starts.append(prev_end + 1)
        values.append(0)

    return starts, values


def _emit_array(
    lines: list[str],
    name: str,
    mojo_type: str,
    entries: list[int],
    fmt_entry: Callable[[int], str],
    per_line: int = 10,
) -> None:
    """Emit a single `comptime` ``InlineArray`` declaration.

    ``fmt_entry`` returns the formatted text for one entry (e.g.
    ``"0x000A,"`` or ``"3,"``).  Entries are packed ``per_line`` per
    row and wrapped in ``# fmt: off`` / ``# fmt: on`` to prevent the
    formatter from exploding the table back to one-per-line.
    """
    lines.append("# fmt: off")
    lines.append(
        f"comptime {name}: InlineArray[{mojo_type}, {len(entries)}] = ["
    )
    for i in range(0, len(entries), per_line):
        chunk = entries[i : i + per_line]
        lines.append("    " + " ".join(fmt_entry(e) for e in chunk))
    lines.append("]")
    lines.append("# fmt: on")
    lines.append("")


def emit_mojo(
    gbp_starts: list[int],
    gbp_values: list[int],
    incb_starts: list[int],
    incb_values: list[int],
) -> str:
    """Emit Mojo source code for the lookup tables."""
    lines = []
    lines.append(f"""\
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
\"\"\"Grapheme Break Property lookup tables generated from Unicode {UNICODE_VERSION}.

Data sources:
    {GBP_URL}
    {EMOJI_URL}
    {INCB_URL}

DO NOT EDIT - Generated by scripts/gen_grapheme_break_tables.py
\"\"\"

""")

    fmt_hex = lambda cp: f"0x{cp:04X},"
    fmt_int = lambda v: f"{v},"

    lines.append(
        f"# {len(gbp_starts)} range entries for Grapheme Break Properties"
    )
    _emit_array(lines, "_GBP_RANGE_STARTS", "UInt32", gbp_starts, fmt_hex)
    _emit_array(lines, "_GBP_RANGE_VALUES", "UInt8", gbp_values, fmt_int)

    if incb_starts:
        lines.append(
            f"# {len(incb_starts)} range entries for Indic_Conjunct_Break"
        )
        _emit_array(lines, "_INCB_RANGE_STARTS", "UInt32", incb_starts, fmt_hex)
        _emit_array(lines, "_INCB_RANGE_VALUES", "UInt8", incb_values, fmt_int)

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help=(
            "Re-download the UCD files into unicode-data/ before generating."
            " Use this when bumping UNICODE_VERSION."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "unicode-data"
    lookups_path = (
        script_dir / "../std/collections/string/_grapheme_break_lookups.mojo"
    ).resolve()

    if args.refresh:
        refresh_local_data(data_dir)

    missing = [
        data_dir / name
        for name, _ in DATA_SOURCES
        if not (data_dir / name).exists()
    ]
    if missing:
        raise SystemExit(
            "Missing UCD data files: "
            + ", ".join(str(p) for p in missing)
            + "\nRun with --refresh to download them."
        )

    gbp_text = (data_dir / GBP_FILE).read_text(encoding="utf-8")
    emoji_text = (data_dir / EMOJI_FILE).read_text(encoding="utf-8")
    dcp_text = (data_dir / INCB_FILE).read_text(encoding="utf-8")

    gbp_ranges = build_gbp_ranges(gbp_text, emoji_text)
    gbp_starts, gbp_values = compress_to_range_starts(gbp_ranges)

    incb_ranges = build_incb_ranges(dcp_text)
    incb_starts, incb_values = compress_to_range_starts(incb_ranges)

    lookups_path.write_text(
        emit_mojo(gbp_starts, gbp_values, incb_starts, incb_values)
    )
    print(
        f"Wrote {lookups_path} ({len(gbp_starts)} GBP ranges,"
        f" {len(incb_starts)} InCB ranges)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
