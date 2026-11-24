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
"""Main for tblgen."""

from __future__ import annotations

import json
import sys
from typing import TextIO

import click

from . import allowlist
from .backends.debug import DebugBackend
from .backends.json import JsonBackend
from .backends.nanobind import NanobindBackend
from .types import Entry, Table

BACKENDS = {
    "debug": DebugBackend,
    "json": JsonBackend,
    "nanobind": NanobindBackend,
}


@click.command()
@click.argument("json_file")
@click.argument("names", nargs=-1)
@click.option(
    "--backend",
    default="nanobind",
    type=click.Choice(list(BACKENDS)),
    help="backend to use for generating entry values",
)
@click.option(
    "--defined-in",
    default=None,
    help="generate only definitions from a particular TD file",
)
@click.option(
    "--dialect", default=None, help="generate only a specific dialect"
)
@click.option(
    "--allowlist",
    "allowlist_file",
    default=None,
    type=click.File(),
    help=(
        "generate only the names listed in this file, which must also name"
        " everything those bindings refer to; generates everything when omitted"
    ),
)
@click.option("--output", "-o", default=sys.stdout, type=click.File(mode="w"))
def generate(
    json_file: str,
    names: tuple[str, ...],
    backend: str,
    defined_in: str | None,
    dialect: str | None,
    allowlist_file: TextIO | None,
    output: TextIO,
) -> None:
    with open(json_file) as td:
        table = Table(json.load(td))

    entry_names = names or [
        name for name in table.data if not name.startswith("!")
    ]
    entries = [table[name] for name in entry_names]

    dialect_entry = dialect and table[dialect]

    def locations(entry: Entry):  # noqa: ANN202
        for loc in entry["!locs"]:
            yield loc.split(":", 1)[0]

    backend_instance = BACKENDS[backend]()

    if allowlist_file is not None:
        keep = allowlist.read_allowlist(allowlist_file.read())
        # Checked against every entry in the table, not just the ones this
        # invocation emits: a reference that crosses dialects is only visible
        # from the referring side.
        if missing := allowlist.missing_references(
            keep, entries, backend_instance
        ):
            raise click.ClickException(
                f"{allowlist_file.name} is missing entries that its own"
                " entries refer to. Unlisted names reach Python as an"
                " unregistered type, which makes the signatures using them"
                " uncallable. Add:\n"
                + "\n".join(
                    f"  {name}  # referenced by {', '.join(sorted(by))}"
                    for name, by in sorted(missing.items())
                )
            )
        entries = [entry for entry in entries if entry.name in keep]

    filters = [
        lambda entry: not dialect_entry or dialect_entry == entry.dialect,
        lambda entry: not defined_in or defined_in in locations(entry),
    ]

    entries = [
        entry for entry in entries if all(filter(entry) for filter in filters)
    ]

    for line in backend_instance.generate_all(entries):
        print(line, file=output)

    # If we never print anything, we still write an empty buffer to create the file.
    output.flush()


if __name__ == "__main__":
    generate()
