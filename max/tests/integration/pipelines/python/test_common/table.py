# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
#
# A utility class for easy creation and printing of tables
#
# ===----------------------------------------------------------------------=== #

from collections.abc import Sequence
from typing import Any, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

CONSOLE = Console()


class PrettyTable:
    """
    A table utility class to reuse and pretty print tables.
    This class supports paginated output (enabled by default).
    This allows the user to navigate the table using the keyboard.

    For eg: if a table spans multiple pages, the first page is
    displayed. User can scroll using arrow keys, go to start
    by pressing `g`, go to end by pressing `G` and search for
    a word using `\\<word>` commands.
    """

    def __init__(
        self,
        headers: Sequence[Any],
        title: Optional[str] = None,
        paginate: bool = True,
    ):
        """
            Initialize table
        Args:
            title(str): Title of the table
            headers(Sequence[Any]): Table headers
            paginate(bool): Paginate the table
        """

        self._table = (
            Table(
                title=Text(f"{title}", no_wrap=True, justify="center"),
                min_width=120,
            )
            if title
            else Table()
        )

        for header in headers:
            self._table.add_column(f"{header}", justify="right")
        self._headers = headers
        self._paginate = paginate

    def add_row(self, row: Sequence[Any]):
        """
        Add a new row to the table
        Args:
            row(Sequence[Any]): List of values in new row
        """
        assert len(row) == len(self._headers), (
            "Number of columns in row should equal number of headers"
        )
        self._table.add_row(*[f"{col}" for col in row])

    def add_rows(self, rows: Sequence[Sequence[Any]]):
        """
        Add multiple rows to the table
        Args:
            rows(Sequence[Sequence[Any]]): List of rows that's to be added to table in
                succession
        """
        for row in rows:
            self.add_row(row)

    def print(self):
        """
        Print the table to console
        """
        if self._paginate:
            with CONSOLE.pager():
                CONSOLE.print(self._table)
        else:
            CONSOLE.print(self._table)
