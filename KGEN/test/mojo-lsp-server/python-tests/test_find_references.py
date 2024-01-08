# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from lsprotocol.types import Position, Range
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def test_find_variable_references(client: LanguageClient):
    doc = Document.from_file("references.mojo")
    requests = Requests(client)
    requests.open_document(doc)

    results = fail_if_none(
        await requests.find_references(doc, Position(line=8, character=20))
    )
    assert len(results) == 3
    expected = [
        Range(Position(7, 12), Position(7, 15)),
        Range(Position(8, 19), Position(8, 22)),
        Range(Position(9, 10), Position(9, 13)),
    ]
    for result in results:
        assert result.range in expected
