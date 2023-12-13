# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def test_doc_string_folding_range(client: LanguageClient):
    requests = Requests(client)

    doc = Document(
        "foo.mojo",
        '''
fn single_line():
  """This is a single line doc string."""
  return

fn multi_line():
  """This is a multi-line doc string.

  It has multiple lines.

  """
''',
    )
    requests.open_document(doc)

    ranges = fail_if_none(await requests.folding_range(doc))
    assert len(ranges) != 0

    assert any(
        rangeIt.start_line == 2
        and rangeIt.start_character == 5
        and rangeIt.end_line == 2
        and rangeIt.end_character == 38
        and rangeIt.kind == "comment"
        for rangeIt in ranges
    )
    assert any(
        rangeIt.start_line == 6
        and rangeIt.start_character == 5
        and rangeIt.end_line == 10
        and rangeIt.end_character == 2
        and rangeIt.kind == "comment"
        for rangeIt in ranges
    )
