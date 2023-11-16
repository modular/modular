# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from typing import List

import pytest_lsp
from lib.utils import Document, Requests, fail_if_none
from lsprotocol.types import InlayHintKind
from pytest_lsp import ClientServerConfig, LanguageClient


@pytest_lsp.fixture(
    config=ClientServerConfig(
        server_command=[os.environ["MOJO_LSP_SERVER"]],
    ),
)
async def client(lsp_client: LanguageClient):
    # Setup
    await Requests(lsp_client).initialize()
    yield
    # Teardown
    await lsp_client.shutdown_session()


async def test_inlay_hint_doc_string_code_block(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        '''
fn foo():
  """Test doc string

  ```mojo
  fn comment_fn():

    return
  ```

  """
  return
''',
    )
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(await requests.inlay_hint(doc, doc.get_full_range()))
    assert len(items) != 0

    # Check that we generate an inlay hint for each line of the code block, at
    # the correct position. Empty lines will need to pad the hint label with
    # spaces.
    assert any(
        item.label == ">>>"
        and item.kind == InlayHintKind.Type
        and item.position.line == 5
        and item.position.character == 2
        for item in items
    )
    assert any(
        item.label == "  >>>"
        and item.kind == InlayHintKind.Type
        and item.position.line == 6
        and item.position.character == 0
        for item in items
    )
    assert any(
        item.label == ">>>"
        and item.kind == InlayHintKind.Type
        and item.position.line == 7
        and item.position.character == 2
        for item in items
    )
