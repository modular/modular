# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import pytest_lsp
from lsprotocol.types import (
    HoverParams,
    Position,
    Range,
    TextDocumentIdentifier,
)
from pytest_lsp import ClientServerConfig, LanguageClient

from utils import initialize, open_document


@pytest_lsp.fixture(
    config=ClientServerConfig(
        server_command=[os.environ["MOJO_LSP_SERVER"]],
    ),
)
async def client(lsp_client: LanguageClient):
    # Setup
    await initialize(lsp_client)
    yield
    # Teardown
    await lsp_client.shutdown_session()


async def test_hover_letvar(client: LanguageClient):
    uri = "test:///foo.mojo"
    open_document(
        client,
        uri,
        """
from IO import print

fn function():
  let foo: Int = 420
  var bar = 1 + foo
  print(bar)
""",
    )

    result = await client.text_document_hover_async(
        params=HoverParams(
            position=Position(line=5, character=17),
            text_document=TextDocumentIdentifier(uri),
        )
    )
    assert (
        result.contents.value
        == """### variable `foo`

---

###
```mojo
let foo: Int
```"""
    )
    assert result.range == Range(
        start=Position(line=4, character=6), end=Position(line=4, character=9)
    )

    result = await client.text_document_hover_async(
        params=HoverParams(
            position=Position(line=6, character=8),
            text_document=TextDocumentIdentifier(uri),
        )
    )
    assert (
        result.contents.value
        == """### variable `bar`

---

###
```mojo
var bar: Int
```"""
    )
    assert result.range == Range(
        start=Position(line=5, character=6), end=Position(line=5, character=9)
    )
