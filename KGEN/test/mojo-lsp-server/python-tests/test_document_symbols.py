# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from typing import List

import pytest_lsp
from lib.utils import Document, Requests, fail_if_none
from lsprotocol.types import SymbolKind
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


async def test_document_symbols(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
alias Value = 10

fn foo():
  let variable = 15
  fn inner_fn():
    return

struct struct_name:
  fn struct_fn():
    return

  var field: Int
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    results = fail_if_none(await requests.document_symbols(doc))
    assert len(results) == 3

    assert results[0].name == "Value"
    assert results[0].kind == SymbolKind.Property
    assert results[0].detail == "10"

    assert results[1].name == "foo"
    assert results[1].kind == SymbolKind.Function
    assert results[1].detail == "foo()"
    assert len(results[1].children) == 1
    assert results[1].children[0].name == "inner_fn"

    assert results[2].name == "struct_name"
    assert results[2].kind == SymbolKind.Struct
    assert len(results[2].children) == 2
    assert results[2].children[0].name == "struct_fn"
    assert results[2].children[1].name == "field"
    assert results[2].children[1].kind == SymbolKind.Field
    assert results[2].children[1].detail == "Int"
