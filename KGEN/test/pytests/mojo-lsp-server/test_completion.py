# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from typing import List

import pytest_lsp
from lib.utils import Document, Requests, fail_if_none
from lsprotocol.types import (
    CompletionItemKind,
    CompletionList,
    CompletionParams,
    Position,
)
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


async def test_completion_import(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
import B
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    results = fail_if_none(
        await client.text_document_completion_async(
            params=CompletionParams(
                position=Position(line=1, character=8),
                text_document=doc.identifier,
            )
        )
    )
    if isinstance(results, CompletionList):
        items = results.items
    else:
        items = results

    assert any(
        item.label == "Builtin" and item.kind == CompletionItemKind.Folder
        for item in items
    )


async def test_completion_nested_import(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
import Builtin.
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    results = fail_if_none(
        await client.text_document_completion_async(
            params=CompletionParams(
                position=Position(line=1, character=15),
                text_document=doc.identifier,
            )
        )
    )
    if isinstance(results, CompletionList):
        items = results.items
    else:
        items = results

    assert any(
        item.label == "Bool" and item.kind == CompletionItemKind.Module
        for item in items
    )


async def test_completion_import_member(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
from Pointer import P
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    results = fail_if_none(
        await client.text_document_completion_async(
            params=CompletionParams(
                position=Position(line=1, character=21),
                text_document=doc.identifier,
            )
        )
    )
    if isinstance(results, CompletionList):
        items = results.items
    else:
        items = results

    assert any(
        item.label == "Pointer" and item.kind == CompletionItemKind.Struct
        for item in items
    )


async def test_completion_member_lookup(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn function(arg: Int):
    arg.
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    results = fail_if_none(
        await client.text_document_completion_async(
            params=CompletionParams(
                position=Position(line=2, character=8),
                text_document=doc.identifier,
            )
        )
    )
    if isinstance(results, CompletionList):
        items = results.items
    else:
        items = results

    assert any(
        item.label == "__add__" and item.kind == CompletionItemKind.Function
        for item in items
    )
    assert any(
        item.label == "value" and item.kind == CompletionItemKind.Field
        for item in items
    )
