# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from typing import List

import pytest_lsp
from lib.utils import Document, Requests, fail_if_none
from lsprotocol.types import CompletionItemKind, Position
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
import b

# This is a comment.
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(
        await requests.completion(doc, Position(line=1, character=8))
    )

    assert any(
        item.label == "builtin"
        and item.kind == CompletionItemKind.Folder
        and "Implements the builtin package" in item.documentation.value
        for item in items
    )


async def test_completion_nested_import(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
import builtin.
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(
        await requests.completion(doc, Position(line=1, character=15))
    )

    assert any(
        item.label == "bool" and item.kind == CompletionItemKind.Module
        for item in items
    )


async def test_completion_relative_import(client: LanguageClient):
    doc = Document.from_file("imports.mojo")
    range = fail_if_none(doc.find_first_range("from .aliases"))

    requests = Requests(client)
    requests.open_document(doc)
    items = fail_if_none(await requests.completion(doc, range.end))

    assert any(
        item.label == "aliases" and item.kind == CompletionItemKind.Module
        for item in items
    )


async def test_completion_import_member(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
from memory.unsafe import P
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(
        await requests.completion(doc, Position(line=1, character=27))
    )

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

    items = fail_if_none(
        await requests.completion(doc, Position(line=2, character=8))
    )

    assert any(
        item.label == "__add__" and item.kind == CompletionItemKind.Function
        for item in items
    )
    assert any(
        item.label == "value" and item.kind == CompletionItemKind.Field
        for item in items
    )


async def test_completion_top_level_lookup(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn function() -> Int:
    let value: Int = 10
    return value
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    # Check that we can complete the `Int` from `I` in the result type.
    items = fail_if_none(
        await requests.completion(doc, doc.find_first_pos("nt"))
    )
    assert any(
        item.label == "Int" and item.kind == CompletionItemKind.Struct
        for item in items
    )

    # Check that we can complete the `value` from `v` in the return statement.
    items = fail_if_none(
        await requests.completion(doc, doc.find_last_pos("alue"))
    )
    assert any(
        item.label == "value" and item.kind == CompletionItemKind.Variable
        for item in items
    )
