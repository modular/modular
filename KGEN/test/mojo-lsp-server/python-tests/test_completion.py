# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from lsprotocol.types import CompletionItemKind, Position
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


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


# This following test checks that we can perform code completions within
# compound statements like `if` and `for`, in partially parsed states.


async def check_partial_compound_statement(
    requests: Requests, doc_name: str, code: str, complete_at: str
):
    doc = Document(doc_name, code)
    requests.open_document(doc)

    # Check that we have completion results.
    items = fail_if_none(
        await requests.completion(doc, doc.find_first_pos(complete_at))
    )
    assert len(items) != 0


async def test_completion_partial_fn(client: LanguageClient):
    await check_partial_compound_statement(
        Requests(client),
        "fn_no_colon.mojo",
        """
fn function(arg: Int)
        """,
        complete_at="nt",
    )


async def test_completion_partial_if(client: LanguageClient):
    await check_partial_compound_statement(
        Requests(client),
        "if_no_colon.mojo",
        """
fn function(arg: Int):
    if arg.value
        """,
        complete_at="value",
    )


async def test_completion_partial_elif(client: LanguageClient):
    await check_partial_compound_statement(
        Requests(client),
        "elif_no_colon.mojo",
        """
fn function(arg: Int):
    if False:
        return
    elif arg.value
        """,
        complete_at="value",
    )


async def test_completion_partial_while(client: LanguageClient):
    await check_partial_compound_statement(
        Requests(client),
        "while_no_colon.mojo",
        """
fn function(arg: Int):
    while arg.value
        """,
        complete_at="value",
    )


async def test_completion_partial_with(client: LanguageClient):
    await check_partial_compound_statement(
        Requests(client),
        "with_no_colon.mojo",
        """
fn function(arg: Int):
    with arg.value
        """,
        complete_at="value",
    )
