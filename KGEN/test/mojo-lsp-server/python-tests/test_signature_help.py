# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from typing import List

import pytest_lsp
from lib.utils import Document, Requests, fail_if_none
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


async def test_signature_help_overload(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn function(): # skip
    return
fn function(arg: Int) -> Int: # skip
    return arg
fn function(arg: Bool, arg2: Int) -> Int: # skip
    return arg2

fn test():
    function()
    function(10)
    function(True, 10)
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    function_calls = list(doc.find_all_ranges("function("))
    assert len(function_calls) == 3

    # Test possible results for `function(`.
    result = fail_if_none(
        await requests.signature_help(doc, function_calls[0].end)
    )
    assert len(result.signatures) == 3
    assert result.active_signature == 0
    assert result.active_parameter == 0
    assert result.signatures[0].label == "fn function()"
    assert result.signatures[1].label == "fn function(arg: Int) -> Int"
    assert (
        result.signatures[2].label == "fn function(arg: Bool, arg2: Int) -> Int"
    )

    # Test possible results for `function(arg`.
    result = fail_if_none(
        await requests.signature_help(doc, function_calls[1].end)
    )
    assert len(result.signatures) == 2
    assert result.active_signature == 0
    assert result.active_parameter == 0
    assert result.signatures[0].label == "fn function(arg: Int) -> Int"
    assert (
        result.signatures[1].label == "fn function(arg: Bool, arg2: Int) -> Int"
    )

    # Test possible results for `function(arg,`.
    result = fail_if_none(
        await requests.signature_help(doc, function_calls[2].end)
    )
    assert len(result.signatures) == 1
    assert result.active_signature == 0
    assert result.active_parameter == 0
    assert (
        result.signatures[0].label == "fn function(arg: Bool, arg2: Int) -> Int"
    )

    # Test possible results for `function(arg,`.
    result = fail_if_none(
        await requests.signature_help(doc, doc.find_last_range("True,").end)
    )
    assert len(result.signatures) == 1
    assert result.active_signature == 0
    assert result.active_parameter == 1
    assert (
        result.signatures[0].label == "fn function(arg: Bool, arg2: Int) -> Int"
    )


async def test_signature_help_type_call(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
struct SomeStruct:
    var a_field: Int

    fn __init__(inout self):
        pass

    fn __init__(inout self, a_field: Int):
        pass

fn test():
    SomeStruct()
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(
        await requests.signature_help(
            doc, doc.find_last_range("SomeStruct(").end
        )
    )
    assert len(items.signatures) == 2
    assert items.active_signature == 0
    assert items.active_parameter == 1
    assert items.signatures[0].label == "fn __init__(inout self: Self)"
    assert (
        items.signatures[1].label
        == "fn __init__(inout self: Self, a_field: Int)"
    )
