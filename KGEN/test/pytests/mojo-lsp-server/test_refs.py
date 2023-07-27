# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import pytest_lsp
from lib.utils import Document, Requests, fail_if_none
from lsprotocol.types import Location
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


async def assert_hover_and_decl_location(requests, doc, arg_name: str):
    """We expect the hover of the reference to be the same as the decl."""
    decl_range = fail_if_none(doc.find_first_range(arg_name))
    ref_range = fail_if_none(doc.find_first_range(arg_name))

    decl_hover = fail_if_none(await requests.hover(doc, decl_range.start))
    ref_hover = fail_if_none(await requests.hover(doc, ref_range.start))

    assert ref_hover == decl_hover

    definition = await requests.definition(doc, ref_range.start)
    assert isinstance(definition, Location)
    assert definition.range == decl_range


async def test_argument_ref(client: LanguageClient):
    doc = Document.from_file("functions.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_hover_and_decl_location(requests, doc, "init_arg")
    await assert_hover_and_decl_location(requests, doc, "init_kargs")


async def test_struct_field_ref(client: LanguageClient):
    doc = Document.from_file("struct_fields.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_hover_and_decl_location(requests, doc, "a_field")


async def test_struct_alias_ref(client: LanguageClient):
    doc = Document.from_file("aliases.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_hover_and_decl_location(requests, doc, "AliasInStruct")


async def test_global_variables_ref(client: LanguageClient):
    doc = Document.from_file("global_variables.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_hover_and_decl_location(requests, doc, "let_global_variable")
    await assert_hover_and_decl_location(requests, doc, "var_global_variable")
