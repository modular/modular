# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import pytest_lsp
from lib.utils import Document, Requests, fail_if_none
from lsprotocol.types import MarkupContent, Range
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


async def assert_ref(
    requests,
    doc,
    identifier: str,
    range: Range,
):
    ref_hover = fail_if_none(await requests.hover(doc, range.start))
    assert isinstance(ref_hover.contents, MarkupContent)
    assert f"struct `{identifier}`" in ref_hover.contents.value
    assert ref_hover.range == range


async def assert_every_ref(
    requests,
    doc,
    identifier: str,
    count: int,
):
    """Issue hover requests on every occurence of `identifier`, then assert the hover response expecting a `struct` type.
    Finally, `identifier_count` is expected to be the number of occurences of `identifier`."""
    identifier_count = 0
    for range in doc.find_all_ranges(identifier):
        await assert_ref(requests, doc, identifier, range)
        identifier_count += 1

    assert identifier_count == count


async def test_argument_ref(client: LanguageClient):
    doc = Document.from_file("types.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_every_ref(requests, doc, "Bool", count=3)
    await assert_every_ref(requests, doc, "StaticIntTuple", count=3)
    await assert_every_ref(requests, doc, "StaticTuple", count=3)

    range = fail_if_none(doc.find_last_range("DType"))
    await assert_ref(requests, doc, "DType", range)
