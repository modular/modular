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


async def test_diagnostics_invalid_import(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
from a.b.c import d
""",
    )
    requests = Requests(client)
    requests.open_document(doc)
    await requests.client.wait_for_notification(
        "textDocument/publishDiagnostics"
    )

    assert (
        doc.uri in requests.client.diagnostics
        and len(requests.client.diagnostics[doc.uri]) == 1
        and requests.client.diagnostics[doc.uri][0].message
        == "unable to locate module 'a'"
    )
