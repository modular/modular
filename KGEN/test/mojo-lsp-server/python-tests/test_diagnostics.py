# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, mojo_lsp_client
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


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
        and len(requests.client.diagnostics[doc.uri]) == 2
        and requests.client.diagnostics[doc.uri][0].message
        == "unable to locate module 'a'"
    )
