# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

from lsprotocol.types import (
    DidOpenTextDocumentParams,
    InitializeParams,
    TextDocumentItem,
)
from pytest_lsp import LanguageClient, client_capabilities


def open_document(client: LanguageClient, uri: str, text: str):
    client.text_document_did_open(
        DidOpenTextDocumentParams(
            text_document=TextDocumentItem(
                uri=uri, language_id="mojo", version=0, text=text
            )
        )
    )


async def initialize(lsp_client: LanguageClient):
    return await lsp_client.initialize_session(
        InitializeParams(
            capabilities=client_capabilities("visual-studio-code"),
            root_uri="file://" + os.environ["MODULAR_PATH"],
        )
    )
