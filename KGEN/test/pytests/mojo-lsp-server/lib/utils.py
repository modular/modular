# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path
from typing import Optional, TypeVar

from lsprotocol.types import (
    DefinitionParams,
    DidOpenTextDocumentParams,
    HoverParams,
    InitializeParams,
    Position,
    Range,
    TextDocumentIdentifier,
    TextDocumentItem,
)
from pytest_lsp import LanguageClient, client_capabilities

T = TypeVar("T")


def fail_if_none(t: Optional[T]) -> T:
    assert t is not None
    return t


class Document:
    """Helper class for dealing with documents, either from files or from memory"""

    @staticmethod
    def from_file(file_name: str):
        with open(
            Path(__file__).parent.parent / "inputs" / file_name, "r"
        ) as file:
            return Document(file_name, file.read())

    def __init__(self, name: str, contents: str):
        self.uri = f"test:///{name}"
        self.contents = contents
        self.lines = contents.splitlines()

    def find_first_range(self, substr: str) -> Optional[Range]:
        """Find the range of the first occurrence of the given `substr` in the document."""
        for line in range(0, len(self.lines)):
            if (character := self.lines[line].find(substr)) != -1:
                return Range(
                    start=Position(line, character),
                    end=Position(line, character + len(substr)),
                )
        return None

    def find_first_pos(self, substr: str) -> Optional[Position]:
        """Find the position of the first occurrence of the given `substr` in the document."""
        range = self.find_first_range(substr)
        return range.start if range else None

    def find_last_range(self, substr: str) -> Optional[Range]:
        """Find the range of the last occurrence of the given `substr` in the document."""
        for line in reversed(range(0, len(self.lines))):
            if (character := self.lines[line].rfind(substr)) != -1:
                return Range(
                    start=Position(line, character),
                    end=Position(line, character + len(substr)),
                )
        return None

    def find_last_pos(self, substr: str) -> Optional[Position]:
        """Find the position of the first occurrence of the given `substr` in the document."""
        range = self.find_first_range(substr)
        return range.start if range else None

    @property
    def identifier(self) -> TextDocumentIdentifier:
        return TextDocumentIdentifier(self.uri)


class Requests:
    """Helper class for issuing requests to the server. It is not intenteded to be a full wrapper of `LanguageClient`."""

    def __init__(self, client: LanguageClient):
        self.client = client

    async def initialize(self):
        return await self.client.initialize_session(
            InitializeParams(
                capabilities=client_capabilities("visual-studio-code"),
                root_uri="file://" + os.environ["MODULAR_PATH"],
            )
        )

    def open_document(self, doc: Document):
        self.client.text_document_did_open(
            DidOpenTextDocumentParams(
                text_document=TextDocumentItem(
                    uri=doc.uri,
                    language_id="mojo",
                    version=0,
                    text=doc.contents,
                )
            )
        )

    async def hover(self, doc: Document, pos: Position):
        return await self.client.text_document_hover_async(
            params=HoverParams(position=pos, text_document=doc.identifier)
        )

    async def definition(self, doc: Document, pos: Position):
        return await self.client.text_document_definition_async(
            params=DefinitionParams(position=pos, text_document=doc.identifier)
        )
