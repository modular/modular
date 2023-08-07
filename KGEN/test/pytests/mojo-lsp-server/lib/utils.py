# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path
from typing import Generator, Optional, TypeVar

from lsprotocol.types import (
    CompletionList,
    CompletionParams,
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
    """Helper class for dealing with documents, either from files or from memory."""

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

    def find_all_ranges(
        self, substr: str, in_line_with: Optional[str] = None
    ) -> Generator[Range, None, None]:
        """Generate all non-overlapping locations where `substr` is found in the document.

        This function, just like all other `find` methods, omits lines that end with `# skip`.

        in_line_with: if provided, this function will only consider lines that contain that substring.
        """
        for line in range(0, len(self.lines)):
            if self.lines[line].strip().endswith("# skip"):
                continue
            if in_line_with and in_line_with not in self.lines[line]:
                continue
            start = 0
            while (character := self.lines[line].find(substr, start)) != -1:
                yield Range(
                    start=Position(line, character),
                    end=Position(line, character + len(substr)),
                )
                start = character + len(substr)

    def find_first_range(
        self, substr: str, in_line_with: Optional[str] = None
    ) -> Optional[Range]:
        """Find the range of the first occurrence of the given `substr` in the document.

        See `find_all_ranges` for additional notes on the `find` family of functions."""
        for range in self.find_all_ranges(substr, in_line_with):
            return range
        return None

    def find_first_pos(self, substr: str) -> Optional[Position]:
        """Find the position of the first occurrence of the given `substr` in the document.

        See `find_all_ranges` for additional notes on the `find` family of functions."""
        for range in self.find_all_ranges(substr):
            return range.start
        return None

    def find_last_range(
        self, substr: str, in_line_with: Optional[str] = None
    ) -> Optional[Range]:
        """Find the range of the last occurrence of the given `substr` in the document.

        See `find_all_ranges` for additional notes on the `find` family of functions."""
        for range in reversed(list(self.find_all_ranges(substr, in_line_with))):
            return range
        return None

    def find_last_pos(self, substr: str) -> Optional[Position]:
        """Find the position of the last occurrence of the given `substr` in the document.

        See `find_all_ranges` for additional notes on the `find` family of functions."""
        for range in reversed(list(self.find_all_ranges(substr))):
            return range.start
        return None

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

    async def completion(self, doc: Document, pos: Position):
        results = await self.client.text_document_completion_async(
            params=CompletionParams(position=pos, text_document=doc.identifier)
        )
        if results is None:
            return None
        return results.items if isinstance(results, CompletionList) else results
