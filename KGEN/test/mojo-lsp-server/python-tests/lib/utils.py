# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os
from pathlib import Path
from typing import Generator, List, Optional, TypeVar

import pytest_lsp
from lsprotocol.types import (
    CompletionList,
    CompletionParams,
    DefinitionParams,
    DidOpenNotebookDocumentParams,
    DidOpenTextDocumentParams,
    DocumentSymbolParams,
    FoldingRangeParams,
    HoverParams,
    InitializeParams,
    InlayHintParams,
)
from lsprotocol.types import NotebookCell as LspNotebookCell
from lsprotocol.types import NotebookCellKind
from lsprotocol.types import NotebookDocument as LspNotebookDocument
from lsprotocol.types import (
    Position,
    Range,
    SemanticTokensParams,
    SignatureHelpParams,
    TextDocumentIdentifier,
    TextDocumentItem,
)
from pytest_lsp import ClientServerConfig, LanguageClient, client_capabilities

T = TypeVar("T")


def fail_if_none(t: Optional[T]) -> T:
    assert t is not None
    return t


# We need to use `--log=error` instead of `debug`, because when stderr is too
# long, mojo-lsp-server hangs trying to send bytes to the test client.
MOJO_LSP_CONFIG = ClientServerConfig(
    server_command=[os.environ["MOJO_LSP_SERVER"], "--log=error"],
)


# pyright: reportUnknownMemberType=false
@pytest_lsp.fixture(config=MOJO_LSP_CONFIG)
async def mojo_lsp_client(
    lsp_client: LanguageClient,
):
    # Setup
    await Requests(lsp_client).initialize()
    yield
    # Teardown
    # Here we should shutdown the session programmatically, but we instead don't
    # do that, which forces pytest_lsp to kill the LSP manually. We have to do
    # this because there's a race condition with version 0.3.1 in which
    # pytest_lsp tries to kill the LSP even though it's already dead, which
    # raises an Error. This only happens in smaller machines like our CI hosts.
    #
    # await lsp_client.shutdown_session()


class Document:
    """Helper class for dealing with documents, either from files or from memory.
    """

    @staticmethod
    def from_file(file_name: str):
        path = Path(__file__).parent.parent / "inputs" / file_name
        with open(path, "r") as file:
            return Document(str(path), file.read())

    def __init__(self, name: str, contents: str):
        self.uri = f"test:///{name}"
        self.set_contents(contents)

    def set_contents(self, contents: str):
        """Update the contents of this document with the given string."""
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

        See `find_all_ranges` for additional notes on the `find` family of functions.
        """
        for range in self.find_all_ranges(substr, in_line_with):
            return range
        return None

    def find_first_pos(self, substr: str) -> Optional[Position]:
        """Find the position of the first occurrence of the given `substr` in the document.

        See `find_all_ranges` for additional notes on the `find` family of functions.
        """
        for range in self.find_all_ranges(substr):
            return range.start
        return None

    def find_last_range(
        self, substr: str, in_line_with: Optional[str] = None
    ) -> Optional[Range]:
        """Find the range of the last occurrence of the given `substr` in the document.

        See `find_all_ranges` for additional notes on the `find` family of functions.
        """
        for range in reversed(list(self.find_all_ranges(substr, in_line_with))):
            return range
        return None

    def find_last_pos(self, substr: str) -> Optional[Position]:
        """Find the position of the last occurrence of the given `substr` in the document.

        See `find_all_ranges` for additional notes on the `find` family of functions.
        """
        for range in reversed(list(self.find_all_ranges(substr))):
            return range.start
        return None

    def get_full_range(self):
        """Return a range covering the entire document."""
        return Range(
            start=Position(line=0, character=0),
            end=Position(line=len(self.lines), character=0),
        )

    @property
    def identifier(self) -> TextDocumentIdentifier:
        return TextDocumentIdentifier(self.uri)


class NotebookDocument:
    """Helper class for dealing with notebook documents."""

    def __init__(self, name: str, cell_contents: List[str]):
        self.uri = f"test:///{str(name)}"

        self.cells = []
        for cell in cell_contents:
            self.cells.append(Document(str(len(self.cells)), cell))


class SemanticToken:
    """High level representation of a semantic token"""

    def __init__(
        self,
        token: List[int],
        token_types: List[str],
        token_modifiers: List[str],
        last_token,
    ):
        line = token[0]
        col = token[1]
        if last_token:
            line += last_token.range.start.line

            # If the line number is 0, we are in the same line as the last token.
            # In that case, we need to add the column offset of the last token.
            if token[0] == 0:
                col += last_token.range.start.character
        self.range = Range(Position(line, col), Position(line, col + token[2]))
        self.token_type = token_types[token[3]]

        # Unpack the modifier bit list into a list of strings.
        self.token_modifiers = []
        for i in range(0, len(token_modifiers)):
            if token[4] & (1 << i):
                self.token_modifiers = token_modifiers[token[i]]


class Requests:
    """Helper class for issuing requests to the server. It is not intended to be a full wrapper of `LanguageClient`.
    """

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

    def open_notebook_document(self, doc: NotebookDocument):
        self.client.notebook_document_did_open(
            DidOpenNotebookDocumentParams(
                notebook_document=LspNotebookDocument(
                    uri=doc.uri,
                    notebook_type="jupyter",
                    version=0,
                    cells=map(
                        lambda cell: LspNotebookCell(
                            kind=NotebookCellKind.Code, document=cell.uri
                        ),
                        doc.cells,
                    ),
                ),
                cell_text_documents=map(
                    lambda cell: TextDocumentItem(
                        uri=cell.uri,
                        language_id="mojo",
                        version=0,
                        text=cell.contents,
                    ),
                    doc.cells,
                ),
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

    async def document_symbols(self, doc: Document):
        return await self.client.text_document_document_symbol_async(
            params=DocumentSymbolParams(text_document=doc.identifier)
        )

    async def completion(self, doc: Document, pos: Position):
        results = await self.client.text_document_completion_async(
            params=CompletionParams(position=pos, text_document=doc.identifier)
        )
        if results is None:
            return None
        return results.items if isinstance(results, CompletionList) else results

    async def folding_range(self, doc: Document):
        return await self.client.text_document_folding_range_async(
            params=FoldingRangeParams(text_document=doc.identifier)
        )

    async def inlay_hint(self, doc: Document, range: Range):
        return await self.client.text_document_inlay_hint_async(
            params=InlayHintParams(text_document=doc.identifier, range=range)
        )

    async def semantic_tokens(
        self, doc: Document, token_types: List[str], token_modifiers: List[str]
    ):
        lsp_result = await self.client.text_document_semantic_tokens_full_async(
            params=SemanticTokensParams(text_document=doc.identifier)
        )
        if lsp_result is None:
            return None

        result = []
        last_token = None
        for i in range(0, len(lsp_result.data), 5):
            last_token = SemanticToken(
                lsp_result.data[i : i + 5],
                token_types,
                token_modifiers,
                last_token,
            )
            result.append(last_token)
        return result

    async def signature_help(self, doc: Document, pos: Position):
        return await self.client.text_document_signature_help_async(
            params=SignatureHelpParams(
                position=pos, text_document=doc.identifier
            )
        )
