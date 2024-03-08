# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import List, Optional

from lib.utils import NotebookDocument, Requests, fail_if_none, mojo_lsp_client
from lsprotocol.types import (
    CompletionItemKind,
    DidChangeNotebookDocumentParams,
    MarkupContent,
    NotebookCell,
    NotebookCellArrayChange,
    NotebookCellKind,
    NotebookDocumentChangeEvent,
    NotebookDocumentChangeEventCellsType,
    NotebookDocumentChangeEventCellsTypeStructureType,
    NotebookDocumentChangeEventCellsTypeTextContentType,
    Position,
    Range,
    TextDocumentContentChangeEvent_Type1,
    TextDocumentContentChangeEvent_Type2,
    VersionedNotebookDocumentIdentifier,
    VersionedTextDocumentIdentifier,
)
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def test_updates(client: LanguageClient):
    cell_contents = [
        """
fn function() -> Int:
  return 10
""",
        """
function()
""",
    ]
    doc = NotebookDocument("test_updates", cell_contents)

    requests = Requests(client)
    requests.open_notebook_document(doc)

    async def wait_for_diags():
        """Wait for diagnostics to be published for all cells."""
        while len(requests.client.diagnostics) <= len(doc.cells):
            await requests.client.wait_for_notification(
                "textDocument/publishDiagnostics"
            )

    # Check that no diagnostics were emitted.
    await wait_for_diags()
    for cell in doc.cells:
        assert (
            cell.uri in requests.client.diagnostics
            and len(requests.client.diagnostics[cell.uri]) == 0
        )

    def build_change_params(
        arrayChange: NotebookCellArrayChange,
        text_content: Optional[
            List[NotebookDocumentChangeEventCellsTypeTextContentType]
        ] = None,
    ) -> DidChangeNotebookDocumentParams:
        return DidChangeNotebookDocumentParams(
            notebook_document=VersionedNotebookDocumentIdentifier(
                uri=doc.uri,
                version=0,
            ),
            change=NotebookDocumentChangeEvent(
                cells=NotebookDocumentChangeEventCellsType(
                    structure=NotebookDocumentChangeEventCellsTypeStructureType(
                        array=arrayChange
                    ),
                    text_content=text_content,
                ),
            ),
        )

    # Send an update to replace the first cell.
    requests.client.diagnostics.clear()
    requests.client.notebook_document_did_change(
        build_change_params(
            NotebookCellArrayChange(
                start=0,
                delete_count=1,
                cells=[
                    NotebookCell(
                        kind=NotebookCellKind.Code,
                        document=doc.cells[0].uri,
                    )
                ],
            )
        )
    )

    # Check that the second cell can't find the called function.
    await wait_for_diags()
    cell1_diags = requests.client.diagnostics[doc.cells[1].uri]
    assert (
        len(cell1_diags) == 1
        and cell1_diags[0].message == "use of unknown declaration 'function'"
    )

    # Add back in the first cell, and change the function called in the second
    # cell.
    requests.client.diagnostics.clear()
    requests.client.notebook_document_did_change(
        build_change_params(
            NotebookCellArrayChange(start=0, delete_count=0, cells=[]),
            [
                NotebookDocumentChangeEventCellsTypeTextContentType(
                    document=VersionedTextDocumentIdentifier(
                        uri=doc.cells[0].uri,
                        version=0,
                    ),
                    changes=[
                        TextDocumentContentChangeEvent_Type2(
                            text=cell_contents[0]
                        ),
                        TextDocumentContentChangeEvent_Type1(
                            range=doc.cells[0].find_first_range("function"),
                            text="renamed_function",
                        ),
                    ],
                ),
                NotebookDocumentChangeEventCellsTypeTextContentType(
                    document=VersionedTextDocumentIdentifier(
                        uri=doc.cells[1].uri,
                        version=0,
                    ),
                    changes=[
                        TextDocumentContentChangeEvent_Type1(
                            range=doc.cells[1].find_first_range("function"),
                            text="renamed_function",
                        ),
                    ],
                ),
            ],
        )
    )
    # Update the contents within the cells in the test document.
    for cell in doc.cells:
        cell.set_contents(cell.contents.replace("function", "renamed_function"))

    # Check that no diagnostics were emitted.
    await wait_for_diags()
    for cell in doc.cells:
        assert (
            cell.uri in requests.client.diagnostics
            and len(requests.client.diagnostics[cell.uri]) == 0
        )

    # Check that the second cell has the renamed function.
    result = fail_if_none(
        await requests.hover(
            doc.cells[1], doc.cells[1].find_first_pos("renamed_function")
        )
    )
    assert isinstance(result.contents, MarkupContent)
    assert (
        result.contents.value
        == """```mojo
(function) fn renamed_function() -> Int
```"""
    )
    assert result.range == Range(
        start=Position(line=1, character=0), end=Position(line=1, character=16)
    )


async def test_completion(client: LanguageClient):
    cell_contents = [
        """
fn function() -> Int:
  return 10
""",
        """
fu
""",
    ]
    doc = NotebookDocument("test_completion", cell_contents)

    requests = Requests(client)
    requests.open_notebook_document(doc)

    items = fail_if_none(
        await requests.completion(doc.cells[1], Position(line=1, character=2))
    )
    assert any(
        item.label == "function" and item.kind == CompletionItemKind.Function
        for item in items
    )


async def test_signature_help(client: LanguageClient):
    cell_contents = [
        """
struct SomeStruct:
    var a_field: Int

    fn __init__(inout self):
        pass

    fn __init__(inout self, a_field: Int):
        pass
""",
        """
SomeStruct()
""",
    ]
    doc = NotebookDocument("test_signature_help", cell_contents)

    requests = Requests(client)
    requests.open_notebook_document(doc)

    result = fail_if_none(
        await requests.signature_help(
            doc.cells[1], doc.cells[1].find_last_range("SomeStruct(").end
        )
    )
    assert len(result.signatures) == 2
    assert result.active_signature == 0
    assert result.active_parameter == 1
    assert result.signatures[0].label == "fn __init__(inout self: Self)"
    assert (
        result.signatures[1].label
        == "fn __init__(inout self: Self, a_field: Int)"
    )


async def test_python(client: LanguageClient):
    cell_contents = [
        """%%python
def function():
  return
""",
        """
function
""",
    ]
    doc = NotebookDocument("test_python", cell_contents)

    requests = Requests(client)
    requests.open_notebook_document(doc)

    result = fail_if_none(
        await requests.hover(doc.cells[1], Position(line=1, character=2))
    )
    assert isinstance(result.contents, MarkupContent)
    assert (
        result.contents.value
        == """```mojo
(argument) inout function: PythonObject
```"""
    )
