# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from lsprotocol.types import CompletionItemKind, MarkupContent, Position, Range
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def test_codeblock_diagnostic(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        '''
fn function():
  """Test doc string.

  ```mojo
  let foo = bar
  ```
  """
''',
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
        == "use of unknown declaration 'bar'"
    )


async def test_codeblock_hover(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        '''
fn function():
  """Test doc string.

  ```mojo
  fn test():
    let foo: Int = 420
    var bar = 1 + `foo`
    print(bar)
  ```

  """
''',
    )
    requests = Requests(client)
    requests.open_document(doc)

    result = fail_if_none(await requests.hover(doc, doc.find_first_pos("foo")))
    assert isinstance(result.contents, MarkupContent)
    assert (
        result.contents.value
        == """```mojo
(variable) let foo: Int
```"""
    )
    assert result.range == Range(
        start=Position(line=6, character=8), end=Position(line=6, character=11)
    )


async def test_codeblock_completion(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        '''
fn function():
  """Test doc string.

  ```mojo
  let value = 10
  ```

  ```mojo
  value.completion
  ```

  """
''',
    )
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(
        await requests.completion(doc, doc.find_first_pos("completion"))
    )

    assert any(
        item.label == "value" and item.kind == CompletionItemKind.Field
        for item in items
    )


async def test_codeblock_end_completion(client: LanguageClient):
    doc = Document.from_file("doc_strings.mojo")
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(
        await requests.completion(
            doc, doc.find_first_range("test_completions.").end
        )
    )

    assert any(
        item.label == "completion_test"
        and item.kind == CompletionItemKind.Function
        for item in items
    )
