# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from lsprotocol.types import SymbolKind
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def test_document_symbols(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
alias Value = 10

fn foo(a: DTypePointer[DType.float32]) -> Float32:
  var variable = 15
  fn inner_fn():
    return
  fn inner_closure(arg: Int) -> Float32:
    return a.load(arg)
  return inner_fn(variable)

struct struct_name:
  fn struct_fn():
    return

  var field: Int
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    results = fail_if_none(await requests.document_symbols(doc))
    assert len(results) == 3

    assert results[0].name == "Value"
    assert results[0].kind == SymbolKind.Property
    assert results[0].detail == "10"

    assert results[1].name == "foo"
    assert results[1].kind == SymbolKind.Function
    assert results[1].detail.startswith("foo(")
    assert len(results[1].children) == 1
    assert results[1].children[0].name == "inner_fn"

    assert results[2].name == "struct_name"
    assert results[2].kind == SymbolKind.Struct
    assert len(results[2].children) == 3
    assert results[2].children[0].name == "struct_fn"
    assert results[2].children[1].name == "field"
    assert results[2].children[1].kind == SymbolKind.Field
    assert results[2].children[1].detail == "Int"
    assert results[2].children[2].name == "__del__"
    assert results[2].children[2].kind == SymbolKind.Function
    assert results[2].children[2].detail == "__del__(owned self: Self)"
