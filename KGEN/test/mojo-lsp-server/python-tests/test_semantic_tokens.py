# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def test_semantic_tokens(client: LanguageClient):
    requests = Requests(client)
    initialize_result = await requests.initialize()
    semantic_token_legend = (
        initialize_result.capabilities.semantic_tokens_provider.legend
    )
    semantic_token_types = semantic_token_legend.token_types
    semantic_token_modifiers = semantic_token_legend.token_modifiers

    doc = Document(
        "foo.mojo",
        """
import builtin
alias builtin_alias = builtin

struct Struct:
  var field: Int

alias struct_alias = Struct

fn foo():
  return

alias int_alias = 10

trait ATrait:
  fn foo(owned self, i: Self):
     ...

struct StructWithTrait(ATrait):
    fn foo(owned self, i: Self):
        pass
""",
    )
    requests.open_document(doc)

    tokens = fail_if_none(
        await requests.semantic_tokens(
            doc, semantic_token_types, semantic_token_modifiers
        )
    )
    assert len(tokens) != 0

    # Check that we generate proper tokens for the different constructs.
    assert any(
        token.range == doc.find_first_range("builtin")
        and token.token_type == "namespace"
        for token in tokens
    )
    assert any(
        token.range == doc.find_first_range("builtin_alias")
        and token.token_type == "namespace"
        for token in tokens
    )

    assert any(
        token.range == doc.find_first_range("Struct")
        and token.token_type == "class"
        for token in tokens
    )
    assert any(
        token.range == doc.find_first_range("struct_alias")
        and token.token_type == "type"
        for token in tokens
    )
    assert any(
        token.range == doc.find_first_range("field")
        and token.token_type == "property"
        for token in tokens
    )

    assert any(
        token.range == doc.find_first_range("foo")
        and token.token_type == "function"
        for token in tokens
    )

    assert any(
        token.range == doc.find_first_range("int_alias")
        and token.token_type == "variable"
        for token in tokens
    )

    assert any(
        token.range == doc.find_first_range("ATrait")
        and token.token_type == "interface"
        for token in tokens
    )

    assert any(
        token.range == doc.find_first_range("Self")
        and token.token_type == "interface"
        for token in tokens
    )

    # Check that we didn't add a token for the synthetic methods of the
    # StructWithTrait struct.
    assert not any(
        token.range.start == doc.find_last_pos("struct StructWithTrait")
        and token.token_type == "function"
        for token in tokens
    )
