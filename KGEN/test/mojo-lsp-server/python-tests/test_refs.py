# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from typing import Optional

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from lsprotocol.types import Location, MarkupContent, Position, Range
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def assert_ref(
    kind: str,
    requests,
    doc,
    identifier: str,
    range: Optional[Range],
):
    assert range is not None
    ref_hover = fail_if_none(await requests.hover(doc, range.start))
    assert isinstance(ref_hover.contents, MarkupContent)
    assert identifier in ref_hover.contents.value
    assert ref_hover.range == range


async def assert_all_refs(
    kind: str,
    requests,
    doc,
    identifier: str,
    count: int,
):
    """Issue hover requests on every occurence of `identifier`, then assert the hover response expecting a `kind` type.
    Finally, `identifier_count` is expected to be the number of occurences of `identifier`.
    """
    identifier_count = 0
    for range in doc.find_all_ranges(identifier):
        await assert_ref(kind, requests, doc, identifier, range)
        identifier_count += 1

    assert identifier_count == count


async def assert_hover_and_decl_location(requests, doc, arg_name: str):
    """We expect the hover of the reference to be the same as the decl."""
    decl_range = fail_if_none(doc.find_first_range(arg_name))
    ref_range = fail_if_none(doc.find_first_range(arg_name))

    decl_hover = fail_if_none(await requests.hover(doc, decl_range.start))
    ref_hover = fail_if_none(await requests.hover(doc, ref_range.start))

    assert ref_hover == decl_hover

    definition = await requests.definition(doc, ref_range.start)
    assert isinstance(definition, list)
    assert len(definition) == 1
    assert isinstance(definition[0], Location)
    assert definition[0].range == decl_range


async def test_argument_ref(client: LanguageClient):
    doc = Document.from_file("functions.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_hover_and_decl_location(requests, doc, "init_arg")
    await assert_hover_and_decl_location(requests, doc, "init_kargs")


async def test_struct_field_ref(client: LanguageClient):
    doc = Document.from_file("struct_fields.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_hover_and_decl_location(requests, doc, "a_field")


async def test_struct_alias_ref(client: LanguageClient):
    doc = Document.from_file("aliases.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_hover_and_decl_location(requests, doc, "AliasInStruct")


async def test_global_variables_ref(client: LanguageClient):
    doc = Document.from_file("global_variables.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    await assert_hover_and_decl_location(requests, doc, "var_global_variable")


async def test_refs(client: LanguageClient):
    requests = Requests(client)

    doc = Document.from_file("types.mojo")
    requests.open_document(doc)

    await assert_all_refs("struct", requests, doc, "Bool", count=3)
    await assert_all_refs("struct", requests, doc, "StaticIntTuple", count=3)
    await assert_all_refs("struct", requests, doc, "StaticTuple", count=3)
    await assert_ref(
        "struct",
        requests,
        doc,
        "DType",
        doc.find_last_range("DType"),
    )
    await assert_ref(
        "struct",
        requests,
        doc,
        "Int",
        doc.find_last_range("Int", in_line_with="builtin.int.Int"),
    )
    await assert_ref(
        "struct",
        requests,
        doc,
        "Int",
        doc.find_first_range("Int", in_line_with="StaticTuple[size, Int]"),
    )
    await assert_ref(
        "module",
        requests,
        doc,
        "Int",
        doc.find_first_range("Int", in_line_with="builtin.int.Int"),
    )

    doc = Document.from_file("struct_fields.mojo")
    requests.open_document(doc)

    await assert_all_refs("struct", requests, doc, "SomeStruct", count=2)
    await assert_all_refs("variable", requests, doc, "someStruct", count=2)

    doc = Document.from_file("aliases.mojo")
    requests.open_document(doc)

    await assert_all_refs("struct", requests, doc, "StructWithAlias", count=2)


async def test_decl_multi_location(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn print(x: StringRef):
    pass

fn print(x: Bool):
    pass

fn function[type: AnyRegType](arg: type):
    print(arg)
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    ref_range = fail_if_none(doc.find_first_range("print(arg"))
    str_range = fail_if_none(doc.find_first_range("print(x: StringRef"))
    bool_range = fail_if_none(doc.find_first_range("print(x: Bool"))

    str_definition = await requests.definition(doc, str_range.start)
    bool_definition = await requests.definition(doc, bool_range.start)

    assert len(str_definition) == 1
    assert len(bool_definition) == 1
    assert isinstance(str_definition[0], Location)
    assert isinstance(bool_definition[0], Location)

    definition = await requests.definition(doc, ref_range.start)
    assert isinstance(definition, list)
    assert len(definition) == 2
    assert isinstance(definition[0], Location)

    assert definition == [str_definition[0], bool_definition[0]]


async def test_module(client: LanguageClient):
    doc = Document("foo.mojo", "")
    requests = Requests(client)
    requests.open_document(doc)

    # Make sure we didn't add a definition for the module itself, its location
    # is technically the start of the document (but it's not defined inside the
    # document).
    assert await requests.definition(doc, Position(0, 0)) == []
