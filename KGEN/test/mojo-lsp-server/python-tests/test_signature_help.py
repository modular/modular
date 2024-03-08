# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from flaky import flaky
from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def test_signature_help_overload(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn function(): # skip
    return
fn function(arg: Int) -> Int: # skip
    return arg
fn function(arg: Bool, arg2: Int) -> Int: # skip
    return arg2

fn test():
    function()
    function(10)
    function(True, 10)
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    function_calls = list(doc.find_all_ranges("function("))
    assert len(function_calls) == 3

    # Test possible results for `function(`.
    result = fail_if_none(
        await requests.signature_help(doc, function_calls[0].end)
    )
    assert len(result.signatures) == 3
    assert result.active_signature == 0
    assert result.active_parameter == 0
    assert result.signatures[0].label == "fn function()"
    assert result.signatures[1].label == "fn function(arg: Int) -> Int"
    assert (
        result.signatures[2].label == "fn function(arg: Bool, arg2: Int) -> Int"
    )

    # Test possible results for `function(arg`.
    result = fail_if_none(
        await requests.signature_help(doc, function_calls[1].end)
    )
    assert len(result.signatures) == 2
    assert result.active_signature == 0
    assert result.active_parameter == 0
    assert result.signatures[0].label == "fn function(arg: Int) -> Int"
    assert (
        result.signatures[1].label == "fn function(arg: Bool, arg2: Int) -> Int"
    )

    # Test possible results for `function(arg,`.
    result = fail_if_none(
        await requests.signature_help(doc, function_calls[2].end)
    )
    assert len(result.signatures) == 1
    assert result.active_signature == 0
    assert result.active_parameter == 0
    assert (
        result.signatures[0].label == "fn function(arg: Bool, arg2: Int) -> Int"
    )

    # Test possible results for `function(arg,`.
    result = fail_if_none(
        await requests.signature_help(doc, doc.find_last_range("True,").end)
    )
    assert len(result.signatures) == 1
    assert result.active_signature == 0
    assert result.active_parameter == 1
    assert (
        result.signatures[0].label == "fn function(arg: Bool, arg2: Int) -> Int"
    )


async def test_signature_help_type_call(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
struct SomeStruct:
    var a_field: Int

    fn __init__(inout self):
        pass

    fn __init__(inout self, a_field: Int):
        pass

fn test():
    SomeStruct()
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(
        await requests.signature_help(
            doc, doc.find_last_range("SomeStruct(").end
        )
    )
    assert len(items.signatures) == 2
    assert items.active_signature == 0
    assert items.active_parameter == 1
    assert items.signatures[0].label == "fn __init__(inout self: Self)"
    assert (
        items.signatures[1].label
        == "fn __init__(inout self: Self, a_field: Int)"
    )


async def test_signature_help_overload_params(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn function[type: DType](): # skip
    return
fn function[type: DType, type2: DType](): # skip
    return

fn test():
    function[DType.bool]()
    function[DType.bool, DType.bool]()
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    # Test possible results for `function[`.
    result = fail_if_none(
        await requests.signature_help(
            doc, doc.find_first_range("function[").end
        )
    )
    assert len(result.signatures) == 2
    assert result.active_signature == 0
    assert result.active_parameter == 0
    assert result.signatures[0].label == "fn function[type: DType]()"
    assert (
        result.signatures[1].label == "fn function[type: DType, type2: DType]()"
    )

    # Test possible results for `function[DType.bool,`.
    result = fail_if_none(
        await requests.signature_help(
            doc, doc.find_last_range("DType.bool,").end
        )
    )
    assert len(result.signatures) == 1
    assert result.active_signature == 0
    assert result.active_parameter == 1
    assert (
        result.signatures[0].label == "fn function[type: DType, type2: DType]()"
    )


async def test_signature_help_type_params(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
struct SomeStruct[dtype: DType]: # skip
    fn __init__(inout self):
        pass

fn test():
    SomeStruct[DType.bool]()
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    items = fail_if_none(
        await requests.signature_help(
            doc, doc.find_last_range("SomeStruct[").end
        )
    )
    assert len(items.signatures) == 1
    assert items.active_signature == 0
    assert items.active_parameter == 0
    assert items.signatures[0].label == "struct SomeStruct[dtype: DType]"
