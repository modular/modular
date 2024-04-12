# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from lib.utils import Document, Requests, fail_if_none, mojo_lsp_client
from lsprotocol.types import HoverParams, MarkupContent, Position, Range
from pytest_lsp import LanguageClient

# This variable is required by the test runner.
# pyright: reportUnknownVariableType=false
client = mojo_lsp_client


async def assert_doc_hover(
    doc: Document, requests: Requests, text: str, expected: str
):
    range = fail_if_none(doc.find_first_range(text))
    result = fail_if_none(await requests.hover(doc, range.start))
    assert isinstance(result.contents, MarkupContent)
    assert result.contents.value == expected


async def test_hover_letvar(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn function():
  var foo: Int = 420
  var bar = 1 + `foo`
  print(bar)
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    result = fail_if_none(
        await requests.hover(doc, Position(line=3, character=17))
    )
    assert isinstance(result.contents, MarkupContent)
    assert (
        result.contents.value
        == """```mojo
(variable) var foo: Int
```"""
    )
    assert result.range == Range(
        start=Position(line=3, character=16), end=Position(line=3, character=21)
    )

    result = await client.text_document_hover_async(
        params=HoverParams(
            position=Position(line=4, character=8),
            text_document=doc.identifier,
        )
    )
    assert result
    assert isinstance(result.contents, MarkupContent)
    assert (
        result.contents.value
        == """```mojo
(variable) var bar: Int
```"""
    )
    assert result.range == Range(
        start=Position(line=4, character=8), end=Position(line=4, character=11)
    )


async def test_hover_function_decls(client: LanguageClient):
    doc = Document.from_file("functions.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    async def assert_decl(func_name: str, expected: str):
        range = fail_if_none(doc.find_first_range(func_name))
        result = fail_if_none(await requests.hover(doc, range.start))
        assert result.range == range
        assert isinstance(result.contents, MarkupContent)
        assert result.contents.value == expected

    await assert_decl(
        "__init__",
        """```mojo
(function) fn __init__(inout self: Self, borrowed_input: Int, init_arg: Int, owned owned_input: Int, *init_kargs: Int)
```
---

###
Init documentation.

#### Args:
&nbsp;&nbsp;borrowed_input: A borrowed argument.
\\
&nbsp;&nbsp;init_arg: An Int argument.
\\
&nbsp;&nbsp;owned_input: An owned argument.
\\
&nbsp;&nbsp;init_kargs: Multiple arguments.

""",
    )

    await assert_decl(
        "static_method",
        """```mojo
(function) fn static_method() -> Int
```""",
    )

    await assert_decl(
        "non_capturing_nested_function",
        """```mojo
(function) fn non_capturing_nested_function()
```""",
    )

    await assert_decl(
        "async_function",
        """```mojo
(function) async fn async_function(inout self: Self)
```""",
    )

    await assert_decl(
        "parameter_nested_function",
        """```mojo
(function) fn parameter_nested_function()
```""",
    )

    await assert_decl(
        "another_nested_function",
        """```mojo
(function) fn another_nested_function()
```""",
    )

    await assert_decl(
        "function_that_raises",
        """```mojo
(function) fn function_that_raises(inout self: Self, arg_in_function_that_raises: Int) raises -> String
```
---

###
A function that raises.

#### Args:
&nbsp;&nbsp;arg_in_function_that_raises: An arg in a function with by-ref result.

""",
    )

    await assert_decl(
        "function_with_param",
        """```mojo
(function) fn function_with_param[Param1: Int, Param2: Int](inout self: Self)
```
---

###
A function with param.

#### Parameters:
&nbsp;&nbsp;Param1: An Int param.
\\
&nbsp;&nbsp;Param2: Another Int param.

""",
    )

    await assert_decl(
        "exported_function",
        """```mojo
(function) fn exported_function()
```
---

###
This is an exported function.

""",
    )

    await assert_decl(
        "def_function",
        """```mojo
(function) def def_function() raises -> Int
```""",
    )


async def test_hover_struct_decls(client: LanguageClient):
    doc = Document.from_file("functions.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    async def assert_decl(func_name: str, expected: str):
        range = fail_if_none(doc.find_first_range(func_name))
        result = fail_if_none(await requests.hover(doc, range.start))
        assert result.range == range
        assert isinstance(result.contents, MarkupContent)
        assert result.contents.value == expected

    await assert_decl(
        "SomeStruct",
        """```mojo
struct SomeStruct[size: Int, other_param: Bool]
```
---

###
Docstring for SomeStruct.

More docstring for SomeStruct.


#### Parameters:
&nbsp;&nbsp;size: The size of SomeStruct.
\\
&nbsp;&nbsp;other_param: Another param.

#### Constraints:
&nbsp;&nbsp;The contraints of SomeStruct.

""",
    )


async def test_hover_alias_decls(client: LanguageClient):
    doc = Document.from_file("aliases.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    async def assert_decl(func_name: str, contents: str):
        range = fail_if_none(doc.find_first_range(func_name))
        result = fail_if_none(await requests.hover(doc, range.start))
        assert result.range == range
        assert isinstance(result.contents, MarkupContent)
        assert contents == result.contents.value

    await assert_decl(
        "IntAlias",
        """```mojo
alias IntAlias = 12
```
---

###
Int alias summary

Int alias description.

""",
    )

    await assert_decl(
        "ExplicitIntAlias",
        """```mojo
alias ExplicitIntAlias = 123
```""",
    )

    await assert_decl(
        "AliasInsideFunction",
        """```mojo
alias AliasInsideFunction = "sdfsdf"
```""",
    )

    await assert_decl(
        "AliasToAlias",
        """```mojo
alias AliasToAlias = 12
```""",
    )

    await assert_decl(
        "AliasInStruct",
        """```mojo
alias AliasInStruct = Int
```""",
    )


async def test_hover_struct_field_decls(client: LanguageClient):
    doc = Document.from_file("struct_fields.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    async def assert_decl(func_name: str, expected: str):
        range = fail_if_none(doc.find_first_range(func_name))
        result = fail_if_none(await requests.hover(doc, range.start))
        assert result.range == range
        assert isinstance(result.contents, MarkupContent)
        assert result.contents.value == expected

    await assert_decl(
        "a_field",
        """```mojo
(field) var a_field: Int
```
---

###
Summary of a_field.

""",
    )


async def test_hover_argument(client: LanguageClient):
    doc = Document.from_file("functions.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    async def assert_decl(func_name: str, expected: str):
        range = fail_if_none(doc.find_first_range(func_name))
        result = fail_if_none(await requests.hover(doc, range.start))
        assert isinstance(result.contents, MarkupContent)
        assert result.contents.value == expected

    await assert_decl(
        "self",
        """```mojo
(argument) inout self: Self
```""",
    )

    await assert_decl(
        "borrowed_input",
        """```mojo
(argument) borrowed_input: Int
```
---

###
A borrowed argument.

""",
    )

    await assert_decl(
        "init_arg",
        """```mojo
(argument) init_arg: Int
```
---

###
An Int argument.

""",
    )

    await assert_decl(
        "init_kargs",
        """```mojo
(variable) var init_kargs: VariadicList[Int]
```""",
    )

    # We currently can't recover an owned argument from its decl, so we just print its name.
    await assert_decl(
        "owned_input",
        """```mojo
(variable) var owned_input: Int
```""",
    )

    await assert_decl(
        "arg_in_function_that_raises",
        """```mojo
(argument) arg_in_function_that_raises: Int
```
---

###
An arg in a function with by-ref result.

""",
    )

    await assert_decl(
        "Param1",
        """```mojo
(parameter) Param1: Int
```
---

###
An Int param.

""",
    )

    await assert_decl(
        "Param2",
        """```mojo
(parameter) Param2: Int
```
---

###
Another Int param.

""",
    )


async def test_hover_global_variables(client: LanguageClient):
    doc = Document.from_file("global_variables.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    async def assert_decl(func_name: str, expected: str):
        range = fail_if_none(doc.find_first_range(func_name))
        result = fail_if_none(await requests.hover(doc, range.start))
        assert isinstance(result.contents, MarkupContent)
        assert result.contents.value == expected

    await assert_decl(
        "var_global_variable",
        """```mojo
(variable) var var_global_variable: Int
```""",
    )


async def test_hover_import(client: LanguageClient):
    doc = Document.from_file("imports.mojo")

    requests = Requests(client)
    requests.open_document(doc)

    async def assert_import(func_name: str, expected: str):
        range = fail_if_none(doc.find_first_range(func_name))
        result = fail_if_none(await requests.hover(doc, range.start))
        assert isinstance(result.contents, MarkupContent)
        assert result.contents.value == expected

    await assert_import(
        "builtin",
        """### package `builtin`

---

###
Implements the builtin package.

""",
    )

    await assert_import(
        "string",
        """### module `string`

---

###
Implements basic object methods for working with strings.

These are Mojo built-ins, so you don't need to import them.

""",
    )

    simd_doc = """### module `simd`

---

###
Implements SIMD struct.

These are Mojo built-ins, so you don't need to import them.

"""
    await assert_import("simd", simd_doc)
    await assert_import("_simd", simd_doc)

    await assert_import(
        "aliases",
        """### module `aliases`
""",
    )

    await assert_import(
        "function",
        """```mojo
(function) fn function() -> Int
```""",
    )

    await assert_import(
        "StructWithAlias",
        """```mojo
struct StructWithAlias
```""",
    )


async def test_hover_external_symbol(client: LanguageClient):
    requests = Requests(client)

    doc = Document.from_file("aliases.mojo")
    requests.open_document(doc)

    async def assert_hover(text: str, expected: str):
        assert_doc_hover(doc, requests, text, expected)

    await assert_hover(
        "LAZY",
        """```mojo
alias LAZY = 1
```
---

###
Load library lazily (defer function resolution until needed).

""",
    )

    await assert_hover(
        "ExternalAlias",
        """```mojo
alias ExternalAlias = 1
```""",
    )


async def test_function_call(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn print(x: StringLiteral):
    pass

fn print(x: Bool):
    pass

fn function[type: AnyRegType](arg: type):
    print("string")
    print(arg)
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    async def assert_hover(text: str, expected: str):
        assert_doc_hover(doc, requests, text, expected)

    await assert_hover(
        'print("',
        """```mojo
(function) fn print(x: StringLiteral)
```""",
    )

    await assert_hover(
        "print(arg",
        """```mojo
(function) fn print(x: StringLiteral)
```
---

```mojo
(function) fn print(x: Bool)
```""",
    )


async def test_hover(client: LanguageClient):
    doc = Document.from_file("traits.mojo")
    requests = Requests(client)
    requests.open_document(doc)

    async def assert_hover(text: str, expected: str):
        assert_doc_hover(doc, requests, text, expected)

    await assert_hover(
        "ATrait:",
        """```mojo
(trait) trait ATrait
```
---

###
Some documentation.

""",
    )

    await assert_hover(
        "ATrait):",
        """```mojo
(trait) trait ATrait
```
---

###
Some documentation.

""",
    )


async def test_function_types(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
def function[
    func: fn (Int) capturing -> Int
]() -> fn (Int) capturing -> Int:
    pass
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    assert_doc_hover(
        doc,
        requests,
        "function",
        """```mojo
(function) def function[func: fn(Int, /) capturing -> Int]() raises -> fn(Int, /) capturing -> Int
```""",
    )


async def test_named_function_types(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
fn fn1[f: fn [p1: DType](foo: Scalar[p1]) -> __type_of(foo)]():
  ...


fn fn2[f: fn [dt: DType, dt2: Int](arg1: Scalar[dt], arg2: Int) -> None]():
  ...
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    async def assert_hover(text: str, expected: str):
        assert_doc_hover(doc, requests, text, expected)

    assert_hover(
        "p1",
        """```mojo
(parameter) p1: DType
```""",
    )
    assert_hover(
        "foo",
        """```mojo
(argument) foo: SIMD[$0, 1]
```""",
    )
    assert_hover(
        "arg2",
        """```mojo
(argument) arg2: Int
```""",
    )
