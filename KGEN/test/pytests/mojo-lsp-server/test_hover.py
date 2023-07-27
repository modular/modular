# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import os

import pytest_lsp
from lib.utils import Document, Requests, fail_if_none
from lsprotocol.types import HoverParams, MarkupContent, Position, Range
from pytest_lsp import ClientServerConfig, LanguageClient


@pytest_lsp.fixture(
    config=ClientServerConfig(
        server_command=[os.environ["MOJO_LSP_SERVER"]],
    ),
)
async def client(lsp_client: LanguageClient):
    # Setup
    await Requests(lsp_client).initialize()
    yield
    # Teardown
    await lsp_client.shutdown_session()


async def test_hover_letvar(client: LanguageClient):
    doc = Document(
        "foo.mojo",
        """
from IO import print

fn function():
  let foo: Int = 420
  var bar = 1 + foo
  print(bar)
""",
    )
    requests = Requests(client)
    requests.open_document(doc)

    result = fail_if_none(
        await requests.hover(doc, Position(line=5, character=17))
    )
    assert isinstance(result.contents, MarkupContent)
    assert (
        result.contents.value
        == """### variable `foo`

---

###
```mojo
let foo: Int
```"""
    )
    assert result.range == Range(
        start=Position(line=4, character=6), end=Position(line=4, character=9)
    )

    result = await client.text_document_hover_async(
        params=HoverParams(
            position=Position(line=6, character=8),
            text_document=doc.identifier,
        )
    )
    assert result
    assert isinstance(result.contents, MarkupContent)
    assert (
        result.contents.value
        == """### variable `bar`

---

###
```mojo
var bar: Int
```"""
    )
    assert result.range == Range(
        start=Position(line=5, character=6), end=Position(line=5, character=9)
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
        """### function `__init__`

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


---

###
```mojo
fn __init__(inout self: Self, borrowed_input: Int, init_arg: Int, owned owned_input: Int, *init_kargs: Int)
```""",
    )

    await assert_decl(
        "static_method",
        """### function `static_method`

---

###
```mojo
fn static_method() -> Int
```""",
    )

    await assert_decl(
        "non_capturing_nested_function",
        """### function `non_capturing_nested_function`

---

###
```mojo
fn non_capturing_nested_function()
```""",
    )

    await assert_decl(
        "async_function",
        """### function `async_function`

---

###
```mojo
async fn async_function(inout self: Self)
```""",
    )

    await assert_decl(
        "parameter_nested_function",
        """### function `parameter_nested_function`

---

###
```mojo
fn parameter_nested_function()
```""",
    )

    await assert_decl(
        "another_nested_function",
        """### function `another_nested_function`

---

###
```mojo
fn another_nested_function()
```""",
    )

    await assert_decl(
        "function_that_raises",
        """### function `function_that_raises`

---

###
A function that raises.

#### Args:
&nbsp;&nbsp;arg_in_function_that_raises: An arg in a function with by-ref result.


---

###
```mojo
fn function_that_raises(inout self: Self, arg_in_function_that_raises: Int) raises -> String
```""",
    )

    await assert_decl(
        "exported_function",
        """### function `exported_function`

---

###
This is an exported function.


---

###
```mojo
fn exported_function()
```""",
    )

    await assert_decl(
        "def_function",
        """### function `def_function`

---

###
```mojo
def def_function() raises -> Int
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
        """### struct `SomeStruct`

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


---

###
```mojo
struct SomeStruct[size: Int, other_param: Bool]
```""",
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
        """### alias `IntAlias`

---

###
Int alias summary

Int alias description.


---

###
```mojo
alias IntAlias = 12
```""",
    )

    await assert_decl(
        "ExplicitIntAlias",
        """### alias `ExplicitIntAlias`

---

###
```mojo
alias ExplicitIntAlias = 123
```""",
    )

    await assert_decl(
        "AliasInsideFunction",
        """### alias `AliasInsideFunction`

---

###
```mojo
alias AliasInsideFunction = "sdfsdf"
```""",
    )

    await assert_decl(
        "AliasToAlias",
        """### alias `AliasToAlias`

---

###
```mojo
alias AliasToAlias = 12
```""",
    )

    await assert_decl(
        "AliasInStruct",
        """### alias `AliasInStruct`

---

###
```mojo
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
        """### field `a_field`

---

###
Summary of a_field.


---

###
```mojo
var a_field: Int
```""",
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
        """### argument `self`

---

###
```mojo
inout self: Self
```""",
    )

    await assert_decl(
        "borrowed_input",
        """### argument `borrowed_input`

---

###
A borrowed argument.


---

###
```mojo
borrowed_input: Int
```""",
    )

    await assert_decl(
        "init_arg",
        """### argument `init_arg`

---

###
An Int argument.


---

###
```mojo
init_arg: Int
```""",
    )

    await assert_decl(
        "init_kargs",
        """### argument `init_kargs`

---

###
Multiple arguments.


---

###
```mojo
*init_kargs: Int
```""",
    )

    # We currently can't recover an owned argument from its decl, so we just print its name.
    await assert_decl("owned_input", "### `owned_input`")

    await assert_decl(
        "arg_in_function_that_raises",
        """### argument `arg_in_function_that_raises`

---

###
An arg in a function with by-ref result.


---

###
```mojo
arg_in_function_that_raises: Int
```""",
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
        "let_global_variable",
        """### variable `let_global_variable`

---

###
```mojo
let let_global_variable: Int
```""",
    )

    await assert_decl(
        "var_global_variable",
        """### variable `var_global_variable`

---

###
```mojo
var var_global_variable: Int
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
        "String",
        """### module `String`

---

###
Implements basic object methods for working with strings.

""",
    )

    await assert_import(
        "SIMD",
        """### module `SIMD`

---

###
Implements SIMD struct.

""",
    )
