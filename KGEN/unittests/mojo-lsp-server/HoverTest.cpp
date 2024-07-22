//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(HoverTest, testHoverVar) {
  Document doc("test:///foo.mojo",
               R"(
fn function():
  var foo: Int = 420
  var bar = 1 + `foo`
  print(bar)
)");

  createTestClient()
      .open(doc)
      .hover(doc, lsp::Position(3, 17),
             [](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(variable) var foo: Int
```)");

               EXPECT_EQ(hover.range, lsp::Range(lsp::Position(3, 16),
                                                 lsp::Position(3, 21)));
             })
      .hover(doc, lsp::Position(4, 8),
             [](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(variable) var bar: Int
```)");

               EXPECT_EQ(hover.range,
                         lsp::Range(lsp::Position(4, 8), lsp::Position(4, 11)));
             })
      .execute();
}

TEST(HoverTest, testHoverFunctionDecls) {
  Document doc = createDocumentFromInputFile("functions.mojo");

  lsp::Range rangeInit = *doc.findFirstRange("__init__");
  lsp::Range rangeStaticMethod = *doc.findFirstRange("static_method");
  lsp::Range rangeNonCapturingNestedFunction =
      *doc.findFirstRange("non_capturing_nested_function");
  lsp::Range rangeAsyncFunction = *doc.findFirstRange("async_function");
  lsp::Range rangeParameterNestedFunction =
      *doc.findFirstRange("parameter_nested_function");
  lsp::Range rangeAnotherNestedFunction =
      *doc.findFirstRange("another_nested_function");
  lsp::Range rangeFunctionThatRaises =
      *doc.findFirstRange("function_that_raises");
  lsp::Range rangeFunctionWithParam =
      *doc.findFirstRange("function_with_param");
  lsp::Range rangeExportedFunction = *doc.findFirstRange("exported_function");
  lsp::Range rangeDefFunction = *doc.findFirstRange("def_function");

  createTestClient()
      .open(doc)
      .hover(doc, rangeInit.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeInit);
               EXPECT_EQ(hover.contents.value,
                         R"(```mojo
(function) fn __init__(inout self: Self, borrowed_input: Int, init_arg: Int, owned owned_input: Int, *init_kargs: Int)
```
---

###
Init documentation.

#### Args:
&nbsp;&nbsp;borrowed_input: A borrowed argument.
\
&nbsp;&nbsp;init_arg: An Int argument.
\
&nbsp;&nbsp;owned_input: An owned argument.
\
&nbsp;&nbsp;init_kargs: Multiple arguments.

)");
             })
      .hover(doc, rangeStaticMethod.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeStaticMethod);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn static_method() -> Int
```)");
             })
      .hover(doc, rangeNonCapturingNestedFunction.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeNonCapturingNestedFunction);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn non_capturing_nested_function()
```)");
             })
      .hover(doc, rangeAsyncFunction.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeAsyncFunction);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) async fn async_function(inout self: Self)
```)");
             })
      .hover(doc, rangeParameterNestedFunction.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeParameterNestedFunction);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn parameter_nested_function()
```)");
             })
      .hover(doc, rangeAnotherNestedFunction.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeAnotherNestedFunction);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn another_nested_function()
```)");
             })
      .hover(doc, rangeFunctionThatRaises.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeFunctionThatRaises);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn function_that_raises(inout self: Self, arg_in_function_that_raises: Int) raises -> String
```
---

###
A function that raises.

#### Args:
&nbsp;&nbsp;arg_in_function_that_raises: An arg in a function with by-ref result.

)");
             })
      .hover(doc, rangeFunctionWithParam.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeFunctionWithParam);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn function_with_param[Param1: Int, Param2: Int](inout self: Self)
```
---

###
A function with param.

#### Parameters:
&nbsp;&nbsp;Param1: An Int param.
\
&nbsp;&nbsp;Param2: Another Int param.

)");
             })
      .hover(doc, rangeExportedFunction.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeExportedFunction);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn exported_function()
```
---

###
This is an exported function.

)");
             })
      .hover(doc, rangeDefFunction.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeDefFunction);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) def def_function() raises -> Int
```)");
             })
      .execute();
}

TEST(HoverTest, testHoverStructDecls) {
  Document doc = createDocumentFromInputFile("functions.mojo");

  lsp::Range rangeSomeStruct = *doc.findFirstRange("SomeStruct");

  createTestClient()
      .open(doc)
      .hover(doc, rangeSomeStruct.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeSomeStruct);
               EXPECT_EQ(hover.contents.value, R"(```mojo
struct SomeStruct[size: Int, other_param: Bool]
```
---

###
Docstring for SomeStruct.

More docstring for SomeStruct.


#### Parameters:
&nbsp;&nbsp;size: The size of SomeStruct.
\
&nbsp;&nbsp;other_param: Another param.

#### Constraints:
&nbsp;&nbsp;The contraints of SomeStruct.

)");
             })
      .execute();
}

TEST(HoverTest, testHoverAliasDecls) {
  Document doc = createDocumentFromInputFile("aliases.mojo");

  lsp::Range rangeIntAlias = *doc.findFirstRange("IntAlias");
  lsp::Range rangeExplicitIntAlias = *doc.findFirstRange("ExplicitIntAlias");
  lsp::Range rangeAliasInsideFunction =
      *doc.findFirstRange("AliasInsideFunction");
  lsp::Range rangeAliasToAlias = *doc.findFirstRange("AliasToAlias");
  lsp::Range rangeAliasInStruct = *doc.findFirstRange("AliasInStruct");

  createTestClient()
      .open(doc)
      .hover(doc, rangeIntAlias.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeIntAlias);
               EXPECT_EQ(hover.contents.value, R"(```mojo
alias IntAlias = 12
```
---

###
Int alias summary

Int alias description.

)");
             })
      .hover(doc, rangeExplicitIntAlias.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeExplicitIntAlias);
               EXPECT_EQ(hover.contents.value, R"(```mojo
alias ExplicitIntAlias = 123
```)");
             })
      .hover(doc, rangeAliasInsideFunction.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeAliasInsideFunction);
               EXPECT_EQ(hover.contents.value, R"(```mojo
alias AliasInsideFunction = "sdfsdf"
```)");
             })
      .hover(doc, rangeAliasToAlias.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeAliasToAlias);
               EXPECT_EQ(hover.contents.value, R"(```mojo
alias AliasToAlias = 12
```)");
             })
      .hover(doc, rangeAliasInStruct.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeAliasInStruct);
               EXPECT_EQ(hover.contents.value, R"(```mojo
alias AliasInStruct = Int
```)");
             })
      .execute();
}

TEST(HoverTest, testHoverStructFieldDecls) {
  Document doc("test:///foo.mojo", R"(
struct SomeStruct:
    var a_field: Int
    """Summary of a_field."""

    fn __init__(inout self):
        pass


fn main():
    var someStruct = SomeStruct()
    _ = someStruct.a_field
)");
  lsp::Range rangeAField = *doc.findFirstRange("a_field");

  createTestClient()
      .open(doc)
      .hover(doc, rangeAField.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeAField);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(field) var a_field: Int
```
---

###
Summary of a_field.

)");
             })
      .execute();
}

TEST(HoverTest, testHoverArgument) {
  Document doc = createDocumentFromInputFile("functions.mojo");

  lsp::Range rangeSelfField = *doc.findFirstRange("self");
  lsp::Range rangeBorrowedInput = *doc.findFirstRange("borrowed_input");
  lsp::Range rangeInitArg = *doc.findFirstRange("init_arg");
  lsp::Range rangeInitKargs = *doc.findFirstRange("init_kargs");
  lsp::Range rangeOwnedInput = *doc.findFirstRange("owned_input");
  lsp::Range rangeArgInRaises =
      *doc.findFirstRange("arg_in_function_that_raises");
  lsp::Range rangeParam1 = *doc.findFirstRange("Param1");
  lsp::Range rangeParam2 = *doc.findFirstRange("Param2");

  createTestClient()
      .open(doc)
      .hover(doc, rangeSelfField.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeSelfField);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) inout self: Self
```)");
             })
      .hover(doc, rangeBorrowedInput.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeBorrowedInput);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) borrowed_input: Int
```
---

###
A borrowed argument.

)");
             })
      .hover(doc, rangeInitArg.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeInitArg);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) init_arg: Int
```
---

###
An Int argument.

)");
             })
      .hover(doc, rangeInitKargs.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeInitKargs);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) *init_kargs: Int
```
---

###
Multiple arguments.

)");
             })
      .hover(doc, rangeOwnedInput.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeOwnedInput);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) owned owned_input: Int
```
---

###
An owned argument.

)");
             })
      .hover(doc, rangeArgInRaises.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeArgInRaises);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) arg_in_function_that_raises: Int
```
---

###
An arg in a function with by-ref result.

)");
             })
      .hover(doc, rangeParam1.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeParam1);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(parameter) Param1: Int
```
---

###
An Int param.

)");
             })
      .hover(doc, rangeParam2.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeParam2);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(parameter) Param2: Int
```
---

###
Another Int param.

)");
             })
      .execute();
}

TEST(HoverTest, testGlobalVariables) {
  Document doc("test:///foo.mojo", R"(
var var_global_variable: Int = 345


fn main():
    var sum = let_global_variable + var_global_variable
)");
  lsp::Range rangeGlobalVar = *doc.findFirstRange("var_global_variable");

  createTestClient()
      .open(doc)
      .hover(doc, rangeGlobalVar.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeGlobalVar);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(variable) var var_global_variable: Int
```)");
             })
      .execute();
}

TEST(HoverTest, testHoverImport) {
  Document doc = createDocumentFromInputFile("imports.mojo");

  lsp::Range rangeBuiltin = *doc.findFirstRange("builtin");
  lsp::Range rangeString = *doc.findFirstRange("string");
  lsp::Range rangeSimd = *doc.findFirstRange("simd");
  lsp::Range range_Simd = *doc.findFirstRange("_simd");
  lsp::Range rangeAliases = *doc.findFirstRange("aliases");
  lsp::Range rangeFunction = *doc.findFirstRange("function");
  lsp::Range rangeStructWithAlias = *doc.findFirstRange("StructWithAlias");

  auto simdDoc = R"(### module `simd`

---

###
Implements SIMD struct.

These are Mojo built-ins, so you don't need to import them.

)";

  createTestClient()
      .open(doc)
      .hover(doc, rangeBuiltin.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeBuiltin);
               EXPECT_EQ(hover.contents.value, R"(### package `builtin`

---

###
Implements the builtin package.

)");
             })
      .hover(doc, rangeString.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeString);
               EXPECT_EQ(hover.contents.value, R"(### module `string`

---

###
Implements basic object methods for working with strings.

These are Mojo built-ins, so you don't need to import them.

)");
             })
      .hover(doc, rangeSimd.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeSimd);
               EXPECT_EQ(hover.contents.value, simdDoc);
             })
      .hover(doc, range_Simd.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, range_Simd);
               EXPECT_EQ(hover.contents.value, simdDoc);
             })
      .hover(doc, rangeAliases.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeAliases);
               EXPECT_EQ(hover.contents.value, R"(### module `aliases`
)");
             })
      .hover(doc, rangeFunction.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeFunction);
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn function() -> Int
```)");
             })
      .hover(doc, rangeStructWithAlias.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeStructWithAlias);
               EXPECT_EQ(hover.contents.value, R"(```mojo
struct StructWithAlias
```)");
             })
      .execute();
}

TEST(HoverTest, testHoverExternalSymbol) {
  Document doc = createDocumentFromInputFile("aliases.mojo");

  lsp::Range rangeLazy = *doc.findFirstRange("LAZY");
  lsp::Range rangeExternalAlias = *doc.findFirstRange("ExternalAlias");

  createTestClient()
      .open(doc)
      .hover(doc, rangeLazy.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeLazy);
               EXPECT_EQ(hover.contents.value, R"(```mojo
alias LAZY = 1
```
---

###
Load library lazily (defer function resolution until needed).

)");
             })
      .hover(doc, rangeExternalAlias.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeExternalAlias);
               EXPECT_EQ(hover.contents.value, R"(```mojo
alias ExternalAlias = 1
```)");
             })
      .execute();
}

TEST(HoverTest, testFunctionCall) {
  Document doc("test:///foo.mojo",
               R"(
fn print(x: StringLiteral):
    pass

fn print(x: Bool):
    pass

fn function[type: AnyTrivialRegType](arg: type):
    print("string")
    print(arg)
  )");

  createTestClient()
      .open(doc)
      .hover(doc, *doc.findFirstPos("print("),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn print(x: StringLiteral)
```)");
             })
      .hover(doc, *doc.findFirstPos("print(arg"),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn print(x: StringLiteral)
```
---

```mojo
(function) fn print(x: Bool)
```)");
             })
      .execute();
}

TEST(HoverTest, testHover) {
  Document doc("test:///foo.mojo",
               R"(
trait ATrait:
    """Some documentation."""

    fn print(owned self, x: StringRef):
        pass


struct Foo(ATrait):
    fn __init__(inout self):
        pass

    fn print(owned self, x: StringRef):
        pass
  )");

  createTestClient()
      .open(doc)
      .hover(doc, *doc.findFirstPos("ATrait:"),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(trait) trait ATrait
```
---

###
Some documentation.

)");
             })
      .hover(doc, *doc.findFirstPos("ATrait):"),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(trait) trait ATrait
```
---

###
Some documentation.

)");
             })
      .execute();
}

TEST(HoverTest, testFunctionTypes) {
  Document doc("test:///foo.mojo",
               R"(
def function[
    func: fn (Int) capturing -> Int
]() -> fn (Int) capturing -> Int:
    pass
  )");

  createTestClient()
      .open(doc)
      .hover(doc, *doc.findFirstPos("function"),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) def function[func: fn(Int) capturing -> Int]() raises -> fn(Int) capturing -> Int
```)");
             })
      .execute();
}

TEST(HoverTest, testNamedFunctionTypes) {
  Document doc("test:///foo.mojo",
               R"(
fn fn1[f: fn [p1: DType](foo: Scalar[p1]) -> __type_of(foo)]():
  ...


fn fn2[f: fn [dt: DType, dt2: Int](arg1: Scalar[dt], arg2: Int) -> None]():
  ...
  )");

  createTestClient()
      .open(doc)
      .hover(doc, *doc.findFirstPos("p1"),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(parameter) p1: DType
```)");
             })
      .hover(doc, *doc.findFirstPos("foo"),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) foo: SIMD[$0, 1]
```)");
             })
      .hover(doc, *doc.findFirstPos("arg2"),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(argument) arg2: Int
```)");
             })
      .execute();
}

TEST(HoverTest, testStructFieldsHoverAndDef) {
  Document doc("test:///foo.mojo",
               R"(
struct SomeStruct:
    var a_field: Int
    """Summary of a_field."""

    fn __init__(inout self):
        pass


fn main():
    var someStruct = SomeStruct()
    _ = someStruct.a_field
  )");

  lsp::Range rangeAField = *doc.findFirstRange("a_field");

  createTestClient()
      .open(doc)
      .hover(
          doc, rangeAField.start,
          [&](const lsp::Hover &hover) { EXPECT_EQ(hover.range, rangeAField); })
      .definition(doc, rangeAField.start,
                  [&](const std::vector<lsp::Location> &locations) {
                    ASSERT_EQ((int)locations.size(), 1);
                    EXPECT_EQ(locations[0].range, rangeAField);
                  })
      .execute();
}

TEST(HoverTest, testStructAliasHoverAndDef) {
  Document doc = createDocumentFromInputFile("aliases.mojo");

  lsp::Range rangeAlias = *doc.findFirstRange("AliasInStruct");

  createTestClient()
      .open(doc)
      .hover(
          doc, rangeAlias.start,
          [&](const lsp::Hover &hover) { EXPECT_EQ(hover.range, rangeAlias); })
      .definition(doc, rangeAlias.start,
                  [&](const std::vector<lsp::Location> &locations) {
                    ASSERT_EQ((int)locations.size(), 1);
                    EXPECT_EQ(locations[0].range, rangeAlias);
                  })
      .execute();
}

TEST(HoverTest, testGlobalVariableHoverAndDef) {
  Document doc("test:///foo.mojo",
               R"(
var var_global_variable: Int = 345


fn main():
    var sum = let_global_variable + var_global_variable

  )");

  lsp::Range rangeGlobalVar = *doc.findFirstRange("var_global_variable");

  createTestClient()
      .open(doc)
      .hover(doc, rangeGlobalVar.start,
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.range, rangeGlobalVar);
             })
      .definition(doc, rangeGlobalVar.start,
                  [&](const std::vector<lsp::Location> &locations) {
                    ASSERT_EQ((int)locations.size(), 1);
                    EXPECT_EQ(locations[0].range, rangeGlobalVar);
                  })
      .execute();
}

TEST(HoverTest, testInferredParameters) {
  Document doc("test:///foo.mojo",
               R"(
@always_inline
fn parametric[
    type: DType, simd_width: Int, //, other: Int
](x: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    return x * x


fn foo():
    var v = SIMD[DType.float16, 4](33)
    _ = parametric[12](v)
  )");

  createTestClient()
      .open(doc)
      .hover(doc, *doc.findLastPos("parametric"),
             [&](const lsp::Hover &hover) {
               EXPECT_EQ(hover.contents.value, R"(```mojo
(function) fn parametric[type: DType, simd_width: Int, //, other: Int](x: SIMD[type, simd_width]) -> SIMD[$0, $1]
```)");
             })
      .execute();
}
