//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(DefinitionTest, testModule) {
  Document doc("test:///foo.mojo", "");

  createTestClient()
      .open(doc)
      .definition(doc, lsp::Position(0, 0),
                  [](const std::vector<mlir::lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 0);
                  })
      .execute();
}

TEST(DefinitionTest, testTypes) {
  Document doc("test:///foo.mojo",
               R"(
from utils.index import StaticIntTuple # skip
from utils.static_tuple import StaticTuple # skip
import builtin
import builtin.dtype


fn functionWithNestedType(x: dtype.DType):
    var y: builtin.int.Int = 12
    pass


fn functionWithBuiltins(x: Bool) -> Bool:
    var copy: Bool = x
    return copy


fn functionWithParametrizedArgument(x: StaticIntTuple[2]) -> StaticIntTuple[2]:
    var copy: StaticIntTuple[2] = x
    return copy


fn parametrizedFunction[
    size: Int
](x: StaticTuple[Int, size]) -> StaticTuple[Int, size]:
    var copy: StaticTuple[Int, size] = x
    return copy


  )");

  createTestClient()
      .open(doc)
      .definition(doc, *doc.findLastPos("Bool"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                  })
      .hover(doc, *doc.findLastPos("Bool"),
             [&](const lsp::Hover &hover) {
               EXPECT_TRUE(StringRef(hover.contents.value).contains("struct"));
             })
      .definition(doc, *doc.findLastPos("StaticIntTuple"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                  })
      .hover(doc, *doc.findLastPos("StaticIntTuple"),
             [&](const lsp::Hover &hover) {
               EXPECT_TRUE(StringRef(hover.contents.value).contains("struct"));
             })
      .definition(doc, *doc.findLastPos("StaticTuple"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                  })
      .hover(doc, *doc.findLastPos("StaticTuple"),
             [&](const lsp::Hover &hover) {
               EXPECT_TRUE(StringRef(hover.contents.value).contains("struct"));
             })
      .definition(doc, *doc.findLastPos("DType"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                  })
      .hover(doc, *doc.findLastPos("DType"),
             [&](const lsp::Hover &hover) {
               EXPECT_TRUE(StringRef(hover.contents.value).contains("struct"));
             })
      .definition(doc, *doc.findLastPos("Int"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                  })
      .hover(doc, *doc.findLastPos("Int"),
             [&](const lsp::Hover &hover) {
               EXPECT_TRUE(StringRef(hover.contents.value).contains("struct"));
             })
      .execute();
}

TEST(DefinitionTest, testMultiLocation) {
  Document doc("test:///foo.mojo", R"(
fn print(x: StringRef):
    pass

fn print(x: Bool):
    pass

fn function[type: AnyTrivialRegType](arg: type):
    print(arg)
  )");

  lsp::Location strLocation, boolLocation;

  createTestClient()
      .open(doc)
      .definition(doc, *doc.findFirstPos("print(x: StringRef"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                    strLocation = locations[0];
                  })
      .definition(doc, *doc.findFirstPos("print(x: Bool"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                    boolLocation = locations[0];
                  })
      .definition(doc, *doc.findFirstPos("print(arg"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 2);
                    EXPECT_EQ(locations[0], strLocation);
                    EXPECT_EQ(locations[1], boolLocation);
                  })
      .execute();
}

TEST(DefinitionTest, testImplicitVariables) {
  Document doc("test:///foo.mojo", R"(
fn test() raises:
    with open("test", "r") as with_var:
        pass

    for for_var in range(5):
        pass

def def_test():
  def_value = 10
  )");

  lsp::Location strLocation, boolLocation;

  createTestClient()
      .open(doc)
      .definition(doc, *doc.findFirstPos("with_var"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                  })
      .definition(doc, *doc.findFirstPos("for_var"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                  })
      .definition(doc, *doc.findFirstPos("def_value"),
                  [&](const std::vector<lsp::Location> &locations) {
                    EXPECT_EQ((int)locations.size(), 1);
                  })
      .execute();
}
