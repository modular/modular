//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(DocumentSymbolTest, testDocumentSymbols) {
  Document doc("test:///foo.mojo",
               R"(
alias Value = 10

fn foo(a: DTypePointer[DType.float32]) -> Float32:
  var variable = 15
  fn inner_fn():
    return
  fn inner_closure(arg: Int, arg2: __type_of(arg)) -> Float32:
    return a.load(arg)
  return inner_fn(variable)

struct struct_name:
  fn struct_fn():
    return

  var field: Int
)");

  createTestClient()
      .open(doc)
      .documentSymbol(
          doc,
          [](const std::vector<lsp::DocumentSymbol> &symbols) {
            ASSERT_EQ((int)symbols.size(), 3);

            EXPECT_EQ(symbols[0].name, "Value");
            EXPECT_EQ(symbols[0].kind, lsp::SymbolKind::Property);
            EXPECT_EQ(symbols[0].detail, "10");

            EXPECT_EQ(symbols[1].name, "foo");
            EXPECT_EQ(symbols[1].kind, lsp::SymbolKind::Function);
            EXPECT_TRUE(StringRef(symbols[1].detail).starts_with("foo("));
            ASSERT_EQ((int)symbols[1].children.size(), 1);
            EXPECT_EQ(symbols[1].children[0].name, "inner_fn");

            EXPECT_EQ(symbols[2].name, "struct_name");
            EXPECT_EQ(symbols[2].kind, lsp::SymbolKind::Struct);
            ASSERT_EQ((int)symbols[2].children.size(), 2);
            EXPECT_EQ(symbols[2].children[0].name, "struct_fn");
            EXPECT_EQ(symbols[2].children[1].name, "field");
            EXPECT_EQ(symbols[2].children[1].kind, lsp::SymbolKind::Field);
            EXPECT_EQ(symbols[2].children[1].detail, "Int");
          })
      .execute();
}
