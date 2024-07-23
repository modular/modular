//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(DocumentSymbolTest, testImportSelf) {
  Document doc("test:///foo.mojo", R"(
import .foo
  )");

  createTestClient()
      .open(doc)
      .documentSymbol(doc,
                      [](const std::vector<lsp::DocumentSymbol> &symbols) {
                        // Nothing at all! We just need to not crash.
                      })
      .execute();
}

TEST(DocumentSymbolTest, testDocumentSymbols) {
  Document doc("test:///foo.mojo",
               R"(
alias Value = 10

fn foo(a: UnsafePointer[Float32]) -> Float32:
  var variable = 15
  fn inner_fn():
    return
  fn inner_closure(arg: Int, arg2: __type_of(arg)) -> Float32:
    return Scalar.load(a, arg)
  return inner_fn(variable)

struct struct_name:
  fn struct_fn():
    return

  var field: Int

trait trait_name:
    fn trait_fn(self):
        ...
)");

  createTestClient()
      .open(doc)
      .documentSymbol(
          doc,
          [](const std::vector<lsp::DocumentSymbol> &symbols) {
            ASSERT_EQ((int)symbols.size(), 4);

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

            EXPECT_EQ(symbols[3].name, "trait_name");
            EXPECT_EQ(symbols[3].kind, lsp::SymbolKind::Interface);
            ASSERT_EQ((int)symbols[3].children.size(), 1);
            EXPECT_EQ(symbols[3].children[0].name, "trait_fn");
          })
      .execute();
}
