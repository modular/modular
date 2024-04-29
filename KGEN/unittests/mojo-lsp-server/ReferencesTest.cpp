//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(ReferencesTest, testFindVariableReferences) {
  Document doc("test:///foo.mojo", R"(
fn function(foo: Int):
    var bar: Int = foo + 420
    print(foo)
    print("foo")
)");

  createTestClient()
      .open(doc)
      .references(doc, lsp::Position(2, 20),
                  /*includeDeclaration=*/true,
                  [](const std::vector<lsp::Location> &references) {
                    ASSERT_EQ((int)references.size(), 3);
                    auto expected = {
                        lsp::Range({1, 12}, {1, 15}),
                        lsp::Range({2, 19}, {2, 22}),
                        lsp::Range({3, 10}, {3, 13}),
                    };
                    for (const lsp::Location &reference : references)
                      EXPECT_TRUE(
                          llvm::is_contained(expected, reference.range));
                  })
      .execute();
}
