//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(SampleTest, testHoverAndDefinition) {
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
             [](const lsp::Hover &response) {
               EXPECT_EQ(response.range, lsp::Range({3, 16}, {3, 21}));
               EXPECT_EQ(response.contents.value, R"(```mojo
(variable) var foo: Int
```)");
             })
      .definition(doc, lsp::Position(3, 17),
                  [](const std::vector<lsp::Location> &response) {
                    EXPECT_EQ(response.size(), 1u);
                    EXPECT_EQ(response[0].range, lsp::Range({2, 6}, {2, 9}));
                  })
      .execute();
}
