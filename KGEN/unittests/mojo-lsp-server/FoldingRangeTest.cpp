//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(FoldingRangeTest, testDocStringFoldingRange) {
  Document doc("test:///foo.mojo",
               R"(
fn single_line():
  """This is a single line doc string."""
  return

fn multi_line():
  """This is a multi-line doc string.

  It has multiple lines.

  """
)");

  createTestClient()
      .open(doc)
      .foldingRange(
          doc,
          [](const std::vector<lsp::FoldingRange> &ranges) {
            ASSERT_TRUE(!ranges.empty());

            EXPECT_TRUE(
                llvm::any_of(ranges, [](const lsp::FoldingRange &range) {
                  return range.startLine == 2 && range.startCharacter == 5 &&
                         range.endLine == 2 && range.endCharacter == 38 &&
                         range.kind == "comment";
                }));

            EXPECT_TRUE(
                llvm::any_of(ranges, [](const lsp::FoldingRange &range) {
                  return range.startLine == 6 && range.startCharacter == 5 &&
                         range.endLine == 10 && range.endCharacter == 2 &&
                         range.kind == "comment";
                }));
          })
      .execute();
}
