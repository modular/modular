//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;
TEST(InitFileTest, testInitModuleIsNotIndexed) {
  Document doc = createDocumentFromInputFileWithinPackage("__init__.mojo");

  createTestClient()
      .open(doc)
      .hoverNullable(doc, {0, 0},
                     [&](const std::optional<lsp::Hover> &hover) {
                       if (hover.has_value()) {
                         EXPECT_NE(hover->contents.value,
                                   "### module `__init__`\n");
                       }
                     })
      .execute();
}
