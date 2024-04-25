//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(DiagnosticsTest, testDiagnosticsInvalidImport) {
  Document doc("test:///foo.mojo",
               R"(
from a.b.c import d
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 2);
                       EXPECT_EQ(diags[0].message,
                                 "unable to locate module 'a'");
                     })
      .execute();
}
