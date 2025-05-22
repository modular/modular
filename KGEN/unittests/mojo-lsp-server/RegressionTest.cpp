//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(RegressionTest, moto1041) {
  // The original issue was caused by the parser emitting a hidden symbol that,
  // when combined with the language server's range calculations, created a
  // range that slightly exceeded the bounds of the document. This caused us to
  // crash because the range was not contained within the document, which meant
  // SourceMgr lookups failed.

  // The whitespace in this source snippet is deliberate and required to
  // reproduce the original crash.
  Document doc("test:///foo.mojo",
               // clang-format off
               R"(
fn main() raises:
  pass)");
  // clang-format on

  // Simply not crashing is sufficient.
  createTestClient()
      .open(doc)
      .semanticTokensFull(doc, [](ArrayRef<Mojo::LSP::SemanticToken>) {})
      .execute();
}
