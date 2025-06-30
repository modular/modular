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

TEST(RegressionTest, moto983) {
  // The original issue was caused by a misinterpretation of the LSP's encoding
  // of column offsets, where we interpreted them as UTF-8 offsets instead of
  // UTF-16 code unit offsets as required by the specification. This caused the
  // server's internal view to diverge from reality as it incorrectly sliced
  // multi-byte code points.

  Document doc("test:///foo.mojo", R"(
fn main():
    var str = "Hello 🔥"

    print(str)

    )");

  createTestClient()
      .open(doc)
      .update(doc, mlir::lsp::Range{{0, 24}, {1, 0}}, "")
      .onDiagnostics(doc, [](auto diags) { EXPECT_TRUE(diags.empty()); })
      .execute();
}
