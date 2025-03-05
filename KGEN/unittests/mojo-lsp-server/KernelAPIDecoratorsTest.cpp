//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(KernelAPIDecoratorsTest, enableFusionForInvalidArgName) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice, OutputTensor, InputTensor

@compiler.register("fusion")
struct Fusion:
    @compiler.enable_fusion_for("output")
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: OutputTensor, x: InputTensor, y: InputTensor):
        ...
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(
                           diags[0].message,
                           "enable_fusion_for decorator: 'output' does not "
                           "name any of the arguments of Fusion::execute");
                     })
      .execute();
}
