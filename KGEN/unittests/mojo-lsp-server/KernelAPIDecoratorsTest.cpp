//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support.h"
#include "gtest/gtest.h"

using namespace M;

TEST(KernelAPIDecoratorsTest, mutableDecoratorInvalidArgName) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice, foreach

@compiler.register("mutable", num_dps_outputs=0)
struct Mutable:
    @compiler.mutable("output")
    @staticmethod
    fn execute[
        type: DType,
    ](input: ManagedTensorSlice[type=type, rank=2]):
        x = input[0, 0]
        x += 1
        input[0, 0] = x
)");

  createTestClient()
      .open(doc)
      .onDiagnostics(doc,
                     [](const std::vector<lsp::Diagnostic> &diags) {
                       ASSERT_EQ((int)diags.size(), 1);
                       EXPECT_EQ(diags[0].message,
                                 "mutable decorator: 'output' does not name "
                                 "any of the arguments of Mutable::execute");
                     })
      .execute();
}

TEST(KernelAPIDecoratorsTest, enableFusionForInvalidArgName) {
  Document doc("test:///foo.mojo", R"(
import compiler_internal as compiler
from tensor import ManagedTensorSlice

@compiler.register("fusion")
struct Fusion:
    @compiler.enable_fusion_for("output")
    @staticmethod
    fn execute[
        synchronous: Bool,
        target: StringLiteral,
    ](z: ManagedTensorSlice, x: ManagedTensorSlice, y: ManagedTensorSlice):
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
