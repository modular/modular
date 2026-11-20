//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "GraphCompiler/MOGGDialect/MOGGAttrs.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/Support/SymbolNames.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace M::MOGG;

namespace {

TEST(SymbolNamesTest, JoinsPartsThatFit) {
  EXPECT_EQ(
      joinNameParts({"matmul", "gelu", "mul"}, TruncationMarker::DroppedCount),
      "matmul_gelu_mul");
}

TEST(SymbolNamesTest, ReplacesInvalidCharacters) {
  EXPECT_EQ(joinNameParts({"reduce.rms_norm", "a-b"}, TruncationMarker::Hash),
            "reduce_rms_norm_a_b");
  EXPECT_EQ(joinNameParts({"2d_conv"}, TruncationMarker::Hash), "_2d_conv");
  EXPECT_EQ(joinNameParts({"shapeof(range)"}, TruncationMarker::Hash),
            "shapeof_range");
}

TEST(SymbolNamesTest, CountsDroppedParts) {
  // 32 characters fit "matmul_gelu_mul_add_sub_relu"; "cast" and "exp" don't.
  EXPECT_EQ(joinNameParts(
                {"matmul", "gelu", "mul", "add", "sub", "relu", "cast", "exp"},
                TruncationMarker::DroppedCount),
            "matmul_gelu_mul_add_sub_relu_and2");
}

TEST(SymbolNamesTest, HashesDroppedParts) {
  std::string a = joinNameParts(
      {"matmul", "gelu", "mul", "add", "sub", "relu", "cast", "exp"},
      TruncationMarker::Hash);
  std::string b = joinNameParts(
      {"matmul", "gelu", "mul", "add", "sub", "relu", "cast", "log"},
      TruncationMarker::Hash);
  // Same kept prefix, then `_` and 16 hex digits that differ.
  EXPECT_EQ(a.substr(0, 29), "matmul_gelu_mul_add_sub_relu_");
  EXPECT_EQ(a.size(), 29u + 16u);
  EXPECT_NE(a, b);
}

TEST(SymbolNamesTest, NeverCutsAPart) {
  std::string longPart(40, 'x');
  EXPECT_EQ(joinNameParts({longPart, "add"}, TruncationMarker::DroppedCount),
            longPart + "_and1");
}

TEST(SymbolNamesTest, LeafNamesSkipGroupsAndDialects) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.loadDialect<MOGGDialect>();
  auto leaf = [&](llvm::StringRef name) -> mlir::Attribute {
    return FusionLeafAttr::get(&ctx, mlir::StringAttr::get(&ctx, name),
                               mlir::UnknownLoc::get(&ctx), mlir::TypeAttr{});
  };
  auto elementwise =
      FusionElementwiseAttr::get(&ctx, {leaf("mo.gelu"), leaf("mo.mul")});
  auto epilogue = FusionEpilogueAttr::get(
      &ctx, {leaf("mo.matmul"), elementwise}, /*operandIndices=*/{0});
  auto fusionInfo = FusionInfoAttr::get(&ctx, epilogue);

  EXPECT_EQ(fusionInfo.getLeafNames(),
            (llvm::SmallVector<std::string>{"matmul", "gelu", "mul"}));

  auto shapeOf = FusionInfoAttr::get(&ctx, leaf("shapeof(mo.range)"));
  EXPECT_EQ(shapeOf.getLeafNames(),
            (llvm::SmallVector<std::string>{"shapeof(range)"}));
}

} // namespace
