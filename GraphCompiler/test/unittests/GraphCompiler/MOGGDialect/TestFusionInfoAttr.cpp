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
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

using namespace M::MOGG;

namespace {

// Test that toString preserves dialect prefixes by default.
TEST(FusionInfoAttrTest, ToStringPreservesDialectPrefix) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.loadDialect<MOGGDialect>();

  auto opName = mlir::StringAttr::get(&ctx, "mo.add");
  auto loc = mlir::UnknownLoc::get(&ctx);
  auto leaf = FusionLeafAttr::get(&ctx, opName, loc, mlir::TypeAttr{});

  auto fusionInfo = FusionInfoAttr::get(&ctx, leaf);

  // Default: no stripping.
  EXPECT_EQ(fusionInfo.toString(), "mo.add");
}

// Test that toString strips dialect prefixes when requested.
TEST(FusionInfoAttrTest, ToStringStripsDialectPrefix) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.loadDialect<MOGGDialect>();

  auto opName = mlir::StringAttr::get(&ctx, "mo.add");
  auto loc = mlir::UnknownLoc::get(&ctx);
  auto leaf = FusionLeafAttr::get(&ctx, opName, loc, mlir::TypeAttr{});

  auto fusionInfo = FusionInfoAttr::get(&ctx, leaf);

  std::string result = fusionInfo.toString(/*includeLineNumbers=*/false,
                                           /*quote=*/false,
                                           /*stripDialectPrefixes=*/true);
  EXPECT_EQ(result, "add");
}

// Test that multi-part names only strip the dialect prefix.
TEST(FusionInfoAttrTest, ToStringStripsOnlyDialectPrefix) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.loadDialect<MOGGDialect>();

  auto opName = mlir::StringAttr::get(&ctx, "mo.distributed.allreduce.sum");
  auto loc = mlir::UnknownLoc::get(&ctx);
  auto leaf = FusionLeafAttr::get(&ctx, opName, loc, mlir::TypeAttr{});

  auto fusionInfo = FusionInfoAttr::get(&ctx, leaf);

  std::string result = fusionInfo.toString(/*includeLineNumbers=*/false,
                                           /*quote=*/false,
                                           /*stripDialectPrefixes=*/true);
  EXPECT_EQ(result, "distributed.allreduce.sum");
}

// Test that other prefixes are also stripped.
TEST(FusionInfoAttrTest, ToStringStripsEpPrefix) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.loadDialect<MOGGDialect>();

  auto opName = mlir::StringAttr::get(&ctx, "ep.dispatch");
  auto loc = mlir::UnknownLoc::get(&ctx);
  auto leaf = FusionLeafAttr::get(&ctx, opName, loc, mlir::TypeAttr{});

  auto fusionInfo = FusionInfoAttr::get(&ctx, leaf);

  std::string result = fusionInfo.toString(/*includeLineNumbers=*/false,
                                           /*quote=*/false,
                                           /*stripDialectPrefixes=*/true);
  EXPECT_EQ(result, "dispatch");
}

// Test that names without a prefix are unchanged.
TEST(FusionInfoAttrTest, ToStringPreservesNamesWithoutPrefix) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.loadDialect<MOGGDialect>();

  auto opName = mlir::StringAttr::get(&ctx, "custom_op");
  auto loc = mlir::UnknownLoc::get(&ctx);
  auto leaf = FusionLeafAttr::get(&ctx, opName, loc, mlir::TypeAttr{});

  auto fusionInfo = FusionInfoAttr::get(&ctx, leaf);

  std::string result = fusionInfo.toString(/*includeLineNumbers=*/false,
                                           /*quote=*/false,
                                           /*stripDialectPrefixes=*/true);
  EXPECT_EQ(result, "custom_op");
}

// Test nested fusion info (sequences) with stripping.
TEST(FusionInfoAttrTest, ToStringStripsPrefixInSequences) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.loadDialect<MOGGDialect>();

  auto loc = mlir::UnknownLoc::get(&ctx);
  auto leaf1 = FusionLeafAttr::get(&ctx, mlir::StringAttr::get(&ctx, "mo.add"),
                                   loc, mlir::TypeAttr{});
  auto leaf2 = FusionLeafAttr::get(&ctx, mlir::StringAttr::get(&ctx, "mo.mul"),
                                   loc, mlir::TypeAttr{});

  auto sequence = FusionSequenceAttr::get(&ctx, {leaf1, leaf2});
  auto fusionInfo = FusionInfoAttr::get(&ctx, sequence);

  std::string result = fusionInfo.toString(/*includeLineNumbers=*/false,
                                           /*quote=*/false,
                                           /*stripDialectPrefixes=*/true);
  EXPECT_EQ(result, "Sequence(add, mul)");

  // Without stripping, prefixes are preserved.
  EXPECT_EQ(fusionInfo.toString(), "Sequence(mo.add, mo.mul)");
}

// Test that quoting works correctly with stripped prefixes.
TEST(FusionInfoAttrTest, ToStringQuoteWithStrippedPrefix) {
  mlir::MLIRContext ctx(mlir::MLIRContext::Threading::DISABLED);
  ctx.loadDialect<MOGGDialect>();

  auto opName = mlir::StringAttr::get(&ctx, "mo.matmul");
  auto loc = mlir::UnknownLoc::get(&ctx);
  auto leaf = FusionLeafAttr::get(&ctx, opName, loc, mlir::TypeAttr{});

  auto fusionInfo = FusionInfoAttr::get(&ctx, leaf);

  std::string result = fusionInfo.toString(/*includeLineNumbers=*/false,
                                           /*quote=*/true,
                                           /*stripDialectPrefixes=*/true);
  EXPECT_EQ(result, "\"matmul\"");

  // Without stripping, prefix is preserved.
  result = fusionInfo.toString(/*includeLineNumbers=*/false, /*quote=*/true);
  EXPECT_EQ(result, "\"mo.matmul\"");
}

} // namespace
