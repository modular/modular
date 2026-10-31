//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENDialect.h"
#include "Mojo/KGENDialect/KGENUtils.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace mlir;
using namespace testing;

namespace {
class ParamPrintDedupTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  ParamPrintDedupTest() { ctx.loadDialect<KGENDialect>(); }
};
} // namespace

// A chain of comptime aliases like
//   a_i = a_{i-1} + 1 if a_{i-1} > 3 else a_{i-1} * 2
// is stored as a shared DAG of ParamOperatorAttr: each `cond` node references
// the previous level three times (then-branch, condition, else-branch).
// Printing that DAG without deduplication expands it into a tree of ~3^n
// leaves, which is what made compile time grow exponentially with the number
// of chained aliases (MOCO-5077).
//
// This test builds a simpler shared DAG (a_i = a_{i-1} + a_{i-1}, giving 2^n
// expansion) and checks that getParamTypeAsString prints it in linear time:
// repeated subexpressions are emitted once and referenced by a short label
// thereafter, so the output stays small instead of exploding. The display
// path uses an `e` label prefix (not a backtick) because these strings feed
// SourceNameAttr, whose encoding treats a backtick as a string delimiter.
TEST_F(ParamPrintDedupTest, ChainedAddDedupsRepeatedSubexpressions) {
  auto indexType = IndexType::get(&ctx);
  // A reference, which is never labeled.
  auto leaf = ParamIndexRefAttr::get(/*depth=*/0, /*index=*/0, indexType);

  // Build a_i = add(a_{i-1}, a_{i-1}) for i in 1..depth. Use
  // getFromBytecode to bypass the builder's constant folding, which would
  // otherwise collapse the chain.
  TypedAttr level = leaf;
  constexpr int depth = 12;
  for (int i = 0; i < depth; ++i) {
    SmallVector<TypedAttr> operands = {level, level};
    level = ParamOperatorAttr::getFromBytecode(POC::Add, operands, indexType);
  }

  std::string printed = getParamTypeAsString(level).str();

  // Dedup must have kicked in: shared subexpressions are labeled.
  EXPECT_NE(std::string::npos, printed.find("e0"))
      << "expected a dedup label, got: " << printed;
  // The repeated subexpression must be referenced by label, not re-expanded.
  EXPECT_NE(std::string::npos, printed.find("e1"))
      << "expected a repeated-subexpression label, got: " << printed;

  // Without dedup this would be ~2^12 leaves (~4096), each several characters
  // wide. With dedup it is linear in `depth`. Bound it well below the
  // exponential size so a regression that re-expands the DAG fails here.
  EXPECT_LT(printed.size(), 500u)
      << "printed string is too large; the DAG was likely re-expanded: "
      << printed.size() << " chars";
}
