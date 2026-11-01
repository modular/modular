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
#include "Mojo/MojoParser/ASTType.h"
#include "gtest/gtest.h"

using namespace M;
using namespace KGEN;
using namespace KGEN::LIT;
using namespace mlir;
using namespace testing;

namespace {
class MangledNameDedupTest : public Test {
protected:
  MLIRContext ctx{MLIRContext::Threading::DISABLED};

  MangledNameDedupTest() { ctx.loadDialect<KGENDialect>(); }

  /// Builds a shared DAG a_i = add(a_{i-1}, a_{i-1}) for i in 1..depth, rooted
  /// at a ParamIndexRefAttr leaf. Each level references the previous one twice,
  /// so printing without deduplication expands to ~2^depth leaves.
  TypedAttr buildSharedDag(int depth) {
    auto indexType = IndexType::get(&ctx);
    TypedAttr level =
        ParamIndexRefAttr::get(/*depth=*/0, /*index=*/0, indexType);
    for (int i = 0; i < depth; ++i) {
      SmallVector<TypedAttr> operands = {level, level};
      level = ParamOperatorAttr::getFromBytecode(POC::Add, operands, indexType);
    }
    return level;
  }

  std::string printWithDedup(TypedAttr value) {
    DedupContext dedup;
    ASTTypePrinterContext ctx;
    ctx.dedup = &dedup;
    std::string result;
    llvm::raw_string_ostream os(result);
    ASTType::printParam(os, value, ctx);
    return dedup.finalize(result);
  }

  std::string printWithoutDedup(TypedAttr value) {
    std::string result;
    llvm::raw_string_ostream os(result);
    ASTType::printParam(os, value, /*ctx=*/{});
    return result;
  }
};
} // namespace

// A shared DAG of parameter expressions must print in linear time when
// dedup is active: repeated subexpressions are emitted in full once and
// referenced by a short label thereafter, instead of being re-expanded at
// every use (which is what made mangled names grow exponentially with the
// number of chained comptime aliases, MOCO-5077).
TEST_F(MangledNameDedupTest, SharedDagPrintsLinearlyWithDedup) {
  constexpr int depth = 12;
  TypedAttr value = buildSharedDag(depth);

  std::string deduped = printWithDedup(value);

  // Dedup must have kicked in: shared subexpressions are labeled.
  EXPECT_NE(std::string::npos, deduped.find("`0"))
      << "expected a dedup label, got: " << deduped;
  // The repeated subexpression must be referenced by label, not re-expanded.
  EXPECT_NE(std::string::npos, deduped.find("`1"))
      << "expected a repeated-subexpression label, got: " << deduped;

  // Without dedup this would be ~2^12 leaves (~4096), each several characters
  // wide. With dedup it is linear in `depth`. Bound it well below the
  // exponential size so a regression that re-expands the DAG fails here.
  EXPECT_LT(deduped.size(), 500u)
      << "printed string is too large; the DAG was likely re-expanded: "
      << deduped.size() << " chars";

  // Sanity check: without dedup the same DAG expands exponentially, which is
  // the behavior this test guards against.
  std::string undeduped = printWithoutDedup(value);
  EXPECT_GT(undeduped.size(), deduped.size())
      << "dedup did not shrink the output; deduped=" << deduped.size()
      << " undeduped=" << undeduped.size();
}
