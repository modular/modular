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

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Patterns/Algebraic.h"
#include "GraphCompiler/MODialect/Patterns/ShapeOpSinking.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Support/SplitSlicePatterns.h"

#define DEBUG_TYPE "symbolic-optimization"

using namespace M;
using namespace MO;

namespace M::MO {
#define GEN_PASS_DEF_SYMBOLICOPTIMIZATION
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

namespace {

struct SymbolicOptimizationPass
    : public MO::impl::SymbolicOptimizationBase<SymbolicOptimizationPass> {
  using SymbolicOptimizationBase::SymbolicOptimizationBase;

  template <typename... PatternTypes>
  LogicalResult applyCorePatternsPlusExtraPatterns() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    if constexpr (sizeof...(PatternTypes)) {
      patterns.add<PatternTypes...>(context);
    }

    populateAlgebraicPatterns(patterns, symbolicFfastMath,
                              symbolicNoUndefinedMath);
    populateShapeOpSinkingPatterns(patterns);

    // TODO(24646): Ensure these use device aware rewriter.
    TransposeOp::getCanonicalizationPatterns(patterns, context);
    Static::BroadcastToOp::getCanonicalizationPatterns(patterns, context);

    // TODO(GEX-540): Top-down traversal avoids high-iteration pattern loops.
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    return applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

  LogicalResult applyForwardMathematicalCanonicalizations() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateForwardCanonicalizationPatterns(patterns);

    // TODO(GEX-540): Top-down traversal avoids high-iteration pattern loops.
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    return applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

  LogicalResult applyBackwardMathematicalCanonicalizations() {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    populateBackwardCanonicalizationPatterns(patterns);

    // TODO(GEX-540): Top-down traversal avoids high-iteration pattern loops.
    mlir::GreedyRewriteConfig config;
    config.setUseTopDownTraversal(true);
    return applyPatternsGreedily(getOperation(), std::move(patterns), config);
  }

  LogicalResult applyCorePatterns() {
    // Split patterns must run before splits are lowered back to slices, or
    // they would never match.
    if (applyCorePatternsPlusExtraPatterns<SliceToSplitPattern<>>().failed())
      return failure();

    return applyCorePatternsPlusExtraPatterns<SplitToSlicePattern<true>>();
  }

  void runOnOperation() override {
    if (applyCorePatterns().failed())
      return signalPassFailure();
    if (applyForwardMathematicalCanonicalizations().failed())
      return signalPassFailure();
    if (applyCorePatterns().failed())
      return signalPassFailure();
    if (applyBackwardMathematicalCanonicalizations().failed())
      return signalPassFailure();
    if (applyCorePatterns().failed())
      return signalPassFailure();
  }
};

} // namespace
