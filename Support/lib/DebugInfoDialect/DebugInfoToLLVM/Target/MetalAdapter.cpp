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

#include "MetalAdapter.h"

#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "Support/LogicalResult.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace M::DebugInfo;

namespace {
struct ConvertLineTableLocOp : public OpRewritePattern<LineTableLocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LineTableLocOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

static void populateMetalConversionPatterns(DIAttrTypeReplacer &,
                                            RewritePatternSet &patterns) {
  patterns.add<ConvertLineTableLocOp>(patterns.getContext());
}

TargetAdapter DebugInfo::getMetalAdapter() {
  return TargetAdapter{populateMetalConversionPatterns, [](ModuleOp) {},
                       [](ModuleOp) {}};
}
