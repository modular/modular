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

#include "GraphCompiler/Conversion/MOTarget/Conversion.h"
#include "GraphCompiler/Conversion/RMOSupport/Activations.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/RMODialect/RMODialect.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace M;

class ConvertPreluPattern : public ConversionPattern {
public:
  ConvertPreluPattern(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, "prelu", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto x = cast<MO::TensorValue>(operands[0]);
    auto alpha = cast<MO::TensorValue>(operands[1]);
    auto result = M::RMO::convertPRelu(rewriter, op->getLoc(), x, alpha);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertReluPattern : public ConversionPattern {
public:
  ConvertReluPattern(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, "relu", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto x = cast<MO::TensorValue>(operands[0]);
    auto result = M::RMO::convertRelu(rewriter, op->getLoc(), x);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class ConvertRelu6Pattern : public ConversionPattern {
public:
  ConvertRelu6Pattern(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, "relu6", /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto x = cast<MO::TensorValue>(operands[0]);
    auto result = M::RMO::convertRelu6(rewriter, op->getLoc(), x);
    rewriter.replaceOp(op, result);
    return success();
  }
};

namespace M::MO {

#define GEN_PASS_DEF_SUPPORTCONVERSIONSTEST
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

class SupportConversionTestPass
    : public impl::SupportConversionsTestBase<SupportConversionTestPass> {
  using SupportConversionsTestBase::SupportConversionsTestBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    TypeConverter converter;
    converter.addConversion([](TensorType t) { return t; });
    RewritePatternSet patterns{ctx};
    patterns.add<ConvertPreluPattern, ConvertReluPattern, ConvertRelu6Pattern>(
        converter, ctx);
    MO::MOConversionTarget target(*ctx);
    target.addLegalDialect<M::RMO::RMODialect>();
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace M::MO
