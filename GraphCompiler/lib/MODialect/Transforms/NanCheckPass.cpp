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

#include "GraphCompiler/Conversion/MOToMAP/MOToMAP.h"
#include "GraphCompiler/MAPDialect/Dialect/MAPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

namespace M::MO {

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_NANCHECKPASS
#include "GraphCompiler/MODialect/Transforms/MOTransforms.h.inc"

} // namespace M::MO

using namespace M;
using namespace M::MO;

namespace {

struct NanCheckPass : public impl::NanCheckPassBase<NanCheckPass> {
  using Base::Base;

  std::string valueToLabel(Value value) {
    std::string str;
    llvm::raw_string_ostream os(str);

    if (auto op = value.getDefiningOp())
      os << " " << op->getName() << " ";

    value.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());

    return str;
  }

  bool isFloatTensor(Value value) {
    auto tensorType = dyn_cast<MO::TensorType>(value.getType());
    if (!tensorType)
      return false;
    return tensorType.getDtype().isFloat();
  }

  /// Lowers a check op in place via the `mo-to-map` opaque lowering. Running
  /// after MAP fusion keeps it from being fused into the kernel it checks.
  /// Returns the `map.iter.opaque` op, or nullptr on failure.
  Operation *lowerCheckOpToMAP(MO::CustomOp customOp,
                               MOGG::MojoRegistry &registry) {
    FailureOr<Operation *> mapOp = MO::lowerMOOpToMAP(customOp, registry);
    if (failed(mapOp))
      return nullptr;
    return *mapOp;
  }

  void runOnOperation() override {
    GraphOp graphOp = getOperation();
    OpBuilder builder(graphOp->getContext());

    auto *ctx = builder.getContext();
    auto chainType = MO::ChainType::get(ctx);
    auto hostDevice = M::getHostDevice(ctx);

    // Single-element int32 tensor types for NaN/Inf counts.
    auto emptyMetadata = builder.getDictionaryAttr({});
    auto countTypeHost = MO::TensorType::get(
        ctx, ArrayRef<int64_t>{1}, M::DType::si32, hostDevice, emptyMetadata);
    assert(countTypeHost && "Failed to create count tensor type");

    // Before `map-to-mogg`, compiled regions are still `map.iter.elementwise` /
    // `map.iter.opaque` ops.
    SmallVector<Operation *> kernelOps;
    graphOp->walk([&kernelOps](Operation *op) {
      if (isa<MAP::IterElementwiseOp, MAP::IterOpaqueOp>(op))
        kernelOps.push_back(op);
    });

    if (kernelOps.empty())
      return;

    MOGG::MojoRegistry registry(
        getOperation()->getParentOfType<mlir::ModuleOp>());

    // Created lazily so it is inserted after the first checked kernel.
    Value currentChain;

    // Sampling stride (GEX-3703): the single-threaded Mojo/KGEN compile can't
    // absorb hundreds of extra kernel call sites (1040 count kernels alone hung
    // past 30 min on B200), so only every `stride`-th float output is checked.
    // stride=1 checks every tensor; large models raise it via the pass option
    // or the `max-debug.nan-check-stride` config key.
    const uint64_t effectiveStride = stride == 0 ? 1 : stride;
    size_t floatResultIdx = 0;
    size_t totalFloatResults = 0;
    size_t checkedFloatResults = 0;

    // Per checked output, the original tensor flows unchanged:
    //   1. nan_check_count: read-only reduction → scalar counts (on device)
    //   2. mo.transfer: move counts to host (chain provides ordering)
    //   3. nan_check_raise: read host counts, raise if > 0 (chain for DCE)
    for (auto kernelOp : kernelOps) {
      builder.setInsertionPointAfter(kernelOp);
      for (Value res : kernelOp->getResults()) {
        if (!isFloatTensor(res))
          continue;

        ++totalFloatResults;
        size_t samplingIndex = floatResultIdx++;
        if (effectiveStride > 1 && (samplingIndex % effectiveStride) != 0)
          continue;
        ++checkedFloatResults;

        auto tensorType = cast<MO::TensorType>(res.getType());
        auto device = getDeviceOrDefaultAny(kernelOp);

        auto countTypeOnDevice = MO::TensorType::get(
            ctx, ArrayRef<int64_t>{1}, M::DType::si32, device, emptyMetadata);

        std::string label = valueToLabel(res);
        std::string typeStr;
        {
          llvm::raw_string_ostream os(typeStr);
          os << tensorType;
        }

        auto countOp = MO::CustomOp::create(
            builder, kernelOp->getLoc(),
            /*resultTypes=*/TypeRange{countTypeOnDevice, countTypeOnDevice},
            /*operands=*/ValueRange{res},
            /*symbol=*/builder.getStringAttr("nan_check_count"),
            /*device=*/device,
            /*parameters=*/builder.getDictionaryAttr({}));

        Value nanCount = countOp.getResult(0);
        Value infCount = countOp.getResult(1);

        if (!currentChain) {
          currentChain = MO::ChainCreateOp::create(builder, kernelOp->getLoc(),
                                                   chainType, ValueRange{})
                             .getResult();
          assert(currentChain && "currentChain is null after creation");
        }

        Value nanCountHost = nanCount;
        Value infCountHost = infCount;
        bool needsTransfer = (device != hostDevice);
        if (needsTransfer) {
          auto nanTransfer = MO::TransferOp::create(
              builder, kernelOp->getLoc(), cast<TensorValue>(nanCount),
              hostDevice, cast<ChainValue>(currentChain),
              /*alwaysElideSameDeviceCopy=*/true);
          nanCountHost = nanTransfer.getResult();
          currentChain = nanTransfer.getOutChain();

          auto infTransfer = MO::TransferOp::create(
              builder, kernelOp->getLoc(), cast<TensorValue>(infCount),
              hostDevice, cast<ChainValue>(currentChain),
              /*alwaysElideSameDeviceCopy=*/true);
          infCountHost = infTransfer.getResult();
          currentChain = infTransfer.getOutChain();
        }

        auto raiseOp = MO::CustomOp::create(
            builder, kernelOp->getLoc(),
            /*resultTypes=*/TypeRange{chainType},
            /*operands=*/ValueRange{nanCountHost, infCountHost, currentChain},
            /*symbol=*/builder.getStringAttr("nan_check_raise"),
            /*device=*/hostDevice,
            /*parameters=*/
            builder.getDictionaryAttr(
                {NamedAttribute(builder.getStringAttr("label"),
                                builder.getStringAttr(label)),
                 NamedAttribute(builder.getStringAttr("type_str"),
                                builder.getStringAttr(typeStr))}));
        currentChain = raiseOp.getResult(0);

        if (!lowerCheckOpToMAP(countOp, registry))
          return signalPassFailure();

        // Mark the raise as writing so DCE keeps it despite its unused chain;
        // the opaque lowering only sets that for an in-place `mo.custom`.
        auto *raiseKernel = lowerCheckOpToMAP(raiseOp, registry);
        if (!raiseKernel)
          return signalPassFailure();
        cast<MAP::IterOpaqueOp>(raiseKernel).setHasWriteEffect(true);

        // Lowering erased raiseOp, so continue from the MAP kernel's chain.
        for (auto result : raiseKernel->getResults()) {
          if (isa<MO::ChainType>(result.getType())) {
            currentChain = result;
            break;
          }
        }
      }
    }

    llvm::errs() << "[GEX-3703] NanCheckPass: checked " << checkedFloatResults
                 << " of " << totalFloatResults << " float kernel outputs "
                 << "(stride=" << effectiveStride << ", graph @"
                 << graphOp.getSymName() << ")\n";
  }
};

} // namespace

std::unique_ptr<Pass> M::MO::createNanCheckPass(uint64_t stride) {
  NanCheckPassOptions opts;
  opts.stride = stride;
  return std::make_unique<NanCheckPass>(std::move(opts));
}
