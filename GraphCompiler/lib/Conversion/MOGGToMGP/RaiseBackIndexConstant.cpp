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

#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/Threading.h"

using namespace M;

namespace M::MGP {

#define GEN_PASS_DEF_RAISEBACKINDEXCONSTANT
#include "GraphCompiler/Conversion/MOGGToMGP/MOGGToMGP.h.inc"

} // namespace M::MGP

namespace M::MGP {

namespace {

struct RaiseBackIndexConstant
    : public MGP::impl::RaiseBackIndexConstantBase<RaiseBackIndexConstant> {
  using RaiseBackIndexConstantBase::RaiseBackIndexConstantBase;
  /// Single-caller stubs were not deduplicated, so their lowered
  /// index.constants are raised back to reduce emitted text. The now-unused
  /// constant resources are left in place.
  void runOnOperation() override {
    auto moduleOp = getOperation();
    // Callee -> callers, built from the model ops rather than a SymbolTable
    // to skip the thousands of KernelStubOps.
    llvm::DenseMap<StringAttr, SmallVector<MOGG::CompiledCallOp>> symbolUses;
    for (auto &&modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      modelOp.walk([&](MOGG::CompiledCallOp callOp) {
        symbolUses[callOp.getCalleeSymbolAttr().getAttr()].push_back(callOp);
      });
    }

    auto stubsRange = moduleOp.getOps<MOGG::FuncOp>();
    SmallVector<MOGG::FuncOp> kernels(stubsRange.begin(), stubsRange.end());
    mlir::parallelForEach(
        moduleOp->getContext(), kernels, [&](MOGG::FuncOp kernelStub) {
          if (kernelStub.getAbi() != MOGG::ABIKind::MGP)
            return;

          auto it = symbolUses.find(kernelStub.getSymNameAttr());
          if (it == symbolUses.end())
            return;

          // With several callers the constant may differ per caller.
          if (it->second.size() != 1)
            return;

          auto resourceArr = it->second.front().getConstantResourcesAttr();

          OpBuilder builder(kernelStub);
          kernelStub.walk([&](MGP::BufferToIndexOp bufferToIndexOp) {
            auto resourceConstantOp =
                bufferToIndexOp.getInBuffer()
                    .getDefiningOp<MGP::ResourceConstantOp>();
            if (!resourceConstantOp)
              return;

            uint64_t slotIndex =
                resourceConstantOp.getResourceSlot().getZExtValue();
            auto resource = dyn_cast<AlignedBytesAttr>(resourceArr[slotIndex]);
            // Assumes index.constant was lowered to an AlignedBytesAttr.
            if (!resource)
              return;

            if (resource.size() != sizeof(int64_t))
              return;

            int64_t value =
                *reinterpret_cast<const int64_t *>(resource.getData().data());
            builder.setInsertionPoint(bufferToIndexOp);
            auto indexConstantOp = mlir::index::ConstantOp::create(
                builder, resourceConstantOp->getLoc(), value);
            bufferToIndexOp->replaceAllUsesWith(indexConstantOp);
          });
        });
  }
};

} // namespace
} // namespace M::MGP
