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

#include "GraphCompiler/Interfaces/LoweringState.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/SymbolNames.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/FormatVariadic.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_OUTLINEASYNCROUTINES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

/// Returns `__<callees>_async`, joining the stubs the region calls in call
/// order, or `__primitive_<number>_async` when it calls none. The suffix keeps
/// a region that calls a single stub from taking that stub's name.
static std::string getAsyncStubName(AsyncRoutineOp region, uint64_t number) {
  SmallVector<std::string> callees;
  region.getBody().walk([&](FuncCallOpInterface call) {
    callees.push_back(call.getCalleeSymbol().getValue().ltrim('_').str());
  });
  std::string base = callees.empty()
                         ? llvm::formatv("primitive_{}", number).str()
                         : joinNameParts(callees, TruncationMarker::Hash);
  return "__" + base + "_async";
}

namespace {

struct OutlineAsyncRoutines
    : public MOGG::impl::OutlineAsyncRoutinesBase<OutlineAsyncRoutines> {
  using OutlineAsyncRoutinesBase::OutlineAsyncRoutinesBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();
    OpBuilder builder(moduleOp->getContext());
    builder.setInsertionPointToStart(moduleOp.getBody(0));
    auto ctx = moduleOp->getContext();

    // The pass manager never invalidates the cached `SymbolTableAnalysis`, so
    // register new stubs in it (and mark it preserved); otherwise passes that
    // read the cached table would not see them.
    SymbolTable &symtab =
        getAnalysis<mlir::SymbolTableAnalysis>().getTopLevelSymbolTable();

    uint64_t stubNumber = 0;
    moduleOp.walk<mlir::WalkOrder::PreOrder>(
        [&](MOGG::AsyncRoutineOp kernelOp) {
          auto loc = kernelOp.getLoc();
          auto resultTypes = kernelOp->getResultTypes();
          // The MGP-ABI stub takes the region body verbatim, so the signature
          // uses its `!mogg.async<T>` block arg types (the void** runtime
          // slots), not the region's `T` operand types.
          auto functionType = TypeAttr::get(
              FunctionType::get(ctx, kernelOp.getBodyBlock().getArgumentTypes(),
                                kernelOp.getResultTypes()));
          auto stubName =
              StringAttr::get(ctx, getAsyncStubName(kernelOp, stubNumber++));
          // Synthesize a leaf FusionInfoAttr from the summary when the region
          // has no fusionInfo, so FuncOp always carries one.
          FusionInfoAttr fusionInfo = kernelOp.getFusionInfoAttr();
          if (!fusionInfo) {
            StringAttr opName = stubName;
            if (auto s = kernelOp->getAttrOfType<StringAttr>("summary"))
              opName = s;
            fusionInfo = FusionInfoAttr::get(
                ctx, FusionLeafAttr::get(ctx, opName, /*loc=*/{}, /*sig=*/{}));
          }
          DeviceRefAttr device = MO::getDeviceOrDefaultAny(kernelOp);
          auto stubOp = MOGG::FuncOp::create(
              builder, loc, stubName, functionType, fusionInfo,
              kernelOp.getDispatchAttr(), kernelOp.getConstantResourcesAttr(),
              MOGG::ABIKindAttr::get(ctx, MOGG::ABIKind::MGP),
              builder.getStringAttr(device.getLabel()), nullptr, nullptr);
          stubOp.getBody().takeBody(kernelOp.getBody());

          // `create` already linked the stub into the module; this updates the
          // table's name map, which may unique the name.
          stubName = symtab.insert(stubOp);

          builder.setInsertionPoint(kernelOp);
          auto callOp = MOGG::CompiledCallOp::create(
              builder, stubOp->getLoc(),
              /*resultTypes=*/resultTypes,
              /*operands=*/kernelOp->getOperands(),
              /*kernelName=*/FlatSymbolRefAttr::get(stubName),
              /*usedParams=*/builder.getArrayAttr({}),
              /*outputParamDecls=*/
              KGEN::ParamDeclArrayAttr::get(builder.getContext(), {}),
              /*fusionInfo=*/stubOp.getFusionInfoAttr(),
              /*dispatch=*/stubOp.getDispatchAttr(),
              /*constantResources=*/stubOp.getConstantResources());
          callOp.setDeviceRef(device);

          kernelOp->replaceAllUsesWith(callOp);
          kernelOp.erase();
          builder.setInsertionPoint(stubOp);

          return mlir::WalkResult::skip();
        });
  }
};

} // namespace

} // namespace M::MOGG
