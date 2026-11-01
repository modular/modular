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
//
// Converts MO ops to MGP and MEF ops.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/Interfaces/LowerableToMGPInterface.h"
#include "GraphCompiler/MGPDialect/Dialect/MGPTypes.h"
#include "GraphCompiler/MGPDialect/Transforms/FixupIsolatedFromAbove.h"
#include "GraphCompiler/MGPDialect/Transforms/Utils.h"
#include "GraphCompiler/MODialect/MOOps.h"
#include "GraphCompiler/MOGGDialect/MOGGDialect.h"
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOGGDialect/Support/FusionUtils.h"
#include "MLRT/GraphRT/GRTDialect/GRTOps.h"
#include "MLRT/GraphRT/MEFDialect/MEFTypes.h"
#include "Support/AssertStream.h"
#include "Support/DeviceSpecs.h"

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_LOWERCOMPILEDROUTINES
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

//===----------------------------------------------------------------------===//
// MOGraphLowering implementation
//===----------------------------------------------------------------------===//

namespace M::MOGG {

namespace {

struct LowerCompiledRoutinesPass
    : public MOGG::impl::LowerCompiledRoutinesBase<LowerCompiledRoutinesPass> {
  using LowerCompiledRoutinesBase::LowerCompiledRoutinesBase;
  void rewriteExecuteCompiledOp(OpBuilder &builder, const Value &context,
                                size_t runtimeSlot, size_t module_index,
                                size_t kernel_index,
                                MOGG::CompiledCallOp &oldKernel);
  void runOnOperation() override;
};

// Rewrite the MOGG::CompiledCallOp into a MGP::ExecuteGenericOp
void LowerCompiledRoutinesPass::rewriteExecuteCompiledOp(
    OpBuilder &builder, const Value &context, size_t runtimeSlot,
    size_t module_index, size_t kernel_index, MOGG::CompiledCallOp &oldKernel) {
  auto ctx = oldKernel.getOperation()->getContext();
  SmallVector<Value> outputs;
  SmallVector<Type> resultTypes;
  for (auto [resTy, res] :
       llvm::zip(oldKernel.getResultTypes(), oldKernel.getResults())) {
    outputs.push_back(res);
    // In emit-mojo path, convert mo.opaque result types to mgp.opaque for
    // the model's external interface. Preserve type parameters.
    auto moOpaqueType = dyn_cast<MO::OpaqueType>(resTy);
    if (!moOpaqueType) {
      resultTypes.push_back(resTy);
      continue;
    }
    resultTypes.push_back(MGP::OpaqueType::get(
        ctx, moOpaqueType.getSymbol(), moOpaqueType.getOptionalParameters()));
  }

  auto device = oldKernel.getExecutionDevice();
  if (!device)
    device = DeviceRefAttr::getDefault(ctx);

  auto newExecuteOp = MGP::ExecuteGenericOp::create(
      builder, oldKernel.getLoc(), resultTypes, context,
      oldKernel->getOperands(), runtimeSlot, module_index, kernel_index,
      StringAttr::get(ctx, oldKernel.getFusionInfoAttr().toString(
                               /*includeLineNumbers=*/false)),
      /*eIsAsync=*/oldKernel.getDispatch() == MOGG::DispatchKind::Deferred,
      oldKernel.getConstantResources());

  for (auto [idx, result] : llvm::enumerate(outputs))
    result.replaceAllUsesWith(newExecuteOp.getResult(idx));

  oldKernel.erase();
}

// Lowers compiled JIT kernels:
// - MOGG::CompiledLoadOp -> MGP::ModuleLoadOp, after the RuntimeCreateOp.
// - MOGG::CompiledCallOp -> MGP::ExecuteGenericOp in the init and execute
//   blocks, preserving init-block ordering.
void LowerCompiledRoutinesPass::runOnOperation() {
  MLIRContext *ctx = getOperation().getContext();
  for (auto model :
       llvm::make_early_inc_range(getOperation().getOps<MGP::ModelOp>())) {
    Block &init = *model.getInitBlock();
    Block &execute = *model.getExecuteBlock();
    OpBuilder builder{ctx};

    MGP::ExecuteState execState(ctx, model);

    // MOGG::CompiledLoadOp is CPU-only; count the right one for
    // moduleIndex.
    const auto hostDevice = DeviceRefAttr::get(ctx, kCPULabel, 0);

    // Collect the chains for all module load and runtime creates.
    SmallVector<Value> moduleLoadChains;

    // `moduleLoadResultByIndex[i]` is the `mgp.module.load` output chain for
    // device module `i`. Used below to sequence each init-block execute after
    // the load of the specific module it calls into.
    SmallVector<Value> moduleLoadResultByIndex;

    // So executions target the right module load and device runtime.
    MGP::RuntimeCreateOp lastDeviceRuntimeOp;
    for (auto op : init.getOps<MGP::RuntimeCreateOp>()) {
      moduleLoadChains.push_back(op.getResult());
      lastDeviceRuntimeOp = op;
    }

    // For each kernel name, track the module index and kernel index
    llvm::StringMap<std::pair<uint64_t, uint64_t>> kernelModuleAndFnIndex;
    uint64_t deviceModuleIndex = 0;
    // Rewrite the MOGG::CompiledLoadOp in init blocks.
    for (auto loadKernels :
         llvm::make_early_inc_range(init.getOps<MOGG::CompiledLoadOp>())) {
      Operation *insertionOp = lastDeviceRuntimeOp.getOperation();
      size_t runtimeSlot = lastDeviceRuntimeOp.getSlot();
      Value opContext = lastDeviceRuntimeOp.getTheContext();
      Value inChain = lastDeviceRuntimeOp.getResult();

      SmallVector<StringAttr> deviceKernelNameAttrs;
      uint64_t deviceKernelIndex = 0;
      for (StringAttr attr :
           loadKernels.getSymbols().getAsRange<StringAttr>()) {
        kernelModuleAndFnIndex[attr] =
            std::make_pair(deviceModuleIndex, deviceKernelIndex++);
        deviceKernelNameAttrs.push_back(attr);
      }

      builder.setInsertionPointAfter(insertionOp);
      auto loadArchive = MGP::ModuleLoadOp::create(
          builder, insertionOp->getLoc(),
          /*inChain=*/inChain,
          /*theContext=*/opContext,
          /*aSlot=*/runtimeSlot,
          /*bArchive=*/loadKernels.getValue(),
          /*cKernelNames=*/
          MultiLineStringAttr::get(ctx, deviceKernelNameAttrs),
          /*dConstantSymbols=*/
          MultiLineStringAttr::get(ctx, SmallVector<StringAttr>()),
          /*eDevice=*/hostDevice,
          /*fModuleIndex=*/deviceModuleIndex,
          /*gDebuginfo=*/loadKernels.getDebuginfo());
      moduleLoadChains.push_back(loadArchive.getResult());
      // Index-aligned with `deviceModuleIndex` (pushed once per module, in
      // order), so `moduleLoadResultByIndex[deviceModuleIndex]` is this
      // module's load chain.
      moduleLoadResultByIndex.push_back(loadArchive.getResult());
      inChain.replaceAllUsesExcept(loadArchive.getResult(), loadArchive);
      loadKernels.erase();
      ++deviceModuleIndex;
    }

    // Remove the placeholder chain.
    builder.setInsertionPointToStart(&init);
    for (auto op : llvm::make_early_inc_range(
             init.getOps<MGP::ModuleLoadPlaceholderOp>())) {

      Value newChain;
      if (moduleLoadChains.size() > 1) {
        newChain =
            GRT::ChainCreateOp::create(builder, op.getLoc(), moduleLoadChains);
      } else {
        newChain = moduleLoadChains[0];
      }
      op.replaceAllUsesWith(newChain);
      op.erase();
    }

    // Rewrite the MOGG::CompiledCallOp in both init and execute blocks.
    // Shunt all the executes for Init stage at the very end of init
    builder.setInsertionPoint(init.getTerminator());
    init.walk([&](MOGG::CompiledCallOp kernel) {
      auto [moduleIndex, kernelIndex] =
          kernelModuleAndFnIndex[kernel.getCalleeSymbol()];
      rewriteExecuteCompiledOp(builder, lastDeviceRuntimeOp.getTheContext(),
                               lastDeviceRuntimeOp.getSlot(), moduleIndex,
                               kernelIndex, kernel);
    });

    WalkResult executeWalkResult =
        execute.walk([&](MOGG::CompiledCallOp kernel) {
          builder.setInsertionPoint(kernel);
          Region *parentRegion = kernel->getParentRegion();
          Value executeContext = nullptr;

          // For execute block, need to account for the bounding region in the
          // case of control flow or other bounding operations.
          while (parentRegion && !executeContext) {
            for (Value arg : parentRegion->getArguments()) {
              if (isa<MGP::ContextType>(arg.getType()))
                executeContext = arg;
            }
            parentRegion = parentRegion->getParentRegion();
          }
          ASSERT_STREAM(executeContext,
                        << "Cannot find execute context in bounding region of "
                           "generic.execute");
          auto kernelIndexIt =
              kernelModuleAndFnIndex.find(kernel.getCalleeSymbol());
          if (kernelIndexIt == kernelModuleAndFnIndex.end()) {
            kernel.emitError()
                << "compiled kernel '" << kernel.getCalleeSymbol()
                << "' is not present in any loaded module";
            signalPassFailure();
            return WalkResult::interrupt();
          }
          auto [moduleIndex, kernelIndex] = kernelIndexIt->second;
          // Execute-block executes need no setup-ordering guard: their slot is
          // populated during the init phase, which always completes before the
          // execute phase runs.
          rewriteExecuteCompiledOp(builder, executeContext,
                                   lastDeviceRuntimeOp.getSlot(), moduleIndex,
                                   kernelIndex, kernel);
          return WalkResult::advance();
        });
    if (executeWalkResult.wasInterrupted())
      return;

    // `mgp.module.load` is a stateful op: it populates the
    // `KernelExecutionState` slot that an init-block `mgp.generic.execute`
    // reads through its `!mgp.context`.

    // Route each chain-less init execute's context through an `mgp.guard` on
    // the load chain of the module it calls into. Gate per module so unrelated
    // init kernels stay free to run in parallel.
    init.walk([&](MGP::ExecuteGenericOp exec) {
      // Already ordered after the load through a chain operand: leave it alone.
      if (llvm::any_of(exec->getOperands(), [](Value v) {
            return isa<MEF::ChainType>(v.getType());
          }))
        return;
      // A chain-less execute resolved its callee to a loaded module above,
      // which also pushed that module's load chain, so the index is in range.
      uint64_t moduleIndex = exec.getBModuleIndex();
      assert(moduleIndex < moduleLoadResultByIndex.size() &&
             "chain-less init execute has no module-load chain to order after");
      if (moduleIndex >= moduleLoadResultByIndex.size())
        return;
      Value context = exec.getTheContext();
      Value moduleLoadChain = moduleLoadResultByIndex[moduleIndex];
      builder.setInsertionPoint(exec);
      auto guard = MGP::GuardOp::create(builder, exec.getLoc(),
                                        TypeRange{context.getType()},
                                        moduleLoadChain, ValueRange{context});
      exec.getTheContextMutable().assign(guard.getResult(0));
    });

    // In the emit-mojo path the emitted Mojo already handles the mo.opaque ->
    // mgp.opaque conversion for graph outputs, so forward ToMGPFromOpaqueOp.
    for (auto toMgpOp : llvm::make_early_inc_range(
             execute.getOps<MOGG::ToMGPFromOpaqueOp>())) {
      toMgpOp.getDataPointer().replaceAllUsesWith(toMgpOp.getInput());
      toMgpOp.erase();
    }
    // The introduction of generic execute may introduce unbound free-values.
    // Use fixupIsolatedFromAbove to rebind these.
    (void)fixupIsolatedFromAbove(builder, model);
    model.erase();
  }
}

} // namespace

} // namespace M::MOGG
