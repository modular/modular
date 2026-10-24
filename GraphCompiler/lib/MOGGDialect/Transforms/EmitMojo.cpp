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

#include "GraphCompiler/MGPDialect/Dialect/MGPOps.h"
#include "GraphCompiler/MODialect/MODevices.h"
#include "GraphCompiler/MOGGDialect/Analysis/MojoRegistry.h"
#include "GraphCompiler/MOGGDialect/MOGGInterfaces.h" // IWYU: needed for type declarations used via MOGG::
#include "GraphCompiler/MOGGDialect/MOGGOps.h"
#include "GraphCompiler/MOGGDialect/MOGGPasses.h"
#include "GraphCompiler/MOSHDialect/MOSHAttrs.h"
#include "Support/CommonCLOptions.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ToolOutputFile.h"

#include "mlir/Dialect/Index/IR/IndexOps.h"

#include <filesystem>

using namespace M;

namespace M::MOGG {

#define GEN_PASS_DEF_EMITMOJO
#define GEN_PASS_DEF_LIFTBUFFERCACHESLOTS
#include "GraphCompiler/MOGGDialect/MOGGPasses.h.inc"

} // namespace M::MOGG

namespace M::MOGG {

namespace {

// Returns true if every use of `constOp` is as the slot operand of an
// `mgp.buffer.get_cached`, so lifting it to a kernel argument breaks no op
// that needs its compile-time value.
static bool allUsesAreCachedBufferSlot(mlir::index::ConstantOp constOp) {
  for (mlir::Operation *user : constOp->getUsers()) {
    auto cached = mlir::dyn_cast<MGP::BufferGetCachedOp>(user);
    if (!cached || cached.getBBufferSlot() != constOp.getResult())
      return false;
  }
  return true;
}

} // namespace

// GEX-3685: Lift `index.constant` ops feeding `mgp.buffer.get_cached` slot
// operands to kernel-stub arguments, materialized at each `CompiledCallOp`
// caller, so kernels differing only in the slot index become identical and
// `MergeIdenticalFuncs` collapses them (e.g. a 537-line near-duplicate
// pair in gemma-4 vision).
//
// Only MGP-ABI kernel bodies contain `get_cached` (cf.
// `RaiseBackIndexConstant`), and their only caller op is `CompiledCallOp`:
// `CallOp` / `CallAsyncOp` target MOGG-ABI kernels only (enforced by
// `CallOp::verifySymbolUses`).
void liftBufferCacheSlots(mlir::ModuleOp moduleOp) {
  mlir::MLIRContext *ctx = moduleOp.getContext();
  mlir::Type indexType = mlir::IndexType::get(ctx);

  // Symbol name -> CompiledCallOp callers.
  llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::Operation *, 4>>
      callersByName;
  moduleOp.walk([&](MOGG::CompiledCallOp exec) {
    callersByName[exec.getCalleeSymbolAttr().getAttr()].push_back(exec);
  });

  for (auto kernel : moduleOp.getOps<MOGG::FuncOp>()) {
    if (kernel.getAbi() != MOGG::ABIKind::MGP)
      continue;
    mlir::Block &body = kernel.getBodyBlock();

    // Collect candidate index.constant ops (sole uses are
    // `mgp.buffer.get_cached`'s slot operand).
    llvm::SmallVector<mlir::index::ConstantOp, 8> toPromote;
    for (mlir::Operation &op : body) {
      auto constOp = mlir::dyn_cast<mlir::index::ConstantOp>(&op);
      if (!constOp || constOp->use_empty())
        continue;
      if (allUsesAreCachedBufferSlot(constOp))
        toPromote.push_back(constOp);
    }

    if (toPromote.empty())
      continue;

    // Capture values BEFORE mutating IR.
    auto values = llvm::map_to_vector(
        toPromote, [](auto cop) { return cop.getValue().getSExtValue(); });

    // Extend the kernel function type with one `index` argument per
    // promoted constant.
    auto fnType = kernel.getFunctionType();
    auto newInputs = llvm::to_vector(fnType.getInputs());
    newInputs.append(toPromote.size(), indexType);
    auto newFnType =
        mlir::FunctionType::get(ctx, newInputs, fnType.getResults());
    kernel.setFunctionTypeAttr(mlir::TypeAttr::get(newFnType));

    // Add Block arguments matching the new function inputs, redirect uses,
    // and erase the original constants.
    for (auto cop : toPromote) {
      mlir::BlockArgument newArg = body.addArgument(indexType, cop->getLoc());
      cop.getResult().replaceAllUsesWith(newArg);
      cop->erase();
    }

    // Update every caller to pass the original constant values as new
    // trailing operands.
    auto it = callersByName.find(kernel.getSymNameAttr());
    if (it == callersByName.end())
      continue;
    for (mlir::Operation *callOp : it->second) {
      mlir::OpBuilder builder(callOp);
      llvm::SmallVector<mlir::Value, 8> newCallOperands;
      for (int64_t value : values) {
        auto cst =
            mlir::index::ConstantOp::create(builder, callOp->getLoc(), value);
        newCallOperands.push_back(cst.getResult());
      }
      callOp->insertOperands(callOp->getNumOperands(), newCallOperands);
    }
  }
}

namespace {

struct LiftBufferCacheSlots
    : public impl::LiftBufferCacheSlotsBase<LiftBufferCacheSlots> {
  using LiftBufferCacheSlotsBase::LiftBufferCacheSlotsBase;
  void runOnOperation() override { liftBufferCacheSlots(getOperation()); }
};

struct EmitMojo : public impl::EmitMojoBase<EmitMojo> {
  using EmitMojoBase::EmitMojoBase;

  void emitMojoText() {
    auto moduleOp = getOperation();
    auto ctx = moduleOp->getContext();
    auto loc = moduleOp->getLoc();
    OpBuilder builder(ctx);
    MojoRegistry registry(moduleOp);
    MOGG::EmitMojoState state(registry, enableOpTraceComments,
                              enableComputeLambdas,
                              moggDebugDumpFusedTensorSpecs);
    state.moduleOp = moduleOp;

    // Import mojo stdlib packages and compiler directives.
    for (auto [package, as] : MOGG::kPackagesToImport) {
      if (as.empty())
        state.builder.import(package);
      else
        state.builder.importAs(package, as);
    }
    // A plain `import` brings in no `__extension`s; importing a name does. A
    // declarative launch calls device context methods that a target can add
    // as extensions.
    state.builder.importFrom("max.gpu.host._legacy.device_context",
                             "DeviceContext as _DeviceContextExtensions");

    // Import mojo packages for the custom kernel libraries.
    for (const auto &kernelLibraryPath : kernelLibraryPaths) {
      auto moduleName = std::filesystem::path(kernelLibraryPath).stem();
      state.builder.import(moduleName.string());
    }

    // Library imports bind only the top-level package, so under
    // submodule-visibility gating a kernel's fully-qualified path
    // (`<module>.<Struct>.<method>`) isn't navigable. Import each kernel's
    // module: its `KernelDeclOp` `sym_name` minus the final segment.
    llvm::StringSet<> importedKernelModules;
    for (auto kernelDecl : moduleOp.getOps<MOGG::KernelDeclOp>()) {
      auto [moduleName, structName] =
          llvm::StringRef(kernelDecl.getSymName()).rsplit('.');
      if (structName.empty())
        continue;
      if (importedKernelModules.insert(moduleName).second)
        state.builder.import(moduleName.str());
    }

    state.builder.newline();

    for (auto &&kernelStub : llvm::make_early_inc_range(
             moduleOp.getBodyRegion().getOps<MOGG::FuncOp>())) {
      if (kernelStub.emitMojo(state).failed()) {
        signalPassFailure();
        return;
      }
    }

    // Append any deferred struct definitions (fusion structs emitted by
    // BindLambdaOp). Mojo processes all module-scope declarations before
    // compiling function bodies, so ordering doesn't matter.
    auto mojoText = state.builder.str() + state.structDefsBuilder.str();

    // Compute a base file name for the dump from the names of non-subgraph
    // ModelOps (user-visible model names) joined by "+", or "graph" when there
    // are none. Mirrors the `<names>.mo.mlir` stem pipeline_to_mo_graph.py
    // writes, so the .mojo dump lands beside the source graph's stages.
    std::string baseFileName;
    for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      if (!modelOp.getIsSubgraph()) {
        if (!baseFileName.empty())
          baseFileName += "+";
        baseFileName += modelOp.getSymName().str();
      }
    }
    if (baseFileName.empty())
      baseFileName = "graph";

    // Save the emitted Mojo code to a file if an IR dump directory is set
    // (via `MODULAR_DEBUG=ir-output-dir=<dir>` or the legacy `max.temps_dir`).
    if (auto outFile = openIntermediateTextFile(baseFileName, ".mojo")) {
      outFile->os() << mojoText;
      outFile->keep();
    }

    auto attrType =
        ArrayType::get((int64_t)mojoText.size(),
                       IntegerType::get(ctx, 8, IntegerType::Unsigned));
    auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyInferAlign<char>(
        {mojoText.data(), mojoText.size()});
    auto resourceManager =
        DenseResourceElementsHandle::getManagerInterface(ctx);
    auto &blobManager = resourceManager.getBlobManager();

    // If there is already a mojo resource, it's probably because of constant
    // folding. Update it instead of inserting a new one that would have a
    // different name.
    DenseResourceElementsAttr mojoTextAttr;
    static constexpr llvm::StringLiteral mojoResourceName = "mojo";
    if (blobManager.lookup(mojoResourceName)) {
      blobManager.update(mojoResourceName, std::move(blob));
      auto blobHandle = mlir::DialectResourceBlobHandle<mlir::BuiltinDialect>(
          blobManager.lookup(mojoResourceName),
          cast<mlir::BuiltinDialect>(resourceManager.getDialect()));
      mojoTextAttr = DenseResourceElementsAttr::get(attrType, blobHandle);
    } else {
      mojoTextAttr = DenseResourceElementsAttr::get(attrType, mojoResourceName,
                                                    std::move(blob));
    }

    // For each model, add a reference to the Mojo symbols in
    // CompiledLoadOp.
    for (auto modelOp : moduleOp.getOps<MGP::ModelOp>()) {
      llvm::SetVector<StringRef> kernelsToLoadInThisModel;
      modelOp.walk([&](MOGG::CompiledCallOp callOp) {
        kernelsToLoadInThisModel.insert(callOp.getCalleeSymbol());
      });

      if (kernelsToLoadInThisModel.empty())
        continue;

      builder.setInsertionPointToStart(modelOp.getInitBlock());
      /// TODO: Add debug info support. For now, defaults to false.
      MOGG::CompiledLoadOp::create(builder, loc,
                                   builder.getArrayAttr(llvm::map_to_vector(
                                       kernelsToLoadInThisModel,
                                       [&](auto name) {
                                         return cast<Attribute>(
                                             StringAttr::get(ctx, name));
                                       })),
                                   mojoTextAttr, /*debug*/ false);
    }

    for (auto &&kernelStub : llvm::make_early_inc_range(
             moduleOp.getBodyRegion().getOps<MOGG::FuncOp>())) {
      kernelStub->erase();
    }

    // The device-info mapping is only needed through Mojo emission (e.g. to
    // resolve per-device `tile_based_fusion`); strip it now so it doesn't
    // surface in downstream formats (e.g. MEF).
    moduleOp->removeAttr(MO::DeviceInfoMapping::kAttrName);
  }

  // Kernel-dedup and try/except preparation run in `buildMgpPipeline`
  // immediately before this pass.
  void runOnOperation() override { emitMojoText(); }
};

} // namespace

} // namespace M::MOGG
