//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

// Host-side shim that lets a compiler plugin attach additional MLIR passes
// to the LLVM-dialect lowering pipeline. Owns a `PluginManager` via the
// caller-injected / pass-local dual-ctor pattern (mirroring
// `LowerPOPToLLVMPass`) so the CLI path (`kgen-opt -lower-to-llvm`) loads
// plugins per pass instance without needing a process-wide static.
//
// The pass itself does nothing target-specific: it forwards to the plugin's
// `M_KGEN_addPostLowerToLLVMPasses` entry point, which populates a nested
// `OpPassManager`. Each plugin-contributed pass self-gates on the target
// triple via `lookupTargetInfo`, so the pass is a safe no-op on
// triples that the plugin doesn't claim.

#include "KGEN/Support/PluginUtils.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"

using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_PLUGINSPECIFICLLVMLOWERING
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {

struct PluginSpecificLLVMLoweringPass
    : public KGEN::impl::PluginSpecificLLVMLoweringBase<
          PluginSpecificLLVMLoweringPass> {
  using PluginSpecificLLVMLoweringBase::PluginSpecificLLVMLoweringBase;

  // Dual-source PluginManager — see `LowerPOPToLLVMPass` for the canonical
  // example of the same idiom. When the caller (the pipeline builder) does
  // not supply a manager, the pass creates its own; that local manager's
  // ctor reads `MODULAR_COMPILER_PLUGINS` and dlopens each .so.
  PluginSpecificLLVMLoweringPass()
      : passLocalPluginManager(std::make_unique<PluginManager>()),
        pluginMgr(passLocalPluginManager.get()) {}

  PluginSpecificLLVMLoweringPass(const PluginSpecificLLVMLoweringPass &other)
      : passLocalPluginManager(other.pluginMgr
                                   ? nullptr
                                   : std::make_unique<PluginManager>(
                                         *other.passLocalPluginManager)),
        pluginMgr(other.pluginMgr ? other.pluginMgr
                                  : passLocalPluginManager.get()) {}

  PluginSpecificLLVMLoweringPass(PluginSpecificLLVMLoweringPass &&other)
      : passLocalPluginManager(std::move(other.passLocalPluginManager)),
        pluginMgr(other.pluginMgr ? other.pluginMgr
                                  : passLocalPluginManager.get()) {}

  PluginSpecificLLVMLoweringPass(const PluginManager *plugin)
      : passLocalPluginManager(plugin ? nullptr
                                      : std::make_unique<PluginManager>()),
        pluginMgr(plugin ? plugin : passLocalPluginManager.get()) {}

  void runOnOperation() override;

private:
  /// Pass-local plugin manager. Only allocated when the pass was constructed
  /// without an injected `PluginManager*` (i.e. the standalone CLI path,
  /// e.g. via kgen-opt's tablegen pass factory).
  std::unique_ptr<PluginManager> passLocalPluginManager = nullptr;

  /// Non-owning pointer to the active manager — either the caller-injected
  /// one or `passLocalPluginManager.get()`.
  const PluginManager *pluginMgr = nullptr;
};

} // namespace

void PluginSpecificLLVMLoweringPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  TargetInfoAttr targetInfo = lookupTargetInfo(module);
  if (!targetInfo)
    return; // No target info — nothing to attach.

  // Select only on a locally-owned manager: respect the caller's selection
  // when one was injected (parallel to `LowerPOPToLLVMPass`).
  if (passLocalPluginManager)
    passLocalPluginManager->selectPluginForTarget(targetInfo.getTriple().str());

  // Skip the rest when no plugin claims this triple. The plugin's late
  // passes self-gate too, but checking here avoids materializing a nested
  // PassManager and an empty pass set.
  if (!pluginMgr->hasPluginForTarget(targetInfo.getTriple().str()))
    return;

  // Build a nested module-anchored pass manager; let the active plugin
  // populate it through the `addPostLowerToLLVMPasses` entry point. The
  // plugin's passes can anchor on `LLVMFuncOp` (or any nested op), and
  // MLIR's per-function parallelism is preserved inside the nested run.
  mlir::PassManager nested(&getContext(), mlir::ModuleOp::getOperationName());
  // Discard the aggregated success — individual plugin failures are already
  // ignored inside `PluginManager::addPostLowerToLLVMPasses`, so the only
  // remaining error here is "no plugins iterated", which is fine.
  (void)pluginMgr->addPostLowerToLLVMPasses(nested);

  if (failed(nested.run(module)))
    signalPassFailure();
}

namespace M::KGEN {
std::unique_ptr<mlir::Pass>
createPluginSpecificLLVMLowering(const PluginManager *plugin) {
  return std::make_unique<PluginSpecificLLVMLoweringPass>(plugin);
}
} // namespace M::KGEN
