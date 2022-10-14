//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LowerToObject.h"
#include "Support/BlobCache.h"
#include "Support/ErrorOr.h"
#include "Support/VCSRevision.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include <filesystem>
#include <mutex>

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ObjectCache
//===----------------------------------------------------------------------===//

namespace {
/// This allows the BlobCache to key off of a module pointer by hashing the
/// contents of the module.
struct ModulePtrKeyInfo {
  using KeyTy = const llvm::Module *;

  /// Hash the signature of each function in the module, as well as each
  /// instruction and its operands. This should produce a stable hash that is
  /// unique between modules.
  static std::string hashKey(KeyTy key) {
    // Make sure we hash the data layout!
    llvm::hash_code moduleHash = llvm::hash_value(key->getDataLayoutStr());
    for (auto &func : key->functions()) {
      moduleHash =
          llvm::hash_combine(moduleHash, func.getReturnType()->getTypeID());
      for (auto *in : func.getFunctionType()->params())
        moduleHash = llvm::hash_combine(moduleHash, in->getTypeID());

      for (auto &instruction : llvm::instructions(func)) {
        // Add any instruction operands to the hash as well.
        for (auto &operand : instruction.operands()) {
          if (auto inst = dyn_cast<llvm::Instruction>(operand.get()))
            moduleHash = llvm::hash_combine(moduleHash, inst->getOpcode());
        }
        // And finally add the opcode for this instruction itself to the hash.
        moduleHash = llvm::hash_combine(moduleHash, instruction.getOpcode());
      }
    }

    return std::to_string(size_t(moduleHash));
  }
};
class LLVMOptimizer {
public:
  LLVMOptimizer() : passmgr(std::make_unique<llvm::legacy::PassManager>()) {
    using namespace llvm;
    passmgr->add(createFunctionInliningPass());
    passmgr->add(createSCCPPass());
    passmgr->add(createLICMPass());
    passmgr->add(createLoopStrengthReducePass());
    passmgr->add(createCFGSimplificationPass());
    passmgr->add(createAggressiveInstCombinerPass());
    passmgr->add(createStripDeadDebugInfoPass());  // Remove dead debug info
    passmgr->add(createStripDeadPrototypesPass()); // Remove dead func decls
  }

  llvm::Expected<llvm::orc::ThreadSafeModule>
  operator()(llvm::orc::ThreadSafeModule tsm,
             llvm::orc::MaterializationResponsibility &r) {
    tsm.withModuleDo([this](llvm::Module &m) { passmgr->run(m); });
    return std::move(tsm);
  }

private:
  std::unique_ptr<llvm::legacy::PassManager> passmgr;
};

} // namespace

/// Setup the machine properties from the current architecture.
static ErrorOr<std::unique_ptr<llvm::TargetMachine>> createHostTargetMachine() {
  auto targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target)
    return Error("no target exists for '" + targetTriple +
                 "': " + errorMessage);

  std::string cpu(llvm::sys::getHostCPUName());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> hostFeatures;

  if (llvm::sys::getHostCPUFeatures(hostFeatures))
    for (auto &f : hostFeatures)
      features.AddFeature(f.first(), f.second);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/None, /*OL=*/llvm::CodeGenOpt::Aggressive, /*JIT=*/true));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

//===----------------------------------------------------------------------===//
// ExecutionEngine implementation
//===----------------------------------------------------------------------===//

M::ErrorOr<ExecutionEngine> ExecutionEngine::create() {
  ExecutionEngine ee(nullptr);

  // Ensure the native target is initialized.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create the target machine.
  auto machineOr = createHostTargetMachine();
  if (machineOr.isError())
    return machineOr.takeError();

  // Create the JIT.
  auto jitOr =
      llvm::orc::LLJITBuilder()
          .setCompileFunctionCreator(
              [&](const llvm::orc::JITTargetMachineBuilder &jtmb) {
                return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
                    std::move(*machineOr));
              })
          .create();
  if (!jitOr)
    return M::Error(llvm::toString(jitOr.takeError()));

  ee.jit = std::move(*jitOr);
  ee.jit->getIRTransformLayer().setTransform(LLVMOptimizer());
  return ee;
}

ExecutionEngine::ExecutionEngine(std::unique_ptr<llvm::orc::LLJIT> jit)
    : ctx(std::make_unique<llvm::LLVMContext>()), jit(std::move(jit)) {}

ExecutionEngine::~ExecutionEngine() = default;
ExecutionEngine::ExecutionEngine(ExecutionEngine &&other) = default;

/// Add the given module to the execution engine. This slices all public funcs
/// out of the module with their dependencies to generate self-contained object
/// files.
// TODO: The slicing -> convert to LLVM -> createJITDylib + compile has natural
//       parallelism that we aren't taking advantage of.
M::ErrorOrSuccess ExecutionEngine::add(mlir::ModuleOp module,
                                       ArrayRef<FuncOp> only) {
  mlir::OwningOpRef<ModuleOp> cloned = module.clone();
  compiler = std::make_unique<ObjectCompiler>(".kgen_cache", *cloned);

  // Loop over all the funcs in the module and perform non-destructive
  // slicing, then push them to LLVM IR and compile them to objects.
  for (auto func : module.getOps<KGEN::FuncOp>()) {
    // Only compile public functions.
    if (func.getLinkage() != Linkage::Public)
      continue;

    // Apply the filter.
    if (!only.empty() && !llvm::is_contained(only, func))
      continue;

    // Lower to LLVM from the cloned module.
    auto llvmOr =
        compiler->lowerToLLVM(cloned->lookupSymbol<FuncOp>(func.getSymName()),
                              TargetInfoAttr::getForHost(func->getContext()));
    if (failed(llvmOr))
      return M::Error("failed to compile to LLVM");

    // Lower to object, again from the cloned module.
    auto objOr = compiler->lowerToObject(*llvmOr, /*isJIT=*/true);
    if (failed(objOr))
      return M::Error("failed to compile to object");

    // Produce a standalone object.
    auto standaloneOr =
        compiler->produceStandaloneObject({func.getSymName()}, /*isJIT=*/true);
    if (failed(standaloneOr))
      return M::Error("failed to produce standalone object");

    // Add this new standalone object to the execution engine.
    if (auto err = add(func.getSymName(), std::move(*standaloneOr)))
      return err;
  }

  return success();
}

ErrorOrSuccess ExecutionEngine::add(StringRef name,
                                    std::unique_ptr<llvm::MemoryBuffer> obj) {
  // Create a new dylib so that we don't have ODR violations.
  auto dylibOr = jit->createJITDylib(name.str());
  if (!dylibOr)
    return M::Error(toString(dylibOr.takeError()));

  // Resolve symbols that are statically linked in the current process.
  dylibOr->addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix())));

  if (auto err = jit->addObjectFile(*dylibOr, std::move(obj)))
    return M::Error(toString(std::move(err)));

  return success();
}

ErrorOr<CompiledFunc> ExecutionEngine::lookup(StringRef libName, FuncOp func) {
  auto *dylib = jit->getJITDylibByName(libName);
  if (!dylib)
    return Error("could not find JITDylib for " + libName);

  auto addr = jit->lookup(*dylib, func.getName());
  if (!addr)
    return M::Error(toString(addr.takeError()));

  return CompiledFunc(addr->toPtr<void *>(), func);
}

ErrorOr<CompiledFunc> ExecutionEngine::lookupOpaqueWrapper(StringRef libName,
                                                           KGEN::FuncOp func) {
  // TODO: The opaque_wrapper attr is added to the llvm.func op, not the
  //       FuncOp so we gotta have a map or something for that - we don't
  //       currently save the LLVM-dialect IR. For now, just suffix it manually.
  //       It'll be in the dylib for the original func.
  auto *dylib = jit->getJITDylibByName(libName);
  if (!dylib)
    return Error("could not find JITDylib for " + libName);

  auto addr = jit->lookup(*dylib, (func.getName() + "_opaque_wrapper").str());
  if (!addr)
    return M::Error(toString(addr.takeError()));

  return CompiledFunc(addr->toPtr<void *>(), func);
}
