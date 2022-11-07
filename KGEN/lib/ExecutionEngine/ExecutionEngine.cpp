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
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ExecutionEngine/JITEventListener.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h"
#include "llvm/ExecutionEngine/SectionMemoryManager.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"

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
  llvm::InitializeNativeTargetAsmParser(); // needed for inline_asm

  // Create the target machine.
  auto machineOr = createHostTargetMachine();
  if (machineOr.isError())
    return machineOr.takeError();

  // Callback to create the object layer with symbol resolution to current
  // process and dynamically linked libraries.
  auto objectLinkingLayerCreator = [&](llvm::orc::ExecutionSession &session,
                                       const llvm::Triple &tt) {
    auto objectLayer =
        std::make_unique<llvm::orc::RTDyldObjectLinkingLayer>(session, []() {
          return std::make_unique<llvm::SectionMemoryManager>();
        });

    // Register JIT event listeners if they are enabled.
    if (ee.gdbListener)
      objectLayer->registerJITEventListener(*ee.gdbListener);
    if (ee.perfListener)
      objectLayer->registerJITEventListener(*ee.perfListener);

    // Make sure the debug info sections aren't stripped.
    objectLayer->setProcessAllSections(true);

    // COFF format binaries (Windows) need special handling to deal with
    // exported symbol visibility.
    if (tt.isOSBinFormatCOFF()) {
      objectLayer->setOverrideObjectFlagsWithResponsibilityFlags(true);
      objectLayer->setAutoClaimResponsibilityForObjectSymbols(true);
    }

    return objectLayer;
  };

  // Create the JIT.
  auto jitOr = llvm::orc::LLJITBuilder()
                   .setObjectLinkingLayerCreator(objectLinkingLayerCreator)
                   .create();
  if (!jitOr)
    return M::Error(llvm::toString(jitOr.takeError()));

  ee.jit = std::move(*jitOr);
  return ee;
}

ExecutionEngine::ExecutionEngine(std::unique_ptr<llvm::orc::LLJIT> jit)
    : ctx(std::make_unique<llvm::LLVMContext>()), jit(std::move(jit)),
      gdbListener(llvm::JITEventListener::createGDBRegistrationListener()),
      perfListener(nullptr) {
  // Attach the perf listener.
  if (auto *listener = llvm::JITEventListener::createPerfJITEventListener())
    perfListener = listener;
  else if (auto *listener =
               llvm::JITEventListener::createIntelJITEventListener())
    perfListener = listener;
}

ExecutionEngine::~ExecutionEngine() = default;
ExecutionEngine::ExecutionEngine(ExecutionEngine &&other) = default;

/// Add the given module to the execution engine. This slices all public funcs
/// out of the module with their dependencies to generate self-contained object
/// files.
M::ErrorOrSuccess ExecutionEngine::add(ModuleOp module,
                                       ArrayRef<FuncOp> exports,
                                       StringRef libName) {
  // Create the set of symbols to export.
  DenseSet<StringAttr> exportedSymbols;
  for (auto e : exports)
    exportedSymbols.insert(e.getSymNameAttr());

  compiler = std::make_unique<ObjectCompiler>(".kgen_cache", module,
                                              std::move(exportedSymbols));

  // Produce a standalone object for all the exports.
  auto objOr = compiler->produceStandaloneObject(
      TargetInfoAttr::getForHost(module->getContext()), true);
  if (failed(objOr))
    return Error("failed to produce standalone object");

  return add(libName, std::move(*objOr));
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
