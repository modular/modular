//===- ExecutionEngine.cpp ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "Support/ErrorOr.h"
#include "mlir/IR/Block.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ObjectCache
//===----------------------------------------------------------------------===//

namespace M::KGEN::detail {
/// Provides a simple object cache. Users shouldn't be interacting directly with
/// this cache, they should interact with `ExecutionEngine` below.
class ObjectCache : public llvm::ObjectCache {
public:
  /// notifyObjectCompiled - Provides a pointer to compiled code for Module M.
  void notifyObjectCompiled(const llvm::Module *m,
                            llvm::MemoryBufferRef obj) override;

  /// Returns a pointer to a newly allocated MemoryBuffer that contains the
  /// object which corresponds with `m`, or `nullptr` if an object is not
  /// available.
  std::unique_ptr<llvm::MemoryBuffer> getObject(const llvm::Module *m) override;
  /// Returns a pointer to a newly allocated MemoryBuffer that contains the
  /// object which corresponds with the kernel named `name`, or `nullptr` if an
  /// object is not available.
  std::unique_ptr<llvm::MemoryBuffer> getObject(llvm::StringRef name);

  /// Check if the cache has the object with the given name.
  bool hasObject(llvm::StringRef name) { return storage.count(name) != 0; }

private:
  /// Map of llvm::Module name to compiled object in the form of a
  /// llvm::MemoryBuffer.
  llvm::StringMap<std::unique_ptr<llvm::MemoryBuffer>> storage;
};
} // namespace M::KGEN::detail

void detail::ObjectCache::notifyObjectCompiled(const llvm::Module *m,
                                               llvm::MemoryBufferRef obj) {
  storage[m->getModuleIdentifier()] = llvm::MemoryBuffer::getMemBufferCopy(
      obj.getBuffer(), obj.getBufferIdentifier());
}

std::unique_ptr<llvm::MemoryBuffer>
detail::ObjectCache::getObject(const llvm::Module *m) {
  auto found = storage.find(m->getModuleIdentifier());
  if (found != storage.end())
    return llvm::MemoryBuffer::getMemBufferCopy(
        found->second->getBuffer(), found->second->getBufferIdentifier());
  return nullptr;
}

std::unique_ptr<llvm::MemoryBuffer>
detail::ObjectCache::getObject(llvm::StringRef name) {
  auto found = storage.find((name + "_module").str());
  if (found != storage.end())
    return llvm::MemoryBuffer::getMemBufferCopy(
        found->second->getBuffer(), found->second->getBufferIdentifier());
  return nullptr;
}

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
      targetTriple, cpu, features.getString(), {}, {}));
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
  ee.targetMachine = std::move(*machineOr);

  auto createCompileFn = [&](llvm::orc::JITTargetMachineBuilder jtmb)
      -> llvm::Expected<
          std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
    jtmb.setCodeGenOptLevel(llvm::CodeGenOpt::Aggressive);
    auto tm = jtmb.createTargetMachine();
    if (!tm)
      return tm.takeError();
    return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(std::move(*tm),
                                                               ee.cache.get());
  };

  // Create the JIT.
  auto jitOr = llvm::orc::LLJITBuilder()
                   .setCompileFunctionCreator(createCompileFn)
                   .create();
  if (!jitOr)
    return M::Error(llvm::toString(jitOr.takeError()));

  ee.jit = std::move(*jitOr);
  return ee;
}

ExecutionEngine::ExecutionEngine(std::unique_ptr<llvm::orc::LLJIT> jit)
    : ctx(std::make_unique<llvm::LLVMContext>()),
      cache(new KGEN::detail::ObjectCache), jit(std::move(jit)) {}

ExecutionEngine::~ExecutionEngine() = default;
ExecutionEngine::ExecutionEngine(ExecutionEngine &&other) = default;

M::ErrorOrSuccess ExecutionEngine::add(mlir::LLVM::LLVMFuncOp kernel) {
  // Short-circuit early if we already have this kernel.
  if (cache->hasObject(kernel.getName()))
    return success();

  // Create a new module for this single kernel. This will go away at the end of
  // this function.
  mlir::OwningOpRef<mlir::ModuleOp> singleModule =
      mlir::ModuleOp::create(kernel->getLoc());
  mlir::Block *body = singleModule->getBody(0);
  mlir::OpBuilder builder(body, body->end());

  // Clone any symbols used by this kernel into the module as well.
  mlir::SymbolTable symtab(kernel->getParentOp());

  // Traverse the call graph and clone all the callees into this module.
  std::function<ErrorOrSuccess(mlir::LLVM::LLVMFuncOp)> dfsCloner =
      [&](mlir::LLVM::LLVMFuncOp func) -> ErrorOrSuccess {
    for (auto call : func.getOps<mlir::LLVM::CallOp>()) {
      auto callee = symtab.lookup<mlir::LLVM::LLVMFuncOp>(
          call.getCalleeAttr().getValue());
      if (!callee || callee.isExternal()) {
        auto error = mlir::emitError(call.getLoc())
                     << "could not find local callee '" << call.getCalleeAttr()
                     << "' in the current module";
        if (callee)
          error.attachNote(callee.getLoc()) << "callee declared here";
        return Error("could not find local callee '" +
                     call.getCalleeAttr().getValue() +
                     "' in the current module.");
      }

      if (auto err = dfsCloner(callee))
        return err.takeError();

      builder.clone(*callee);
    }
    return success();
  };

  if (auto err = dfsCloner(kernel))
    return err.takeError();

  // Clone the kernel into this new module. We don't want to remove it from the
  // current module.
  builder.clone(*kernel);

  auto llvmModule = mlir::translateModuleToLLVMIR(
      *singleModule, *ctx.getContext(), (kernel.getName() + "_module").str());

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetMachine->getTargetTriple().normalize());

  // Create a new dylib so we don't have ODR violations.
  auto dylibOr = jit->createJITDylib(kernel.getName().str());
  if (!dylibOr)
    return M::Error(toString(dylibOr.takeError()));

  // Resolve symbols that are statically linked in the current process.
  dylibOr->addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix())));

  if (auto err = jit->addIRModule(*dylibOr, {std::move(llvmModule), ctx}))
    return M::Error(toString(std::move(err)));

  return success();
}

ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ExecutionEngine::getObject(StringRef kernel) {
  auto *dylib = jit->getJITDylibByName(kernel);
  if (!dylib)
    return Error("could not find JITDylib for " + kernel);

  // Do the lookup to ensure it's compiled. We don't actually care about the
  // address of the result.
  auto addr = jit->lookup(*dylib, kernel);
  if (!addr)
    return M::Error(toString(addr.takeError()));

  if (auto mbuf = cache->getObject(kernel))
    return mbuf;

  return Error("could not find kernel '" + kernel +
               "' in cache, please call Executor::addKernel.");
}

ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ExecutionEngine::getObject(mlir::LLVM::LLVMFuncOp kernel) {
  return getObject(kernel.getName());
}
