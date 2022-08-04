//===- ExecutionEngine.cpp ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "mlir/IR/Block.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"

using namespace M;
using namespace KGEN;

void ObjectCache::notifyObjectCompiled(const llvm::Module *M,
                                       llvm::MemoryBufferRef Obj) {
  storage[M->getModuleIdentifier()] = llvm::MemoryBuffer::getMemBufferCopy(
      Obj.getBuffer(), Obj.getBufferIdentifier());
}

std::unique_ptr<llvm::MemoryBuffer>
ObjectCache::getObject(const llvm::Module *M) {
  if (auto found = storage.find(M->getModuleIdentifier());
      found != storage.end())
    return llvm::MemoryBuffer::getMemBufferCopy(
        found->second->getBuffer(), found->second->getBufferIdentifier());
  return nullptr;
}

std::unique_ptr<llvm::MemoryBuffer>
ObjectCache::getObject(llvm::StringRef name) {
  if (auto found = storage.find((name + "_module").str());
      found != storage.end())
    return llvm::MemoryBuffer::getMemBufferCopy(
        found->second->getBuffer(), found->second->getBufferIdentifier());
  return nullptr;
}

static ErrorOr<std::unique_ptr<llvm::TargetMachine>> createHostTargetMachine() {
  // Setup the machine properties from the current architecture.
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

ErrorOr<ExecutionEngine> ExecutionEngine::create() {
  ExecutionEngine ee(nullptr);

  // Ensure the native target is initialized.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Create the target machine.
  auto machineOr = createHostTargetMachine();
  if (machineOr.isError())
    return machineOr.takeError();
  ee.targetMachine = std::move(*machineOr);

  // Create the JIT.
  auto jitOr =
      llvm::orc::LLJITBuilder()
          .setCompileFunctionCreator(
              [&](llvm::orc::JITTargetMachineBuilder jtmb)
                  -> llvm::Expected<
                      std::unique_ptr<llvm::orc::IRCompileLayer::IRCompiler>> {
                jtmb.setCodeGenOptLevel(llvm::CodeGenOpt::None);
                auto tm = jtmb.createTargetMachine();
                if (!tm)
                  return tm.takeError();
                return std::make_unique<llvm::orc::TMOwningSimpleCompiler>(
                    std::move(*tm), ee.cache.get());
              })
          .create();
  if (!jitOr)
    return M::Error(llvm::toString(jitOr.takeError()));

  ee.jit = std::move(*jitOr);

  // Resolve symbols that are statically linked in the current process.
  llvm::orc::JITDylib &mainJD = ee.jit->getMainJITDylib();
  mainJD.addGenerator(
      cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          ee.targetMachine->createDataLayout().getGlobalPrefix())));

  return ee;
}

ExecutionEngine::ExecutionEngine(std::unique_ptr<llvm::orc::LLJIT> jit)
    : ctx(std::make_unique<llvm::LLVMContext>()), cache(new KGEN::ObjectCache),
      jit(std::move(jit)) {}

M::ErrorOrSuccess ExecutionEngine::add(mlir::LLVM::LLVMFuncOp kernel) {
  // Short-circuit early if we already have this kernel.
  if (cache->getObject(kernel.getName()))
    return success();

  // Create a new module for this single kernel. This will go away at the end of
  // this function.
  mlir::OwningOpRef<mlir::ModuleOp> singleModule =
      mlir::ModuleOp::create(kernel->getLoc());
  mlir::Block *body = singleModule->getBody(0);
  mlir::OpBuilder builder(body, body->end());

  // Clone any symbols used by this kernel into the module as well.
  mlir::SymbolTable symtab(kernel->getParentOp());
  for (auto call : kernel.getOps<mlir::LLVM::CallOp>()) {
    mlir::Operation *callee = symtab.lookup(call.getCalleeAttr().getValue());
    if (!callee)
      return Error("could not find callee '" + call.getCalleeAttr().getValue() +
                   "' in the current module.");

    builder.clone(*callee);
  }

  // Clone the kernel into this new module. We don't want to remove it from the
  // current module.
  builder.clone(*kernel);

  auto llvmModule = mlir::translateModuleToLLVMIR(
      *singleModule, *ctx.getContext(), (kernel.getName() + "_module").str());

  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(llvm::sys::getDefaultTargetTriple());

  if (auto err = jit->addIRModule({std::move(llvmModule), ctx}))
    return M::Error(toString(std::move(err)));

  return success();
}

ErrorOr<std::unique_ptr<llvm::MemoryBuffer>>
ExecutionEngine::getObject(llvm::StringRef kernel) {
  if (auto mbuf = cache->getObject(kernel))
    return mbuf;

  return Error("could not find kernel '" + kernel +
               "' in cache, please call Executor::addKernel.");
}
