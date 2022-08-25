//===- ExecutionEngine.cpp ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/ErrorOr.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
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

/// Slice a kernel and all its dependencies out of the existing module. This
/// operates using FunctionOpInterface as the 'function' op type so that we can
/// use LLVMFuncOps as well as KGEN::KernelOp and friends. This also uses
/// CallOpInterface to capture all callees.
static ErrorOrSuccess kernelSlicer(mlir::FunctionOpInterface kernel,
                                   OpBuilder &builder,
                                   const mlir::SymbolTable &symtab) {
  for (auto call : kernel.getBody().getOps<mlir::CallOpInterface>()) {
    auto callableForCallee = call.getCallableForCallee();
    if (auto val = callableForCallee.dyn_cast<Value>()) {
      auto err = mlir::emitError(call.getLoc())
                 << "dynamic callee is not supported";
      err.attachNote(val.getLoc()) << "dynamic callee here";
      return Error("dynamic callee is not supported");
    }
    // This is safe because in KGEN all symbols are flattened, and we don't
    // support recursion in KGEN.
    StringAttr calleeRef =
        callableForCallee.get<SymbolRefAttr>().getLeafReference();
    auto callee = symtab.lookup<mlir::FunctionOpInterface>(calleeRef);
    if (!callee || callee.isExternal()) {
      auto error = mlir::emitError(call.getLoc())
                   << "could not find local callee '@" << calleeRef.getValue()
                   << "' in the current module";
      if (callee)
        error.attachNote(callee.getLoc()) << "callee defined here";

      return Error("could not find local callee '@" + calleeRef.getValue() +
                   "' in the current module.");
    }

    if (auto err = kernelSlicer(callee, builder, symtab))
      return err.takeError();

    builder.clone(*callee);
  }
  return success();
}

/// Set up a pass manager with the *ToLLVM passes and run it. This has the
/// effect of taking `module` and converting it fully to LLVM.
static LogicalResult convertToLLVM(ModuleOp module, StringRef name) {
  mlir::PassManager pm(module.getContext());

  pm.addNestedPass<KGEN::KernelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<KGEN::KernelOp>(KGEN::createConvertPOPToLLVMPass());

  pm.addNestedPass<KGEN::KernelOp>(
      mlir::arith::createConvertArithmeticToLLVMPass());
  pm.addNestedPass<KGEN::KernelOp>(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::cf::createConvertControlFlowToLLVMPass());
  pm.addPass(KGEN::createConvertKGENToLLVMPass(name));

  // And finally canonicalize again before running through the JIT.
  pm.addPass(mlir::createCanonicalizerPass());

  return pm.run(module);
}

/// Add the given module to the execution engine. This slices all the kernels
/// out of the module with their dependencies to generate self-contained object
/// files.
// TODO: The slicing -> convert to LLVM -> createJITDylib + compile has natural
//       parallelism that we aren't taking advantage of.
M::ErrorOrSuccess ExecutionEngine::add(mlir::ModuleOp module) {
  // Loop over all the kernels in the module and perform non-destructive
  // slicing, then push them to LLVM IR and compile them to objects.
  for (auto kernel : module.getOps<KGEN::KernelOp>()) {
    // Short-circuit early if we already have this kernel.
    if (cache->hasObject(kernel.getName()))
      return success();

    // Create a new module for this single kernel. This will go away at the end
    // of this function.
    mlir::OwningOpRef<mlir::ModuleOp> singleModule =
        mlir::ModuleOp::create(kernel->getLoc());
    mlir::Block *body = singleModule->getBody(0);
    mlir::OpBuilder builder(body, body->end());

    // Clone any symbols used by this kernel into the module as well.
    mlir::SymbolTable symtab(kernel->getParentOp());

    // Traverse the call graph and clone all the callees into this module.
    if (auto err = kernelSlicer(kernel, builder, symtab))
      return err.takeError();

    // Clone the kernel into this new module. We don't want to remove it from
    // the current module.
    builder.clone(*kernel);

    if (failed(convertToLLVM(*singleModule, kernel.getName())))
      return Error("could not convert kernel '@" + kernel.getName() +
                   "' to LLVM");

    auto llvmModule = mlir::translateModuleToLLVMIR(
        *singleModule, *ctx.getContext(), (kernel.getName() + "_module").str());

    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetMachine->getTargetTriple().normalize());

    // Create a new dylib so that we don't have ODR violations.
    auto dylibOr = jit->createJITDylib(kernel.getName().str());
    if (!dylibOr)
      return M::Error(toString(dylibOr.takeError()));

    // Resolve symbols that are statically linked in the current process.
    dylibOr->addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            jit->getDataLayout().getGlobalPrefix())));

    if (auto err = jit->addIRModule(*dylibOr, {std::move(llvmModule), ctx}))
      return M::Error(toString(std::move(err)));
  }

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
ExecutionEngine::getObject(KGEN::KernelOp kernel) {
  return getObject(kernel.getName());
}
