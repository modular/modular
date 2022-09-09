//===- ExecutionEngine.cpp ------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ExecutionEngine.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENPasses.h"
#include "Support/BlobCache.h"
#include "Support/ErrorOr.h"
#include "Support/IndexToLLVM/IndexToLLVM.h"
#include "Support/VCSRevision.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Analysis/IRSimilarityIdentifier.h"
#include "llvm/ExecutionEngine/ObjectCache.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetSelect.h"
#include <filesystem>

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

namespace M::KGEN::detail {
/// Provides a simple object cache. Users shouldn't be interacting directly with
/// this cache, they should interact with `ExecutionEngine` below.
class ObjectCache : public llvm::ObjectCache {
public:
  template <typename... Args>
  explicit ObjectCache(Args &&...args)
      : storage(std::forward<Args &&>(args)...) {}

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
  std::unique_ptr<llvm::MemoryBuffer> getObject(KernelOp kernel);

  /// Check if the cache has the object corresponding to the given kernel.
  bool hasObject(KernelOp kernel);

  /// Map a KernelOp to an llvm::Module.
  void mapKernelToModule(KernelOp kernel, const llvm::Module *module) {
    auto didEmplace = kernelToModule.try_emplace(kernel, module);
    assert(
        didEmplace.second ||
        (didEmplace.first->getFirst() == kernel &&
         didEmplace.first->getSecond() == module) &&
            "tried to overwrite a kernel/module pair with a new kernel/module "
            "pair");
  }

private:
  /// Lookup the module corresponding to the provided kernel. Returns nullptr if
  /// no such module exists.
  const llvm::Module *getModuleForKernel(KernelOp kernel) const {
    auto found = kernelToModule.find(kernel);
    if (found == kernelToModule.end())
      return nullptr;
    return found->getSecond();
  }

  /// Map of llvm::Module name to compiled object in the form of a
  /// llvm::MemoryBuffer.
  BlobCache<ModulePtrKeyInfo> storage;
  /// Map from a KernelOp to the corresponding LLVM module.
  llvm::DenseMap<KernelOp, const llvm::Module *> kernelToModule;
};
} // namespace M::KGEN::detail

void detail::ObjectCache::notifyObjectCompiled(const llvm::Module *m,
                                               llvm::MemoryBufferRef obj) {
  // report_fatal_error here is not great, but the API doesn't allow us to
  // report an error any other way!
  if (auto err = storage.insert(m, obj))
    llvm::report_fatal_error(err.getError());
}

std::unique_ptr<llvm::MemoryBuffer>
detail::ObjectCache::getObject(const llvm::Module *m) {
  auto found = storage.find(m);
  if (failed(found))
    return nullptr;
  return std::move(*found);
}

std::unique_ptr<llvm::MemoryBuffer>
detail::ObjectCache::getObject(KernelOp kernel) {
  const llvm::Module *module = getModuleForKernel(kernel);
  if (!module)
    return nullptr;

  auto storageFound = storage.find(module);
  if (failed(storageFound))
    return nullptr;
  return std::move(*storageFound);
}

bool detail::ObjectCache::hasObject(KernelOp kernel) {
  const llvm::Module *module = getModuleForKernel(kernel);
  if (!module)
    return false;
  return storage.contains(module);
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
// CompiledKernel implementation
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> CompiledKernel::getObject() {
  if (auto mbuf = cache.getObject(kernel))
    return mbuf;

  return Error("could not find kernel '" + kernel.getName() +
               "' in cache, please call `ExecutionEngine::add`.");
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
    : ctx(std::make_unique<llvm::LLVMContext>()), jit(std::move(jit)) {
  auto basePath =
      llvm::sys::Process::GetEnv("MODULAR_DERIVED_PATH").value_or("");

  std::filesystem::path filepath(basePath);
  filepath /= ".kgen_cache";
  filepath /= MODULAR_VCS_VERSION;

  cache = std::make_unique<KGEN::detail::ObjectCache>(
      getDefaultBackendChain(filepath.string()));
}

ExecutionEngine::~ExecutionEngine() = default;
ExecutionEngine::ExecutionEngine(ExecutionEngine &&other) = default;

/// Slice a kernel and all its dependencies out of the existing module. This
/// operates using FunctionOpInterface as the 'function' op type so that we can
/// use LLVMFuncOps as well as KGEN::KernelOp and friends. This also uses
/// CallOpInterface to capture all callees.
static ErrorOrSuccess kernelSlicer(mlir::FunctionOpInterface kernel,
                                   OpBuilder &builder,
                                   const mlir::SymbolTable &symtab) {
  Optional<Error> error;
  auto extractDependencies = [&](mlir::CallOpInterface call) {
    auto callableForCallee = call.getCallableForCallee();
    if (auto val = callableForCallee.dyn_cast<Value>()) {
      auto err = mlir::emitError(call.getLoc())
                 << "dynamic callee is not supported";
      err.attachNote(val.getLoc()) << "dynamic callee here";
      error = Error("dynamic callee is not supported");
      return WalkResult::interrupt();
    }
    // This is safe because in KGEN all symbols are flattened, and we don't
    // support recursion in KGEN.
    StringAttr calleeRef =
        callableForCallee.get<SymbolRefAttr>().getLeafReference();
    auto callee = symtab.lookup<mlir::FunctionOpInterface>(calleeRef);
    if (!callee || callee.isExternal()) {
      auto err = mlir::emitError(call.getLoc())
                 << "could not find local callee '@" << calleeRef.getValue()
                 << "' in the current module";
      if (callee)
        err.attachNote(callee.getLoc()) << "callee defined here";

      error = Error("could not find local callee '@" + calleeRef.getValue() +
                    "' in the current module.");
      return WalkResult::interrupt();
    }

    if (auto err = kernelSlicer(callee, builder, symtab)) {
      error = err.takeError();
      return WalkResult::interrupt();
    }

    builder.clone(*callee);
    return WalkResult::advance();
  };
  if (kernel->walk(extractDependencies).wasInterrupted())
    return std::move(*error);
  return success();
}

/// Set up a pass manager with the *ToLLVM passes and run it. This has the
/// effect of taking `module` and converting it fully to LLVM.
static LogicalResult convertToLLVM(ModuleOp module, StringRef name) {
  mlir::PassManager pm(module.getContext());

  pm.addNestedPass<KGEN::KernelOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<KGEN::KernelOp>(KGEN::createConvertPOPToLLVMPass());

  pm.addNestedPass<KGEN::KernelOp>(index::createIndexToLLVM());
  // FIXME: We don't necessarily always want to emit opaque wrappers. Split this
  //        code up better because there's 2 semi-separate compilation models
  //        here.
  pm.addPass(KGEN::createConvertKGENToLLVMPass(name, {}, true));
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(KGEN::createConvertSCFToLLVMPass());

  // And finally canonicalize again before running through the JIT.
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createCanonicalizerPass());

  return pm.run(module);
}

/// Add the given module to the execution engine. This slices all the kernels
/// out of the module with their dependencies to generate self-contained object
/// files.
// TODO: The slicing -> convert to LLVM -> createJITDylib + compile has natural
//       parallelism that we aren't taking advantage of.
M::ErrorOrSuccess ExecutionEngine::add(mlir::ModuleOp module,
                                       ArrayRef<KernelOp> only) {
  // Loop over all the kernels in the module and perform non-destructive
  // slicing, then push them to LLVM IR and compile them to objects.
  for (auto kernel : module.getOps<KGEN::KernelOp>()) {
    // If we've added a filter and this kernel isn't one we want, don't deal
    // with it.
    if (!only.empty() && !llvm::is_contained(only, kernel))
      continue;

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
        *singleModule, *ctx.getContext(), kernel.getName());

    // Map this kernel to the llvm::Module pointer we just got. This will allow
    // us to look up objects in the cache with the KernelOp as the key.
    cache->mapKernelToModule(kernel, llvmModule.get());

    // Create a new dylib so that we don't have ODR violations.
    auto dylibOr = jit->createJITDylib(kernel.getName().str());
    if (!dylibOr)
      return M::Error(toString(dylibOr.takeError()));

    // Short-circuit if we already have this kernel. We have to do this here
    // because we key off the module contents, which aren't known until here.
    if (auto mbuf = cache->getObject(kernel)) {
      // Add the object file to the JIT so it can be looked-up later.
      if (auto err = jit->addObjectFile(*dylibOr, std::move(mbuf)))
        return Error(toString(std::move(err)));

      // And hold onto the module in our vector of modules.
      compiledModules.emplace_back(std::move(llvmModule), ctx);
      continue;
    }

    llvmModule->setDataLayout(targetMachine->createDataLayout());
    llvmModule->setTargetTriple(targetMachine->getTargetTriple().normalize());

    // Resolve symbols that are statically linked in the current process.
    dylibOr->addGenerator(
        cantFail(llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
            jit->getDataLayout().getGlobalPrefix())));

    llvm::orc::ThreadSafeModule tsm(std::move(llvmModule), ctx);

    jit->getIRCompileLayer().setNotifyCompiled(
        [=](auto &&r, llvm::orc::ThreadSafeModule tsm) {
          compiledModules.push_back(std::move(tsm));
        });

    // Map the kernel to the module we just created. This pointer should be
    // stable; the JIT just takes ownership of the pointer.
    cache->mapKernelToModule(kernel, tsm.getModuleUnlocked());

    if (auto err = jit->addIRModule(*dylibOr, std::move(tsm)))
      return M::Error(toString(std::move(err)));
  }

  return success();
}

ErrorOr<CompiledKernel> ExecutionEngine::lookup(KGEN::KernelOp kernel) {
  auto *dylib = jit->getJITDylibByName(kernel.getName());
  if (!dylib)
    return Error("could not find JITDylib for " + kernel.getName());

  auto addr = jit->lookup(*dylib, kernel.getName().str());
  if (!addr)
    return M::Error(toString(addr.takeError()));

  return CompiledKernel(addr->toPtr<void *>(), kernel, *cache);
}

ErrorOr<CompiledKernel>
ExecutionEngine::lookupOpaqueWrapper(KGEN::KernelOp kernel) {
  // TODO: The opaque_wrapper attr is added to the llvm.func op, not the
  //       KernelOp so we gotta have a map or something for that - we don't
  //       currently save the LLVM-dialect IR. For now, just suffix it manually.
  //       It'll be in the dylib for the original kernel.
  auto *dylib = jit->getJITDylibByName(kernel.getName());
  if (!dylib)
    return Error("could not find JITDylib for " + kernel.getName());

  auto addr = jit->lookup(*dylib, kernel.getName().str() + "_opaque_wrapper");
  if (!addr)
    return M::Error(toString(addr.takeError()));

  return CompiledKernel(addr->toPtr<void *>(), kernel, *cache);
}
