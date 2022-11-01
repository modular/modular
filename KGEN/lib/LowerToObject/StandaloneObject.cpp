//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "LowerToObjectImpl.h"
#include "Support/TempFile.h"
#include "Support/TimeProfiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include <utility>

#define DEBUG_TYPE "standalone-object"

using namespace M;
using namespace KGEN;

/// Convenience typedefs for many-templated types.
using LLVMModuleSet = SmallVector<std::unique_ptr<llvm::Module>>;
using ObjectSet = SmallVector<std::unique_ptr<llvm::MemoryBuffer>>;

/// This struct provides the context necessary to provide incremental raising
/// and call graph slicing. Its explicit purpose is to provide a recursive
/// slice, which will walk the call graph and raise anything that isn't already
/// in the IR.
namespace {
struct CallGraphSlicer {
  llvm::LLVMContext ctx;
  /// List of llvm::Modules that we have managed to slice out of the call graph.
  LLVMModuleSet moduleSet;

  /// The compiler instance we're currently using.
  ObjectCompiler &compiler;

  /// Use dense sets to check if we've already seen something.
  DenseSet<StringAttr> seenLLVMSymbols;

  /// Construct a CallGraphSlicer with a location for error reporting.
  CallGraphSlicer(ObjectCompiler &compiler)
      : compiler(compiler), dbgs(llvm::dbgs()) {}

  /// Slice the PrecompiledLLVMOp's dependencies out of the IR by recursively
  /// raising it and its callees to gather the whole list of modules we need to
  /// combine together.
  LogicalResult slice(Location loc, StringRef symbol);

  mlir::raw_indented_ostream dbgs;
};
} // namespace

//===----------------------------------------------------------------------===//
// CallGraphSlicer::slice
//===----------------------------------------------------------------------===//

LogicalResult CallGraphSlicer::slice(Location loc, StringRef symbol) {
  LLVM_DEBUG(dbgs << "Slicing for " << symbol << "...\n");
  auto llvmOp = compiler.getSymbolTable().lookup<PrecompiledLLVMOp>(symbol);

  // If we don't have an llvm op for this, we've already visited it.
  if (!llvmOp) {
    if (compiler.getSymbolTable().lookup<FuncOp>(symbol)) {
      LLVM_DEBUG(dbgs << "Already have LLVM for " << symbol << "\n");
      return success();
    }
    return mlir::emitError(loc)
           << "no function named '@" << symbol << "' found";
  }

  // Read the LLVM module out of the LLVM cache.
  CacheFindResult llvmModuleBufOr = compiler.getCaches().getLLVM().find(llvmOp);
  if (llvmModuleBufOr.isError())
    return emitError(llvmOp.getLoc()) << llvmModuleBufOr.getError();
  // If the LLVM module is not in the cache, then we don't have to continue, but
  // we have gathered the object we wanted so it's not a failure.
  if (!llvmModuleBufOr.hasValue())
    return success();

  // Parse the module to an in-memory object.
  auto moduleOr = llvm::parseBitcodeFile(*llvmModuleBufOr, ctx);
  if (auto err = moduleOr.takeError())
    return emitError(llvmOp.getLoc()) << toString(std::move(err));
  std::unique_ptr<llvm::Module> module = std::move(*moduleOr);

  // Store the module in the moduleSet if we haven't already.
  if (seenLLVMSymbols.insert(llvmOp.getNameAttr()).second)
    moduleSet.push_back(std::move(module));

  // Get the kgen.func out of the LLVM object.
  FailureOr<FuncOp> funcOr = compiler.raiseFromLLVM(llvmOp);
  if (failed(funcOr))
    return failure();

  LLVM_DEBUG(dbgs << "Raised to kgen.func:\n" << *funcOr << "\n");

  // Now for each call, slice again.
  auto walkDependency = [&](CallOp call) -> WalkResult {
    auto callee = compiler.getSymbolTable().lookup(call.getCallee());
    if (!callee)
      return emitError(call.getLoc())
             << "could not find callee " << call.getCallee();

    LLVM_DEBUG(dbgs << "Found callee:\n"; callee->print(dbgs); dbgs << "\n");

    // Now slice out the callers of the callee too.
    LLVM_DEBUG(
        dbgs
        << "//"
           "===---------------------------------------------------------------"
           "-------===//\n");
    dbgs.indent();
    if (failed(slice(callee->getLoc(), call.getCallee())))
      return WalkResult::interrupt();
    dbgs.unindent();
    LLVM_DEBUG(
        dbgs
        << "//"
           "===---------------------------------------------------------------"
           "-------===//\n");

    return WalkResult::advance();
  };
  return failure(funcOr->walk(walkDependency).wasInterrupted());
}

//===----------------------------------------------------------------------===//
// produceStandaloneObject
//===----------------------------------------------------------------------===//

FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
ObjectCompiler::produceStandaloneObject(ArrayRef<StringRef> symbols,
                                        bool isJIT) {
  TimeTraceScope<> traceScope("produce-standalone-object");
  Location loc = module.getLoc();

  // Grab the target information.
  TargetInfoAttr theTarget =
      (*module.getOps<PrecompiledLLVMOp>().begin()).getCompiledFor();

  // Slice all of the precompiled objects into this set.
  CallGraphSlicer slicer(*this);
  for (auto symbol : symbols)
    if (failed(slicer.slice(loc, symbol)))
      return failure();

  // Create the target machine.
  auto machineOr = createTargetMachine(theTarget, isJIT);
  if (failed(machineOr))
    return emitError(loc) << machineOr.getError();
  auto &firstModule = slicer.moduleSet.front();

  // If we have multiple modules, we have to link them together.
  if (slicer.moduleSet.size() > 1) {
    llvm::Linker linker(*firstModule);
    for (auto &llvmModule : llvm::drop_begin(slicer.moduleSet)) {
      // Set to linkonce because otherwise the private symbols get inserted as
      // undefined symbols in the final object, which doesn't make a ton of
      // sense, but there it is.
      for (auto &f : llvmModule->functions())
        if (!f.isIntrinsic() && !f.isDeclarationForLinker() &&
            f.getLinkage() != llvm::GlobalValue::ExternalLinkage)
          f.setLinkage(llvm::GlobalValue::LinkOnceAnyLinkage);

      if (linker.linkInModule(std::move(llvmModule)))
        return emitError(loc) << "could not link LLVM modules together";
    }

    // Erase the "llvm.used" global value, we don't need it because we have a
    // single module and this will stymie inlining in cases where we might
    // actually want it to stay.
    if (llvm::GlobalVariable *used = firstModule->getNamedGlobal("llvm.used"))
      used->eraseFromParent();
  }

  CacheFindResult objOr = getCaches().getComposite().find(&*firstModule);
  if (objOr.hasValue())
    return objOr.takeValue();

  SmallVector<char, 0> objBuf;
  if (failed(compileLLVMToObject(*firstModule, **machineOr, objBuf)))
    return failure();

  // Turn it into a memory buffer so we can put it into the cache.
  auto obj = std::make_unique<llvm::SmallVectorMemoryBuffer>(
      std::move(objBuf), /*RequiresNullTerminator=*/false);

  // Insert the compiled object into the cache.
  if (auto err = getCaches().getComposite().insert(&*firstModule, *obj))
    return mlir::emitError(loc) << err.getError();

  // Return the object itself.
  return std::unique_ptr<llvm::MemoryBuffer>(std::move(obj));
}

//===----------------------------------------------------------------------===//
// produceStandaloneObject(ModuleOp)
//===----------------------------------------------------------------------===//

FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
ObjectCompiler::produceStandaloneObject(bool isJIT) {
  // Collect all of the `kgen.precompiled.llvm`.
  SmallVector<StringRef> objs;
  for (auto op : module.getOps<PrecompiledLLVMOp>())
    objs.push_back(op.getName());

  return produceStandaloneObject(objs, isJIT);
}
