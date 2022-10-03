//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "LowerToObjectImpl.h"
#include "lld/Common/Driver.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Target/TargetMachine.h"
#include <filesystem>

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
  /// List of llvm::Module and object MemoryBuffers that we have managed to
  /// slice out of the call graph.
  LLVMModuleSet moduleSet;
  ObjectSet objSet;

  /// The compiler instance we're currently using.
  ObjectCompiler &compiler;

  /// Use dense sets to check if we've already seen something.
  DenseSet<StringAttr> seenObjectSymbols, seenLLVMSymbols;

  /// Construct a CallGraphSlicer with a location for error reporting.
  CallGraphSlicer(ObjectCompiler &compiler) : compiler(compiler) {}

  /// A slice can result in 3 states. Because the fundamental operation of a
  /// slice includes raising, we have to distinguish between "not in cache" and
  /// "failure". Not having something in the cache is not a failure - it could
  /// be a dylib on-disk that we want to load.
  struct SliceResult {
    enum State {
      notInCache,
      failed,
      succeeded,
    } state;

    /*implicit*/ SliceResult(State s) : state(s) {}
    /*implicit*/ SliceResult(LogicalResult r)
        : state(mlir::failed(r) ? failed : succeeded) {}
    /*implicit*/ SliceResult(InFlightDiagnostic r)
        : SliceResult(LogicalResult(r)) {}

    /*implicit*/ operator LogicalResult() {
      return mlir::failure(state == State::failed);
    }
  };

  /// Returns true if we have the LLVM IR for every symbol we set out to get.
  bool haveAllLLVM() const { return moduleSet.size() == objSet.size(); }

  /// Slice the PrecompiledObjectOp's dependencies out of the IR by recursively
  /// raising it and its callees to gather the whole list of objects we need to
  /// combine together.
  SliceResult slice(StringRef symbol);
};
} // namespace

//===----------------------------------------------------------------------===//
// CallGraphSlicer::slice
//===----------------------------------------------------------------------===//

CallGraphSlicer::SliceResult CallGraphSlicer::slice(StringRef symbol) {
  auto objOp = compiler.getSymbolTable().lookup<PrecompiledObjectOp>(symbol);
  // If we don't have an object for this, we've already visited it.
  if (!objOp)
    return SliceResult::succeeded;

  // First try to find the object in the cache.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objOr =
      compiler.getCaches().getObject().find(objOp);
  if (failed(objOr))
    return SliceResult::notInCache;

  // Insert this object into the set if we haven't already.
  if (seenObjectSymbols.insert(objOp.getNameAttr()).second)
    objSet.push_back(std::move(*objOr));

  // First, we decompile the object.
  FailureOr<PrecompiledLLVMOp> llvmOr = compiler.raiseFromObject(objOp);
  if (failed(llvmOr))
    return SliceResult::failed;

  // Read the LLVM module out of the LLVM cache.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> llvmModuleBufOr =
      compiler.getCaches().getLLVM().find(*llvmOr);
  if (failed(llvmModuleBufOr))
    return emitError(llvmOr->getLoc()) << llvmModuleBufOr.getError();

  // Parse the module to an in-memory object.
  auto moduleOr = llvm::parseBitcodeFile(**llvmModuleBufOr, ctx);
  if (auto err = moduleOr.takeError())
    return emitError(llvmOr->getLoc()) << toString(std::move(err));
  std::unique_ptr<llvm::Module> module = std::move(*moduleOr);

  // Store the module in the moduleSet if we haven't already.
  if (seenLLVMSymbols.insert(llvmOr->getNameAttr()).second)
    moduleSet.push_back(std::move(module));

  // Get the kgen.func out of the LLVM object.
  FailureOr<FuncOp> funcOr = compiler.raiseFromLLVM(*llvmOr);
  if (failed(funcOr))
    return SliceResult::failed;

  // Now for each call, slice again.
  auto walkDependency = [&](CallOp call) -> mlir::WalkResult {
    auto callee = compiler.getSymbolTable().lookup(call.getCallee());
    if (!callee)
      return emitError(call.getLoc())
             << "could not find callee " << call.getCallee();

    // If the callee was already raised then we don't have to do anything.
    if (llvm::isa<FuncOp>(callee))
      return mlir::WalkResult::advance();

    // Now slice out the callers of the callee too.
    SliceResult res = slice(call.getCallee());
    if (failed(res))
      return mlir::WalkResult::interrupt();

    return mlir::WalkResult::advance();
  };
  if (funcOr->walk(walkDependency).wasInterrupted())
    return SliceResult::failed;

  return SliceResult::succeeded;
}

//===----------------------------------------------------------------------===//
// produceStandaloneObject
//===----------------------------------------------------------------------===//

FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
ObjectCompiler::produceStandaloneObject(ArrayRef<StringRef> symbols) {
  // Grab the first one so we can use it for locations, etc.
  Location loc = module.getLoc();
  TargetInfoAttr theTarget =
      (*module.getOps<PrecompiledObjectOp>().begin()).getCompiledFor();

  CallGraphSlicer slicer(*this);

  // Slice all of the precompiled objects into this set.
  for (auto symbol : symbols)
    if (failed(slicer.slice(symbol)))
      return failure();

  // Create the target machine.
  auto machineOr = createTargetMachine(theTarget);
  if (failed(machineOr))
    return emitError(loc) << machineOr.getError();
  std::unique_ptr<llvm::TargetMachine> machine = std::move(*machineOr);

  // If we have all the objects as LLVM modules, then we should invoke the llvm
  // optimizer and lower it to an object file. If there's a duplicate symbol,
  // it's because we copied the symbol to do something specific.
  if (slicer.haveAllLLVM()) {
    auto &firstModule = slicer.moduleSet.front();
    for (auto &llvmModule : llvm::drop_begin(slicer.moduleSet))
      if (llvm::Linker::linkModules(*firstModule, std::move(llvmModule),
                                    llvm::Linker::OverrideFromSrc))
        return emitError(loc) << "could not link LLVM modules together";

    // Mark all symbols as external linkage. We will need to do a better job
    // with this in the near future, but for now this is OK.
    for (auto &f : firstModule->functions())
      f.setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);

    SmallVector<char, 0> objBuf;
    if (failed(compileLLVMToObject(*firstModule, *machine, objBuf)))
      return failure();

    // Return a copy of this thing, that's the object file.
    return llvm::MemoryBuffer::getMemBufferCopy({objBuf.data(), objBuf.size()});
  }

  // TODO: fix this path!
  return emitError(loc) << "TODO: we currently require all objects to "
                           "have LLVM IR in the cache";
}

//===----------------------------------------------------------------------===//
// produceStandaloneObject(ModuleOp)
//===----------------------------------------------------------------------===//

FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
ObjectCompiler::produceStandaloneObject() {
  // Collect all the `kgen.precompiled.object`.
  SmallVector<StringRef> objs;
  for (auto obj : module.getOps<PrecompiledObjectOp>())
    objs.push_back(obj.getName());

  return produceStandaloneObject(objs);
}
