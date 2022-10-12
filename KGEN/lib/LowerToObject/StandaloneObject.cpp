//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "LowerToObjectImpl.h"
#include "Support/TempFile.h"
#include "Support/VCSRevision.h"
#include "lld/Common/Driver.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Target/TargetMachine.h"
#include <filesystem>
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
  /// List of llvm::Module and object MemoryBuffers that we have managed to
  /// slice out of the call graph.
  LLVMModuleSet moduleSet;
  ObjectSet objSet;

  /// The compiler instance we're currently using.
  ObjectCompiler &compiler;

  /// Use dense sets to check if we've already seen something.
  DenseSet<StringAttr> seenObjectSymbols, seenLLVMSymbols;

  /// Construct a CallGraphSlicer with a location for error reporting.
  CallGraphSlicer(ObjectCompiler &compiler)
      : compiler(compiler), dbgs(llvm::dbgs()) {}

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
  SliceResult slice(Location loc, StringRef symbol);

  mlir::raw_indented_ostream dbgs;
};
} // namespace

//===----------------------------------------------------------------------===//
// CallGraphSlicer::slice
//===----------------------------------------------------------------------===//

CallGraphSlicer::SliceResult CallGraphSlicer::slice(Location loc,
                                                    StringRef symbol) {
  LLVM_DEBUG(dbgs << "Slicing for " << symbol << "...\n");
  auto objOp = compiler.getSymbolTable().lookup<PrecompiledObjectOp>(symbol);
  // If we don't have an object for this, we've already visited it.
  if (!objOp) {
    if (compiler.getSymbolTable().lookup<FuncOp>(symbol)) {
      LLVM_DEBUG(dbgs << "Already have object for " << symbol << "\n");
      return SliceResult::succeeded;
    }
    return mlir::emitError(loc)
           << "no function named '@" << symbol << "' found";
  }

  // First try to find the object in the cache.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> objOr =
      compiler.getCaches().getObject().find(objOp);
  if (failed(objOr))
    return SliceResult::notInCache;

  // Insert this object into the set if we haven't already.
  if (seenObjectSymbols.insert(objOp.getNameAttr()).second) {
    LLVM_DEBUG(dbgs << "Inserting object for " << symbol << "\n");
    objSet.push_back(std::move(*objOr));
  }

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

  LLVM_DEBUG(dbgs << "Raised to kgen.func:\n" << *funcOr << "\n");

  // Now for each call, slice again.
  auto walkDependency = [&](CallOp call) -> mlir::WalkResult {
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
    SliceResult res = slice(callee->getLoc(), call.getCallee());
    if (failed(res))
      return mlir::WalkResult::interrupt();
    dbgs.unindent();
    LLVM_DEBUG(
        dbgs
        << "//"
           "===---------------------------------------------------------------"
           "-------===//\n");

    return mlir::WalkResult::advance();
  };
  if (funcOr->walk(walkDependency).wasInterrupted())
    return SliceResult::failed;

  return SliceResult::succeeded;
}

/// Combine all the LLVM modules into a single module and emit that to an object
/// file. This is effectively LTO on steroids, as we recover the original LLVMIR
/// and optimize/emit it directly.
/// TODO: We ultimately want to be caching the composite modules.
static FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
constructSingleModule(Location loc, LLVMModuleSet &moduleSet,
                      llvm::TargetMachine &machine) {
  auto &firstModule = moduleSet.front();

  llvm::Linker linker(*firstModule);
  for (auto &llvmModule : llvm::drop_begin(moduleSet)) {
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

  SmallVector<char, 0> objBuf;
  if (failed(compileLLVMToObject(*firstModule, machine, objBuf)))
    return failure();

  // Return a copy of this thing, that's the object file.
  return llvm::MemoryBuffer::getMemBufferCopy({objBuf.data(), objBuf.size()});
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
    if (failed(slicer.slice(loc, symbol)))
      return failure();

  if (slicer.objSet.empty())
    return mlir::emitError(loc) << "no objects found for slicing";

  // If there's only one object then just return it.
  if (slicer.objSet.size() == 1)
    return std::move(slicer.objSet.front());

  // Create the target machine.
  auto machineOr = createTargetMachine(theTarget);
  if (failed(machineOr))
    return emitError(loc) << machineOr.getError();
  std::unique_ptr<llvm::TargetMachine> machine = std::move(*machineOr);

  // If we have all the objects as LLVM modules, then we should invoke the llvm
  // optimizer and lower it to an object file. Then we pass it through the
  // linker to clean up all the symbols we don't want to export.
  if (slicer.haveAllLLVM()) {
    auto modOr = constructSingleModule(loc, slicer.moduleSet, *machine);
    if (mlir::succeeded(modOr)) {
      slicer.objSet.clear();
      slicer.objSet.push_back(std::move(*modOr));
    }
  }

  SmallVector<std::string> tmpFileNames;
  SmallVector<TempFile> tmpFiles;
  for (auto &obj : slicer.objSet) {
    auto fileOr =
        TempFile::create("kgen-standalone-object-input-%%%%%%%%%%%.o");
    if (failed(fileOr))
      return mlir::emitError(loc) << fileOr.getError();

    // Write the object to this temp file.
    llvm::raw_fd_ostream os(fileOr->getFD(), /*shouldClose=*/false);
    LLVM_DEBUG(llvm::dbgs()
               << "Writing " << obj->getBufferSize() << " bytes\n");
    os.write(obj->getBufferStart(), obj->getBufferSize());

    LLVM_DEBUG(llvm::dbgs() << "Keeping file " << fileOr->getPath()
                            << " for debugging\n";
               fileOr->keep());

    // And save the temp file.
    tmpFileNames.push_back(fileOr->getPath().str());
    tmpFiles.push_back(std::move(*fileOr));
  }

  auto outFileOr =
      TempFile::create("kgen-standalone-object-output-%%%%%%%%%%%.o");
  if (failed(outFileOr))
    return mlir::emitError(loc) << outFileOr.getError();

  auto reportFailure = [&]() -> InFlightDiagnostic {
    return mlir::emitError(loc);
  };

  llvm::Triple triple(theTarget.getTriple());

  // Start off the arguments with the tmp file names.
  SmallVector<std::string> args(tmpFileNames.begin(), tmpFileNames.end());

  // Otherwise, use lld's ::link method for whichever target we're on.
  bool worked = true;
  // TODO: We probably also want WASM and COFF
  if (triple.isOSBinFormatELF()) {
    // Add the requested AND public symbols as retained.
    auto retainOr =
        TempFile::create("kgen-standalone-object-retain-syms-%%%%%%%%%%%.txt");
    if (failed(retainOr))
      return mlir::emitError(loc) << retainOr.getError();

    llvm::raw_fd_ostream retainStream(retainOr->getFD(), false);

    for (auto f : cast<ModuleOp>(symtab.getOp()).getOps<FuncOp>()) {
      if (llvm::is_contained(symbols, f.getName())) {
        if (f.getLinkage() != Linkage::Public)
          return mlir::emitError(f.getLoc())
                 << "requested export of private symbol, aborting";

        LLVM_DEBUG(llvm::dbgs()
                   << "Exporting symbol: " << f.getSymName() << "\n");
        retainStream << f.getSymName() << "\n";
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "Keeping version-script for debugging: "
                   << retainOr->getPath() << "\n";
      retainOr->keep();
    });

    args.append({
        "--retain-symbols-file",
        retainOr->getPath().str(),
        "-r", //< Get a relocatable object
        "-o",
        outFileOr->getPath().str(),
    });

    auto ldOr = llvm::sys::findProgramByName("ld");
    if (!ldOr)
      return mlir::emitError(loc)
             << "could not find ld: " << ldOr.getError().message();

    // The first argument must be the program's name.
    SmallVector<StringRef> cstrArgs = {*ldOr};
    for (const auto &a : args) {
      LLVM_DEBUG(llvm::dbgs() << a << "\n");
      cstrArgs.push_back(a);
    }

    std::string err;
    if (llvm::sys::ExecuteAndWait(*ldOr, cstrArgs, /*Env=*/None,
                                  /*Redirects=*/{}, /*SecondsToWait=*/0,
                                  /*MemoryLimit=*/0, /*ErrMsg=*/&err) != 0) {
      worked = false;
      emitError(loc) << err;
    }
  } else if (triple.isOSBinFormatMachO()) {
    // Add the requested AND public symbols as exported.
    for (auto f : cast<ModuleOp>(symtab.getOp()).getOps<FuncOp>()) {
      if (llvm::is_contained(symbols, f.getName())) {
        if (f.getLinkage() != Linkage::Public)
          return mlir::emitError(f.getLoc())
                 << "requested export of private symbol, aborting";

        args.push_back("-exported_symbol");
        args.push_back("_" + f.getName().str());
        LLVM_DEBUG(llvm::dbgs() << "Exporting symbol: " << args.back() << "\n");
      }
    }

    // Append mac-specific arguments to the linker args.
    args.append({
        "-r", //< "-r" means "give me a new relocatable object".
        "-arch",
        triple.getArchName().str(),
        "-flat_namespace",
        "-o",
        outFileOr->getPath().str(),
    });

    auto ldOr = llvm::sys::findProgramByName("ld");
    if (!ldOr)
      return mlir::emitError(loc)
             << "could not find ld: " << ldOr.getError().message();

    // The first argument must be the program's name.
    SmallVector<StringRef> cstrArgs = {*ldOr};
    for (const auto &a : args) {
      LLVM_DEBUG(llvm::dbgs() << a << "\n");
      cstrArgs.push_back(a);
    }

    std::string err;
    if (llvm::sys::ExecuteAndWait(*ldOr, cstrArgs, /*Env=*/None,
                                  /*Redirects=*/{}, /*SecondsToWait=*/0,
                                  /*MemoryLimit=*/0, /*ErrMsg=*/&err) != 0) {
      worked = false;
      emitError(loc) << err;
    }
  } else {
    return reportFailure() << "could not detect target";
  }

  // If something broke, discard the output and print an error.
  if (!worked)
    return reportFailure() << "linking failed";

  // Now, open the output tmp file as a memory buffer.
  auto objFileOr = llvm::MemoryBuffer::getFile(outFileOr->getPath());
  if (!objFileOr)
    return reportFailure() << objFileOr.getError().message();

  // Copy the memory buffer so we have ownership of it and return from this
  // function.
  LLVM_DEBUG(llvm::dbgs() << "Keeping file " << outFileOr->getPath()
                          << " for debugging\n";
             outFileOr->keep(););

  return std::move(*objFileOr);
}

//===----------------------------------------------------------------------===//
// produceStandaloneObject(ModuleOp)
//===----------------------------------------------------------------------===//

FailureOr<std::unique_ptr<llvm::MemoryBuffer>>
ObjectCompiler::produceStandaloneObject() {
  // Collect all the `kgen.precompiled.object`.
  SmallVector<StringRef> objs;
  for (auto obj : module.getOps<PrecompiledObjectOp>())
    if (obj.getLinkage() == Linkage::Public)
      objs.push_back(obj.getName());

  return produceStandaloneObject(objs);
}
