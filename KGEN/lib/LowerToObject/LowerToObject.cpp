//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "BytecodeUtils.h"
#include "LowerToObjectImpl.h"
#include "Support/ErrorOr.h"
#include "Support/TempFile.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SmallVectorMemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"

using namespace M;
using namespace KGEN;

#define DEBUG_TYPE "lower-to-object"

//===----------------------------------------------------------------------===//
// compileLLVMToObject
//===----------------------------------------------------------------------===//

LogicalResult KGEN::compileLLVMToObject(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine,
                                        SmallVectorImpl<char> &buf) {
  module.setDataLayout(targetMachine.createDataLayout());

  llvm::legacy::PassManager passManager;
  llvm::PassManagerBuilder passManagerBuilder;

  // Set up the pass manager builder to populate the passes we want.
  passManagerBuilder.OptLevel = 3;

  // Set up the pass manager and populate it.
  targetMachine.adjustPassManager(passManagerBuilder);
  passManagerBuilder.populateModulePassManager(
      llvm::cast<llvm::PassManagerBase>(passManager));

  // Add passes to emit an object file.
  llvm::raw_svector_ostream objStream(buf);
  targetMachine.addPassesToEmitFile(passManager, objStream, nullptr,
                                    llvm::CGFT_ObjectFile);
  // Run the pass manager to compile the module.
  passManager.run(module);

  return success();
}

//===----------------------------------------------------------------------===//
// createTargetMachine
//===----------------------------------------------------------------------===//

ErrorOr<std::unique_ptr<llvm::TargetMachine>>
KGEN::createTargetMachine(TargetInfoAttr targetInfo, bool isJIT) {
  { // TODO: remove this once we have more cross-compilation capability.
    auto targetTriple = llvm::sys::getDefaultTargetTriple();
    assert(targetInfo.getTriple() == targetTriple &&
           "TODO: target info must match host for now");
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  std::string errorMessage;
  const llvm::Target *target = llvm::TargetRegistry::lookupTarget(
      targetInfo.getTriple().str(), errorMessage);
  if (!target)
    return Error("no target exists for '" + targetInfo.getTriple() +
                 "': " + errorMessage);

  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetInfo.getTriple(), targetInfo.getCpu(), targetInfo.getFeatures(),
      /*Options=*/{},
      /*RM=*/llvm::Reloc::Model::PIC_,
      /*CM=*/None, /*OL=*/llvm::CodeGenOpt::Aggressive, /*JIT=*/isJIT));
  if (!machine)
    return Error("unable to create target machine");

  return machine;
}

//===----------------------------------------------------------------------===//
// ObjectCacheKeyInfo implementation
//===----------------------------------------------------------------------===//

std::string ObjectCacheKeyInfo::hashKey(ObjectCacheKeyInfo::KeyTy key) {
  return std::visit(
      [](auto &&key) -> std::string {
        using T = std::decay_t<decltype(key)>;
        if constexpr (std::is_same_v<T, PrecompiledLLVMOp>) {
          return std::to_string(size_t(llvm::hash_combine(
              key.getSymName(), key.getCompiledFor().hash(), key.getLlvm())));
        } else if constexpr (std::is_same_v<T, llvm::MemoryBufferRef>) {
          return std::to_string(size_t(llvm::hash_value(key.getBuffer())));
        } else {
          return key.getObject().str();
        }
      },
      key);
}

//===----------------------------------------------------------------------===//
// lowerToObject implementation
//===----------------------------------------------------------------------===//

/// This pulls in the LLVM IR for the `kgen.precompiled.llvm` from `llvmCache`,
/// and replaces the op with a `kgen.precompiled.object`. This also stores the
/// compiled object in `cache` and stores a back-pointer from the precompiled
/// object to the precompiled LLVM in `backtrackCache`.
FailureOr<PrecompiledObjectOp>
ObjectCompiler::lowerToObject(PrecompiledLLVMOp func, bool isJIT) {
  // So first, check if the result is already in the cache.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> precompiledOr =
      caches.getObject().find(func);
  if (succeeded(precompiledOr)) {
    auto precompiled = replaceSymbolFromBytecode(func, symtab, **precompiledOr);
    if (succeeded(precompiled))
      return llvm::cast<PrecompiledObjectOp>(*precompiled);
  }

  // Read the LLVM module out of the LLVM cache.
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> llvmModuleBufOr =
      caches.getLLVM().find(func);
  if (failed(llvmModuleBufOr))
    return mlir::emitError(func->getLoc()) << llvmModuleBufOr.getError();

  // Parse the module to an in-memory object.
  llvm::LLVMContext ctx;
  llvm::Expected<std::unique_ptr<llvm::Module>> moduleOr =
      llvm::parseBitcodeFile(**llvmModuleBufOr, ctx);
  if (auto err = moduleOr.takeError())
    return mlir::emitError(func->getLoc()) << toString(std::move(err));
  std::unique_ptr<llvm::Module> llvmModule = std::move(*moduleOr);

  // The `kgen.precompiled.llvm` has a compiledFor attr that must be the same as
  // the one we use.
  TargetInfoAttr targetInfo = func.getCompiledFor();

  // Create the target machine.
  ErrorOr<std::unique_ptr<llvm::TargetMachine>> machineOr =
      createTargetMachine(targetInfo, isJIT);
  if (failed(machineOr))
    return mlir::emitError(func->getLoc()) << machineOr.getError();
  std::unique_ptr<llvm::TargetMachine> machine = std::move(*machineOr);

  // TODO: We should really just keep the binary code and discard the other
  //       object file trappings, then assemble everything together later.
  SmallVector<char, 0> objBuf;
  if (failed(compileLLVMToObject(*llvmModule, *machine, objBuf)))
    return failure();

  // Turn it into a memory buffer so we can put it into the cache.
  auto objectMemBuf = llvm::SmallVectorMemoryBuffer(
      std::move(objBuf), /*RequiresNullTerminator=*/false);

  // If we're debugging, save the object file for the symbol.
  LLVM_DEBUG({
    auto fileOr = TempFile::create("lower-to-object-%%%%%%%%%%%.o");
    if (failed(fileOr))
      return mlir::emitError(func->getLoc()) << fileOr.getError();

    // Write the object to this temp file.
    llvm::raw_fd_ostream os(fileOr->getFD(), /*shouldClose=*/false);

    llvm::dbgs() << "Keeping file " << fileOr->getPath() << " for symbol "
                 << func.getSymName() << " for debugging, writing "
                 << objectMemBuf.getBufferSize() << " bytes\n";
    os.write(objectMemBuf.getBufferStart(), objectMemBuf.getBufferSize());
    fileOr->keep();
  });

  // Now we have to stuff it into the cache. We're going to key off of the
  // object file itself and store that, then place the key into the
  // `kgen.precompiled.object`.
  ErrorOr<std::string> keyOr =
      caches.getObject().insert(objectMemBuf, objectMemBuf);
  if (failed(keyOr))
    return mlir::emitError(func->getLoc()) << keyOr.getError();
  std::string keyHash = std::move(*keyOr);

  // Now we can create the new op at the location of the old op.
  OpBuilder b(func->getContext());

  // Remove the previous function from the symbol table.
  symtab.remove(func);

  // Create the new op.
  auto newOp = b.create<PrecompiledObjectOp>(func->getLoc(), func, keyHash);

  // Finally, we'll also cache from this new op to the existing
  // `kgen.precompiled.llvm`.
  std::string buf;
  llvm::raw_string_ostream stream(buf);

  // Cache from the func to this new op so we can skip doing this in the future.
  mlir::writeBytecodeToFile(newOp, stream);
  auto precompiledBuf = llvm::MemoryBuffer::getMemBuffer(stream.str());
  if (auto err = caches.getObject().insert(func, *precompiledBuf))
    return mlir::emitError(func->getLoc()) << err.getError();

  // Make sure to clear out the stream.
  stream.str().clear();
  mlir::writeBytecodeToFile(func, stream);
  auto funcBuf = llvm::MemoryBuffer::getMemBuffer(stream.str());

  // Insert into the raising cache.
  if (auto err = caches.getRaising().insert(newOp, *funcBuf))
    return mlir::emitError(func->getLoc()) << err.getError();

  // RAUW (replace all uses with) and delete the original func.
  symtab.insert(newOp, ++Block::iterator(func));
  func.erase();

  // And return the new op we just created.
  return newOp;
}

//===----------------------------------------------------------------------===//
// raiseFromObject
//===----------------------------------------------------------------------===//

/// Backtrack to a `kgen.precompiled.llvm`.
FailureOr<PrecompiledLLVMOp>
ObjectCompiler::raiseFromObject(PrecompiledObjectOp precompiled) {
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> llvmBufOr =
      caches.getRaising().find(precompiled);
  if (failed(llvmBufOr))
    return mlir::emitError(precompiled->getLoc()) << llvmBufOr.getError();

  // It was in the cache, so do the replacement.
  FailureOr<Operation *> func =
      replaceSymbolFromBytecode(precompiled, symtab, **llvmBufOr);
  if (failed(func))
    return failure();

  return llvm::cast<PrecompiledLLVMOp>(*func);
}

//===----------------------------------------------------------------------===//
// lowerAllFuncsToObject
//===----------------------------------------------------------------------===//

LogicalResult ObjectCompiler::lowerAllFuncsToObject(TargetInfoAttr target,
                                                    bool isJIT) {
  for (auto f : llvm::make_early_inc_range(module.getOps<FuncOp>())) {
    auto llvmFuncOr = lowerToLLVM(f, target);
    if (failed(llvmFuncOr))
      return mlir::emitError(f->getLoc()) << "lowering to llvm failed";
    if (failed(lowerToObject(*llvmFuncOr, isJIT)))
      return mlir::emitError(llvmFuncOr->getLoc())
             << "lowering to object failed";
  }
  return success();
}
