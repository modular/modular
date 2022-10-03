//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "BytecodeUtils.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LowerToObject.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// Hashers
//===----------------------------------------------------------------------===//

/// The hash consists of the OperationName, the input types, the output types,
/// and the attributes.
static llvm::hash_code hashNoRegionOperation(Operation *op) {
  auto hashTypeOrAttr = [&](auto t) {
    llvm::SmallString<64> tmp;
    llvm::raw_svector_ostream stringStream(tmp);
    stringStream << t;
    return llvm::hash_value(stringStream.str());
  };

  llvm::hash_code opHash = llvm::hash_value(op->getName().getStringRef());

  for (Type t : op->getOperandTypes())
    opHash = llvm::hash_combine(opHash, hashTypeOrAttr(t));
  for (Type t : op->getResultTypes())
    opHash = llvm::hash_combine(opHash, hashTypeOrAttr(t));

  for (auto attr : op->getAttrs())
    opHash = llvm::hash_combine(opHash, attr.getName().getValue(),
                                hashTypeOrAttr(attr.getValue()));

  return opHash;
}

/// The hash consists of the symbol name, the signature and any attrs on the op,
/// and the body.
static std::string hashOpWithRegions(FuncOp f) {
  llvm::hash_code opHash = 0;
  // Hash the body of the operation.
  f.walk([&](Operation *op) { opHash = hashNoRegionOperation(op); });
  return std::to_string(size_t(opHash));
}

static std::string hashLLVMModule(llvm::Module *m) {
  // Make sure we hash the data layout!
  llvm::hash_code moduleHash = llvm::hash_value(m->getDataLayoutStr());
  for (auto &func : m->functions()) {
    moduleHash = llvm::hash_combine(moduleHash, func.getName(),
                                    func.getReturnType()->getTypeID());
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

//===----------------------------------------------------------------------===//
// LLVMCacheKeyInfo implementation
//===----------------------------------------------------------------------===//

std::string LLVMCacheKeyInfo::hashKey(LLVMCacheKeyInfo::KeyTy key) {
  return std::visit(
      [](auto &&key) -> std::string {
        using T = std::decay_t<decltype(key)>;
        if constexpr (std::is_same_v<T, FuncOp>)
          return hashOpWithRegions(key);
        else if constexpr (std::is_same_v<T, llvm::Module *>)
          return hashLLVMModule(key);
        else
          return key.getLlvm().str();
      },
      key);
}

//===----------------------------------------------------------------------===//
// RaisingCacheKeyInfo implementation
//===----------------------------------------------------------------------===//

std::string RaisingCacheKeyInfo::hashKey(RaisingCacheKeyInfo::KeyTy key) {
  return std::visit(
      [](auto &&key) -> std::string {
        using T = std::decay_t<decltype(key)>;
        if constexpr (std::is_same_v<T, PrecompiledLLVMOp>) {
          return std::to_string(size_t(hashNoRegionOperation(key)));
        } else {
          return std::to_string(size_t(hashNoRegionOperation(key)));
        }
      },
      key);
}

//===----------------------------------------------------------------------===//
// lowerToLLVM implementation
//===----------------------------------------------------------------------===//

/// Slice a func and all its dependencies out of the existing module. This
/// operates using FunctionOpInterface as the 'function' op type so that we can
/// use LLVMFuncOps as well as KGEN::FuncOp and friends. This also uses
/// CallOpInterface to capture all callees.
static ErrorOrSuccess funcSlicer(mlir::FunctionOpInterface func,
                                 mlir::SymbolTable &sliceSymtab,
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
    if (!callee || (!isa<PrecompiledLLVMOp, PrecompiledObjectOp>(callee) &&
                    callee.isExternal())) {
      auto err = mlir::emitError(call.getLoc())
                 << "could not find local callee '@" << calleeRef.getValue()
                 << "' in the current module";
      if (callee)
        err.attachNote(callee.getLoc()) << "callee defined here";

      error = Error("could not find local callee '@" + calleeRef.getValue() +
                    "' in the current module.");
      return WalkResult::interrupt();
    }

    // Don't copy the function if it already was.
    if (sliceSymtab.lookup(calleeRef))
      return WalkResult::advance();

    if (auto err = funcSlicer(callee, sliceSymtab, symtab)) {
      error = err.takeError();
      return WalkResult::interrupt();
    }

    // Mark copied dependencies as private.
    Operation *dependency = callee.clone();
    SymbolTable::setSymbolVisibility(dependency,
                                     SymbolTable::Visibility::Private);
    sliceSymtab.insert(dependency);
    return WalkResult::advance();
  };
  if (func->walk(extractDependencies).wasInterrupted())
    return std::move(*error);
  return success();
}

static LogicalResult convertToLLVM(ModuleOp module, StringRef name) {
  mlir::PassManager pm(module.getContext());
  LowerToLLVMOptions options;

  pm.addPass(createLowerZAPToPOPPass());

  if (!name.empty())
    options.topLevelKernel = name;

  options.emitOpaqueWrappers = true;
  buildLowerToLLVMPipeline(pm, options);
  return pm.run(module);
}

/// This compiles a given `kgen.func` to LLVM IR, and then stores the LLVM IR
/// and the function itself in the two caches. It also replaces `func` in the IR
/// with a new `kgen.precompiled.llvm` that is returned for convenience.
FailureOr<PrecompiledLLVMOp>
ObjectCompiler::lowerToLLVM(FuncOp func, TargetInfoAttr target) {
  // So first, check if the result is already in the cache.
  auto precompiledOr = caches.getLLVM().find(func);
  if (succeeded(precompiledOr)) {
    auto precompiled = replaceSymbolFromBytecode(func, symtab, **precompiledOr);
    if (succeeded(precompiled))
      return llvm::cast<PrecompiledLLVMOp>(*precompiled);
  }

  // Create a new module for this single func. This will go away at the end
  // of this function.
  mlir::OwningOpRef<mlir::ModuleOp> singleModule =
      mlir::ModuleOp::create(func->getLoc());

  // Clone any symbols used by this func into the module as well.
  mlir::SymbolTable sliceSymtab(*singleModule);

  // Traverse the call graph and clone all the callees into this module.
  if (auto err = funcSlicer(func, sliceSymtab, symtab))
    return mlir::emitError(func->getLoc()) << err.getError();

  // Clone the func into this new module. We don't want to remove it from
  // the current module.
  sliceSymtab.insert(func.clone());

  // Only generate wrappers for the func if it's public.
  if (failed(convertToLLVM(*singleModule,
                           /*name=*/func.isPublic() ? func.getName() : "")))
    return failure();

  // Turn the thing into an LLVM module.
  llvm::LLVMContext ctx;
  auto llvmModule =
      mlir::translateModuleToLLVMIR(*singleModule, ctx, func.getName());

  std::string bytes;
  llvm::raw_string_ostream stream(bytes);
  llvm::WriteBitcodeToFile(*llvmModule, stream);

  // Get the memory buffer and write it into the cache.
  auto moduleBuf = llvm::MemoryBuffer::getMemBuffer(stream.str());
  auto keyOr = caches.getLLVM().insert(&*llvmModule, *moduleBuf);
  if (failed(keyOr))
    return mlir::emitError(func->getLoc()) << keyOr.getError();

  std::string keyHash = std::move(*keyOr);

  // Now we can create the new op at the location of the old op.
  OpBuilder b(func->getContext());

  // Remove the previous function from the symbol table.
  symtab.remove(func);

  // Create the new op.
  auto newOp =
      b.create<PrecompiledLLVMOp>(func->getLoc(), func, target, keyHash);

  // Cache from the func to this new op so we can skip doing this in the future.
  stream.str().clear();
  mlir::writeBytecodeToFile(newOp, stream);
  auto precompiledBuf = llvm::MemoryBuffer::getMemBuffer(stream.str());
  if (auto err = caches.getLLVM().insert(func, *precompiledBuf))
    return mlir::emitError(func->getLoc()) << err.getError();

  // Finally, we'll also cache from this new op to the existing
  // function.
  stream.str().clear();
  mlir::writeBytecodeToFile(func, stream);
  auto funcBuf = llvm::MemoryBuffer::getMemBuffer(stream.str());
  if (auto err = caches.getRaising().insert(newOp, *funcBuf))
    return mlir::emitError(func->getLoc()) << err.getError();

  // RAUW and delete the original func.
  symtab.insert(newOp, ++Block::iterator(func));
  func.erase();

  // And return the new op we just created.
  return newOp;
}

//===----------------------------------------------------------------------===//
// raiseFromLLVM
//===----------------------------------------------------------------------===//

/// Backtrack up the compilation stack and get back the function that was used
/// to generate the `kgen.precompiled.llvm`.
FailureOr<FuncOp> ObjectCompiler::raiseFromLLVM(PrecompiledLLVMOp precompiled) {
  ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> funcBufOr =
      caches.getRaising().find(precompiled);
  if (failed(funcBufOr))
    return mlir::emitError(precompiled->getLoc()) << funcBufOr.getError();

  // It was in the cache, so do the replacement.
  FailureOr<Operation *> func =
      replaceSymbolFromBytecode(precompiled, symtab, **funcBufOr);
  if (failed(func))
    return failure();

  return llvm::cast<FuncOp>(*func);
}

//===----------------------------------------------------------------------===//
// EmitLLVMPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_EMITLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
class EmitLLVMPass : public M::KGEN::impl::EmitLLVMBase<EmitLLVMPass> {
public:
  using EmitLLVMBase::EmitLLVMBase;

  void runOnOperation() override;
};
} // namespace

void EmitLLVMPass::runOnOperation() {
  ObjectCompiler compiler(".kgen_cache", getOperation());
  TargetInfoAttr target = TargetInfoAttr::getForHost(&getContext());
  // Lower all functions to LLVM.
  for (auto func : llvm::make_early_inc_range(getOperation().getOps<FuncOp>()))
    if (failed(compiler.lowerToLLVM(func, target)))
      return signalPassFailure();

  // We might have an output file.
  std::unique_ptr<llvm::ToolOutputFile> outputFile = nullptr;
  if (!output.empty()) {
    std::string err;
    outputFile = mlir::openOutputFile(output.getValue(), &err);
    if (!outputFile) {
      mlir::emitError(getOperation()->getLoc()) << err;
      return signalPassFailure();
    }
  }

  // Get the compiled modules and print each one.
  llvm::LLVMContext ctx;
  SmallVector<std::unique_ptr<llvm::Module>> modules;
  for (auto llvm : getOperation().getOps<PrecompiledLLVMOp>()) {
    auto moduleOr = compiler.getCaches().getLLVM().find(llvm);
    if (failed(moduleOr)) {
      mlir::emitError(llvm.getLoc()) << moduleOr.getError();
      return signalPassFailure();
    }

    auto llvmModuleOr = llvm::parseBitcodeFile(**moduleOr, ctx);
    if (auto err = llvmModuleOr.takeError()) {
      mlir::emitError(llvm->getLoc()) << toString(std::move(err));
      return signalPassFailure();
    }
    modules.push_back(std::move(*llvmModuleOr));
  }

  auto &firstModule = modules.front();
  for (auto &m : llvm::drop_begin(modules))
    if (llvm::Linker::linkModules(*firstModule, std::move(m),
                                  llvm::Linker::OverrideFromSrc)) {
      mlir::emitError(getOperation().getLoc())
          << "could not link LLVM modules together";
      return signalPassFailure();
    }

  if (outputFile) {
    firstModule->print(outputFile->os(), nullptr);
    outputFile->keep();
    return;
  }

  firstModule->print(llvm::outs(), nullptr);
}
