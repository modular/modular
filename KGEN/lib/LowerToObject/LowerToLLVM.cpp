//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENPasses.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Runtime/Algorithms.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace M;
using namespace KGEN;
using namespace Cache;

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
static std::string hashOpWithRegions(Operation *f) {
  llvm::hash_code opHash = 0;
  // Hash the body of the operation.
  f->walk([&](Operation *op) {
    opHash = llvm::hash_combine(opHash, hashNoRegionOperation(op));
  });
  return std::to_string(size_t(opHash));
}

//===----------------------------------------------------------------------===//
// LLVMCacheKeyInfo implementation
//===----------------------------------------------------------------------===//

std::string LLVMCacheKeyInfo::hashKey(ModuleOp key) {
  return hashOpWithRegions(key);
}

//===----------------------------------------------------------------------===//
// CompositeObjectCacheKeyInfo implementation
//===----------------------------------------------------------------------===//

std::string CompositeObjectCacheKeyInfo::hashKey(ModuleOp key) {
  return hashOpWithRegions(key);
}

//===----------------------------------------------------------------------===//
// lowerToLLVM implementation
//===----------------------------------------------------------------------===//

static LogicalResult convertToLLVM(ModuleOp module) {
  mlir::PassManager pm(module.getContext());
  LowerToLLVMOptions options;
  pm.addPass(createLowerZAPToPOP());
  buildLowerToLLVMPipeline(pm, options);
  return pm.run(module);
}

//===----------------------------------------------------------------------===//
// lowerAllFuncsToLLVM
//===----------------------------------------------------------------------===//

std::unique_ptr<llvm::Module>
ObjectCompiler::lowerKGENToLLVM(ModuleOp module, llvm::LLVMContext &ctx) {
  if (failed(convertToLLVM(module)))
    return nullptr;

  // Turn the thing into an LLVM module.
  return mlir::translateModuleToLLVMIR(module, ctx);
}

std::unique_ptr<llvm::Module>
ObjectCompiler::lowerAllFuncsToLLVM(llvm::LLVMContext &ctx) {
  OwningOpRef<ModuleOp> singleModule = produceStandaloneModule();
  auto foundModule = caches.getLLVM().find(*singleModule);
  // TODO: this is making an async process sync - fix this!
  LLCL::await(foundModule);
  if (foundModule->hasValue()) {
    BufferRef moduleBuf = foundModule->takeValue();
    // Get the composite module.
    std::unique_ptr<llvm::MemoryBuffer> mbuf =
        llvm::MemoryBuffer::getMemBuffer(moduleBuf->getBuffer());
    auto llvmModuleOr = llvm::parseBitcodeFile(*mbuf, ctx);
    if (auto err = llvmModuleOr.takeError()) {
      mlir::emitError(singleModule->getLoc()) << toString(std::move(err));
      return nullptr;
    }
    std::unique_ptr<llvm::Module> llvmModule = std::move(*llvmModuleOr);
    return llvmModule;
  }

  auto llvmModule = lowerKGENToLLVM(*singleModule, ctx);
  if (!llvmModule)
    return nullptr;

  WriteableBufferRef stream = WriteableBuffer::get();
  llvm::WriteBitcodeToFile(*llvmModule, *stream);

  // Get the memory buffer and write it into the cache.
  auto keyOr = caches.getLLVM().insert(*singleModule, std::move(stream));
  // TODO: this is making an async process sync - fix this!
  LLCL::await(keyOr);
  if (failed(*keyOr)) {
    mlir::emitError(singleModule->getLoc()) << keyOr->getError();
    return nullptr;
  }

  return llvmModule;
}

//===----------------------------------------------------------------------===//
// EmitLLVMPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_EMITLLVM
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

// TODO: delete this in favor of passing in an LLCL runtime to the pass.
static Runtime getDefaultRuntime() {
  return {LLCL::createLeakCheckAllocator(LLCL::createMallocAllocator()),
          LLCL::createSingleThreadWorkQueue(), llvm::StringLiteral(__FILE__)};
}

namespace {
class EmitLLVMPass : public M::KGEN::impl::EmitLLVMBase<EmitLLVMPass> {
public:
  using EmitLLVMBase::EmitLLVMBase;

  void runOnOperation() override;
};
} // namespace

void EmitLLVMPass::runOnOperation() {
  Runtime rt = getDefaultRuntime();
  ObjectCompiler compiler(rt, ".kgen_cache", getOperation());
  // Lower all functions to LLVM.
  llvm::LLVMContext ctx;
  auto llvmModule = compiler.lowerAllFuncsToLLVM(ctx);
  if (!llvmModule)
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

  if (outputFile) {
    llvmModule->print(outputFile->os(), nullptr);
    outputFile->keep();
    return;
  }

  llvmModule->print(llvm::outs(), nullptr);
}
