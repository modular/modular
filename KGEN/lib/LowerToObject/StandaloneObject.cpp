//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LowerToObjectImpl.h"
#include "Support/Compiler/MLIRDenseAttrStorage.h"
#include "Support/TempFile.h"
#include "Support/TimeProfiler.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/BLAKE3.h"
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
using namespace Cache;

//===----------------------------------------------------------------------===//
// produceStandaloneModule
//===----------------------------------------------------------------------===//

/// Slice the dependencies of an operation out of the existing module into the
/// self-contained slice module.
static void sliceDependencies(Operation *op, SymbolTable &sliceSymtab,
                              const SymbolTable &symtab) {
  // Extract a dependency from the IR parent module and place it into the slice
  // module if it does not already exist. If a symbol was copied, return it.
  auto extractDependency = [&](StringAttr name) -> Operation * {
    // Don't copy the symbol if it is already copied.
    if (sliceSymtab.lookup(name))
      return nullptr;

    Operation *symbol = symtab.lookup(name);
    // If the symbol reference attribute doesn't reference a symbol, ignore it.
    // Missing symbol references are caught by the verifier.
    if (!symbol)
      return nullptr;

    // Clone the symbol into the new symbol table.
    Operation *copy = symbol->clone();
    sliceSymtab.insert(copy);
    return copy;
  };

  mlir::AttrTypeWalker walker;
  walker.addWalk([&](Type type) {
    if (auto ref = dyn_cast<DeclRefType>(type)) {
      // Recurse on the type declaration.
      if (Operation *decl = extractDependency(ref.getName()))
        sliceDependencies(decl, sliceSymtab, symtab);
    }
  });
  auto extractDependencies = [&](Operation *op) {
    // Extract references to type declarations.
    walker.walk(op->getAttrDictionary());
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions())
      for (Type type : region.getArgumentTypes())
        walker.walk(type);

    // Extract references to functions. Mark copied functions as module private
    // and recurse.
    StringAttr ref =
        llvm::TypeSwitch<Operation *, StringAttr>(op)
            .Case<CallOp, AddressOfOp>([&](auto op) {
              return op.getCalleeSymbol().getRootReference();
            })
            .Case([&](ParamConstantOp op) {
              if (auto symbol = dyn_cast<SymbolConstantAttr>(op.getValue()))
                return symbol.getSymbol().getRootReference();
              return StringAttr();
            })
            .Default({});
    if (ref) {
      Operation *symbol = extractDependency(ref);
      if (auto func = dyn_cast_if_present<FuncOp>(symbol))
        sliceDependencies(func, sliceSymtab, symtab);
    }
  };
  op->walk(extractDependencies);
}

OwningOpRef<ModuleOp> ObjectCompiler::produceStandaloneModule() {
  // Create a new module for these funcs. This will go away at the end
  // of this function.
  OwningOpRef<ModuleOp> singleModule = ModuleOp::create(module->getLoc());

  // Propagate the target info.
  TargetInfoAttr target = getTargetInfo(module);
  assert(target && "module to compile is missing target specification");
  setTargetInfo(*singleModule, target);

  // Create a new symbol table for the sliced module.
  SymbolTable sliceSymtab(*singleModule);

  // Re-export exported functions.
  auto builder = OpBuilder::atBlockBegin(singleModule->getBody());

  SmallVector<FlatSymbolRefAttr> exportedSymbolVec;
  for (auto [sym, alias] : exportedSymbols) {
    builder.create<ExportOp>(module->getLoc(), FlatSymbolRefAttr::get(sym),
                             alias);
    auto func = symtab.lookup<FuncOp>(sym);
    assert(func && "Unknown exported symbol");

    // Traverse the call graph and clone all the callees into this module.
    sliceDependencies(func, sliceSymtab, symtab);

    // Clone the func into this new module. We don't want to remove it from
    // the current module.
    if (!sliceSymtab.lookup(sym))
      sliceSymtab.insert(func.clone());
  }

  return singleModule;
}

//===----------------------------------------------------------------------===//
// produceStandaloneObject
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef> ObjectCompiler::produceStandaloneObject(bool isJIT) {
  TimeTraceScope<> traceScope("produce-standalone-object");

  // Perform a cache aware transformation to translate the module to an object
  // file.
  llvm::LLVMContext ctx;
  auto runTransformation = [&](Operation *op, WriteableBufferRef buf,
                               LLCL::AnyAsyncValueRef chain) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
    chain.andThenSync([this, op, isJIT, &ctx, output = output.copy(),
                       buf = buf.copy()]() mutable {
      auto llvmModule = lowerAllFuncsToLLVM(ctx, cast<ModuleOp>(op), isJIT);
      if (!llvmModule) {
        return std::move(output).setToError(LLCL::getMLIRDiagnostic(
            "failed to lower module to LLVM IR for object compilation",
            op->getLoc()));
      }

      // Create the target machine.
      auto machineOr = createTargetMachine(options, isJIT);
      if (failed(machineOr)) {
        return std::move(output).setToError(
            LLCL::getMLIRDiagnostic(machineOr.takeError(), op->getLoc()));
      }

      // Set the data layout on the module.
      llvmModule->setDataLayout((*machineOr)->createDataLayout());

      // Set all external and defined functions to hidden visibility.
      for (llvm::Function &func : llvmModule->getFunctionList())
        if (func.hasExternalLinkage() && !func.empty())
          func.setVisibility(llvm::GlobalValue::HiddenVisibility);

      // Lower the LLVM to an object file.
      if (failed(compileLLVMToObject(*llvmModule, **machineOr, *buf))) {
        return std::move(output).setToError(LLCL::getMLIRDiagnostic(
            "failed to lower LLVM IR to object file", op->getLoc()));
      }
      std::move(output).emplace(buf.copy());
    });
    return output;
  };
  auto onCacheHit = [](Operation *op, BufferRef buf) { return buf.copy(); };

  WriteableBufferRef produceStandaloneObjectKey = WriteableBuffer::get();
  options.print(*produceStandaloneObjectKey << "produceStandaloneObject(");
  *produceStandaloneObjectKey << ")";

  OwningOpRef<ModuleOp> slicedModule = produceStandaloneModule();
  auto output = cachedTransform(
      *slicedModule, transformCache.copy(),
      LLCL::AsyncValueRef<Chain>::createReady(runtime),
      std::move(produceStandaloneObjectKey), runTransformation, onCacheHit);
  await(output);

  if (output.isError())
    return {std::move(output.takeDiagnostic().getMessage())};
  return {std::move(output.get<BufferRef>())};
}

ErrorOr<ElementsAttr>
ObjectCompiler::produceStandaloneObjectAttr(TargetInfoAttr target, bool isJIT) {
  auto bufferOr = produceStandaloneObject(isJIT);
  if (bufferOr.isError())
    return bufferOr.takeError();
  BufferRef buffer = bufferOr.takeValue();

  // Get the standalone object key to use as the object name.
  WriteableBufferRef produceStandaloneObjectKey = WriteableBuffer::get();
  options.print(*produceStandaloneObjectKey << "produceStandaloneObject(");
  *produceStandaloneObjectKey << ")";
  mlir::writeBytecodeToFile(module.getOperation(), *produceStandaloneObjectKey);
  // Hash it so the object name isn't enormous.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)produceStandaloneObjectKey->getBufferStart(),
               produceStandaloneObjectKey->getBufferSize()));

  // Produce a DenseResourceElementsAttr from the object file.
  auto resourceManager =
      DenseResourceElementsHandle::getManagerInterface(target.getContext());

  // Pretend this is a "tensor" of data.
  // TODO (#6986) It would be much nicer if we didn't have to clone this data
  //   and we could just reference the data already in the CAS. That would also
  //   prevent us from having to hash the module above.
  return getAttrForTensorData(
      RankedTensorType::get(
          {(int64_t)buffer->getBufferSize()},
          IntegerType::get(target.getContext(), 8, IntegerType::Unsigned)),
      "object_" + llvm::toHex(hash, /*LowerCase=*/true),
      ArrayRef<char>(buffer->getBufferStart(), buffer->getBufferSize()),
      resourceManager, /*optAlignment=*/8, /*forceOutOfLine=*/true);
}

//===----------------------------------------------------------------------===//
// produceStandaloneAssembly
//===----------------------------------------------------------------------===//

ErrorOrSuccess
ObjectCompiler::produceStandaloneAssembly(TargetInfoAttr target,
                                          llvm::raw_pwrite_stream &os) {
  TimeTraceScope<> traceScope("produce-standalone-assembly");

  OwningOpRef<ModuleOp> slicedModule = produceStandaloneModule();
  llvm::LLVMContext ctx;
  auto llvmModule = lowerAllFuncsToLLVM(ctx, *slicedModule, /*isJIT=*/false);
  if (!llvmModule)
    return Error("failed to lower module to LLVM IR");

  auto machineOr = createTargetMachine(options, /*isJIT=*/false);
  if (failed(machineOr))
    return machineOr.takeError();

  // Set the data layout on the module.
  llvmModule->setDataLayout((*machineOr)->createDataLayout());

  // Emit the assembly.
  if (failed(compileLLVMToObject(*llvmModule, **machineOr, os,
                                 /*emitAssembly=*/true))) {
    return Error("failed to lower LLVM IR to assembly");
  }
  return success();
}
