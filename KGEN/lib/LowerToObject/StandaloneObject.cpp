//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/LowerToObject.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LowerToObjectImpl.h"
#include "Support/TempFile.h"
#include "Support/TimeProfiler.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/TypeSwitch.h"
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
using namespace Cache;

//===----------------------------------------------------------------------===//
// produceStandaloneModule
//===----------------------------------------------------------------------===//

/// Slice the dependencies of an operation out of the existing module into the
/// self-contained slice module.
static void sliceDependencies(Operation *op, mlir::SymbolTable &sliceSymtab,
                              const mlir::SymbolTable &symtab) {
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

  std::function<void(Type)> checkForRefType = [&](Type type) {
    if (auto ref = dyn_cast<DeclRefType>(type)) {
      Operation *decl = extractDependency(ref.getName());
      // Recurse on the type declaration.
      if (decl)
        sliceDependencies(decl, sliceSymtab, symtab);
    } else if (auto itf = dyn_cast<mlir::SubElementTypeInterface>(type)) {
      itf.walkSubTypes(checkForRefType);
    }
  };
  auto extractDependencies = [&](Operation *op) {
    // Extract references to type declarations.
    op->getAttrDictionary().walkSubTypes(checkForRefType);
    llvm::for_each(op->getResultTypes(), checkForRefType);
    for (Region &region : op->getRegions())
      llvm::for_each(region.getArgumentTypes(), checkForRefType);

    // Extract references to functions. Mark copied functions as module private
    // and recurse.
    llvm::TypeSwitch<Operation *>(op).Case<CallOp, AddressOfOp>([&](auto op) {
      Operation *symbol =
          extractDependency(op.getCalleeAttr().getRootReference());
      if (auto func = dyn_cast_if_present<FuncOp>(symbol))
        sliceDependencies(func, sliceSymtab, symtab);
    });
  };
  op->walk(extractDependencies);
}

OwningOpRef<ModuleOp> ObjectCompiler::produceStandaloneModule() {
  // Create a new module for these funcs. This will go away at the end
  // of this function.
  mlir::OwningOpRef<mlir::ModuleOp> singleModule =
      mlir::ModuleOp::create(module->getLoc());

  mlir::SymbolTable sliceSymtab(*singleModule);

  // Re-export exported functions.
  auto builder = OpBuilder::atBlockBegin(singleModule->getBody());

  SmallVector<FlatSymbolRefAttr> exportedSymbolVec;
  for (auto exportedSym : exportedSymbols)
    exportedSymbolVec.push_back(FlatSymbolRefAttr::get(exportedSym));

  if (!exportedSymbolVec.empty()) {
    builder.create<ExportOp>(
        module->getLoc(),
        builder.getArrayAttr(ArrayRef<Attribute>{exportedSymbolVec.begin(),
                                                 exportedSymbolVec.end()}));
  }

  for (auto sym : exportedSymbols) {
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

ErrorOr<BufferRef>
ObjectCompiler::produceStandaloneObject(TargetInfoAttr target, bool isJIT) {
  TimeTraceScope<> traceScope("produce-standalone-object");

  // Perform a cache aware transformation to translate the module to an object
  // file.
  llvm::LLVMContext ctx;
  auto runTransformation = [&](Operation *op, WriteableBufferRef buf,
                               LLCL::AnyAsyncValueRef chain) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
    chain->andThen([this, op, target, isJIT, &ctx, output = output.copy(),
                    buf = buf.copy()] {
      auto llvmModule = lowerAllFuncsToLLVM(ctx, cast<ModuleOp>(op));
      if (!llvmModule) {
        return output.setToError(LLCL::getMLIRDiagnostic(
            "failed to lower module to LLVM IR for object compilation",
            op->getLoc()));
      }

      // Create the target machine.
      auto machineOr = createTargetMachine(target, options, isJIT);
      if (failed(machineOr)) {
        return output.setToError(
            LLCL::getMLIRDiagnostic(machineOr.takeError(), op->getLoc()));
      }

      // Set the data layout on the module.
      llvmModule->setDataLayout((*machineOr)->createDataLayout());

      // Lower the LLVM to an object file.
      if (failed(compileLLVMToObject(*llvmModule, **machineOr, *buf))) {
        return output.setToError(LLCL::getMLIRDiagnostic(
            "failed to lower LLVM IR to object file", op->getLoc()));
      }
      output.emplace(buf.copy());
    });
    return std::move(output);
  };
  auto onCacheHit = [this](Operation *op, BufferRef buf) {
    return LLCL::AsyncValueRef<BufferRef>::createReady(runtime, buf.copy());
  };

  WriteableBufferRef produceStandaloneObjectKey = WriteableBuffer::get();
  options.print(*produceStandaloneObjectKey << "produceStandaloneObject(");
  *produceStandaloneObjectKey << ")";

  OwningOpRef<ModuleOp> slicedModule = produceStandaloneModule();
  auto output = cachedTransform(
      *slicedModule, transformCache,
      LLCL::AsyncValueRef<Chain>::createReady(runtime),
      std::move(produceStandaloneObjectKey), runTransformation, onCacheHit);
  await(output);

  if (output->isError())
    return ErrorOr<BufferRef>(std::move(output->takeDiagnostic().getMessage()));
  return {std::move(output->get<BufferRef>())};
}
