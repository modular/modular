//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Cache/CacheTelemetryContext.h"
#include "KGEN/Compiler/LLVMIRUtils.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENVersion/KGENVersion.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/Support/CompilerProfiling.h"
#include "LLCL/CompilerSupport/Context.h"
#include "LLCL/CompilerSupport/MLIRLocationDecoder.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LowerToObject.h"
#include "Support/Context.h"
#include "Support/FileSystemExtras.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugStringHelper.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/IndentedOstream.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/SplitModule.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
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
    // If the symbol reference attribute doesn't reference a symbol, somehow
    // invalid IR made it to the ObjectCompiler.
    assert(symbol && "invalid IR?");

    // Clone the symbol into the new symbol table.
    Operation *copy = symbol->clone();
    sliceSymtab.insert(copy);
    return copy;
  };

  std::vector<Operation *> worklist;
  mlir::AttrTypeWalker walker;
  walker.addWalk([&](FlatSymbolRefAttr ref) {
    if (Operation *decl = extractDependency(ref.getAttr()))
      worklist.push_back(decl);
  });
  auto extractDependencies = [&](Operation *op) {
    // Extract references to type declarations.
    walker.walk(op->getAttrDictionary());
    for (Type type : op->getResultTypes())
      walker.walk(type);
    for (Region &region : op->getRegions())
      for (Type type : region.getArgumentTypes())
        walker.walk(type);
  };

  worklist.push_back(op);
  while (!worklist.empty()) {
    Operation *op = worklist.back();
    worklist.pop_back();
    op->walk(extractDependencies);
  }
}

OwningOpRef<ModuleOp>
ObjectCompiler::produceStandaloneModule(const SymbolTable &symtab,
                                        const ExportMap &exportedSymbols) {
  IRMapping unused;
  return produceStandaloneModule(symtab, exportedSymbols, unused);
}

OwningOpRef<ModuleOp>
ObjectCompiler::produceStandaloneModule(const SymbolTable &symtab,
                                        const ExportMap &exportedSymbols,
                                        IRMapping &mapping) {
  CompilerTimeTraceScope traceScope("produceStandaloneModule");
  auto module = cast<ModuleOp>(symtab.getOp());
  // Create a new module for these funcs. This will go away at the end
  // of this function.
  OwningOpRef<ModuleOp> singleModule = ModuleOp::create(module->getLoc());
  singleModule.get()->setAttrs(module->getAttrDictionary());

  // Create a new symbol table for the sliced module.
  SymbolTable sliceSymtab(*singleModule);

  for (auto [sym, exportVal] : exportedSymbols) {
    auto func = symtab.lookup<ExportInterface>(sym);
    assert(func && "Unknown exported symbol");

    // Traverse the call graph and clone all the callees into this module.
    sliceDependencies(func, sliceSymtab, symtab);

    // Clone the func into this new module. We don't want to remove it from
    // the current module. Make sure the function is also exported in the slice.
    auto sliceFn = sliceSymtab.lookup<ExportInterface>(sym);
    if (!sliceFn) {
      sliceFn = cast<ExportInterface>(func->clone(mapping));
      sliceSymtab.insert(sliceFn);
    }
    ExportKind kind = func.getExportKind();
    sliceFn.setExportKind(kind == ExportKind::NotExported ? exportVal.kind
                                                          : kind);
  }

  return singleModule;
}

//===----------------------------------------------------------------------===//
// produceArchive
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef>
ObjectCompiler::produceArchive(const SymbolTable &symtab,
                               const ExportMap &exportedSymbols,
                               bool standalone) {
  CompilerTimeTraceScope traceScope("produce-archive");

  // Slice out a standalone module for the exported symbols.
  OwningOpRef<ModuleOp> slicedModule =
      produceStandaloneModule(symtab, exportedSymbols);

  // Perform a cache aware transformation to translate the module to an archive
  // file.
  LLCL::Runtime &runtime = *loadContext(&context)->get<LLCL::Runtime>();
  auto runTransformation = [&](Operation *op, WriteableBufferRef buf,
                               LLCL::AnyAsyncValueRef chain) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
#ifdef MODULAR_ENABLE_TELEMETRY
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheMiss("ObjectCompiler::produceArchive");
#endif
    chain.andThenSync([this, &runtime, op, output = output.copy(), standalone,
                       buf = buf.copy()]() mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
      [[maybe_unused]] auto timeScope =
          loadContext(op->getContext())
              ->get<M::Telemetry::TelemetryContext>()
              ->createUInt64Timer<std::chrono::milliseconds>(
                  "mojo.compile.cache.miss.time", M::Telemetry::Level::L2,
                  {{"pipeline", "ObjectCompiler::produceArchive"}});

#endif

      SmallVector<BufferRef> archiveBuffers;
      // Lower the module to LLVM.
      llvm::LLVMContext ctx;
      auto llvmModule = lowerAllFuncsToLLVM(ctx, cast<ModuleOp>(op));
      if (!llvmModule) {
        return std::move(output).setToError(LLCL::getMLIRDiagnostic(
            "failed to lower module to LLVM IR for archive compilation",
            op->getLoc()));
      }
      CompilerTimeTraceScope traceScope("split-input-module");
      StringRef moduleName = llvmModule->getName();

      // HACK HACK HACK https://github.com/modularml/modular/issues/22959
      // HACK: If we are generating PTX we don't want to split.
      bool generatingPtx =
          options.targetTriple.find("nvptx") != std::string::npos;

      // Split the module into multiple slices and compile each in parallel.
      // FIXME(#25622): Disable module splitting for non-standalone archives.
      SmallVector<LLCL::AnyAsyncValueRef> cacheResults;
      bool noSplitting = runtime.getWorkQueue()->getParallelismLevel() < 2 ||
                         generatingPtx || !standalone;

      auto processSync = [&]() {
        // If sync, await cacheResults so that the cloned sub-module
        // can be released before launching the next batch to reduce
        // memory pressure.
        await(cacheResults);

        // If any of the cache results failed, propagate the error.
        for (auto &result : cacheResults) {
          if (result.isError())
            return std::move(output).setToError(result.takeDiagnostic());
        }
        for (LLCL::AnyAsyncValueRef &result : cacheResults) {
          // Move the result buffer to archiveBuffers so that we can
          // concatenate them later together.
          archiveBuffers.emplace_back(std::move(result.get<BufferRef>()));
        }
        // Clear cacheResults for next batch.
        cacheResults.clear();
      };

      bool parLLC = !generatingPtx && options.enableParallelLLC;
      if (noSplitting) {
        SmallVector<AnyAsyncValueRef> results = lowerLLVMModuleToObjects(
            *llvmModule, op->getLoc(), op->getContext(), parLLC);

        for (AnyAsyncValueRef &result : results)
          cacheResults.emplace_back(std::move(result));

      } else {
        if (!options.saveTempsPrefix.empty()) {
          std::string outPath = options.saveTempsPrefix + ".pre-split.ll";
          std::unique_ptr<llvm::ToolOutputFile> outFile =
              mlir::openOutputFile(outPath);
          if (outFile) {
            outFile->os() << *llvmModule;
            outFile->keep();
          }
        }

        if (options.enableLLVMPerFunctionSplitting) {
          splitPerFunction(
              *llvmModule, runtime.getWorkQueue()->getParallelismLevel(),
              [&](llvm::Module *inputModule, int64_t idx, bool sync) {
                if (inputModule) {
                  SmallVector<AnyAsyncValueRef> results =
                      lowerLLVMModuleToObjects(*inputModule, op->getLoc(),
                                               op->getContext(),
                                               /*parLLC=*/false, idx);
                  for (AnyAsyncValueRef &result : results)
                    cacheResults.emplace_back(std::move(result));
                }
                if (sync)
                  processSync();
              });
        } else {
          // TODO: Keep this less aggressive splitting for:
          // - REPL which has different object layout requirements layouts
          // (#35345).
          // - Other cases where aggressive splitting actually slow down
          // compilation and needs better heuristics to improve.
          splitPerExported(*llvmModule, [&](llvm::Module &inputModule,
                                            int64_t idx) {
            SmallVector<AnyAsyncValueRef> results = lowerLLVMModuleToObjects(
                inputModule, op->getLoc(), op->getContext(), parLLC, idx);
            for (AnyAsyncValueRef &result : results)
              cacheResults.emplace_back(std::move(result));
          });
        }
      }

      if (noSplitting || !options.enableLLVMPerFunctionSplitting) {
        andThenSyncMoving(
            cacheResults,
            [moduleName = moduleName.str(), op, buf = buf.copy(),
             output = output.copy(),
             generatingPtx](MutableArrayRef<AnyAsyncValueRef> values) mutable {

#ifdef MODULAR_ENABLE_TELEMETRY
              [[maybe_unused]] auto timeScope =
                  loadContext(op->getContext())
                      ->get<M::Telemetry::TelemetryContext>()
                      ->createUInt64Timer<std::chrono::milliseconds>(
                          "mojo.compile.cache.miss.time",
                          M::Telemetry::Level::L2,
                          {{"pipeline", "ObjectCompiler::produceArchive"}});
#endif

              // If any of the cache results failed, propagate the error.
              for (auto &result : values) {
                if (result.isError())
                  return std::move(output).setToError(result.takeDiagnostic());
              }
              CompilerTimeTraceScope traceScope("concatenate-object-files");

              if (generatingPtx) {
                // If we're not splitting just copy directly to the output
                // buffer.
                assert(values.size() == 1 &&
                       "should have one result if generating PTX");
                *buf << values[0].get<BufferRef>()->getBuffer();
                std::move(output).emplace(buf.copy());
                return;
              }

              // Now that all the object files have been compiled, merge them
              // all into a single archive.
              SmallVector<std::string> archiveMemberNames(values.size());
              SmallVector<llvm::NewArchiveMember> archiveMembers;
              for (auto [idx, result] : llvm::enumerate(values)) {
                auto &resultBuf = result.get<BufferRef>();
                archiveMemberNames[idx] =
                    (moduleName + "." + Twine(idx) + ".o").str();

                archiveMembers.emplace_back(llvm::MemoryBufferRef(
                    resultBuf->getBuffer(), archiveMemberNames[idx]));
              }

              auto result = llvm::writeArchiveToBuffer(
                  archiveMembers,
                  /*WriteSymtab=*/llvm::SymtabWritingMode::NormalSymtab,
                  archiveMembers.front().detectKindFromObject(),
                  /*Deterministic=*/true, /*Thin=*/false);
              if (!result) {
                return std::move(output).setToError(LLCL::getMLIRDiagnostic(
                    "failed to concatenate object files into archive",
                    op->getLoc()));
              }

              // Copy the result into the output buffer.
              *buf << (*result)->getBuffer();
              std::move(output).emplace(buf.copy());
            });
      } else {
        CompilerTimeTraceScope traceScope("concatenate-object-files");
        // Now that all the object files have been compiled,
        // merge them all into a single archive.
        SmallVector<std::string> archiveMemberNames(archiveBuffers.size());
        SmallVector<llvm::NewArchiveMember> archiveMembers;

        for (auto [index, resultBuf] : llvm::enumerate(archiveBuffers)) {
          archiveMemberNames[index] =
              (moduleName + "." + Twine(index) + ".o").str();
          archiveMembers.emplace_back(llvm::MemoryBufferRef(
              resultBuf->getBuffer(), archiveMemberNames[index]));
        }
        auto result = llvm::writeArchiveToBuffer(
            archiveMembers,
            /*WriteSymtab=*/llvm::SymtabWritingMode::NormalSymtab,
            archiveMembers.front().detectKindFromObject(),
            /*Deterministic=*/true, /*Thin=*/false);
        if (!result) {
          return std::move(output).setToError(LLCL::getMLIRDiagnostic(
              "failed to concatenate object files into archive", op->getLoc()));
        }
        // Copy the result into the output buffer.
        *buf << (*result)->getBuffer();
        std::move(output).emplace(buf.copy());
      }
    });
    return output;
  };
  auto onCacheHit = [](Operation *op, BufferRef buf) {
#ifdef MODULAR_ENABLE_TELEMETRY
    CacheTelemetryContext::getCacheTelemetryContext(
        loadContext(op->getContext()))
        .recordCacheHit("ObjectCompiler::produceArchive");
#endif
    return buf.copy();
  };

  WriteableBufferRef produceArchiveKey = WriteableBuffer::get();
  options.print(*produceArchiveKey << "produceArchive(");
  *produceArchiveKey << ", isJIT=" << isJIT
                     << ", enableLLVMPerFunctionSplitting="
                     << options.enableLLVMPerFunctionSplitting << ')';

  auto output = cachedTransform(
      *slicedModule, transformCache.copy(),
      LLCL::AsyncValueRef<Chain>::createReady(runtime),
      std::move(produceArchiveKey), runTransformation, onCacheHit);
  await(output);

  if (output.isError())
    return {std::move(output.takeDiagnostic().getMessage())};
  return {std::move(output.get<BufferRef>())};
}

//===----------------------------------------------------------------------===//
// produceStandaloneArchive
//===----------------------------------------------------------------------===//

ErrorOr<BufferRef>
ObjectCompiler::produceStandaloneArchive(const SymbolTable &symtab,
                                         const ExportMap &exportedSymbols) {
  return produceArchive(symtab, exportedSymbols, /*standalone=*/true);
}

SmallVector<LLCL::AnyAsyncValueRef>
ObjectCompiler::lowerLLVMModuleToObjects(llvm::Module &module, Location loc,
                                         MLIRContext *mlirContext, bool parLLC,
                                         std::optional<size_t> moduleIdx) {
  LLCL::Runtime &runtime = *loadContext(&context)->get<LLCL::Runtime>();
  SmallVector<LLCL::AnyAsyncValueRef> results;

  // Create the target machine.
  auto machineOr = createTargetMachine(options, isJIT);
  if (failed(machineOr)) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
    std::move(output).setToError(
        LLCL::getMLIRDiagnostic(machineOr.takeError(), loc));
    results.emplace_back(std::move(output));
    return results;
  }

  // Set the data layout on the module.
  module.setDataLayout((*machineOr)->createDataLayout());

  // Optimize the llvm Module.
  if (failed(optimizeLLVMModule(module, **machineOr, options, runtime,
                                moduleIdx))) {
    auto output = LLCL::AsyncValueRef<BufferRef>::allocate(runtime);
    std::move(output).setToError(
        LLCL::getMLIRDiagnostic("failed to optimize LLVM IR.", loc));
    results.emplace_back(std::move(output));
    return results;
  }

  // HACK HACK HACK https://github.com/modularml/modular/issues/22959
  // HACK: Some targets like PTX don't support object files so can only
  // emit assembly.
  bool emitAssembly =
      (*machineOr)->getTargetTriple().str().find("nvptx") != std::string::npos;

  // Codegen optimized llvm module to object files.
  return compileOptimizedLLVMToObjects(module, loc, options, runtime,
                                       transformCache, parLLC, isJIT,
                                       emitAssembly, moduleIdx);
}

ErrorOr<ElementsAttr>
ObjectCompiler::produceStandaloneArchiveAttr(const SymbolTable &symtab,
                                             const ExportMap &exportedSymbols,
                                             TargetInfoAttr target) {
  auto bufferOr = produceStandaloneArchive(symtab, exportedSymbols);
  if (bufferOr.isError())
    return bufferOr.takeError();
  BufferRef buffer = bufferOr.takeValue();

  auto module = cast<ModuleOp>(symtab.getOp());

  // Get the standalone archive key to use as the archive name.
  WriteableBufferRef produceStandaloneArchiveKey = WriteableBuffer::get();
  options.print(*produceStandaloneArchiveKey << "produceStandaloneArchive(");
  *produceStandaloneArchiveKey << ")";
  if (failed(mlir::writeBytecodeToFile(module.getOperation(),
                                       *produceStandaloneArchiveKey)))
    return Error("failed to write bytecode file");
  // Hash it so the name isn't enormous.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)produceStandaloneArchiveKey->getBufferStart(),
               produceStandaloneArchiveKey->getBufferSize()));

  // Produce a DenseResourceElementsAttr from the file.
  auto resourceManager =
      DenseResourceElementsHandle::getManagerInterface(target.getContext());

  // Pretend this is a "tensor" of data.
  // TODO (#6986) It would be much nicer if we didn't have to clone this data
  //   and we could just reference the data already in the CAS. That would also
  //   prevent us from having to hash the module above.
  auto attrType = RankedTensorType::get(
      {(int64_t)buffer->getBufferSize()},
      IntegerType::get(target.getContext(), 8, IntegerType::Unsigned));
  auto attrName = "archive_" + llvm::toHex(hash, /*LowerCase=*/true);
  ArrayRef<char> blobData(buffer->getBufferStart(), buffer->getBufferSize());
  auto blob = mlir::HeapAsmResourceBlob::allocateAndCopyWithAlign(blobData,
                                                                  /*align=*/8);
  return DenseResourceElementsAttr::get(
      attrType, resourceManager.insert(attrName, std::move(blob)));
}

//===----------------------------------------------------------------------===//
// produceStandaloneAssembly
//===----------------------------------------------------------------------===//

ErrorOrSuccess
ObjectCompiler::produceStandaloneAssembly(const SymbolTable &symtab,
                                          const ExportMap &exportedSymbols,
                                          llvm::raw_pwrite_stream &os) {
  CompilerTimeTraceScope traceScope("produce-standalone-assembly");

  OwningOpRef<ModuleOp> slicedModule =
      produceStandaloneModule(symtab, exportedSymbols);
  llvm::LLVMContext ctx;
  auto llvmModule = lowerAllFuncsToLLVM(ctx, *slicedModule);
  if (!llvmModule)
    return Error("failed to lower module to LLVM IR");

  auto machineOr = createTargetMachine(options, /*isJIT=*/false);
  if (failed(machineOr))
    return machineOr.takeError();

  // Set the data layout on the module.
  llvmModule->setDataLayout((*machineOr)->createDataLayout());

  // Emit the assembly.
  LLCL::Runtime &runtime = *loadContext(&context)->get<LLCL::Runtime>();
  if (failed(KGEN::compileLLVMToObject(*llvmModule, **machineOr, os, options,
                                       runtime, /*emitAssembly=*/true)))
    return Error("failed to lower LLVM IR to assembly");

  return success();
}
