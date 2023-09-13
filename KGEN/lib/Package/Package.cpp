//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Package/Package.h"
#include "Cache/Buffer.h"
#include "KGEN/CompilationOptions.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LowerToObject.h"
#include "LLCL/Runtime/Runtime.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"

using namespace M;
using namespace KGEN;

ErrorOr<DenseResourceElementsAttr>
M::KGEN::createElaboratedBytecodeAttr(const SymbolTable &symtab,
                                      FlatSymbolRefAttr packageName) {
  ModuleOp theModule = cast<ModuleOp>(symtab.getOp());

  // Prepare the functions within the module for use when importing the package.
  for (KGEN::FuncOp func : theModule.getOps<KGEN::FuncOp>()) {
    // Attach a reference to the precompiled body to the KGEN::FuncOp.
    func.setPrecompiledBodyRefAttr(packageName);
    func.setExported();
  }

  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  Cache::WriteableBufferRef str = Cache::WriteableBuffer::get();
  if (failed(mlir::writeBytecodeToFile(symtab.getOp(), *str)))
    return Error("could not write bytecode for package module");

  // Reset the precompiled references now that we've written to bytecode.
  for (KGEN::FuncOp func : theModule.getOps<KGEN::FuncOp>())
    func.removePrecompiledBodyRefAttr();

  // Hash the bytecode itself - this will give us a unique'd attr name that
  // shouldn't clash even when a large number of packages get imported - and
  // if they do clash, they're guaranteed to be exactly the same.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef<uint8_t>((const uint8_t *)str->getBufferStart(),
                        (const uint8_t *)str->getBufferEnd()));
  return createResourceAttr(symtab.getOp()->getContext(), str->getBuffer(),
                            "bytecode_" +
                                llvm::toHex(hash, /*LowerCase=*/true));
}

ErrorOr<DenseResourceElementsAttr> M::KGEN::createPackageArchive(
    const SymbolTable &symtab, const ExportMap &exportedSymbols,
    const CompilationOptions &compileOptions, LLCL::Runtime &runtime) {
  ModuleOp theModule = cast<ModuleOp>(symtab.getOp());

  // Now we can start to generate the archive.
  mlir::PassManager archivePM(theModule->getContext());
  auto objectCompiler = ObjectCompiler::create(
      runtime, archivePM, ".mojo_cache", compileOptions, /*isJIT=*/false);
  if (failed(objectCompiler))
    return objectCompiler.takeError();

  ErrorOr<Cache::BufferRef> archiveOr =
      objectCompiler->produceStandaloneArchive(symtab, exportedSymbols);
  if (failed(archiveOr))
    return archiveOr.takeError();
  Cache::BufferRef archive = std::move(*archiveOr);

  // Get the standalone archive key to use as the archive name.
  Cache::WriteableBufferRef produceStandaloneArchiveKey =
      Cache::WriteableBuffer::get();
  compileOptions.print(*produceStandaloneArchiveKey
                       << "produceStandaloneArchive(");
  *produceStandaloneArchiveKey << ")";
  if (failed(
          mlir::writeBytecodeToFile(theModule, *produceStandaloneArchiveKey)))
    return Error("failed to write bytecode file");
  // Hash it so the name isn't enormous.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)produceStandaloneArchiveKey->getBufferStart(),
               produceStandaloneArchiveKey->getBufferSize()));

  return createResourceAttr(
      theModule.getContext(),
      ArrayRef(archive->getBufferStart(), archive->getBufferSize()),
      "archive_" + llvm::toHex(hash, /*LowerCase=*/true));
}
