//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Package/Package.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/Compiler/ObjectCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Buffer.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/BLAKE3.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// createElaboratedBytecodeAttr
//===----------------------------------------------------------------------===//

ErrorOr<DenseResourceElementsAttr>
M::KGEN::createElaboratedBytecodeAttr(const SymbolTable &symtab,
                                      FlatSymbolRefAttr packageName) {
  ModuleOp theModule = cast<ModuleOp>(symtab.getOp());

  // Prepare the functions within the module for use when importing the package.
  for (KGEN::FuncOp func : theModule.getOps<KGEN::FuncOp>()) {
    // Attach a reference to the precompiled body to the KGEN::FuncOp.
    func.setPrecompiledBodyRefAttr(packageName);
    func.setWeakExported();
  }

  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  WriteableBufferRef str = WriteableBuffer::get();
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

//===----------------------------------------------------------------------===//
// createPackageArchive
//===----------------------------------------------------------------------===//

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

  ErrorOr<BufferRef> archiveOr =
      objectCompiler->produceStandaloneArchive(symtab, exportedSymbols);
  if (failed(archiveOr))
    return archiveOr.takeError();
  BufferRef archive = std::move(*archiveOr);

  // Get the standalone archive key to use as the archive name.
  WriteableBufferRef produceStandaloneArchiveKey = WriteableBuffer::get();
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

//===----------------------------------------------------------------------===//
// loadAndElaborateBytecode
//===----------------------------------------------------------------------===//

/// Given a pre-elaborated module representing a package linked to by the given
/// package_link op, elaborates the module for the given target and creates an
/// attribute for the resulting bytecode. If successful, this returns the
/// attribute, the elaborated module's symbol table, and a map of its exported
/// symbols, or if unsuccessful this returns an error.
static ErrorOr<std::tuple<DenseResourceElementsAttr, SymbolTable, ExportMap>>
elaborateBytecode(ModuleOp packageModule, PackageLinkOp packageLink,
                  TargetInfoAttr targetInfo, BuildInfoAttr buildInfo,
                  const CompilationOptions &compileOptions,
                  LLCL::Runtime &runtime) {
  // Run elaboration passes on the pre-elaborated module.
  auto cacheBackends = getMojoCacheBackends(runtime);
  if (cacheBackends.isError())
    return cacheBackends.takeError();
  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(cacheBackends->first));
  auto regionCache =
      RCRef<Cache::RegionCache>::create(std::move(cacheBackends->second));

  setTargetInfo(packageModule, targetInfo);
  setBuildInfo(packageModule, buildInfo);
  mlir::PassManager elaboratePM(packageModule->getContext());
  populateElaborateModulePasses(elaboratePM, runtime, targetInfo, buildInfo,
                                compileOptions);
  LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
      packageModule, regionCache.copy(), transformCache.copy(),
      runtime.getReadyChain().copy(), elaboratePM, /*deflateTarget=*/false);
  LLCL::await(ready);
  if (ready.isError())
    return ready.takeDiagnostic().getMessage().copy();

  // Construct the symbol table and export map.
  SymbolTable symtab(packageModule);
  ExportMap exportMap = getExportedSymbols(packageModule);

  // Create the elaborated bytecode attribute, and set it on the link op.
  auto bytecodeResourceOr = createElaboratedBytecodeAttr(
      symtab, FlatSymbolRefAttr::get(packageLink.getSymNameAttr()));
  if (bytecodeResourceOr.isError())
    return bytecodeResourceOr.takeError();
  DenseResourceElementsAttr bytecodeResource = bytecodeResourceOr.takeValue();

  return std::make_tuple(bytecodeResource, symtab, exportMap);
}

/// Loads serialized MLIR bytecode representing a pre-elaborated module for a
/// package, elaborates it, and generates a static archive. If successful, the
/// given package_link op will have its elaborated bytecode and static archive
/// attributes set.
ErrorOr<DenseResourceElementsAttr> M::KGEN::loadAndElaborateBytecode(
    PackageLinkOp packageLink, TargetInfoAttr targetInfo,
    BuildInfoAttr buildInfo, const CompilationOptions &compileOptions,
    LLCL::Runtime &runtime) {
  // Load the pre-elaborated bytecode, which contains the package module.
  // We'll run the elaborator on this bytecode module.
  mlir::AsmResourceBlob *blob =
      packageLink.getPreElaborationModuleAttr().getRawHandle().getBlob();
  if (!blob)
    return Error("unable to find the pre-elaborated module blob");
  ArrayRef<char> bytecode = blob->getData();
  llvm::MemoryBufferRef bufferRef(StringRef(bytecode.begin(), bytecode.size()),
                                  "");

  // Read the entirety of the bytecode in the buffer; no lazy loading.
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  mlir::ParserConfig parserConfig(packageLink.getContext());
  mlir::BytecodeReader reader(bufferRef, parserConfig, /*lazyLoad=*/false,
                              sourceMgr);
  Block block;
  if (failed(reader.readTopLevel(&block)))
    return Error("unable to read pre-elaborated module blob");
  ModuleOp packageModule = cast<ModuleOp>(block.front());

  // Elaborate the bytecode for the given target, and set the resulting bytecode
  // as an attribute on the package_link op.
  auto elaborateOr = elaborateBytecode(packageModule, packageLink, targetInfo,
                                       buildInfo, compileOptions, runtime);
  if (elaborateOr.isError())
    return elaborateOr.takeError();
  auto [bytecodeResource, symtab, exportedSymbols] = elaborateOr.takeValue();

  // Create the compiled archive of the package, and add the resulting archive
  // bytes to the package link op's archives attribute.
  auto archiveOr =
      createPackageArchive(symtab, exportedSymbols, compileOptions, runtime);
  if (archiveOr.isError())
    return archiveOr.takeError();

  // Insert the new archive into the array of archives on the package link op.
  SmallVector<PackageArchiveAttr> archives{PackageArchiveAttr::get(
      targetInfo, bytecodeResource, archiveOr.takeValue())};
  if (PackageArchiveArrayAttr existing = packageLink.getArchivesAttr())
    llvm::append_range(archives, existing.getValue());
  packageLink.setArchives(archives);

  return bytecodeResource;
}

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

void M::KGEN::populateElaborateModulePasses(mlir::PassManager &pm,
                                            LLCL::Runtime &runtime,
                                            TargetInfoAttr target,
                                            BuildInfoAttr build,
                                            const CompilationOptions &options) {
  populateElaborateModulePasses(
      pm, runtime, target, build, options,
      [=, &runtime](PackageLinkOp packageLink, TargetInfoAttr targetInfo,
                    BuildInfoAttr buildInfo) {
        return loadAndElaborateBytecode(packageLink, targetInfo, buildInfo,
                                        options, runtime);
      });
}
