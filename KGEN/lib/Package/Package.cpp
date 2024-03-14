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
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Compiler/MLIRDenseAttr.h"
#include "llvm/Support/BLAKE3.h"

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
  SmallVector<KGEN::FuncOp> exportedFuncs;
  for (KGEN::FuncOp func : theModule.getOps<KGEN::FuncOp>()) {
    // Only tag functions that will be exported from the generated archive with
    // a reference to this package.
    if (!func.isExported() || func.getPrecompiledBodyRef())
      continue;

    func.setPrecompiledBodyRefAttr(packageName);
    exportedFuncs.push_back(func);
  }

  // Write the package bytecode to the given buffer. This will be attached to
  // the exported high level functions.
  WriteableBufferRef str = WriteableBuffer::get();
  if (failed(mlir::writeBytecodeToFile(symtab.getOp(), *str)))
    return Error("could not write bytecode for package module");

  // Reset the precompiled references now that we've written to bytecode.
  for (KGEN::FuncOp func : exportedFuncs)
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

/// Loop over each `kgen.link` op in the given module, adding each of them (and
/// their dependencies) to the collection of `linkDependencies`. As dependencies
/// are added to the collection, a hash of their contents is made and appended
/// to the library name.
static void collectLinkDependencies(
    ModuleOp theModule,
    llvm::MapVector<StringAttr, LinkDependencyAttr> &linkDependencies) {
  for (LinkOp linkOp : theModule.getOps<LinkOp>()) {
    // Collect the link dependencies: first the link op's dependencies, since
    // they ought to be loaded first.
    if (LinkDependencyArrayAttr dependencies = linkOp.getDependenciesAttr())
      for (LinkDependencyAttr dependency : dependencies.getValue())
        linkDependencies.insert({dependency.getName(), dependency});

    // Then, collect the linked library itself: first compute a hash of its
    // contents to form its name, then insert it into the collection.
    ArrayRef<char> bytes =
        linkOp.getLinkBytes().getRawHandle().getBlob()->getData();
    auto hash = llvm::BLAKE3::hash(
        ArrayRef((const uint8_t *)bytes.begin(), bytes.size()));
    StringAttr name = StringAttr::get(
        theModule.getContext(), linkOp.getSymName().str() + '_' +
                                    llvm::toHex(hash, /*LowerCase=*/true));
    linkDependencies.insert(
        {name, LinkDependencyAttr::get(name, linkOp.getLinkBytes())});
  }
}

ErrorOr<PackageArchiveAttr> M::KGEN::createPackageArchive(
    const SymbolTable &symtab, const ExportMap &exportedSymbols,
    TargetInfoAttr targetInfo, DenseResourceElementsAttr elaboratedBytecode,
    const CompilationOptions &compileOptions, LLCL::Runtime &runtime) {
  ModuleOp theModule = cast<ModuleOp>(symtab.getOp());

  // Before elaborating the package module, collect its link dependencies,
  // uniquing them based on their given name.
  llvm::MapVector<StringAttr, LinkDependencyAttr> linkDependencies;
  collectLinkDependencies(theModule, linkDependencies);

  // Now we can start to generate the archive.
  mlir::PassManager archivePM(theModule->getContext());
  auto objectCompiler = ObjectCompiler::create(
      runtime, archivePM, ".mojo_cache", compileOptions, /*isJIT=*/false);
  if (failed(objectCompiler))
    return objectCompiler.takeError();

  ErrorOr<BufferRef> archiveOr = objectCompiler->produceArchive(
      symtab, exportedSymbols, /*standalone=*/false);
  if (failed(archiveOr))
    return archiveOr.takeError();
  BufferRef archive = std::move(*archiveOr);

  // Get the archive key to use as the archive name.
  WriteableBufferRef produceArchiveKey = WriteableBuffer::get();
  compileOptions.print(*produceArchiveKey << "produceArchive(");
  *produceArchiveKey << ")";
  if (failed(mlir::writeBytecodeToFile(theModule, *produceArchiveKey)))
    return Error("failed to write bytecode file");
  // Hash it so the name isn't enormous.
  auto hash = llvm::BLAKE3::hash(
      ArrayRef((const uint8_t *)produceArchiveKey->getBufferStart(),
               produceArchiveKey->getBufferSize()));

  DenseResourceElementsAttr archiveBytes = createResourceAttr(
      theModule.getContext(),
      ArrayRef(archive->getBufferStart(), archive->getBufferSize()),
      "archive_" + llvm::toHex(hash, /*LowerCase=*/true));

  // Collect and return the archive and its dependencies.
  SmallVector<LinkDependencyAttr> dependencies;
  for (auto &[name, dependency] : linkDependencies)
    dependencies.push_back(dependency);
  return PackageArchiveAttr::get(targetInfo, elaboratedBytecode, archiveBytes,
                                 dependencies);
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
                  TargetInfoAttr targetInfo,
                  const CompilationOptions &compileOptions,
                  LLCL::Runtime &runtime, RCRef<Cache::RegionCache> regionCache,
                  RCRef<Cache::TransformCache> transformCache) {
  setTargetInfo(packageModule, targetInfo);
  mlir::PassManager elaboratePM(packageModule->getContext());
  populateElaborateModulePasses(elaboratePM, runtime, targetInfo,
                                compileOptions);
  LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
      packageModule, regionCache.copy(), transformCache.copy(),
      AsyncValueRef<Chain>::createReady(runtime), elaboratePM,
      /*deflateTarget=*/false);
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
ErrorOr<PackageArchiveAttr> M::KGEN::loadAndElaborateBytecode(
    PackageLinkOp packageLink, TargetInfoAttr targetInfo,
    const CompilationOptions &compileOptions, LLCL::Runtime &runtime) {
  DenseResourceElementsAttr preElaborationModuleAttr =
      packageLink.getPreElaborationModuleAttr();
  if (!preElaborationModuleAttr)
    return Error("package link does not have a pre-elaboration module");

  // Run elaboration passes on the pre-elaborated module.
  auto cacheBackends = getMojoCacheBackends(runtime);
  if (cacheBackends.isError())
    return cacheBackends.takeError();
  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(cacheBackends->first));
  auto regionCache =
      RCRef<Cache::RegionCache>::create(std::move(cacheBackends->second));

  // Build a cache key for the elaboration transformation.
  WriteableBufferRef key = WriteableBuffer::get();
  *key << "loadAndElaborateBytecode(" << targetInfo;
  compileOptions.print(*key);
  if (failed(writeAttrToBytecodeFile(preElaborationModuleAttr, *key)))
    return Error("failed to write pre-elaborated module bytecode");
  *key << ")";

  // Functor to adapt the transform functor to the required API.
  auto runTransformation = [packageLink, preElaborationModuleAttr, targetInfo,
                            transformCache = transformCache.copy(),
                            regionCache = regionCache.copy(), &compileOptions,
                            &runtime](WriteableBufferRef buf,
                                      AnyAsyncValueRef chain) mutable {
    auto output =
        AsyncValueRef<PackageArchiveAttr>::allocate(chain.getRuntime());
    std::move(chain).andThenSync(
        [packageLink, preElaborationModuleAttr, targetInfo, &compileOptions,
         &runtime, transformCache = transformCache.copy(),
         regionCache = regionCache.copy(), output = output.copy(),
         buf = std::move(buf)](AnyAsyncValueRef &&chain) mutable {
          if (chain.isError())
            return std::move(output).setToError(chain.takeDiagnostic());

          OwningOpRef<ModuleOp> packageModuleOr =
              readOpFromBytecodeFile<ModuleOp>(preElaborationModuleAttr);
          if (!packageModuleOr) {
            return std::move(output).setToError(LLCL::getMLIRDiagnostic(
                Error("failed to load pre-specialized module bytecode"),
                packageLink->getLoc()));
          }

          // Elaborate the bytecode for the given target, and set the resulting
          // bytecode as an attribute on the package_link op.
          auto elaborateOr = elaborateBytecode(
              *packageModuleOr, packageLink, targetInfo, compileOptions,
              runtime, regionCache.copy(), transformCache.copy());
          if (elaborateOr.isError()) {
            return std::move(output).setToError(LLCL::getMLIRDiagnostic(
                elaborateOr.takeError(), packageLink->getLoc()));
          }
          auto [bytecodeResource, symtab, exportedSymbols] =
              elaborateOr.takeValue();

          // Create the compiled archive of the package, and add the resulting
          // archive bytes to the package link op's archives attribute.
          auto archiveOr =
              createPackageArchive(symtab, exportedSymbols, targetInfo,
                                   bytecodeResource, compileOptions, runtime);
          if (archiveOr.isError()) {
            return std::move(output).setToError(LLCL::getMLIRDiagnostic(
                archiveOr.takeError(), packageLink->getLoc()));
          }

          if (failed(writeAttrToBytecodeFile(*archiveOr, *buf))) {
            return std::move(output).setToError(LLCL::getMLIRDiagnostic(
                "failed to write archive bytecode", packageLink->getLoc()));
          }
          return std::move(output).emplace(*archiveOr);
        });
    return output;
  };
  // On cache hit, just return the assembly buffer.
  auto onCacheHit = [packageLink](BufferRef buf) {
    return readAttrFromBytecodeFile<PackageArchiveAttr>(
        buf->getMemBufferRef(), packageLink->getContext());
  };

  AnyAsyncValueRef result = Cache::cachedTransform(
      LLCL::MLIRLocationDecoder::getEncodedLocation(packageLink->getLoc()),
      transformCache.copy(), AsyncValueRef<Chain>::createReady(runtime),
      std::move(key), std::move(runTransformation), onCacheHit);
  await(result);
  if (result.isError())
    return std::move(result.takeDiagnostic().getMessage());
  auto archiveAttr = result.get<PackageArchiveAttr>();

  // Set an updated list of archives on the package link.
  SmallVector<PackageArchiveAttr> archives{archiveAttr};
  if (PackageArchiveArrayAttr existing = packageLink.getArchivesAttr())
    llvm::append_range(archives, existing.getValue());
  packageLink.setArchives(archives);

  return archiveAttr;
}

//===----------------------------------------------------------------------===//
// specializeModuleForPreElaborationLinking
//===----------------------------------------------------------------------===//

/// Loads the serialized MLIR bytecode representing a post-parser module in
/// `bytecodeAttr`, and prepare to link it into directly another module. Returns
/// the module if successful, or an error. If `exportedPreElaborationAttr` is
/// non-null, it will be set to the exported pre-elaboration bytecode.
static ErrorOr<OwningOpRef<ModuleOp>> specializeModuleForPreElaboration(
    DenseResourceElementsAttr bytecodeAttr, LLCL::Runtime &runtime,
    const KGEN::CompilationOptions &compileOptions,
    DenseResourceElementsAttr *exportedPreElaborationAttr = nullptr) {
  OwningOpRef<ModuleOp> packageModuleOr =
      readOpFromBytecodeFile<ModuleOp>(bytecodeAttr);
  if (!packageModuleOr)
    return Error("unable to load parsed module bytecode");
  auto cacheBackends = getMojoCacheBackends(runtime);
  if (cacheBackends.isError())
    return cacheBackends.takeError();
  auto transformCache =
      RCRef<Cache::TransformCache>::create(std::move(cacheBackends->first));
  auto regionCache =
      RCRef<Cache::RegionCache>::create(std::move(cacheBackends->second));

  // Generate a library from the module, processing the pipeline up to the
  // elaboration phase.
  mlir::PassManager genLibPM(packageModuleOr->getContext());
  buildGenerateLibraryPipeline(genLibPM, runtime, compileOptions);
  genLibPM.addPass(
      createMaterializePackagesWithDefaultGen(runtime, compileOptions));
  LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
      *packageModuleOr, regionCache.copy(), transformCache.copy(),
      AsyncValueRef<Chain>::createReady(runtime), genLibPM,
      /*deflateTarget=*/false);
  LLCL::await(ready);
  if (ready.isError())
    return ready.takeDiagnostic().getMessage().copy();

  // Build the exported pre-elaboration bytecode if requested.
  if (exportedPreElaborationAttr) {
    DenseResourceElementsAttr bytecodeResource =
        writeModuleToBytecodeAttr(*packageModuleOr);
    if (!bytecodeResource)
      return Error("failed to write bytecode for package module");
    *exportedPreElaborationAttr = bytecodeResource;
  }

  // Strip the implicit package exports, we don't need these because we're going
  // to link the package into an existing module as-is.
  for (ExportInterface op : packageModuleOr->getOps<ExportInterface>())
    if (op.isPackageExported())
      op.setNotExported();

  return std::move(packageModuleOr);
}

ErrorOr<OwningOpRef<ModuleOp>> KGEN::specializeModuleForPreElaborationLinking(
    DenseResourceElementsAttr bytecodeAttr, LLCL::Runtime &runtime,
    const KGEN::CompilationOptions &compileOptions) {
  return specializeModuleForPreElaboration(bytecodeAttr, runtime,
                                           compileOptions);
}

//===----------------------------------------------------------------------===//
// specializePackageLinkForPreElaborationLinking
//===----------------------------------------------------------------------===//

ErrorOr<DenseResourceElementsAttr>
KGEN::specializePackageLinkForPreElaborationLinking(
    PackageLinkOp packageLink, LLCL::Runtime &runtime,
    const KGEN::CompilationOptions &compileOptions) {
  DenseResourceElementsAttr bytecodeAttr = packageLink.getPostParseModuleAttr();
  DenseResourceElementsAttr preElaborationBytecode;
  ErrorOr<OwningOpRef<ModuleOp>> packageModuleOr =
      specializeModuleForPreElaboration(bytecodeAttr, runtime, compileOptions,
                                        &preElaborationBytecode);
  if (packageModuleOr.isError())
    return packageModuleOr.takeError();
  packageLink.setPreElaborationModuleAttr(preElaborationBytecode);

  DenseResourceElementsAttr bytecodeResource =
      writeModuleToBytecodeAttr(cast<ModuleOp>(**packageModuleOr));
  if (!bytecodeResource)
    return Error("failed to write bytecode for package module");
  return bytecodeResource;
}

//===----------------------------------------------------------------------===//
// populateElaborateModulePasses
//===----------------------------------------------------------------------===//

void M::KGEN::populateElaborateModulePasses(mlir::PassManager &pm,
                                            LLCL::Runtime &runtime,
                                            TargetInfoAttr target,
                                            const CompilationOptions &options) {
  populateElaborateModulePasses(
      pm, runtime, target, options,
      [=, &runtime](PackageLinkOp packageLink) {
        return specializePackageLinkForPreElaborationLinking(packageLink,
                                                             runtime, options);
      },
      [=, &runtime](PackageLinkOp packageLink, TargetInfoAttr targetInfo) {
        return loadAndElaborateBytecode(packageLink, targetInfo, options,
                                        runtime);
      });
}
//===----------------------------------------------------------------------===//
// createElaborateGeneratorsWithDefaultJIT
//===----------------------------------------------------------------------===//

/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass>
KGEN::createElaborateGeneratorsWithDefaultJIT(LLCL::Runtime &runtime) {
  CompilationOptions options;
  return createElaborateGenerators(
      runtime, /*target=*/{}, /*options=*/{},
      [=, &runtime](FuncOp evaluator, const SymbolTable &symtab,
                    TargetInfoAttr target, ArrayRef<FuncOp> specializations) {
        return evaluateSpecializations(evaluator, symtab, runtime, target,
                                       options, specializations);
      },
      [=, &runtime](GeneratorOp func, SymbolConstantAttr symbol,
                    StringAttr name, const SymbolTable &symtab,
                    TargetInfoAttr target, EmissionKind emissionKind) {
        return compileElaboratorAsm(func, symbol, name, symtab, runtime, target,
                                    emissionKind, options);
      },
      [=, &runtime](PackageLinkOp link, TargetInfoAttr target) {
        return loadAndElaborateBytecode(link, target, options, runtime);
      });
}

//===----------------------------------------------------------------------===//
// createMaterializePackagesWithDefaultGen
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> KGEN::createMaterializePackagesWithDefaultGen(
    LLCL::Runtime &runtime, const CompilationOptions &options) {
  return createMaterializePackages([&](KGEN::PackageLinkOp packageLink) {
    return specializePackageLinkForPreElaborationLinking(packageLink, runtime,
                                                         options);
  });
}
