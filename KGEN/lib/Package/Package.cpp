//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Package/Package.h"
#include "KGEN/Compiler/KGENCompiler.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "LLCL/Runtime/Algorithms.h"
#include "LLCL/Runtime/Runtime.h"
#include "Support/Compiler/BytecodeReaderWriter.h"
#include "Support/Config.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

using namespace M;
using namespace KGEN;

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
  configurePassManager(genLibPM);
  buildGenerateLibraryPipeline(genLibPM, runtime, compileOptions);
  genLibPM.addPass(
      createMaterializePackagesWithDefaultGen(runtime, compileOptions));
  LLCL::AnyAsyncValueRef ready = Cache::cachedTransform(
      *packageModuleOr, regionCache.copy(), transformCache.copy(),
      AsyncValueRef<Chain>::createReady(runtime), genLibPM);
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
      pm, runtime, target, options, [=, &runtime](PackageLinkOp packageLink) {
        return specializePackageLinkForPreElaborationLinking(packageLink,
                                                             runtime, options);
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
