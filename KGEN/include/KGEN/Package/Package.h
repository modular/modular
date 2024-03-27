//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_PACKAGE_PACKAGE_H
#define KGEN_PACKAGE_PACKAGE_H

#include "KGEN/KGENDialect/KGENUtils.h"
#include "Support/ErrorOr.h"
#include "Support/LLVMCompilerForwardDecls.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
class BytecodeReader;
} // namespace mlir

namespace M {
namespace LLCL {
class Runtime;
} // namespace LLCL
} // namespace M

namespace M::KGEN {

class CompilationOptions;
class PackageLinkOp;

/// Loads the serialized MLIR bytecode representing a post-parser module in
/// `bytecodeAttr`, and prepare to link it directly into another module. Returns
/// the module if successful, or an error.
ErrorOr<OwningOpRef<ModuleOp>> specializeModuleForPreElaborationLinking(
    DenseResourceElementsAttr bytecodeAttr, LLCL::Runtime &runtime,
    const KGEN::CompilationOptions &compileOptions);

/// Loads the serialized MLIR bytecode representing a post-parser module in
/// `packageLink`, and prepare to link it directly into another module.
/// The preElaborationModule of `packageLink` is set to the result of the
/// preparation. Returns the bytecode if successful, or an error.
ErrorOr<DenseResourceElementsAttr>
specializePackageLinkForPreElaborationLinking(
    PackageLinkOp packageLink, LLCL::Runtime &runtime,
    const KGEN::CompilationOptions &compileOptions);

/// This populates the passes to produce a fully concrete KGEN module. It's the
/// equivalent of the `buildElaborateModulePipeline` function defined in
/// KGENCompiler, but with a default handler for package link ops.
void populateElaborateModulePasses(mlir::PassManager &pm,
                                   LLCL::Runtime &runtime,
                                   TargetInfoAttr target,
                                   const CompilationOptions &options);

/// This creates the materialize packages pass with the default library
/// generation pipeline, i.e. `specializePackageLinkForPreElaborationLinking`.
std::unique_ptr<Pass>
createMaterializePackagesWithDefaultGen(LLCL::Runtime &runtime,
                                        const CompilationOptions &options);

/// Create an instance of the elaborator pass using the given configuration.
/// The created elaborator pass uses a default specialization executor that
/// JITs and executes in-process.
std::unique_ptr<Pass>
createElaborateGeneratorsWithDefaultJIT(LLCL::Runtime &runtime);

} // namespace M::KGEN

#endif // KGEN_PACKAGE_PACKAGE_H
