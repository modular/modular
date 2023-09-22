//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoTooling/ParserDriver.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/MojoParser/ASTDecl.h"
#include "KGEN/MojoParser/DeclResolver.h"
#include "KGEN/MojoParser/DocString.h"
#include "KGEN/MojoParser/EntryPoint.h"
#include "KGEN/MojoParser/Lexer.h"
#include "KGEN/MojoParser/ParserBase.h"
#include "KGEN/MojoParser/ParserParamEvaluator.h"
#include "KGEN/MojoParser/SharedState.h"
#include "KGEN/MojoTooling/ASTDeclRef.h"
#include "KGEN/MojoTooling/ASTDeclView.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ToolCommon/CompilationOptions.h"

#include "LLCL/Runtime/Runtime.h"
#include "ParserDriverImpl.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/Telemetry/Telemetry.h"
#include "mlir/Bytecode/Encoding.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/SourceMgr.h"

#include <filesystem>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// MojoParserContext::Impl
//===----------------------------------------------------------------------===//

MojoParserContext::Impl::Impl(llvm::SourceMgr &sourceMgr, ParserConfig &config)
    : sharedState(sourceMgr, config), replLocMapper(sourceMgr) {
  // Create the top-level outer decl, which will contain all things we parse.
  module = ModuleOp::create(UnknownLoc::get(sharedState.getContext()));
  topLevelDecl = &sharedState.declResolver->addDecl(
      *module, SMLoc(), StringAttr(), /*parentDecl=*/nullptr, LexerCursor(),
      LexerCursor(), /*indentation=*/-1);
  sharedState.initialize(*topLevelDecl);
}

//===----------------------------------------------------------------------===//
// MojoParserContext
//===----------------------------------------------------------------------===//

MojoParserContext::MojoParserContext(SourceMgr &sourceMgr, ParserConfig &config)
    : impl(std::make_unique<Impl>(sourceMgr, config)) {}

MojoParserContext::~MojoParserContext() {
  // Finalize any imported bytecode now that we've resolved everything. This
  // avoids dangling references to operations from the bytecode.
  (void)impl->sharedState.finalizeImportedBytecodeModules();
}

ModuleOp MojoParserContext::getModule() {
  return cast<ModuleOp>(impl->topLevelDecl->getIfOperation());
}

llvm::SourceMgr &MojoParserContext::getSourceMgr() {
  return impl->sharedState.getSourceMgr();
}

SharedState &MojoParserContext::getSharedState() { return impl->sharedState; }

std::vector<std::string>
MojoParserContext::getModuleSearchDirectories(unsigned fileId) {
  std::vector<std::string> searchDirs;
  impl->sharedState.traverseImportDirectories(fileId, [&](StringRef dir) {
    searchDirs.push_back(dir.str());
    return WalkResult::advance();
  });
  return searchDirs;
}

const KGEN::CompilationOptions &MojoParserContext::getCompilationOptions() {
  return impl->sharedState.options;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

MojoASTDeclRef MojoParserContext::parseFile(unsigned fileId) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();

  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(fileId);
  StringRef filepathStr = sourceBuf->getBufferIdentifier();
  std::filesystem::path filepath(filepathStr.str());

  // If the file is within a package, we create a decl for the outermost package
  // and import this decl from there. This ensures we process relative imports
  // and other package-level constructs correctly.
  ASTDecl *moduleDecl = nullptr;
  if (isMojoSourcePackagePath(filepath.parent_path())) {
    // Collect all of the sub-package names and find the outer most package.
    SmallVector<std::string, 4> packageNames;
    while (isMojoSourcePackagePath(filepath.parent_path())) {
      packageNames.emplace_back(filepath.stem().string());
      filepath = filepath.parent_path();
    }

    // Create the package using the outermost name.
    ASTDecl &packageDecl = impl->sharedState.createPackage(
        filepath.string(), filepath.stem().string());

    // Import the file from within the package.
    std::reverse(packageNames.begin(), packageNames.end());
    moduleDecl = &impl->sharedState.importModule(
        "." + llvm::join(packageNames, "."), cast<PackageOp>(packageDecl),
        SMLoc::getFromPointer(sourceBuf->getBufferStart()));

    // Otherwise, create a decl specifically for the module.
  } else {
    auto fileLoc =
        FileLineColLoc::get(impl->sharedState.getContext(), filepathStr,
                            /*line=*/0, /*column=*/0);
    moduleDecl = &impl->sharedState.createModule(filepath.stem().string(),
                                                 sourceBuf, fileLoc);
  }
  impl->sharedState.declResolver->resolveAllReferencedFrom(*moduleDecl);

  // Now that resolution is finished, cache the state of modules we have parsed.
  // TODO: We should be able to cache even in the presence of warnings and
  // errors. We can store the diagnostics and replay on cache load.
  if (!impl->sharedState.diags.isDiagnosticEmitted())
    impl->sharedState.cacheParsedModules();

  return MojoASTDeclRef(moduleDecl);
}

MojoASTDeclRef MojoParserContext::getDecl(MojoASTTypeRef type) {
  return type.getDecl(impl->sharedState);
}

MojoASTTypeRef
MojoParserContext::concretizeType(KGEN::ParamBindArrayAttr params,
                                  MojoASTTypeRef type) {
  KGEN::LIT::ParserParamEvaluator evaluator(*(impl->sharedState.declResolver));
  for (KGEN::ParamBindAttr paramVal : params)
    evaluator.setParameterValue(paramVal.getName(), paramVal.getValue());

  MojoASTTypeRef concreteType = evaluator.getReboundType(type.getMLIRType());
  concreteType = evaluator.refineType(concreteType.getMLIRType());
  return concreteType;
}
