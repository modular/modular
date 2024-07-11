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

#include "AsyncRT/Runtime/Runtime.h"
#include "ParserDriverImpl.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/Filesystem/Paths.h"
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

MojoASTDeclRef MojoParserContext::getDecl(MojoASTTypeRef type) {
  return type.getDecl(impl->sharedState);
}

MojoASTTypeRef MojoParserContext::concretizeType(MojoASTTypeRef base,
                                                 ArrayRef<TypedAttr> params,
                                                 MojoASTTypeRef type) {
  KGEN::LIT::ParserParamEvaluator evaluator(
      *(impl->sharedState.declResolver),
      cast<StructDeclOp>(base.getDecl(getSharedState()).decl).getInputParams(),
      params);

  return evaluator.refine(evaluator.getReboundType(type.getMLIRType()));
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Return the package name to use for the given Mojo source package directory.
static std::string getNameForSourcePackage(const std::filesystem::path &path) {
  // FIXME: This is kind of a huge hack, but works around the fact that the
  // Mojo standard library was open sourced with a different package name than
  // the directory, and our tools aren't ready to support that yet. Until we can
  // properly support this case, special case the common project layout of
  // having a `src` directory (that contains the package), under a directory
  // with the package name.
  std::string name = path.stem().string();
  if (name == "src")
    return path.parent_path().stem().string();
  return name;
}

/// Import a module or package that is nested within a source package.
static ASTDecl *buildNestedModuleDecl(std::filesystem::path filepath,
                                      SharedState &sharedState) {
  // Collect all of the sub-package names and find the outer most package.
  SmallVector<std::string, 4> packageNames;
  while (Filesystem::isMojoSourcePackagePath(filepath.parent_path())) {
    packageNames.emplace_back(filepath.stem().string());
    filepath = filepath.parent_path();
  }

  // Create the package using the outermost name.
  ASTDecl &packageDecl = sharedState.createPackage(
      filepath.string(), getNameForSourcePackage(filepath));

  // Import the file from within the package.
  std::reverse(packageNames.begin(), packageNames.end());
  return &sharedState.importModule("." + llvm::join(packageNames, "."),
                                   cast<PackageOp>(packageDecl), SMLoc());
}

/// Create an ASTDecl for the given file module.
static ASTDecl *buildModuleDecl(const std::filesystem::path &filepath,
                                const llvm::MemoryBuffer *sourceBuf,
                                SharedState &sharedState) {
  // If the file is within a package, we create a decl for the outermost package
  // and import this decl from there. This ensures we process relative imports
  // and other package-level constructs correctly.
  if (Filesystem::isMojoSourcePackagePath(filepath.parent_path()))
    return buildNestedModuleDecl(filepath, sharedState);

  // Otherwise, create a decl specifically for the module.
  auto fileLoc =
      FileLineColLoc::get(sharedState.getContext(), filepath.string(),
                          /*line=*/1, /*column=*/1);
  return &sharedState.createModule(filepath.stem().string(), sourceBuf,
                                   fileLoc);
}

/// Create an ASTDecl for the given package.
static ASTDecl *buildPackageDecl(const std::filesystem::path &filepath,
                                 SharedState &sharedState) {
  // If the file is a binary package, just import it.
  if (Filesystem::isMojoBinaryPackagePath(filepath)) {
    return &sharedState.createBinaryPackage(filepath.string(),
                                            filepath.stem().string());
  }
  // If this isn't a source package, bail out.
  if (!Filesystem::isMojoSourcePackagePath(filepath))
    return nullptr;

  // If the file is within a package, we create a decl for the outermost package
  // and import this decl from there. This ensures we process relative imports
  // and other package-level constructs correctly.
  if (Filesystem::isMojoSourcePackagePath(filepath.parent_path()))
    return buildNestedModuleDecl(filepath, sharedState);

  // Otherwise, create a new package.
  return &sharedState.createPackage(filepath.string(),
                                    getNameForSourcePackage(filepath));
}

/// Create an ASTDecl for the given module or package.
static ASTDecl *buildModuleOrPackageDecl(const std::filesystem::path &path,

                                         SharedState &sharedState) {
  // Handle the case of a file.
  if (path.extension() == ".mojo" || path.extension() == ".🔥") {
    SourceMgr &sourceMgr = sharedState.getSourceMgr();
    std::string fullPath;
    int fileId = sourceMgr.AddIncludeFile(path.string(), SMLoc(), fullPath);
    if (!fileId)
      return nullptr;
    return buildModuleDecl(path, sourceMgr.getMemoryBuffer(fileId),
                           sharedState);
  }
  return buildPackageDecl(path, sharedState);
}

MojoASTDeclRef MojoParserContext::parseFile(unsigned fileId,
                                            bool eraseUnparsedDecls) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();

  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(fileId);
  StringRef filepathStr = sourceBuf->getBufferIdentifier();
  std::filesystem::path filepath(filepathStr.str());

  ASTDecl *moduleDecl = buildModuleDecl(filepath, sourceBuf, impl->sharedState);
  impl->sharedState.declResolver->resolveAllReferencedFrom(*moduleDecl,
                                                           eraseUnparsedDecls);

  return MojoASTDeclRef(moduleDecl);
}

MojoASTDeclRef
MojoParserContext::parsePackage(const std::filesystem::path &path) {
  // Check that the path is actually a package.
  if (!(Filesystem::isMojoSourcePackagePath(path) ||
        Filesystem::isMojoBinaryPackagePath(path)))
    return nullptr;
  return parseFileOrPackage(path);
}

MojoASTDeclRef
MojoParserContext::parseFileOrPackage(const std::filesystem::path &path) {
  ASTDecl *moduleDecl = buildModuleOrPackageDecl(path, impl->sharedState);
  if (!moduleDecl)
    return nullptr;
  impl->sharedState.declResolver->resolveAllReferencedFrom(*moduleDecl);

  return MojoASTDeclRef(moduleDecl);
}

MojoASTDeclRef MojoParserContext::parseFileOrPackageNonRecursive(
    const std::filesystem::path &path) {
  ASTDecl *moduleDecl = buildModuleOrPackageDecl(path, impl->sharedState);
  if (!moduleDecl)
    return nullptr;

  // Resolve just the top-level decl.
  (void)impl->sharedState.declResolver->resolveFully(*moduleDecl, SMLoc());
  return MojoASTDeclRef(moduleDecl);
}

bool MojoParserContext::wasErrorEmitted() const {
  return impl->sharedState.diags.isErrorEmitted();
}
