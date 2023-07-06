//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the Mojo parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/MojoParser.h"

#include "ASTDecl.h"
#include "DeclResolver.h"
#include "DocString.h"
#include "KGEN/CompilationOptions.h"
#include "Lexer.h"
#include "ParserBase.h"
#include "ParserDriverImpl.h"
#include "SharedState.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"

#include <filesystem>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// MojoASTDeclRef
//===----------------------------------------------------------------------===//

/// Unwrap a raw ASTDecl pointer.
static ASTDecl *unwrapMojoASTDecl(void *declImpl) {
  assert(declImpl && "expected valid MojoASTDeclRef impl");
  return reinterpret_cast<ASTDecl *>(declImpl);
}

Operation *MojoASTDeclRef::getIfOperation() const {
  return unwrapMojoASTDecl(impl)->getIfOperation();
}

MojoASTTypeRef MojoASTDeclRef::getType() const {
  return TypeSwitch<ASTDecl &, MojoASTTypeRef>(*unwrapMojoASTDecl(impl))
      .Case<VarLetDeclOp, LetRegDeclOp>(
          [&](auto op) { return MojoASTTypeRef(op.getType()); })
      .Default([](auto &&) { return MojoASTTypeRef(nullptr); });
}

std::optional<StringRef> MojoASTDeclRef::getName() const {
  return TypeSwitch<ASTDecl &, std::optional<StringRef>>(
             *unwrapMojoASTDecl(impl))
      .Case<VarLetDeclOp, LetRegDeclOp>([&](auto op) { return op.getName(); });
}

llvm::SMLoc MojoASTDeclRef::getLoc() const {
  return unwrapMojoASTDecl(impl)->getLoc();
}

//===----------------------------------------------------------------------===//
// MojoASTTypeRef
//===----------------------------------------------------------------------===//

/// Unwrap a raw ASTDecl pointer.
static ASTType unwrapMojoASTType(void *declImpl) {
  assert(declImpl && "expected valid MojoASTDeclRef impl");
  return ASTType(Type::getFromOpaquePointer(declImpl));
}

MojoASTTypeRef::MojoASTTypeRef(const mlir::Type &type)
    : MojoASTTypeRef(const_cast<void *>(type.getAsOpaquePointer())) {}

MojoASTDeclRef MojoASTTypeRef::getDecl(SharedState &sharedState) {
  return MojoASTDeclRef(unwrapMojoASTType(impl).getDecl(sharedState));
}

std::string MojoASTTypeRef::getAsString() const {
  return unwrapMojoASTType(impl).getAsString(/*forDiag=*/true);
}

//===----------------------------------------------------------------------===//
// MojoParserContext::Impl
//===----------------------------------------------------------------------===//

MojoParserContext::Impl::Impl(llvm::SourceMgr &sourceMgr,
                              MojoParserConfig &config)
    : sharedState(sourceMgr, config, /*enableCaching=*/false) {
  // Create the top-level outer decl, which will contain all things we parse.
  module = ModuleOp::create(UnknownLoc::get(sharedState.getContext()));
  topLevelDecl = &sharedState.declResolver->addDecl(
      *module, SMLoc(), StringAttr(), /*parentDecl=*/nullptr, LexerCursor(),
      LexerCursor(), /*indentation=*/-1);
  sharedState.initialize(*topLevelDecl);

  // Auto-import the core Lang modules.
  for (StringRef moduleName : SharedState::kBuiltinModuleNames) {
    auto builtinStrAttr = StringAttr::get(module->getContext(), moduleName);
    if (failed(sharedState.declResolver->importModule(
            *topLevelDecl, builtinStrAttr, builtinStrAttr, SMLoc())))
      return;
  }
}

//===----------------------------------------------------------------------===//
// MojoParserContext
//===----------------------------------------------------------------------===//

MojoParserContext::MojoParserContext(SourceMgr &sourceMgr,
                                     MojoParserConfig &config)
    : impl(std::make_unique<Impl>(sourceMgr, config)) {}
MojoParserContext::~MojoParserContext() = default;

ModuleOp MojoParserContext::getModule() {
  return cast<ModuleOp>(impl->topLevelDecl->getIfOperation());
}

llvm::SourceMgr &MojoParserContext::getSourceMgr() {
  return impl->sharedState.getSourceMgr();
}

const KGEN::CompilationOptions &MojoParserContext::getCompilationOptions() {
  return impl->sharedState.options;
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Parse the specified Mojo file into the specified MLIR context. Returns the
/// resultant IR, and the decl for the module represented by the input file.
static std::tuple<OwningOpRef<mlir::ModuleOp>, ASTDecl *>
importMojoFileImpl(SourceMgr &sourceMgr, SharedState &sharedState,
                   mlir::TimingScope &ts,
                   SmallVectorImpl<std::string> *includedFiles = nullptr) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  MLIRContext *context = sharedState.getContext();

  // This is the result module we are parsing into.
  auto fileLoc =
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0);
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(fileLoc));

  // Ensure we finalize bytecode modules even in the case of failure.
  auto clearOnError = llvm::make_scope_exit(
      [&] { (void)sharedState.finalizeImportedBytecodeModules(); });

  Lexer lexer(sharedState, sourceBuf);
  auto startSMLoc = lexer.getToken().getLoc();

  // Create the top-level outer decl, which will contain all things we parse.
  ASTDecl &topLevelDecl = sharedState.declResolver->addDecl(
      *module, startSMLoc, StringAttr(), /*parentDecl=*/nullptr,
      lexer.getCursor(), LexerCursor::getEOF(sourceBuf), -1);
  sharedState.initialize(topLevelDecl);

  // If we are emitting debug info, create a file entry for this file.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (sharedState.diBuilder)
    fileGuard = sharedState.diBuilder->pushFile(fileLoc.getFilename(), "/");

  // Grab a module name for the current input, choosing a dummy name if we don't
  // have one that's valid.
  std::string moduleName =
      std::filesystem::path(fileLoc.getFilename().str()).stem().string();
  if (moduleName.empty())
    moduleName = "<input>";

  // Parse the input module.
  ASTDecl &moduleDecl =
      sharedState.createModule(moduleName, sourceBuf, fileLoc);

  // Resolve everything within the main input module.
  sharedState.declResolver->resolveAllReferencedFrom(moduleDecl);

  // We fail either if we have a non-recoverable parse error, or if we emitted
  // an error and then recovered.  In either case, the IR will not be valid and
  // the caller should not verify it.
  if (sharedState.diags.isErrorEmitted())
    return {nullptr, nullptr};

  // Finalize the imported bytecode now that we've resolved everything. This
  // will drop bytecode operations that never got referenced.
  if (failed(sharedState.finalizeImportedBytecodeModules()))
    return {nullptr, nullptr};

  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  {
    auto verificationTimer = ts.nest("Verify module");
    if (failed(verify(*module)))
      return {};
  }
  clearOnError.release();

  // Now that resolution is finished, cache the state of modules we have parsed.
  // TODO: We should be able to cache even in the presence of warnings and
  // errors. We can store the diagnostics and replay on cache load.
  if (!sharedState.diags.isDiagnosticEmitted())
    sharedState.cacheParsedModules();

  // Set the included files if requested.
  if (includedFiles)
    llvm::append_range(*includedFiles, sharedState.getIncludedFiles());
  return {std::move(module), &moduleDecl};
}

OwningOpRef<mlir::ModuleOp> M::importMojoFile(
    llvm::SourceMgr &sourceMgr, MojoParserConfig &config, mlir::TimingScope &ts,
    SmallVectorImpl<std::string> *includedFiles, bool enableCaching) {
  SharedState sharedState(sourceMgr, config, enableCaching);
  auto [module, topLevelDecl] =
      importMojoFileImpl(sourceMgr, sharedState, ts, includedFiles);
  return std::move(module);
}

MojoASTDeclRef MojoParserContext::parseFile(unsigned fileId) {
  llvm::SourceMgr &sourceMgr = getSourceMgr();

  const llvm::MemoryBuffer *sourceBuf = sourceMgr.getMemoryBuffer(fileId);

  StringRef filepath = sourceBuf->getBufferIdentifier();
  auto fileLoc = FileLineColLoc::get(impl->sharedState.getContext(), filepath,
                                     /*line=*/0, /*column=*/0);
  std::string moduleName =
      std::filesystem::path(filepath.data()).stem().string();
  ASTDecl &moduleDecl =
      impl->sharedState.createModule(moduleName, sourceBuf, fileLoc);
  impl->sharedState.declResolver->resolveAllReferencedFrom(moduleDecl);
  return MojoASTDeclRef(&moduleDecl);
};

LogicalResult M::generateMojoDoc(llvm::SourceMgr &sourceMgr,
                                 MojoParserConfig &config,
                                 raw_ostream &outputOS, mlir::TimingScope &ts) {
  // TODO: We should be able to cache when processing doc strings, but we need
  // to define when/how they get cached to not negatively affect the non-doc
  // string caring path.
  SharedState sharedState(sourceMgr, config, /*enableCaching=*/false);
  auto [module, moduleDecl] = importMojoFileImpl(sourceMgr, sharedState, ts);
  if (!module)
    return failure();

  auto docTS = ts.nest("Mojo Documentation Generation");
  generateMojoDocJSON(*moduleDecl, outputOS);
  return success();
}
