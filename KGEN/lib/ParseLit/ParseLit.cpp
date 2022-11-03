//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the main entrypoints for the lit parser.
//
//===----------------------------------------------------------------------===//

#include "KGEN/ParseLit.h"

#include "LitASTDecl.h"
#include "LitDecls.h"
#include "LitExprs.h"
#include "LitLexer.h"
#include "LitParserBase.h"
#include "LitSharedState.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/Timing.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SourceMgr;

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Add declarations for magic things to the builtins decl.
static void addBuiltinDecls(LitSharedState &sharedState,
                            ASTDecl &builtinsDecl) {
  auto &resolver = *sharedState.declResolver;

  // Make the error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  sharedState.typeCheckErrorTypeDecl =
      &resolver.addMagicDecl("<<type check error>>",
                             MagicDeclKind::kTypeCheckErrorType, &builtinsDecl);
  sharedState.typeCheckErrorTypeDecl->hasReferenceError = true;

  // Add a declarations for builtin types.
  sharedState.typeTypeDecl =
      &resolver.addMagicDecl("type", MagicDeclKind::kTypeType, &builtinsDecl);
  sharedState.indexDecl =
      &resolver.addMagicDecl("index", MagicDeclKind::kIndexType, &builtinsDecl);
  sharedState.noneDecl =
      &resolver.addMagicDecl("None", MagicDeclKind::kNoneType, &builtinsDecl);
  sharedState.pointerDecl = &resolver.addMagicDecl(
      "Pointer", MagicDeclKind::kPointerType, &builtinsDecl);
  sharedState.signatureDecl = &resolver.addMagicDecl(
      "Signature", MagicDeclKind::kSignatureType, &builtinsDecl);

  /// FIXME: These should be a user declared types in the standard library,
  /// which are looked up here instead of being synthesized.

  auto b = builtinsDecl.getDeclEndBuilder();
  auto loc = builtinsDecl.getLoc();

  // Add a declaration for an "object" struct.  This should be written in the
  // standard library.
  auto objectDecl = b.create<LITStructDeclOp>(loc, b.getStringAttr("object"));
  sharedState.objectDecl = &resolver.addDecl(
      objectDecl, &builtinsDecl, LitLexerCursor(), LitLexerCursor(), 0);
  sharedState.objectDecl->setResolvedType(
      sharedState.objectDecl->computeSelfTypeForStruct(sharedState));
  sharedState.objectDecl->resolvedness = DeclResolvedness::fullyResolved;
}

// Parse the specified .lit file into the specified MLIR context.
OwningOpRef<mlir::ModuleOp> M::importLitFile(SourceMgr &sourceMgr,
                                             MLIRContext *context,
                                             mlir::TimingScope &ts) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  context->loadDialect<POP::POPDialect, LITDialect, mlir::index::IndexDialect,
                       KGENDialect, mlir::scf::SCFDialect>();

  // This is the result module we are parsing into.
  auto fileLoc =
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0);
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(fileLoc));

  LitSharedState sharedState(sourceMgr, context);
  LitLexer lexer(sharedState);
  auto startSMLoc = lexer.getToken().getLoc();

  // The outermost scope contains the __builtins__ function definitions.
  // TODO: Add these:
  // https://docs.python.org/3/library/functions.html#built-in-funcs
  // https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
  ASTDecl &builtinsDecl = sharedState.declResolver->addDecl(
      *module, nullptr, lexer.getCursor(), lexer.getCursor(), -1);
  addBuiltinDecls(sharedState, builtinsDecl);

  // Create the module scope which will contain all things we parse.  These
  // shadow the builtins module during name lookup.
  ASTDecl &fileScope = sharedState.declResolver->addDecl(
      *module, &builtinsDecl, lexer.getCursor(), lexer.getCursor(), -1);

  // Parse the file.
  /// file ::= statements
  if (LitParserBase::parseSuite(fileScope, lexer))
    return nullptr;

  // With the top-level of the file parsed, we can now go ahead and resolve all
  // of the deferred declarations.
  sharedState.declResolver->resolveAll(startSMLoc);

  // We fail either if we have a non-recoverable parse error, or if we emitted
  // an error and then recovered.  In either case, the IR will not be valid and
  // the caller should not verify it.
  if (sharedState.errorOccurred)
    return nullptr;
  // Make sure the parse module has no other structural problems detected by
  // the verifier.
  auto verificationTimer = ts.nest("Verify module");
  if (failed(verify(*module)))
    return {};
  return module;
}
