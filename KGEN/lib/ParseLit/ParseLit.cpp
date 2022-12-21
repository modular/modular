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

#include "ASTDecl.h"
#include "KGEN/CompilationOptions.h"
#include "LitDecls.h"
#include "LitLexer.h"
#include "LitParserBase.h"
#include "LitSharedState.h"

#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/ZAPDialect/ZAPDialect.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "Support/HLCFDialect/HLCFDialect.h"
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

// Parse the specified .lit file into the specified MLIR context.
OwningOpRef<mlir::ModuleOp>
M::importLitFile(SourceMgr &sourceMgr, MLIRContext *context,
                 mlir::TimingScope &ts,
                 const KGEN::CompilationOptions &options) {
  auto sourceBuf = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  context->loadDialect<DebugInfo::DebugInfoDialect, HLCF::HLCFDialect,
                       POP::POPDialect, LITDialect, mlir::index::IndexDialect,
                       KGENDialect, ZAP::ZAPDialect, mlir::scf::SCFDialect>();

  // This is the result module we are parsing into.
  auto fileLoc =
      FileLineColLoc::get(context, sourceBuf->getBufferIdentifier(), /*line=*/0,
                          /*column=*/0);
  mlir::OwningOpRef<ModuleOp> module(ModuleOp::create(fileLoc));

  LitSharedState sharedState(sourceMgr, *module, options);
  LitLexer lexer(sharedState, sourceBuf);
  auto startSMLoc = lexer.getToken().getLoc();

  // The outermost scope contains the __builtins__ function definitions.
  // TODO: Add these:
  // https://docs.python.org/3/library/functions.html#built-in-funcs
  // https://docs.python.org/3/reference/executionmodel.html#naming-and-binding
  ASTDecl &builtinsDecl = sharedState.declResolver->addDecl(
      *module, startSMLoc, StringAttr(), nullptr, lexer.getCursor(),
      lexer.getCursor(), -1);
  sharedState.addBuiltinTypes(builtinsDecl);
  builtinsDecl.resolvedness = DeclResolvedness::fullyResolved;

  // Create the module scope which will contain all things we parse.  These
  // shadow the builtins module during name lookup.
  ASTDecl &fileScope = sharedState.declResolver->addDecl(
      *module, startSMLoc, StringAttr(), &builtinsDecl, lexer.getCursor(),
      lexer.getCursor(), -1);

  // If we are emitting debug info, create a file entry for this file.
  DebugInfo::DIBuilder::ScopeGuard fileGuard;
  if (sharedState.diBuilder)
    fileGuard = sharedState.diBuilder->pushFile(fileLoc.getFilename(), "/");

  // Parse the file.
  /// file ::= statements
  if (LitParserBase::parseSuite(fileScope, lexer))
    return nullptr;

  // With the top-level of the file parsed, we can now go ahead and resolve all
  // of the deferred declarations.
  sharedState.declResolver->resolveAll();

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
