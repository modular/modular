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

#include "LitDecls.h"
#include "LitExprs.h"
#include "LitParserBase.h"
#include "LitScope.h"
#include "LitSharedState.h"

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
// LitSharedState
//===----------------------------------------------------------------------===//

/// Get the name of the main buffer so we can rapidly build Location objects
/// on demand.
static StringAttr getMainBufferNameIdentifier(const SourceMgr &sourceMgr,
                                              MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

LitSharedState::LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context),
      declResolver(std::make_unique<DeclResolver>(*this)),
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)) {}

LitSharedState::~LitSharedState() { declResolver.reset(); }

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location LitSharedState::translateLocation(SMLoc loc) {
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Add a declaration for an "index" struct, which is used as a transitionary
/// thing as we bring up full type support.  This should be eliminated.
static void makeIndexDecl(LitSharedState &sharedState, Scope &builtinsScope) {
  auto b = builtinsScope.getDeclEndBuilder();
  auto loc = builtinsScope.getLoc();
  auto indexDecl = b.create<LITStructDeclOp>(loc, b.getStringAttr("index"));
  indexDecl.getRegion().push_back(new Block());
  sharedState.indexScope = &sharedState.declResolver->addFullyResolvedDecl(
      indexDecl, &builtinsScope);
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
  Scope &builtinsScope = sharedState.declResolver->addDecl(
      *module, nullptr, lexer.getCursor(), lexer.getCursor(), -1);

  // Add 'index' as a magic type for testing/transition.
  // TODO: Remove this eventually.
  makeIndexDecl(sharedState, builtinsScope);

  // Create the module scope which will contain all things we parse.  These
  // shadow the builtins module during name lookup.
  Scope &fileScope = sharedState.declResolver->addDecl(
      *module, &builtinsScope, lexer.getCursor(), lexer.getCursor(), -1);

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
