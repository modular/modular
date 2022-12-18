//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file provides the implementation of the LitSharedState class.
//
//===----------------------------------------------------------------------===//

#include "LitSharedState.h"
#include "ASTDecl.h"
#include "ASTType.h"
#include "IRValues.h"
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitDecls.h"
#include "Support/DebugInfoDialect/IR/DIBuilder.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Location.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

class LitSharedState::Impl {
public:
  SymbolTableCollection symbolTables;

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType typeCheckErrorType;
  /// This is the decl for the builtin 'kgen.none' type.
  ASTType noneType;

  // These should move the standard library and be looked up from there on
  // demand.

  /// This is the decl for the builtin lit.object type.
  ASTDecl *objectDecl = nullptr;
};

/// Get the name of the main buffer so we can rapidly build Location objects
/// on demand.
static StringAttr getBufferNameIdentifier(const SourceMgr &sourceMgr,
                                          unsigned bufferID,
                                          MLIRContext *context) {
  auto mainBuffer = sourceMgr.getMemoryBuffer(bufferID);
  StringRef bufferName = mainBuffer->getBufferIdentifier();
  if (bufferName.empty())
    bufferName = "<unknown>";
  return StringAttr::get(context, bufferName);
}

LitSharedState::LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context,
                               const CompilationOptions &options)
    : sourceMgr(sourceMgr), context(context),
      declResolver(std::make_unique<DeclResolver>(*this)), options(options),
      bufferNameIdentifier(getBufferNameIdentifier(
          sourceMgr, sourceMgr.getMainFileID(), context)),
      impl(std::make_unique<Impl>()) {
  if (options.getDebugInfoLevelForInput()) {
    diBuilder = std::make_unique<DebugInfo::DIBuilder>(context);

    // TODO: Dwarf technically has a language for python, but it's not really
    // what we want here AFAICT (our compilation model isn't the same as
    // python's). Figure out what we actually want here (though C works well
    // enough for now).
    diBuilder->initializeCompileUnit(
        llvm::dwarf::DW_LANG_C,
        diBuilder->createFile(bufferNameIdentifier, "/"), "Lit",
        /*isOptimized=*/true, options.getDIEmissionKind());
  }
}

LitSharedState::~LitSharedState() { declResolver.reset(); }

/// Emit an error through the parser's logic.
InFlightDiagnostic LitSharedState::emitError(Location loc, const Twine &twine) {
  errorOccurred = true;
  return mlir::emitError(loc, twine);
}

/// Emit an error through the parser's logic.
InFlightDiagnostic LitSharedState::emitError(llvm::SMLoc loc,
                                             const Twine &twine) {
  return emitError(translateLocation(loc), twine);
}

/// Encode the specified source location information into a Location object
/// for attachment to the IR or error reporting.
Location LitSharedState::translateLocation(SMLoc loc) const {
  // TODO: Implement a cache here to speed up location translation.
  unsigned bufferID = sourceMgr.FindBufferContainingLoc(loc);
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, bufferID);

  StringAttr bufferName;
  if (bufferID == sourceMgr.getMainFileID())
    bufferName = bufferNameIdentifier;
  else
    bufferName = getBufferNameIdentifier(sourceMgr, bufferID, getContext());

  auto fileLoc = FileLineColLoc::get(bufferName, lineAndColumn.first,
                                     lineAndColumn.second);
  return diBuilder ? diBuilder->createScopedLoc(fileLoc) : fileLoc;
}

ASTType LitSharedState::getTypeCheckErrorType() const {
  return impl->typeCheckErrorType;
}
ASTType LitSharedState::getNoneType() const { return impl->noneType; }

ASTType LitSharedState::getObjectType() const {
  return impl->objectDecl->getSelfType();
}

/// Add declarations for magic things to the builtins decl.
void LitSharedState::addBuiltinTypes(ASTDecl &builtinsDecl) {
  DeclResolver &resolver = *declResolver;

  // Add a declarations for builtin types.
  impl->noneType = LIT::NoneType::get(context);

  // Make the type check error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorType = TypeCheckErrorType::get(context);

  OpBuilder b = builtinsDecl.getDeclEndBuilder();
  Location loc = translateLocation(builtinsDecl.getLoc());

  // Add an empty struct with the specified name to the resolver.
  auto addEmptyStructDecl = [&](StringRef name, ASTDecl *&decl) {
    auto structOp = b.create<StructDeclOp>(loc, b.getStringAttr(name));
    decl = &resolver.addDecl(structOp, builtinsDecl.getLoc(),
                             structOp.getNameAttr(), &builtinsDecl,
                             LitLexerCursor(), LitLexerCursor(), 0);
    decl->setSelfType(decl->computeSelfTypeForStruct(*this));
    decl->resolvedness = DeclResolvedness::fullyResolved;
  };

  ASTDecl *mlirAttrDecl = nullptr, *mlirOpDecl = nullptr,
          *mlirTypeDecl = nullptr;
  addEmptyStructDecl("__mlir_attr", mlirAttrDecl);
  mlirAttrDecl->setSelfType(MagicMLIRAttrType::get(context));

  addEmptyStructDecl("__mlir_op", mlirOpDecl);
  mlirOpDecl->setSelfType(MagicMLIROpType::get(context));

  addEmptyStructDecl("__mlir_type", mlirTypeDecl);
  mlirTypeDecl->setSelfType(MagicMLIRTypeType::get(context));

  // Add a declaration for an "object" struct.  This should be written in the
  // standard library.
  addEmptyStructDecl("object", impl->objectDecl);
}

/// Set the symbol for the specified declaration (known to be an operation)
/// into the MLIR symbol table for its container.  If the symbol is already
/// declared in the same MLIR scope, then return the conflicting operation.
Operation *LitSharedState::setResolvedDeclSymbol(Operation *declOp) {
  assert(declOp && "Cannot set a symbol for non-operation decl");

  // We look up the symbol in the enclosing symbol table.  For example, for a
  // method in a struct, we use the struct as the symbol table.  For a top-level
  // function we use the global module.
  Operation *parentSymbolTableOp =
      SymbolTable::getNearestSymbolTable(declOp->getParentOp());
  SymbolTable &symTab = impl->symbolTables.getSymbolTable(parentSymbolTableOp);

  // Insert the operation into the symbol table and see if it got renamed.
  auto origName = SymbolTable::getSymbolName(declOp);
  if (symTab.insert(declOp) == origName)
    return nullptr; // No conflict, done.

  return symTab.lookup(origName);
}

//===----------------------------------------------------------------------===//
// Name Lookup
//===----------------------------------------------------------------------===//

/// Perform a name lookup in the specified scope and return the named
/// declaration as a LookupResult.
auto LitSharedState::lookupAndResolveDecl(StringRef name, SMLoc loc,
                                          ASTDecl &scope) -> LookupResult {

  // Ensure the context is fully resolved, so all its members are known.  It
  // would be bad to look something up in a scope without all members known.
  // FIXME(Issue#5975): FuncOp shouldn't be special cased.
  if (!isa<FuncOp>(scope)) {
    if (failed(
            declResolver->resolve(scope, DeclResolvedness::fullyResolved, loc)))
      return LookupResult::getErroneous();
  }

  // Look up the name.
  const TinyPtrVector<ASTDecl *> *entry = scope.lookup(name);
  // If nothing was found, return a failure.
  if (!entry)
    return LookupResult::getFailure();

  // If the lookup succeeded, make sure the signature for the referenced decls
  // are understood.
  for (auto *decl : *entry) {
    if (failed(declResolver->resolve(*decl, DeclResolvedness::signatureResolved,
                                     loc))) {
      // If the decl was erroneous somehow, then don't form a reference to it,
      // the error has already been diagnosed.
      return LookupResult::getErroneous();
    }
  }

  // We return a pointer into the TinyPtrVector entry in the scope.  This should
  // be stable because you can't perform a lookup into a decl that has unknown
  // entries, and we just resolved all the signatures for all the decls.
  return LookupResult::getSuccess(*entry);
}

/// Perform a name lookup for a member in the specified type.
auto LitSharedState::lookupAndResolveDecl(StringRef name, SMLoc loc,
                                          ASTType scope) -> LookupResult {
  if (auto *decl = scope.getDecl(*this))
    return lookupAndResolveDecl(name, loc, *decl);
  return LookupResult::getFailure();
}

/// Lookup the `Error` type in the current context and return it if found,
/// otherwise emit an error and return null.
ASTType LitSharedState::lookupErrorType(SMLoc loc, ASTDecl &context) {
  LookupResult result = lookupAndResolveDecl(
      StringAttr::get(getContext(), "Error"), loc, context);
  if (result.isErroneous())
    return {};
  if (result.isFailure()) {
    emitError(loc, "could not find an 'Error' type");
    return {};
  }
  // The overload set may contain multiple entries, but if it is a struct, it
  // must be a single entry and therefore we can just check that one.
  ASTDecl &firstDecl = *result.getIfSuccess()[0];
  auto structOp = dyn_cast<StructDeclOp>(firstDecl);
  if (!structOp) {
    auto diag = emitError(loc, "'Error' doesn't resolve to a type");
    diag.attachNote(translateLocation(firstDecl.getLoc()))
        << "'Error' declared here";
    return {};
  }
  if (!structOp.getInputParamDecls().empty()) {
    auto diag = emitError(loc, "'Error' resolves to a parameterized type");
    diag.attachNote(translateLocation(firstDecl.getLoc()))
        << "'Error' declared here";
    return {};
  }
  return firstDecl.getSelfType();
}

/// Lookup the Error type and wrap it in a variant with the specified normal
/// value type.  Return the result, or error if the Error type couldn't be
/// found.
ASTType LitSharedState::lookupErrorOrType(ASTType valueType, SMLoc loc,
                                          ASTDecl &context) {
  if (auto errorType = lookupErrorType(loc, context))
    return POP::VariantType::get({errorType, valueType});
  return {};
}
