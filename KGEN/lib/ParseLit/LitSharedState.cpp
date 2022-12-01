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
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/SourceMgr.h"

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

class LitSharedState::Impl {
public:
  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTType typeCheckErrorType;
  /// This is the decl for the builtin 'kgen.none' type.
  ASTType noneType;
  /// This is the decl for the builtin error type.
  ASTType errorType;

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

LitSharedState::LitSharedState(llvm::SourceMgr &sourceMgr, MLIRContext *context)
    : sourceMgr(sourceMgr), context(context),
      declResolver(std::make_unique<DeclResolver>(*this)),
      bufferNameIdentifier(getBufferNameIdentifier(
          sourceMgr, sourceMgr.getMainFileID(), context)),
      impl(std::make_unique<Impl>()) {}

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

  return FileLineColLoc::get(bufferName, lineAndColumn.first,
                             lineAndColumn.second);
}

ASTType LitSharedState::getTypeCheckErrorType() const {
  return impl->typeCheckErrorType;
}
ASTType LitSharedState::getNoneType() const { return impl->noneType; }
ASTType LitSharedState::getErrorType() const { return impl->errorType; }

ASTType LitSharedState::getObjectType() const {
  return impl->objectDecl->getSelfType();
}

ASTType LitSharedState::getErrorOrType(ASTType valueType) const {
  return RaisesOrType::get(valueType);
}

/// Add declarations for magic things to the builtins decl.
void LitSharedState::addBuiltinTypes(ASTDecl &builtinsDecl) {
  auto &resolver = *declResolver;

  // Add a declarations for builtin types.
  impl->noneType = KGEN::NoneType::get(context);

  // Make the type check error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorType = TypeCheckErrorType::get(context);

  // The builtin error type always references the library `Error` type.
  impl->errorType =
      DeclRefType::get(FlatSymbolRefAttr::get(getContext(), "Error"));

  auto b = builtinsDecl.getDeclEndBuilder();
  auto loc = translateLocation(builtinsDecl.getLoc());

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
