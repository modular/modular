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
  /// This is the "type" type, which can bind to any lit type.
  ASTType typeType;

  /// This is the __mlir_type declaration.
  ASTDecl *mlirTypeDecl = nullptr;
  /// This is the decl for the builtin signature type.
  ASTDecl *functionDecl = nullptr;

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

ASTDecl &LitSharedState::getMLIRTypeScope() const {
  return *impl->mlirTypeDecl;
}

ASTType LitSharedState::getTypeCheckErrorType() const {
  return impl->typeCheckErrorType;
}
ASTType LitSharedState::getTypeType() const { return impl->typeType; }
ASTType LitSharedState::getNoneType() const { return impl->noneType; }

ASTType LitSharedState::getObjectType() const {
  return impl->objectDecl->getResolvedType();
}

// FIXME: This isn't correctly parameterized; we need variadics.
ASTType LitSharedState::getFunctionType(MValue resultType) {
  auto functionStruct = cast<LITStructDeclOp>(*impl->functionDecl);
  assert(functionStruct.getParamDecls().size() == 1 && "Have a result type");
  ParamDeclAttr resultDecl = functionStruct.getParamDecls()[0];
  return getASTType(*impl->functionDecl, ParamBinding{resultDecl, resultType});
}

/// Add declarations for magic things to the builtins decl.
void LitSharedState::addBuiltinTypes(ASTDecl &builtinsDecl, SMLoc smLoc) {
  auto &resolver = *declResolver;

  /// FIXME: These should be a user declared types in the standard library,
  /// which are looked up here instead of being synthesized.
  auto b = builtinsDecl.getDeclEndBuilder();
  auto loc = builtinsDecl.getLoc();

  // Given a LITStructDeclOp that is completely initialized, add it to the
  // resolver.
  auto addCompletedStructDecl = [&](LITStructDeclOp structOp, ASTDecl *&decl) {
    decl = &resolver.addDecl(structOp, structOp.getNameAttr(), &builtinsDecl,
                             LitLexerCursor(), LitLexerCursor(), 0);
    decl->setResolvedType(decl->computeSelfTypeForStruct(*this));
    decl->resolvedness = DeclResolvedness::fullyResolved;
  };

  auto addEmptyStructDecl = [&](StringRef name, ASTDecl *&decl) {
    auto structOp = b.create<LITStructDeclOp>(loc, b.getStringAttr(name));
    addCompletedStructDecl(structOp, decl);
  };

  addEmptyStructDecl("__mlir_type", impl->mlirTypeDecl);
  impl->mlirTypeDecl->magicKind = MagicDeclKind::k__mlir_type;

  ASTDecl *mlirOpDecl = nullptr, *mlirAttrDecl = nullptr;
  addEmptyStructDecl("__mlir_op", mlirOpDecl);
  mlirOpDecl->magicKind = MagicDeclKind::k__mlir_op;
  addEmptyStructDecl("__mlir_attr", mlirAttrDecl);
  mlirAttrDecl->magicKind = MagicDeclKind::k__mlir_attr;

  // Add a declarations for builtin types.
  impl->typeType = resolver
                       .addFullyResolvedDecl(
                           MLIRTypeType::get(context), "type", loc,
                           impl->mlirTypeDecl->getResolvedType(), &builtinsDecl)
                       .getResolvedType();
  impl->noneType = KGEN::NoneType::get(context);

  // Make the error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorType = TypeCheckErrorType::get(context);

  // Add a declaration for `struct Function<ResultType: type>:` which gets a
  // magic lowering to KGEN::SignatureType.
  // TODO: This currently only carries result type, it should carry variadic
  // meta parameter and argument packs.
  auto functionOp = b.create<LITStructDeclOp>(loc, b.getStringAttr("Function"));
  functionOp.setParamDecls(
      ParamDeclAttr::get("ResultType", b.getType<MLIRTypeType>()));
  addCompletedStructDecl(functionOp, impl->functionDecl);
  impl->functionDecl->magicKind = MagicDeclKind::kFunctionType;

  // Add a declaration for an "object" struct.  This should be written in the
  // standard library.
  addEmptyStructDecl("object", impl->objectDecl);
}

/// Get a uniqued and pointer sized reference to an ASTType.
ASTType LitSharedState::getASTType(ASTDecl &decl,
                                   ArrayRef<ParamBinding> params) {
  // If this decl is just an MLIR type already, return it.
  if (auto type = decl.getIfMLIRType()) {
    assert(params.empty() && "Cannot parameterize an fixed mlir type");
    return ASTType(type);
  }

  auto symbol = decl.getSymbolRef();
  assert(symbol && "cannot get type for decl without a symbol");

  SmallVector<ParamBindAttr> paramValues;
  for (auto param : params) {
    TypedAttr value = param.second.getIfMAValue();
    if (!value) {
      auto mlirType = param.second.getIfMTValue().getMLIRType();
      value = ParameterizedTypeConstantAttr::get(mlirType);
    }

    paramValues.push_back(ParamBindAttr::get(param.first, value));
  }

  return ASTType(LITDeclRefType::get(
      symbol, ParamBindArrayAttr::get(getContext(), paramValues)));
}

/// Return the MLIR type that corresponds to this AST type, emitting an error
/// if malformed at the specified location and returning a null type.
Type LitSharedState::getMLIRType(MValue typeVal, Location loc) {
  assert(typeVal && "Cannot get MLIR type from a null value");

  // If this value is an attribute for the type, then return it as a Type.
  if (auto attrVal = typeVal.getIfMAValue())
    return ParamRefType::get(attrVal);

  return typeVal.getIfMTValue().getMLIRType();
}

Type LitSharedState::getMLIRType(MValue type, SMLoc loc) {
  return getMLIRType(type, translateLocation(loc));
}
