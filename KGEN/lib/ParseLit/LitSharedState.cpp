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
#include "KGEN/KGENDialect/KGENAttrs.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITTypes.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "LitDecls.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/SourceMgr.h"
#include <set>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;
static const char *plural(size_t value) { return value == 1 ? "" : "s"; }

namespace {
struct AttributeVectorComparison {
  bool operator()(const std::vector<ParamBindAttr> &lhs,
                  const std::vector<ParamBindAttr> &rhs) const {
    if (lhs.size() != rhs.size())
      return lhs.size() < rhs.size();
    for (size_t i = 0, e = lhs.size(); i != e; ++i) {
      if (lhs[i] != rhs[i])
        return lhs[i].getAsOpaquePointer() < rhs[i].getAsOpaquePointer();
    }
    return false;
  }
};
} // namespace

class LitSharedState::Impl {
public:
  DenseMap<std::pair<ASTDecl *, const ParamBindAttr *>, ASTTypeStorage *>
      uniquedASTTypes;

  // TODO(complile time): This is horribly inefficient.
  // Switch to StorageUniquer or something else?
  std::set<std::vector<ParamBindAttr>, AttributeVectorComparison> uniquedParams;

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTDecl *typeCheckErrorTypeDecl = nullptr;
  /// This is the "type" type, which can bind to any lit type.
  ASTDecl *typeTypeDecl = nullptr;
  // TODO: Add IntegerLiteralType.
  ASTDecl *floatLiteralTypeDecl = nullptr;
  ASTDecl *stringLiteralTypeDecl = nullptr;
  /// This is the decl for the builtin 'index' type.
  ASTDecl *indexDecl = nullptr;
  /// This is the decl for the builtin 'kgen.none' type.
  ASTDecl *noneDecl = nullptr;
  // This is a PointerType type which is lowered into POP::PointerType.
  ASTDecl *pointerDecl = nullptr;
  /// This is the decl for the builtin signature type.
  ASTDecl *signatureDecl = nullptr;
  /// This is the decl for the builtin lit.object type.
  ASTDecl *objectDecl = nullptr;
};

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
      bufferNameIdentifier(getMainBufferNameIdentifier(sourceMgr, context)),
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
  unsigned mainFileID = sourceMgr.getMainFileID();
  auto lineAndColumn = sourceMgr.getLineAndColumn(loc, mainFileID);
  return FileLineColLoc::get(bufferNameIdentifier, lineAndColumn.first,
                             lineAndColumn.second);
}

ASTType LitSharedState::getTypeCheckErrorType() const {
  return impl->typeCheckErrorTypeDecl->getResolvedType();
}

ASTType LitSharedState::getTypeType() const {
  return impl->typeTypeDecl->getResolvedType();
}

ASTType LitSharedState::getFloatLiteralType() const {
  return impl->floatLiteralTypeDecl->getResolvedType();
}

ASTType LitSharedState::getStringLiteralType() const {
  return impl->stringLiteralTypeDecl->getResolvedType();
}

ASTType LitSharedState::getIndexType() const {
  return impl->indexDecl->getResolvedType();
}

ASTType LitSharedState::getNoneType() const {
  return impl->noneDecl->getResolvedType();
}

// FIXME: This isn't correctly parameterized.
ASTType LitSharedState::getPointerType(TypedAttr elementTypeParam) {
  auto pointerStruct = cast<LITStructDeclOp>(*impl->pointerDecl);
  assert(pointerStruct.getParamDecls().size() == 1 && "Have an element type");
  ParamDeclAttr elementDecl = pointerStruct.getParamDecls()[0];
  return getASTType(*impl->pointerDecl,
                    ParamBindAttr::get(elementDecl, elementTypeParam));
}

/// PointerType can be parameterized on arbitrary meta expressions, but a common
/// thing is to use a literal type, this provides that.
ASTType LitSharedState::getPointerType(ASTType elementType, SMLoc loc) {
  return getPointerType(
      ParameterizedTypeConstantAttr::get(getMLIRType(elementType, loc)));
}

ASTType LitSharedState::getObjectType() const {
  return impl->objectDecl->getResolvedType();
}

// FIXME: This isn't correctly parameterized; we need variadics.
ASTType LitSharedState::getSignatureType() const {
  return impl->signatureDecl->getResolvedType();
}

/// Add declarations for magic things to the builtins decl.
void LitSharedState::addBuiltinTypes(ASTDecl &builtinsDecl) {
  auto &resolver = *declResolver;

  // Make the error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorTypeDecl =
      &resolver.addMagicDecl("<<type check error>>",
                             MagicDeclKind::kTypeCheckErrorType, &builtinsDecl);
  impl->typeCheckErrorTypeDecl->hasReferenceError = true;

  // Add a declarations for builtin types.
  impl->typeTypeDecl =
      &resolver.addMagicDecl("type", MagicDeclKind::kTypeType, &builtinsDecl);
  impl->floatLiteralTypeDecl = &resolver.addMagicDecl(
      "FloatLiteralType", MagicDeclKind::kFloatLiteralType, &builtinsDecl);
  impl->stringLiteralTypeDecl = &resolver.addMagicDecl(
      "StringLiteralType", MagicDeclKind::kStringLiteralType, &builtinsDecl);
  impl->indexDecl =
      &resolver.addMagicDecl("index", MagicDeclKind::kIndexType, &builtinsDecl);
  impl->noneDecl =
      &resolver.addMagicDecl("None", MagicDeclKind::kNoneType, &builtinsDecl);
  impl->signatureDecl = &resolver.addMagicDecl(
      "Signature", MagicDeclKind::kSignatureType, &builtinsDecl);

  /// FIXME: These should be a user declared types in the standard library,
  /// which are looked up here instead of being synthesized.
  auto b = builtinsDecl.getDeclEndBuilder();
  auto loc = builtinsDecl.getLoc();

  // Given a LITStructDeclOp that is completely initialized, add it to the
  // resolver.
  auto addCompletedStructDecl = [&](LITStructDeclOp structOp, ASTDecl *&decl) {
    decl = &resolver.addDecl(structOp, &builtinsDecl, LitLexerCursor(),
                             LitLexerCursor(), 0);
    decl->setResolvedType(decl->computeSelfTypeForStruct(*this));
    decl->resolvedness = DeclResolvedness::fullyResolved;
  };

  // Add a declaration for `struct Pointer<ElementType: type>:` which gets a
  // magic lowering to POP::PointerType.
  auto pointerOp = b.create<LITStructDeclOp>(loc, b.getStringAttr("Pointer"));
  pointerOp.setParamDecls(
      ParamDeclAttr::get("ElementType", b.getType<MLIRTypeType>()));

  addCompletedStructDecl(pointerOp, impl->pointerDecl);
  impl->pointerDecl->magicKind = MagicDeclKind::kPointerType;

  // Add a declaration for an "object" struct.  This should be written in the
  // standard library.
  auto objectOp = b.create<LITStructDeclOp>(loc, b.getStringAttr("object"));
  addCompletedStructDecl(objectOp, impl->objectDecl);
}

//===----------------------------------------------------------------------===//
// ASTType
//===----------------------------------------------------------------------===//

/// Convert this type to a human readable string representation so it can be
/// printed out for diagnostics.
std::string ASTType::getAsString() const {
  if (!pointer)
    return "<<NULL ASTTYPE>>";

  std::string result;
  llvm::raw_string_ostream os(result);
  os << "'";

  if (auto typeDecl = dyn_cast<LITStructDeclOp>(getDecl())) {
    // TODO: Could include name scope information.
    os << typeDecl.getName();
  } else if (getDecl().isMagic()) {
    switch (getDecl().magicKind) {
    case MagicDeclKind::kNormal:
      llvm_unreachable("not a magic declaration?");
    case MagicDeclKind::kPointerType:
      llvm_unreachable("Implemented as a struct, so should be handled");
    case MagicDeclKind::kTypeType:
      os << "type";
      break;
    case MagicDeclKind::kFloatLiteralType:
      os << "FloatLiteralType";
      break;
    case MagicDeclKind::kStringLiteralType:
      os << "StringLiteralType";
      break;
    case MagicDeclKind::kIndexType:
      os << "!builtin.index";
      break;
    case MagicDeclKind::kNoneType:
      os << "!lit.none";
      break;
    case MagicDeclKind::kTypeCheckErrorType:
      os << "<<TypeCheckError>>";
      break;
    case MagicDeclKind::kSignatureType:
      os << "<<FnSignature>>";
      break;
    }
  } else {
    // TODO: Add "aka" information when we have "type defs".
    os << "<<unknown ASTType>>";
  }

  ArrayRef<ParamBindAttr> params = getParamValues();
  if (!params.empty()) {
    os << '[';
    llvm::interleaveComma(params, os, [&](ParamBindAttr bind) {
      // TODO: This isn't really right, but will work enough for now.
      printParamValue(bind.getValue(), os);
    });
    os << ']';
  }

  os << '\'';
  return os.str();
}

mlir::Diagnostic &M::KGEN::LIT::operator<<(mlir::Diagnostic &diag,
                                           ASTType type) {
  return diag << type.getAsString();
}

/// Print to standard error with newline after it, for use in a debugger.
void ASTType::dump() const { llvm::errs() << getAsString() << '\n'; }

ArrayRef<ParamBindAttr>
LitSharedState::getUniquedParams(ArrayRef<ParamBindAttr> params) {
  // std::set produces stable pointers.
  return *impl->uniquedParams.insert(params.vec()).first;
}

/// Get a uniqued and pointer sized reference to an ASTType.
ASTType LitSharedState::getASTType(ASTDecl &decl,
                                   ArrayRef<ParamBindAttr> params) {
  params = getUniquedParams(params);
  auto &entry = impl->uniquedASTTypes[{&decl, params.data()}];
  if (entry)
    return ASTType(entry);

  // Ok, the entry hasn't been established, make it now.
  return entry = allocPersistent<ASTTypeStorage>(decl, params);
}

/// Return the MLIR type that corresponds to this AST type, emitting an error
/// if malformed at the specified location and returning a null type.
Type LitSharedState::getMLIRType(ASTType type, Location loc) {
  assert(type && "Cannot get MLIR type from a null ASTType");
  ASTDecl &decl = type.getDecl();

  // Check to see if we've already converted this type.  If so, return it.
  Type &result = type.pointer->mlirType;
  if (result)
    return result;

  // If this is a magic declaration, provide custom lowering for it.
  if (decl.isMagic()) {
    switch (decl.magicKind) {
    case MagicDeclKind::kNormal:
      llvm_unreachable("not a magic declaration?");
    case MagicDeclKind::kTypeType:
      return result = MLIRTypeType::get(context);
    case MagicDeclKind::kFloatLiteralType:
      return result = Float64Type::get(context);
    case MagicDeclKind::kStringLiteralType:
      // FIXME: Add a sensible type.
      return result = IndexType::get(context);
    case MagicDeclKind::kIndexType:
      return result = IndexType::get(context);
    case MagicDeclKind::kNoneType:
      return result = KGEN::NoneType::get(context);
    case MagicDeclKind::kTypeCheckErrorType:
      return result = TypeCheckErrorType::get(context);
    case MagicDeclKind::kPointerType: {
      assert(type.getParamValues().size() == 1 &&
             "PointerType should have one parameter");
      auto eltType = type.getParamValues()[0].getValue();
      return result = POP::PointerType::get(eltType);
    }

    case MagicDeclKind::kSignatureType:
      // TODO: Support qualified types.
      emitError(loc, "TODO: Cannot emit parameterized builtin type yet");
      return result = TypeCheckErrorType::get(context);
    }
    llvm_unreachable("unknown case");
  }

  // If we have a reference to a struct, check the signatures match.
  if (auto typeDecl = dyn_cast<LITStructDeclOp>(decl)) {
    size_t numDeclParams = typeDecl.getParamDecls().size();
    size_t numTypeParams = type.getParamValues().size();
    if (numDeclParams != numTypeParams) {
      emitError(loc, "'" + typeDecl.getName() + "' requires ")
          << numDeclParams << " meta parameter" << plural(numDeclParams)
          << " but " << numTypeParams << " were bound";
      return result = TypeCheckErrorType::get(context);
    }

    for (auto [decl, bindValue] :
         llvm::zip(typeDecl.getParamDecls(), type.getParamValues())) {
      if (decl == bindValue.getDecl())
        continue;
      emitError(loc, "'" + typeDecl.getName() + "' expected ")
          << decl.getName() << " of type " << decl.getType()
          << " but use bound " << bindValue.getDecl().getName() << " of type "
          << bindValue.getDecl().getType();
      return TypeCheckErrorType::get(context);
    }

    // Everything looks good, go forth!
    auto typeParams = ParamBindArrayAttr::get(context, type.getParamValues());
    return result = RefType::get(FlatSymbolRefAttr::get(typeDecl.getNameAttr()),
                                 typeParams);
  }

  // Otherwise it is something unknown.
  emitError(decl.getLoc(), "cannot emit a value as a type");
  return result = TypeCheckErrorType::get(context);
}

Type LitSharedState::getMLIRType(ASTType type, SMLoc loc) {
  return getMLIRType(type, translateLocation(loc));
}
