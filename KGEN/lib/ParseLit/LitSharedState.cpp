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
  bool operator()(const std::vector<LitSharedState::ParamBinding> &lhs,
                  const std::vector<LitSharedState::ParamBinding> &rhs) const {
    if (lhs.size() != rhs.size())
      return lhs.size() < rhs.size();
    for (size_t i = 0, e = lhs.size(); i != e; ++i) {
      if (lhs[i].first != rhs[i].first)
        return lhs[i].first.getAsOpaquePointer() <
               rhs[i].first.getAsOpaquePointer();
      if (lhs[i].second.getStorage() != rhs[i].second.getStorage())
        return lhs[i].second.getStorage() < rhs[i].second.getStorage();
    }
    return false;
  }
};
} // namespace

class LitSharedState::Impl {
public:
  DenseMap<std::pair<ASTDecl *, const LitSharedState::ParamBinding *>,
           ASTTypeStorage *>
      uniquedASTTypes;

  // TODO(complile time): This is horribly inefficient.
  // Switch to StorageUniquer or something else?
  std::set<std::vector<LitSharedState::ParamBinding>, AttributeVectorComparison>
      uniquedParams;

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
  /// This is the decl for the builtin lit.object type.
  ASTDecl *objectDecl = nullptr;
  /// This is a PointerType type which is lowered into POP::PointerType.
  ASTDecl *pointerDecl = nullptr;
  /// This is the decl for the builtin signature type.
  ASTDecl *functionDecl = nullptr;
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

ASTType LitSharedState::getObjectType() const {
  return impl->objectDecl->getResolvedType();
}

ASTType LitSharedState::getPointerType(MValue elementType) {
  auto pointerStruct = cast<LITStructDeclOp>(*impl->pointerDecl);
  assert(pointerStruct.getParamDecls().size() == 1 && "Have an element type");
  ParamDeclAttr elementDecl = pointerStruct.getParamDecls()[0];
  return getASTType(*impl->pointerDecl, ParamBinding{elementDecl, elementType});
}

// FIXME: This isn't correctly parameterized; we need variadics.
ASTType LitSharedState::getFunctionType(MValue resultType) {
  auto functionStruct = cast<LITStructDeclOp>(*impl->functionDecl);
  assert(functionStruct.getParamDecls().size() == 1 && "Have a result type");
  ParamDeclAttr resultDecl = functionStruct.getParamDecls()[0];
  return getASTType(*impl->functionDecl, ParamBinding{resultDecl, resultType});
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
  auto objectOp = b.create<LITStructDeclOp>(loc, b.getStringAttr("object"));
  addCompletedStructDecl(objectOp, impl->objectDecl);
}

auto LitSharedState::getUniquedParams(ArrayRef<ParamBinding> params)
    -> ArrayRef<ParamBinding> {
  // std::set produces stable pointers.
  return *impl->uniquedParams.insert(params.vec()).first;
}

/// Get a uniqued and pointer sized reference to an ASTType.
ASTType LitSharedState::getASTType(ASTDecl &decl,
                                   ArrayRef<ParamBinding> params) {
  params = getUniquedParams(params);
  auto &entry = impl->uniquedASTTypes[{&decl, params.data()}];
  if (entry)
    return ASTType(entry);

  // Ok, the entry hasn't been established, make it now.
  return entry = allocPersistent<ASTTypeStorage>(decl, params);
}

/// Return the MLIR type that corresponds to this AST type, emitting an error
/// if malformed at the specified location and returning a null type.
Type LitSharedState::getMLIRType(MValue typeVal, Location loc) {
  assert(typeVal && "Cannot get MLIR type from a null value");

  // If this value is an attribute for the type, then return it as a Type.
  if (auto attrVal = typeVal.getIfMAValue())
    return ParamRefType::get(attrVal);

  auto type = typeVal.getIfMTValue();

  // Check to see if we've already converted this type.  If so, return it.
  Type &result = type.pointer->mlirType;
  if (result)
    return result;

  // If this is a magic declaration, provide custom lowering for it.
  ASTDecl &decl = type.getDecl();
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
      auto eltType = getMLIRType(type.getParamValues()[0].second, loc);
      return result = POP::PointerType::get(eltType);
    }
    case MagicDeclKind::kFunctionType:
      // TODO: Support argument signature.
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

    SmallVector<ParamBindAttr> attrBindings;
    for (auto [decl, binding] :
         llvm::zip(typeDecl.getParamDecls(), type.getParamValues())) {
      if (decl == binding.first) {
        attrBindings.push_back(ParamBindAttr::get(
            decl, binding.second.lowerToAttribute(*this, loc)));
        continue;
      }
      emitError(loc, "'" + typeDecl.getName() + "' expected ")
          << decl.getName() << " of type " << decl.getType()
          << " but use bound " << binding.first.getName() << " of type "
          << binding.first.getType();
      return TypeCheckErrorType::get(context);
    }

    // Everything looks good, go forth!
    auto typeParams = ParamBindArrayAttr::get(context, attrBindings);
    return result = RefType::get(FlatSymbolRefAttr::get(typeDecl.getNameAttr()),
                                 typeParams);
  }

  // Otherwise it is something unknown.
  emitError(decl.getLoc(), "cannot emit a value as a type");
  return result = TypeCheckErrorType::get(context);
}

Type LitSharedState::getMLIRType(MValue type, SMLoc loc) {
  return getMLIRType(type, translateLocation(loc));
}
