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
#include <set>

using namespace M;
using namespace M::KGEN;
using namespace M::KGEN::LIT;

using llvm::SMLoc;
using llvm::SourceMgr;

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

  // TODO(compile time): This is horribly inefficient.
  // Switch to StorageUniquer or something else?
  std::set<std::vector<LitSharedState::ParamBinding>, AttributeVectorComparison>
      uniquedParams;

  /// This is the AST type that corresponds to TypeCheckErrorType.
  ASTDecl *typeCheckErrorTypeDecl = nullptr;

  /// This is the type of values like "__mlir_op.`pop.add`".
  ASTDecl *unboundMLIROperatorTypeDecl = nullptr;
  /// This is the __mlir_type declaration.
  ASTDecl *mlirTypeDecl = nullptr;
  /// This is the "type" type, which can bind to any lit type.
  ASTDecl *typeTypeDecl = nullptr;
  // TODO: Add IntegerLiteralType.
  ASTDecl *floatLiteralTypeDecl = nullptr;
  ASTDecl *stringLiteralTypeDecl = nullptr;
  /// This is the decl for the builtin 'kgen.none' type.
  ASTDecl *noneDecl = nullptr;
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

ASTType LitSharedState::getTypeCheckErrorType() const {
  return impl->typeCheckErrorTypeDecl->getResolvedType();
}

/// This is the type of values like "__mlir_op.`pop.add`"
ASTType LitSharedState::getUnboundMLIROperatorType() const {
  return impl->unboundMLIROperatorTypeDecl->getResolvedType();
}

ASTDecl &LitSharedState::getMLIRTypeScope() const {
  return *impl->mlirTypeDecl;
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

ASTType LitSharedState::getNoneType() const {
  return impl->noneDecl->getResolvedType();
}

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

  // Make the error type.  Anything that references this will
  // considering it erroneous and already declared as such.
  impl->typeCheckErrorTypeDecl =
      &resolver.addMagicDecl("<<type check error>>",
                             MagicDeclKind::kTypeCheckErrorType, &builtinsDecl);
  impl->typeCheckErrorTypeDecl->hasReferenceError = true;

  // This is the type used by unbound MLIR operator types.
  impl->unboundMLIROperatorTypeDecl = &resolver.addMagicDecl(
      "<<unbound MLIR operator type>>", MagicDeclKind::kUnboundMLIROperatorType,
      &builtinsDecl);

  impl->mlirTypeDecl = &resolver.addMagicDecl(
      "__mlir_type", MagicDeclKind::k__mlir_type, &builtinsDecl);
  resolver.addMagicDecl("__mlir_op", MagicDeclKind::k__mlir_op, &builtinsDecl);
  resolver.addMagicDecl("__mlir_attr", MagicDeclKind::k__mlir_attr,
                        &builtinsDecl);

  // Add a declarations for builtin types.
  impl->typeTypeDecl =
      &resolver.addMagicDecl("type", MagicDeclKind::kTypeType, &builtinsDecl);
  impl->floatLiteralTypeDecl = &resolver.addMagicDecl(
      "FloatLiteralType", MagicDeclKind::kFloatLiteralType, &builtinsDecl);
  impl->stringLiteralTypeDecl = &resolver.addMagicDecl(
      "StringLiteralType", MagicDeclKind::kStringLiteralType, &builtinsDecl);
  impl->noneDecl =
      &resolver.addMagicDecl("None", MagicDeclKind::kNoneType, &builtinsDecl);

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
  ASTDecl &decl = type.getDecl(*this);
  if (decl.isMagic()) {
    switch (decl.magicKind) {
    case MagicDeclKind::kNormal:
      llvm_unreachable("not a magic declaration?");
    case MagicDeclKind::k__mlir_op:
    case MagicDeclKind::k__mlir_attr:
      emitError(loc, "__mlir_* is not a type");
      return result = TypeCheckErrorType::get(context);
    case MagicDeclKind::k__mlir_type:
      emitError(
          loc, "cannot use __mlir_type directly, use properties of it instead");
      return result = TypeCheckErrorType::get(context);
    case MagicDeclKind::kUnboundMLIROperatorType:
      emitError(loc, "cannot use __mlir_op operation as a type");
      return result = TypeCheckErrorType::get(context);
    case MagicDeclKind::kTypeType:
      return result = MLIRTypeType::get(context);
    case MagicDeclKind::kFloatLiteralType:
      return result = Float64Type::get(context);
    case MagicDeclKind::kStringLiteralType:
      // FIXME: Add a sensible type.
      return result = IndexType::get(context);
    case MagicDeclKind::kNoneType:
      return result = KGEN::NoneType::get(context);
    case MagicDeclKind::kTypeCheckErrorType:
      return result = TypeCheckErrorType::get(context);
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
    return result = LITDeclRefType::get(decl.getSymbolRef(), typeParams);
  }

  // If this is a direct reference to an MLIR type, use it.
  if (auto type = decl.getIfMLIRType())
    return result = type;

  // Otherwise it is something unknown.
  emitError(decl.getLoc(), "cannot emit a value as a type");
  return result = TypeCheckErrorType::get(context);
}

Type LitSharedState::getMLIRType(MValue type, SMLoc loc) {
  return getMLIRType(type, translateLocation(loc));
}

/// When a lookup in __mlir_type fails for a named field, this method tries to
/// resolve it.  On success, it lazily creates a resolved declaration.  On
/// failure, it bails out.
ASTDecl *LitSharedState::synthesizeMLIRTypeDeclEntry(StringRef name, SMLoc loc,
                                                     ASTDecl &scope) {
  Type result;
  {
    // Capture errors thrown by parseType and ignore them.
    // FIXME: This doesn't silence errors!
    mlir::ScopedDiagnosticHandler handler(getContext(),
                                          [](Diagnostic &diag) {});

    // FIXME(https://github.com/llvm/llvm-project/issues/58964)
    // Copy the string into a temporary smallvector so we can make sure it is
    // nul terminated for the MLIR asmparser.
    SmallString<64> tmpBuf(name.begin(), name.end());
    tmpBuf.push_back(0);
    result = mlir::parseType(StringRef(tmpBuf).drop_back(), getContext());
  }
  if (!result) {
    emitError(loc, "unknown MLIR type: ") << name;
    return nullptr;
  }

  return &declResolver->addFullyResolvedDecl(
      result, StringAttr::get(getContext(), name), translateLocation(loc),
      getTypeType(), &scope);
}

/// Given an MLIR type, return an ASTType that we can use for type system
/// processing.  This should only be used for low level operations touching
/// MLIR, it isn't efficient and shouldn't be used for general user defined
/// types.
ASTType LitSharedState::getASTTypeForMLIRType(Type mlirType, SMLoc loc) {
  // To get an ASTType from an MLIR type, we stringify the MLIR type and look
  // it up on the __mlir_type declaration.
  std::string typeStr;
  llvm::raw_string_ostream(typeStr) << mlirType;

  // See if we already have this declaration.
  auto &mlirTypeScope = getMLIRTypeScope();
  ASTDecl *typeDecl =
      mlirTypeScope.lookup(StringAttr::get(getContext(), typeStr));

  // If not, synthesize it.
  if (!typeDecl) {
    typeDecl = synthesizeMLIRTypeDeclEntry(typeStr, loc, mlirTypeScope);
    if (!typeDecl)
      return {};
  }

  return getASTType(*typeDecl, {});
}
