//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers a variety of high level Mojo types in the 'lit' dialect
// to lower level KGEN abstractions.  Notably, this eliminates symbol based
// struct references (in favor of `!kgen.struct`), `!lit.ref` => `!kgen.pointer`
// etc.  This runs immediately after the LowerLIT pass.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace llvm {
template <>
struct PointerLikeTypeTraits<KGEN::StructType>
    : public PointerLikeTypeTraits<mlir::Type> {
  static inline KGEN::StructType getFromVoidPointer(void *p) {
    return KGEN::StructType::getFromOpaquePointer(p);
  }
};
} // namespace llvm

//===----------------------------------------------------------------------===//
// Struct Lowering
//===----------------------------------------------------------------------===//

namespace {
/// Information about a struct declaration.
struct StructDeclarations {
  /// A map from struct name and field name to index. Used for lowering `insert`
  /// and `extract` ops.
  DenseMap<std::pair<StringAttr, StringAttr>, int64_t> fieldIndices;

  /// A struct declaration representing its field names and types and input
  /// parameters, as well as register passability.
  struct Decl {
    /// The field names and types.
    SmallVector<std::pair<StringAttr, Type>> fields;
    /// The struct input parameters.
    ParamDeclArrayAttr decls;
    /// A flag indicating if the struct is register passable or not.
    bool isRegisterPassable;
  };

  /// A map of all struct declarations by name.
  DenseMap<StringAttr, Decl> decls;

  /// Get the declaration for the type reference.
  const Decl &getDecl(DeclRefType ref) { return decls.at(ref.getName()); }
};

/// Struct operations need to refer to the struct declaration symbol.
struct StructOperationLowerer : public mlir::IRRewriter {
  explicit StructOperationLowerer(MLIRContext *ctx,
                                  StructDeclarations &structDecls);

  /// Get the index of the struct field.
  int64_t getField(StringAttr name, DeclRefType ref) const {
    return structDecls.fieldIndices.lookup({ref.getName(), name});
  }

  /// Replace a LIT struct with a KGEN struct or an arbitrary type if it is was
  /// a single-element type that got flattened.
  /// This function returns PointerUnion<KGEN::StructType, Type> to
  /// distinguish between a flattened single-element struct and a struct that
  /// has multiple elements.
  /// Using PointerUnion<KGEN::StructType, Type> instead of Type because
  /// the single-element itself can also be a struct.
  PointerUnion<KGEN::StructType, Type> substituteStructRef(DeclRefType ref);

  /// Try to build debug information for the given struct ref.
  DebugInfo::DIType
  buildDebugInfoForStructRef(DeclRefType ref,
                             DebugInfo::DebugInfoTypeConverter &converter);

  /// Materialize destination conversions.
  template <typename OpT>
  LogicalResult materializeLowering(OpT op);

  /// Attr replace functions that give more control on what and when to cache
  Type replace(Type type);

  /// Type replace functions that give more control on what and when to cache
  Attribute replace(Attribute attr);

  /// AttrType replace implementation helper.
  /// This function is doing the replace traversal the same as
  /// mlir::AttrTypeReplacer, but has more controlled caching mechanism.
  /// We don't want to cache the replaced result of erased Pointer type for
  /// recursive nested struct field to avoid the same DeclRefType
  /// being replace in general cases. We only want to replace the one that is
  /// used to access the struct field.
  template <typename T, typename U = std::conditional_t<
                            std::is_base_of_v<Type, T>, Type, Attribute>>
  U replaceImpl(T value);

  /// Replace elements in an Operation.
  void replaceElementsIn(Operation *op);

  /// The struct decl map.
  StructDeclarations &structDecls;

  /// Set to the value of an invalid DeclRefType.
  DeclRefType errDeclRef;

  /// The empty `#kgen.struct<>` attribute, which has empty struct type.
  KGEN::StructAttr emptyStructAttr;

  /// Cache to memorize AttrType replacement results.
  DenseMap<const void *, const void *> attrTypeReplaceCache;

  /// Flag to erase the type for a struct field of recursive pointer.
  bool eraseRecursivePointerField = false;

  /// Flag to track recursive AttrType replacement when erasing recursive struct
  /// pointer type.
  bool erasedType = false;

  /// Seen Types when replacing.
  DenseSet<Type> seenTypes;

  /// Seen Attributes when replacing.
  DenseSet<Attribute> seenAttrs;

  /// Flag to run debug type conversion on updated types.
  bool runDebugTypeConversion = false;

  /// Debug type converter.
  DebugInfo::DebugInfoTypeConverter debugTypeConverter;

  /// Cached anyRegTypeType so we don't need to recreate it.
  AnyRegTypeType anyRegTypeType;
};
} // namespace

StructOperationLowerer::StructOperationLowerer(MLIRContext *ctx,
                                               StructDeclarations &structDecls)
    : IRRewriter(ctx), structDecls(structDecls),
      anyRegTypeType(AnyRegTypeType::get(ctx)) {

  // Get the empty `#kgen.struct<>` attribute, which has empty struct type.
  auto emptyStructType = StructType::get(ctx, {});
  emptyStructAttr = KGEN::StructAttr::get({}, emptyStructType);

  // Build a converter to handle updating converted types within debug info
  // constructs.
  debugTypeConverter.addConversion([&](Type type) -> std::optional<Type> {
    Type newType = replace(type);
    if (newType != type)
      return debugTypeConverter.convertDebugType(newType);
    return std::nullopt;
  });
  debugTypeConverter.addConversion([&](DeclRefType type) -> DebugInfo::DIType {
    return buildDebugInfoForStructRef(type, debugTypeConverter);
  });
  debugTypeConverter.addConversion([&](PointerType type) -> DebugInfo::DIType {
    DebugInfo::DIType elementType =
        debugTypeConverter.convertDebugType(type.getElementType());
    auto resultType =
        DebugInfo::DITargetIndependentPointerType::get(elementType);
    return resultType;
  });
  debugTypeConverter.addConversion([&](RefType type) -> DebugInfo::DIType {
    return debugTypeConverter.convertDebugType(type.getAsPointerType());
  });
}

template <typename T, typename U>
U StructOperationLowerer::replaceImpl(T value) {
  SmallVector<Attribute, 16> replAttrs;
  SmallVector<Type, 16> replTypes;
  bool changed = false;
  bool failed = false;
  U result = value;
  value.walkImmediateSubElements(
      [&](Attribute attr) {
        if (failed)
          return;

        Attribute result = replace(attr);
        if (!result)
          failed = true;

        replAttrs.push_back(result);
        changed |= result != attr;
      },
      [&](Type type) {
        if (failed)
          return;
        Type result = replace(type);
        if (!result)
          failed = true;
        replTypes.push_back(result);
        changed |= result != type;
      });

  if (failed)
    return nullptr;

  if (changed)
    result = value.replaceImmediateSubElements(replAttrs, replTypes);

  return result;
}

Attribute StructOperationLowerer::replace(Attribute attr) {
  auto iter = attrTypeReplaceCache.find(attr.getAsOpaquePointer());
  if (iter != attrTypeReplaceCache.end())
    return Attribute::getFromOpaquePointer(iter->second);

  bool foundRecursion = seenAttrs.contains(attr);
  if (foundRecursion && !eraseRecursivePointerField) {
    // Found illegal recursion.
    return nullptr;
  }

  // Keep track of ancestor attributes.
  seenAttrs.insert(attr);

  auto processStructAttr = [&](LITStructAttr attr) -> Attribute {
    PointerUnion<KGEN::StructType, Type> newType =
        substituteStructRef(attr.getType());
    // Handle flattened single-element structs.
    if (auto type = dyn_cast<Type>(newType)) {
      auto &decl = structDecls.getDecl(attr.getType());
      ParameterEvaluator evaluator(decl.decls, attr.getType().getParamValues());
      Attribute value =
          evaluator.getReboundAttribute(std::get<1>(attr.getValues()[0]));

      return replace(value);
    }

    SmallVector<TypedAttr> values;
    for (auto [name, value] : attr.getValues())
      values.push_back(cast<TypedAttr>(replace(value)));
    return KGEN::StructAttr::get(values, cast<KGEN::StructType>(newType));
  };

  auto processStructExtractAttr =
      [&](LIT::StructExtractAttr attr) -> Attribute {
    auto litStructType = cast<DeclRefType>(attr.getStructValue().getType());
    int64_t fieldNo = getField(attr.getField(), litStructType);

    Attribute structValue = replace(attr.getStructValue());

    // If this is an extract of element 0, check to see if it
    // is a flattened struct.
    if (fieldNo == 0)
      if (isa<Type>(substituteStructRef(litStructType)))
        return structValue;

    return KGEN::StructExtractAttr::get(cast<TypedAttr>(structValue), fieldNo);
  };

  auto processSymbolConstantAttr = [&](SymbolConstantAttr attr) -> Attribute {
    // Strip out any lifetime parameters being bound.
    SmallVector<TypedAttr> paramValues;
    for (auto param : attr.getParamValues())
      if (!isa<LIT::LifetimeType>(param.getType()))
        paramValues.push_back(param);
    if (paramValues.size() != attr.getParamValues().size())
      attr = SymbolConstantAttr::get(attr.getSymbol(), paramValues,
                                     attr.getType());
    return replaceImpl<Attribute, Attribute>(attr);
  };

  // Partially bound types never have uses in KGEN.
  // TODO: Need to codegen here when parametric traits are a thing.
  auto processBindType = [this](BindTypeAttr bind) {
    MetaTypeType metatype = bind.getType();
    return TypeConstantAttr::get(
        replace(DeclRefType::get(metatype.getSymbol(),
                                 metatype.getParamValues(), anyRegTypeType)),
        anyRegTypeType);
  };

  auto processParamOperatorAttr = [&](ParamOperatorAttr attr) -> Attribute {
    Attribute result = attr;
    if (attr.getOpcode() == POC::BindSignature) {

      // Strip out any lifetime parameters being bound by a "bind" operator,
      // KGEN will drop this entirely if all the operands are removed.
      SmallVector<TypedAttr> operands;
      for (auto param : attr.getOperands())
        if (!isa<LIT::LifetimeType>(param.getType()))
          operands.push_back(param);
      if (operands.size() != attr.getOperands().size())
        result = ParamOperatorAttr::get(POC::BindSignature, operands);
    }
    return replaceImpl<Attribute, Attribute>(result);
  };

  Attribute result = attr;
  if (auto sattr = dyn_cast<LITStructAttr>(attr)) {
    result = processStructAttr(sattr);
  } else if (auto sattr = dyn_cast<LIT::StructExtractAttr>(attr)) {
    result = processStructExtractAttr(sattr);
  } else if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(attr)) {
    result = processSymbolConstantAttr(symbolConstant);
  } else if (auto bind = dyn_cast<BindTypeAttr>(attr)) {
    result = processBindType(bind);
  } else if (auto paramOperator = dyn_cast<ParamOperatorAttr>(attr)) {
    result = processParamOperatorAttr(paramOperator);
  } else if (isa<LIT::LifetimeAttr>(attr)) {
    result = emptyStructAttr; // #lit.lifetime => #kgen.struct<>
  } else if (auto paramDRE = dyn_cast<ParamDeclRefAttr>(attr)) {
    // References to parameters of lifetime type are folded to their singleton
    // value, completely eliminating any use of them, allowing us to just delete
    // them from signatures.
    if (isa<LIT::LifetimeType>(paramDRE.getType())) {
      // p => #kgen.struct<>
      result = emptyStructAttr;
    } else {
      result = replaceImpl<Attribute, Attribute>(attr);
    }
  } else {
    // Recursively replace attributes.
    result = replaceImpl<Attribute, Attribute>(attr);
  }

  attrTypeReplaceCache.try_emplace(attr.getAsOpaquePointer(),
                                   result.getAsOpaquePointer());
  seenAttrs.erase(attr);

  return result;
}

Type StructOperationLowerer::replace(Type type) {
  auto iter = attrTypeReplaceCache.find(type.getAsOpaquePointer());
  if (iter != attrTypeReplaceCache.end())
    return Type::getFromOpaquePointer(iter->second);

  bool foundRecursion = seenTypes.contains(type);
  if (foundRecursion && !eraseRecursivePointerField) {
    // Found illegal recursion.
    return nullptr;
  }

  bool cacheResult = true;

  // Keep track of types.
  seenTypes.insert(type);

  auto processPointer = [&](PointerType ptr) -> Type {
    if (!foundRecursion)
      return replaceImpl(ptr);
    // Handle a struct (Foo) that has a pointer to a recursive struct, i.e.:
    // 1. the pointer points to the struct itself Foo
    // struct Foo:
    //    var x: Pointer[Foo]
    //
    // 2. the pointer points to another struct that has a chain of field of
    // structs that recurses back to Foo, and one of the field is a Pointer
    // before Foo shows up again in the chain
    //
    // struct Foo:
    //    var x: Pointer[Bar]
    // struct Bar:
    //    var x: Foo
    //
    // or
    // struct Foo:
    //    var x: Bar
    // struct Bar:
    //    var x: Pointer[Foo]
    //
    // or
    // struct Foo:
    //    var x: Pointer[Bar]
    // struct Bar:
    //    var x: Pointer[Foo]
    cacheResult = false;
    erasedType = true;
    // Erase the type of Pointer[Foo] to Pointer[NoneType] to break the
    // recursive chain.
    return PointerType::get(POP::SIMDType::get(
        1, KGEN::DTypeConstantAttr::get(ptr.getContext(), DType::invalid)));
  };

  auto processDeclRefType = [&](DeclRefType ref) -> Type {
    PointerUnion<KGEN::StructType, Type> result = substituteStructRef(ref);
    if (erasedType) {
      // If Pointer type erase happened, don't cache this type replacement
      // result.
      cacheResult = false;
      erasedType = false;
    }

    if (!result)
      return nullptr;

    if (auto type = dyn_cast<Type>(result))
      return type;

    return cast<KGEN::StructType>(result);
  };

  // Signature processing checks to see if there are any lifetime parameters;
  // if so, they are dropped.
  auto processSignatureType = [&](SignatureType signature) -> Type {
    // Just remove any lifetime parameters.
    SmallVector<Type, 8> inputParamTypes;
    for (auto type : signature.getInputParamTypes()) {
      if (!isa<LIT::LifetimeType>(type))
        inputParamTypes.push_back(type);
    }
    if (inputParamTypes.size() != signature.getNumInputParams())
      signature = SignatureType::get(
          signature.getValues(), inputParamTypes,
          signature.getResultParamTypes(), signature.getInputConventions(),
          signature.getFnEffects(), signature.getMetadata());

    return replaceImpl(signature);
  };

  Type result = type;
  if (auto ditype = dyn_cast<DebugInfo::DIType>(type)) {
    if (runDebugTypeConversion)
      result = debugTypeConverter.convertDebugType(ditype);
  } else if (auto ptr = dyn_cast<PointerType>(type)) {
    result = processPointer(ptr);
  } else if (auto ref = dyn_cast<DeclRefType>(type)) {
    result = processDeclRefType(ref);
  } else if (isa<MetaTypeType, TraitType>(type)) {
    // Erase metatypes and reg-passable anytypes. Passability information is
    // encoded elsewhere so this won't be needed.
    result = anyRegTypeType;
  } else if (auto signature = dyn_cast<SignatureType>(type)) {
    result = processSignatureType(signature);
  } else if (isa<LIT::LifetimeType>(type)) {
    // !lit.lifetime => !kgen.struct<()>
    result = emptyStructAttr.getType();
  } else if (auto ref = dyn_cast<LIT::RefType>(type)) {
    // !lit.ref<@T, life> => !kgen.pointer<@T>
    result = replaceImpl(ref.getAsPointerType());
  } else {
    // Recursively replace types.
    result = replaceImpl(type);
  }

  if (cacheResult) {
    attrTypeReplaceCache.try_emplace(type.getAsOpaquePointer(),
                                     result.getAsOpaquePointer());
  }

  if (!foundRecursion)
    seenTypes.erase(type);

  return result;
}

PointerUnion<KGEN::StructType, Type>
StructOperationLowerer::substituteStructRef(DeclRefType ref) {
  auto &decl = structDecls.getDecl(ref);
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
  SmallVector<Type> elementTypes;
  for (Type type : llvm::make_second_range(decl.fields)) {
    Type reboundType = evaluator.getReboundType(type);
    Type substituteReboundType = replace(reboundType);
    elementTypes.push_back(substituteReboundType);
  }

  // Flatten register-passable, single-element structs.
  if (elementTypes.size() == 1 && decl.isRegisterPassable)
    return elementTypes[0];
  return StructType::get(ref.getContext(), elementTypes,
                         !decl.isRegisterPassable);
}

DebugInfo::DIType StructOperationLowerer::buildDebugInfoForStructRef(
    DeclRefType ref, DebugInfo::DebugInfoTypeConverter &converter) {
  // Substitute parameters into the field types.
  auto &decl = structDecls.getDecl(ref);
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());

  auto getDebugInfoType = [&](const std::pair<StringAttr, Type> &nameAndType) {
    auto [name, type] = nameAndType;
    return DebugInfo::DIMemberType::get(
        name, converter.convertDebugType(evaluator.getReboundType(type)));
  };

  // Flatten register-passable, single-element structs.
  // TODO(#23914): Track this optimization with DWARF expressions.
  if (decl.fields.size() == 1 && decl.isRegisterPassable)
    return getDebugInfoType(decl.fields.front()).getType();

  SmallVector<DebugInfo::DIMemberType> elementTypes =
      llvm::map_to_vector(decl.fields, getDebugInfoType);

  // Mangle the struct name.
  std::string name;
  llvm::raw_string_ostream os(name);
  printNestedSymbolReference(os, ref.getSymbol());
  if (!ref.getParamValues().empty()) {
    os << '[';
    auto eachFn = [&os](auto bind) {
      auto [name, value] = bind;
      os << demangleParameterName(name.getName()) << '='
         << getParamAsString(value);
    };
    llvm::interleaveComma(llvm::zip(decl.decls, ref.getParamValues()), os,
                          eachFn);
    os << ']';
  }

  return DebugInfo::DIStructType::get(StringAttr::get(ref.getContext(), name),
                                      elementTypes);
}

static Value lowerOp(LIT::StructCreateOp op, LIT::StructCreateOpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  PointerUnion<KGEN::StructType, Type> newType =
      lowerer.substituteStructRef(op.getType());

  if (isa<Type>(newType)) {
    assert(adaptor.getOperands().size() == 1 &&
           "Flattening non-one element struct");
    return adaptor.getOperands()[0];
  }

  return lowerer.create<KGEN::StructCreateOp>(
      op.getLoc(), cast<KGEN::StructType>(newType), adaptor.getOperands());
}

static Value lowerOp(StructInsertOp op, StructInsertOpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  int64_t index =
      lowerer.getField(op.getFieldAttr(), op.getContainer().getType());

  // Check to see if we need to flatten this.  Flattening an insert just
  // replaces the value.

  PointerUnion<KGEN::StructType, Type> resultStructType =
      lowerer.substituteStructRef(op.getType());
  if (index == 0) {
    if (isa<Type>(resultStructType))
      return adaptor.getValue();
  }

  auto result = lowerer.create<KGEN::StructReplaceOp>(
      op.getLoc(), adaptor.getValue(), adaptor.getContainer(),
      lowerer.getIndexAttr(index));

  auto structType = cast<KGEN::StructType>(result.getResult().getType());
  Type fieldType = structType.getElementTypes()[index];

  if (result->getOperand(0).getType() != fieldType) {
    // If a Pointer type of the struct field is erased to NoneType because of
    // recursive nested type, when inserting the new value to the field, a
    // PointerBitcast is needed here so that the created StructReplaceOp won't
    // complain about the types.
    OpBuilder builder(result);
    auto cast = builder.create<POP::PointerBitcastOp>(
        result.getLoc(), fieldType, result->getOperand(0));
    result.setOperand(0, cast.getResult());
  }

  return result;
}

static Value lowerOp(LIT::StructExtractOp op,
                     LIT::StructExtractOpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  int64_t index =
      lowerer.getField(op.getFieldAttr(), op.getContainer().getType());

  // Check to see if we need to flatten this.  Flattening an extract just
  // returns the value.
  if (index == 0) {
    if (isa<Type>(lowerer.substituteStructRef(op.getContainer().getType())))
      return adaptor.getContainer();
  }

  return lowerer.create<KGEN::StructExtractOp>(
      op.getLoc(), adaptor.getContainer(), lowerer.getIndexAttr(index));
}

static Value lowerOp(RefImmutOp op, RefImmutOpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  assert(isa<PointerType>(adaptor.getRef().getType()) &&
         "operand should be lowered");
  return adaptor.getRef();
}

static Value lowerOp(RefToPointerOp op, RefToPointerOpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  assert(isa<PointerType>(adaptor.getRef().getType()) &&
         "operand should be lowered");
  return adaptor.getRef();
}

static Value lowerOp(RefFromPointerOp op, RefFromPointerOpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  return adaptor.getPtr();
}

static Value lowerOp(RefLoadOp op, RefLoadOpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  assert(isa<PointerType>(adaptor.getRef().getType()) &&
         "operand should be lowered");

  return lowerer.create<POP::LoadOp>(op.getLoc(), adaptor.getRef());
}

static Value lowerOp(RefStoreOp op, RefStoreOpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  assert(isa<PointerType>(adaptor.getRef().getType()) &&
         "operand should be lowered");

  lowerer.create<POP::StoreOp>(op.getLoc(), adaptor.getArg(), adaptor.getRef());
  lowerer.eraseOp(op);
  return {};
}

static Value lowerOp(RefStructGEROp op, RefStructGEROpAdaptor adaptor,
                     StructOperationLowerer &lowerer) {
  auto structType =
      cast<DeclRefType>(op.getContainer().getType().getElementType());
  int64_t index = lowerer.getField(op.getFieldAttr(), structType);

  // Check to see if we need to flatten this.  A flattened gep is a noop.
  if (index == 0) {
    if (isa<Type>(lowerer.substituteStructRef(structType)))
      return adaptor.getContainer();
  }

  return lowerer.create<KGEN::StructGEPOp>(op.getLoc(), adaptor.getContainer(),
                                           lowerer.getIndexAttr(index));
}

static Value getCastedToType(Value value, Type destType, OpBuilder &b) {
  // If already casted, done.
  if (value.getType() == destType)
    return value;

  // If coming from a cast, use input.
  if (auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    if (castOp.getOperand(0).getType() == destType)
      return castOp.getOperand(0);

  // Otherwise create a new cast.
  auto cast = b.create<mlir::UnrealizedConversionCastOp>(value.getLoc(),
                                                         destType, value);
  return cast.getResult(0);
}

template <typename OpT>
LogicalResult StructOperationLowerer::materializeLowering(OpT op) {
  setInsertionPoint(op);
  SmallVector<Value> castedOperands;
  castedOperands.reserve(op->getNumOperands());
  // Get type adjusted values into the adaptor to simplify clients.
  for (OpOperand &operand : op->getOpOperands()) {
    Value value = operand.get();
    Type newType = replace(value.getType());

    if (!newType) {
      // Found recursive nested struct so that replace failed to get a new type.
      // This is very unlikely to happen because the recursion should be
      // detected in earlier passes already.
      op.emitError("Found recursive nested structs in operand: " +
                   std::to_string(operand.getOperandNumber()));
      return failure();
    }

    castedOperands.push_back(getCastedToType(value, newType, *this));
  }

  typename OpT::Adaptor adaptor(castedOperands, op->getAttrDictionary());
  if (op->getNumResults() == 1) {
    auto resultType = op->getResult(0).getType();
    Value result = lowerOp(op, adaptor, *this);
    if (result.getType() != resultType)
      result = getCastedToType(result, resultType, *this);
    replaceOp(op, {result});
  } else {
    assert(op->getNumResults() == 0);
    Value result = lowerOp(op, adaptor, *this);
    (void)result;
    assert(!result && "nullary lowering shouldn't produce an op");
  }

  if (LLVM_UNLIKELY(errDeclRef)) {
    return op.emitError("operation contains a declref type that does not refer "
                        "to a struct: ")
           << errDeclRef;
  }
  return success();
}

void StructOperationLowerer::replaceElementsIn(Operation *op) {
  // Functor that replaces the given element if the new value is different,
  // otherwise returns nullptr.
  auto replaceIfDifferent = [&](auto element) {
    auto replacement = replace(element);
    return (replacement && replacement != element) ? replacement : nullptr;
  };

  // Update the attribute dictionary.
  if (auto newAttrs = replaceIfDifferent(op->getAttrDictionary()))
    op->setAttrs(cast<DictionaryAttr>(newAttrs));

  // Update the location.
  if (Attribute newLoc = replaceIfDifferent(op->getLoc()))
    op->setLoc(cast<mlir::LocationAttr>(newLoc));

  // Update the result types.
  for (OpResult result : op->getResults())
    if (Type newType = replaceIfDifferent(result.getType()))
      result.setType(newType);

  // Update any nested block arguments.
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    // Our IR has at most single-block regions.
    Block &block = region.front();
    for (BlockArgument &arg : block.getArguments()) {
      if (Attribute newLoc = replaceIfDifferent(arg.getLoc()))
        arg.setLoc(cast<mlir::LocationAttr>(newLoc));

      if (Type newType = replaceIfDifferent(arg.getType()))
        arg.setType(newType);
    }
  }
}

/// Do any lowerings needed for a function op.
static LogicalResult lowerFuncOp(GeneratorOp func) {
  // The only specific lowering we do here is to remove input parameters of
  // lifetime type from the signature of the function.  This is because this
  // pass strips the lifetime parameters out.
  SmallVector<ParamDeclAttr, 8> inputParams;
  for (ParamDeclAttr paramDecl : func.getInputParams()) {
    if (!isa<LIT::LifetimeType>(paramDecl.getType()))
      inputParams.push_back(paramDecl);
  }
  if (inputParams.size() != func.getInputParams().size())
    func.setInputParams(inputParams);

  return success();
}

/// Type references can be used in nested types. Walk through all the types and
/// rewrite them in-place to use the lowered types. Walk pre-order, and while
/// doing so, erase any trivial casts left over from the type conversion.
static LogicalResult replaceTypes(Operation *op,
                                  StructOperationLowerer &structLowerer) {
  structLowerer.replaceElementsIn(op);

  if (LLVM_UNLIKELY(structLowerer.errDeclRef)) {
    return op->emitError("operation contains a declref type that does not "
                         "refer to a struct: ")
           << structLowerer.errDeclRef;
  }
  if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
    // Fold trivial casts.
    if (cast.getOperandTypes() == cast.getResultTypes()) {
      cast.replaceAllUsesWith(cast.getOperands());
      cast.erase();
    } else {
      if (cast->getNumResults() == 1 && cast->getNumOperands() == 1) {
        if (isa<PointerType>(cast.getResult(0).getType()) &&
            isa<PointerType>(cast.getOperand(0).getType())) {
          // Change into a PointerBitcastOp for Pointer whose type is erased
          // to be NoneType.
          OpBuilder b(cast);
          auto ptrBCast = b.create<POP::PointerBitcastOp>(
              cast.getLoc(), cast.getResult(0).getType(), cast.getOperand(0));
          cast->replaceAllUsesWith(ptrBCast->getResults());
          cast.erase();
        }
      }
    }

    return success();
  }
  for (Region &region : op->getRegions()) {
    for (Operation &op : llvm::make_early_inc_range(region.getOps()))
      if (failed(replaceTypes(&op, structLowerer)))
        return failure();
  }
  return success();
};

//===----------------------------------------------------------------------===//
// LowerLITTypesPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERLITTYPES
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerLITTypesPass
    : public KGEN::impl::LowerLITTypesBase<LowerLITTypesPass> {
  using LowerLITTypesBase::LowerLITTypesBase;

  void runOnOperation() override;
};
} // namespace

void LowerLITTypesPass::runOnOperation() {
  // Collect all struct declarations and erase them.
  auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();

  StructDeclarations structDecls;

  for (Operation &op : llvm::make_early_inc_range(getOperation().getOps())) {
    if (auto structOp = dyn_cast<StructDeclOp>(op)) {
      SmallVector<std::pair<StringAttr, Type>> fields;
      for (auto [idx, field] : llvm::enumerate(structOp.getFieldDecls())) {
        fields.emplace_back(field.getNameAttr(), field.getType());
        structDecls.fieldIndices.try_emplace(
            {structOp.getNameAttr(), field.getNameAttr()}, idx);
      }

      structDecls.decls.insert(
          {structOp.getNameAttr(),
           {std::move(fields), structOp.getInputParamsAttr(),
            structOp.isRegisterPassable()}});
      analysis.getTopLevelSymbolTable().erase(structOp);
    } else if (isa<TraitDeclOp>(op)) {
      analysis.getTopLevelSymbolTable().erase(&op);
    }
  }

  StructOperationLowerer structLowerer(&getContext(), structDecls);

  // Lower KGEN struct operations.
  structLowerer.eraseRecursivePointerField = true;
  WalkResult result = getOperation()->walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<LIT::StructCreateOp, StructInsertOp, LIT::StructExtractOp,
              RefImmutOp, RefToPointerOp, RefFromPointerOp, RefStructGEROp,
              RefLoadOp, RefStoreOp>(
            [&](auto op) { return structLowerer.materializeLowering(op); })
        .Case<GeneratorOp>([&](auto op) { return lowerFuncOp(op); })
        .Default([](auto) { return success(); });
  });
  structLowerer.eraseRecursivePointerField = false;
  if (result.wasInterrupted())
    return signalPassFailure();

  // update converted types with debug info
  structLowerer.runDebugTypeConversion = true;

  if (failed(replaceTypes(getOperation(), structLowerer)))
    return signalPassFailure();
}
