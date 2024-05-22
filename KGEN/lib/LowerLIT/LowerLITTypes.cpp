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
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/PatternMatch.h"
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
// Type Lowering
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
    /// The un-parameterized SourceNameAttr for the struct decl.
    DebugInfo::SourceNameAttr sourceName;
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
} // end anonymous namespace

namespace {
/// Struct operations need to refer to the struct declaration symbol.
struct LITTypeLowerer : public mlir::IRRewriter {
  explicit LITTypeLowerer(MLIRContext *ctx, StructDeclarations &structDecls);

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

  /// The empty `#kgen.struct<>` attribute, which has empty struct type.
  KGEN::StructAttr emptyStructAttr;

  /// Cache to memorize AttrType replacement results.
  DenseMap<const void *, const void *> attrTypeReplaceCache;

  /// Flag to run debug type conversion on updated types.
  bool runDebugTypeConversion = false;

  /// Debug type converter.
  DebugInfo::DebugInfoTypeConverter debugTypeConverter;

  /// Cached anyRegTypeType so we don't need to recreate it.
  TypeType anyRegTypeType;

  /// Flag to tell the replacer if it is replacing a `#lit.struct` attr where
  /// the pointer type should be erased as if they are in a
  /// `!lit.declref` because these are struct fields.
  int lowerLitStructValues = 0;

  /// Whether replace should use the cache or not.
  bool useReplaceCache();
};
} // namespace

bool LITTypeLowerer::useReplaceCache() { return lowerLitStructValues == 0; }

LITTypeLowerer::LITTypeLowerer(MLIRContext *ctx,
                               StructDeclarations &structDecls)
    : IRRewriter(ctx), structDecls(structDecls),
      anyRegTypeType(TypeType::get(ctx)) {

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
    return DebugInfo::DITargetIndependentPointerType::get(elementType);
  });
  debugTypeConverter.addConversion([&](RefType type) -> DebugInfo::DIType {
    return debugTypeConverter.convertDebugType(type.getAsPointerType());
  });
}

template <typename T, typename U>
U LITTypeLowerer::replaceImpl(T value) {
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

Attribute LITTypeLowerer::replace(Attribute attr) {
  auto iter = attrTypeReplaceCache.find(attr.getAsOpaquePointer());

  if (useReplaceCache() && iter != attrTypeReplaceCache.end())
    return Attribute::getFromOpaquePointer(iter->second);

  auto processStructAttr = [&](LITStructAttr attr) -> Attribute {
    PointerUnion<KGEN::StructType, Type> newType =
        substituteStructRef(attr.getType());
    // Handle flattened single-element structs.
    ++lowerLitStructValues;
    if (auto type = dyn_cast<Type>(newType)) {
      auto &decl = structDecls.getDecl(attr.getType());
      ParameterEvaluator evaluator(decl.decls, attr.getType().getParamValues());
      Attribute value =
          evaluator.getReboundAttribute(std::get<1>(attr.getValues()[0]));

      Attribute newValue = replace(value);
      --lowerLitStructValues;
      return newValue;
    }

    SmallVector<TypedAttr> values;
    for (auto [name, value] : attr.getValues())
      values.push_back(cast<TypedAttr>(replace(value)));

    --lowerLitStructValues;

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

  // Partially bound types never have uses in KGEN.
  // TODO: Need to codegen here when parametric traits are a thing.
  auto processBindType = [this](BindTypeAttr bind) {
    AnyStructType metatype = bind.getType();
    return TypeConstantAttr::get(
        replace(DeclRefType::get(metatype.getSymbol(),
                                 metatype.getParamValues(), anyRegTypeType)),
        anyRegTypeType);
  };

  // #lit.ref.pack => #kgen.pack
  auto processRefPackAttr = [&](RefPackAttr refPack) {
    SmallVector<TypedAttr> loweredElts;
    loweredElts.reserve(refPack.getValues().size());
    for (auto elt : refPack.getValues())
      loweredElts.push_back(cast<TypedAttr>(replace(elt)));
    // for (auto elt : elts
    return KGEN::PackAttr::get(loweredElts,
                               cast<PackType>(replace(refPack.getType())));
  };

  Attribute result = attr;
  if (auto sattr = dyn_cast<LITStructAttr>(attr)) {
    result = processStructAttr(sattr);
  } else if (auto sattr = dyn_cast<LIT::StructExtractAttr>(attr)) {
    result = processStructExtractAttr(sattr);
  } else if (auto bind = dyn_cast<BindTypeAttr>(attr)) {
    result = processBindType(bind);
  } else if (isa<LifetimeAttr, LifetimeUnionAttr, LifetimeMutCastAttr,
                 InvalidRefLifetimeAttr>(attr)) {
    result = emptyStructAttr; // #lit.lifetime => #kgen.struct<>
  } else if (auto refPack = dyn_cast<RefPackAttr>(attr)) {
    result = processRefPackAttr(refPack);
  } else {
    // Recursively replace attributes.
    result = replaceImpl<Attribute, Attribute>(attr);
  }

  attrTypeReplaceCache.try_emplace(attr.getAsOpaquePointer(),
                                   result.getAsOpaquePointer());

  return result;
}

Type LITTypeLowerer::replace(Type type) {
  auto iter = attrTypeReplaceCache.find(type.getAsOpaquePointer());

  if (useReplaceCache() && iter != attrTypeReplaceCache.end())
    return Type::getFromOpaquePointer(iter->second);

  auto processPointerType = [&](PointerType ptr) -> Type {
    if (lowerLitStructValues == 0)
      return replaceImpl(ptr);
    // Erase elementType if a field is of pointer type in a struct.
    // Here updates the value type in a `#lit.struct` which is a constant value
    // for a struct.
    auto newAddrSpace = replace(ptr.getAddressSpace());
    return PointerType::get(KGEN::NoneType::get(ptr.getContext()),
                            cast<TypedAttr>(newAddrSpace));
  };

  auto processDeclRefType = [&](DeclRefType ref) -> Type {
    PointerUnion<KGEN::StructType, Type> result = substituteStructRef(ref);
    if (!result)
      return nullptr;

    if (auto type = dyn_cast<Type>(result))
      return type;

    return cast<KGEN::StructType>(result);
  };

  // !lit.ref.pack<:variadic<!kgen.type> types, owned_in_mem, mut life, 42>
  // => !kgen.pack<variadic_ptr_map(types), 42>
  auto processRefPackType = [&](RefPackType ref) -> Type {
    auto newVariadic = cast<TypedAttr>(replace(ref.getVariadic()));
    auto newAddrSpace = cast<TypedAttr>(replace(ref.getAddressSpace()));
    auto mappedTypes =
        ParamOperatorAttr::get(POC::VariadicPtrMap, newVariadic, newAddrSpace);
    return PackType::get(mappedTypes);
  };

  Type result = type;
  if (auto ditype = dyn_cast<DebugInfo::DIType>(type)) {
    if (runDebugTypeConversion)
      result = debugTypeConverter.convertDebugType(ditype);
  } else if (auto ptr = dyn_cast<PointerType>(type)) {
    result = processPointerType(ptr);
  } else if (auto ref = dyn_cast<DeclRefType>(type)) {
    result = processDeclRefType(ref);
  } else if (isa<AnyStructType, AnyTraitType, TraitType>(type)) {
    // Erase metatypes and reg-passable anytypes. Passability information is
    // encoded elsewhere so this won't be needed.
    result = anyRegTypeType;
  } else if (isa<LIT::LifetimeType>(type)) {
    // !lit.lifetime => !kgen.struct<()>
    result = emptyStructAttr.getType();
  } else if (auto ref = dyn_cast<LIT::RefType>(type)) {
    // !lit.ref<@T, life> => !kgen.pointer<@T>
    result = replaceImpl(ref.getAsPointerType());
  } else if (auto ref = dyn_cast<LIT::RefPackType>(type)) {
    // !lit.ref.pack => !kgen.pack
    result = processRefPackType(ref);
  } else {
    // Recursively replace types.
    result = replaceImpl(type);
  }

  attrTypeReplaceCache.try_emplace(type.getAsOpaquePointer(),
                                   result.getAsOpaquePointer());

  return result;
}

PointerUnion<KGEN::StructType, Type>
LITTypeLowerer::substituteStructRef(DeclRefType ref) {
  auto &decl = structDecls.getDecl(ref);
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
  SmallVector<Type> elementTypes;
  for (Type type : llvm::make_second_range(decl.fields)) {
    Type reboundType = evaluator.getReboundType(type);

    if (auto ptr = dyn_cast<PointerType>(reboundType)) {
      // Erase elementType if a field is of pointer type in a struct.
      auto newAddrSpace = replace(ptr.getAddressSpace());
      Type substituteReboundType = PointerType::get(
          KGEN::NoneType::get(ptr.getContext()), cast<TypedAttr>(newAddrSpace));
      elementTypes.push_back(substituteReboundType);
      continue;
    }
    Type substituteReboundType = replace(reboundType);
    elementTypes.push_back(substituteReboundType);
  }
  // Flatten register-passable, single-element structs.
  if (elementTypes.size() == 1 && decl.isRegisterPassable)
    return elementTypes[0];
  return StructType::get(ref.getContext(), elementTypes,
                         !decl.isRegisterPassable);
}

DebugInfo::DIType LITTypeLowerer::buildDebugInfoForStructRef(
    DeclRefType ref, DebugInfo::DebugInfoTypeConverter &converter) {
  // Substitute parameters into the field types.
  auto &decl = structDecls.getDecl(ref);
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());

  auto getDebugInfoType = [&](const std::pair<StringAttr, Type> &nameAndType) {
    auto [name, type] = nameAndType;
    DebugInfo::DIType fieldDIType =
        converter.convertDebugType(evaluator.getReboundType(type));
    return DebugInfo::DIMemberType::get(name, fieldDIType);
  };

  // Flatten register-passable, single-element structs.
  // TODO(#23914): Track this optimization with DWARF expressions.
  if (decl.fields.size() == 1 && decl.isRegisterPassable)
    return getDebugInfoType(decl.fields.front()).getType();

  SmallVector<DebugInfo::DIMemberType> elementTypes =
      llvm::map_to_vector(decl.fields, getDebugInfoType);

  // Parameterize the raw source name.
  DebugInfo::SourceNameAttr sourceName = decl.sourceName;
  // TODO: Make StructDeclOp's sourceName a DefaultValuedAttr once properties
  // play nicely with it.
  if (!sourceName) {
    std::string name;
    llvm::raw_string_ostream os(name);
    printNestedSymbolReference(os, ref.getSymbol());
    sourceName = DebugInfo::SourceNameAttr::get(
        StringAttr::get(getContext(), name), DebugInfo::SourceNameKind::Struct);
  }

  SmallVector<StringAttr> paramValues;
  for (TypedAttr value : ref.getParamValues())
    paramValues.push_back(getParamTypeAsString(value));
  sourceName = DebugInfo::SourceNameAttr::get(
      sourceName.getName(), sourceName.getParamTypes(),
      sourceName.getArgTypes(), paramValues, sourceName.getParent(),
      sourceName.getKind(), sourceName.getDecorators());

  return DebugInfo::DIStructType::get(sourceName.encode(), elementTypes);
}

static Value lowerOp(LIT::StructCreateOp op, LIT::StructCreateOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
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
                     LITTypeLowerer &lowerer) {
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
                     LITTypeLowerer &lowerer) {
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
                     LITTypeLowerer &lowerer) {
  assert(isa<PointerType>(adaptor.getRef().getType()) &&
         "operand should be lowered");
  return adaptor.getRef();
}

static Value lowerOp(RefToPointerOp op, RefToPointerOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
  assert(isa<PointerType>(adaptor.getRef().getType()) &&
         "operand should be lowered");
  return adaptor.getRef();
}

static Value lowerOp(RefFromPointerOp op, RefFromPointerOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
  return adaptor.getPtr();
}

static Value lowerOp(RefFromPointerREPLOp op,
                     RefFromPointerREPLOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
  return adaptor.getPtr();
}

static Value lowerOp(RefLoadOp op, RefLoadOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
  assert(isa<PointerType>(adaptor.getRef().getType()) &&
         "operand should be lowered");

  return lowerer.create<POP::LoadOp>(op.getLoc(), adaptor.getRef());
}

static Value lowerOp(RefStoreOp op, RefStoreOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
  assert(isa<PointerType>(adaptor.getRef().getType()) &&
         "operand should be lowered");

  lowerer.create<POP::StoreOp>(op.getLoc(), adaptor.getArg(), adaptor.getRef());
  lowerer.eraseOp(op);
  return {};
}

static Value lowerOp(RefStructGEROp op, RefStructGEROpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
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

/// Squash noop rebinds exposed by ref -> ptr lowering.
static Value lowerOp(RebindOp op, RebindOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
  // If this is a noop after lowering, squish it
  if (adaptor.getInput().getType() == lowerer.replace(op.getType()))
    return adaptor.getInput();
  // Otherwise just leave it and type replacement will form a valid rebind in
  // the new type domain.
  return op.getResult();
}

// lit.ref.pack.create => kgen.pack.create
static Value lowerOp(RefPackCreateOp op, RefPackCreateOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
  return lowerer.create<PackCreateOp>(
      op.getLoc(), lowerer.replace(op.getType()), adaptor.getOperands());
}

// lit.ref.pack.extract => kgen.pack.extract
static Value lowerOp(RefPackExtractOp op, RefPackExtractOpAdaptor adaptor,
                     LITTypeLowerer &lowerer) {
  return lowerer.create<PackExtractOp>(op.getLoc(), adaptor.getOperands()[0],
                                       op.getIndex());
}

static Value getCastedToType(Location newLoc, Value value, Type destType,
                             OpBuilder &b) {
  // If already casted, done.
  if (value.getType() == destType)
    return value;

  // If coming from a cast, use input.
  if (auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    if (castOp.getOperand(0).getType() == destType)
      return castOp.getOperand(0);

  // Otherwise create a new cast.
  auto cast =
      b.create<mlir::UnrealizedConversionCastOp>(newLoc, destType, value);
  return cast.getResult(0);
}

template <typename OpT>
LogicalResult LITTypeLowerer::materializeLowering(OpT op) {
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

    // When value is a function argument, location info's function scope is
    // different from the operations in the function body. Use op->getLoc() for
    // new cast op's location instead of using value.loc().
    castedOperands.push_back(
        getCastedToType(op->getLoc(), value, newType, *this));
  }

  typename OpT::Adaptor adaptor(castedOperands, op->getAttrDictionary());
  if (op->getNumResults() == 1) {
    auto resultType = op->getResult(0).getType();
    Value result = lowerOp(op, adaptor, *this);
    if (result.getType() != resultType)
      result = getCastedToType(result.getLoc(), result, resultType, *this);

    if (op->getResult(0) != result)
      replaceOp(op, {result});
  } else {
    assert(op->getNumResults() == 0);
    Value result = lowerOp(op, adaptor, *this);
    (void)result;
    assert(!result && "nullary lowering shouldn't produce an op");
  }

  return success();
}

void LITTypeLowerer::replaceElementsIn(Operation *op) {
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
    op->setLoc(cast<LocationAttr>(newLoc));

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
        arg.setLoc(cast<LocationAttr>(newLoc));

      if (Type newType = replaceIfDifferent(arg.getType()))
        arg.setType(newType);
    }
  }
}

/// Type references can be used in nested types. Walk through all the types and
/// rewrite them in-place to use the lowered types. Walk pre-order, and while
/// doing so, erase any trivial casts left over from the type conversion.
static LogicalResult replaceTypes(Operation *op,
                                  LITTypeLowerer &structLowerer) {
  structLowerer.replaceElementsIn(op);

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
           {structOp.getSourceNameAttr(), std::move(fields),
            structOp.getParamsAttr(), structOp.isRegisterPassable()}});
      analysis.getTopLevelSymbolTable().erase(structOp);
    } else if (isa<TraitDeclOp>(op)) {
      analysis.getTopLevelSymbolTable().erase(&op);
    }
  }

  LITTypeLowerer structLowerer(&getContext(), structDecls);

  // Lower KGEN struct operations.
  WalkResult result = getOperation()->walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<LIT::StructCreateOp, StructInsertOp, LIT::StructExtractOp,
              RefImmutOp, RefToPointerOp, RefFromPointerOp,
              RefFromPointerREPLOp, RefStructGEROp, RefLoadOp, RefStoreOp,
              RebindOp, RefPackCreateOp, RefPackExtractOp>(
            [&](auto op) { return structLowerer.materializeLowering(op); })
        .Default([](auto) { return success(); });
  });
  if (result.wasInterrupted())
    return signalPassFailure();

  // update converted types with debug info
  structLowerer.runDebugTypeConversion = true;

  if (failed(replaceTypes(getOperation(), structLowerer)))
    return signalPassFailure();
}
