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

#include "LowerLITTypes.h"

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENParameters.h"
#include "KGEN/KGENDialect/KGENUtils.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/LITDialect/LITUtils.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/ToolCommon/KGENPasses.h"
#include "Support/Compiler/DomainAwareReplacer.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Analysis/SymbolTableAnalysis.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

namespace {
/// A DomainAwareReplacer that distinguishes the two roles of a mojo type:
/// As a value or as a type itself. The different roles require different Type
/// representations in KGEN.
class LowerLITReplacer : public DomainAwareReplacer {
public:
  enum TypeDomain : DomainId {
    AsType,  // Types are used as types.
    AsValue, // Types are used as values.
  };

  LowerLITReplacer() {
    // Parameters should always use the AsType domain so that their types are
    // lowered as types.
    addNonRecursiveReplacement(
        [&](TypedAttr attr) -> Attribute {
          return replace(attr, TypeDomain::AsType);
        },
        TypeDomain::AsValue);
  }

  /// Add a replacement that skips recursing down the replaced result.
  /// The replacement callback itself must handle any further replacing by
  /// calling back into this DomainAwareReplacer. This way the exact replacer
  /// domain can be controlled at each replacement step.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>,
            typename ResultT = std::invoke_result_t<FnT, T>>
  std::enable_if_t<std::is_convertible_v<ResultT, BaseT>>
  addNonRecursiveReplacement(FnT &&callback, DomainId domain) {
    addReplacement(mlir::AttrTypeReplacer::ReplaceFn<BaseT>(
                       [f = std::forward<FnT>(callback)](BaseT base)
                           -> mlir::AttrTypeReplacer::ReplaceFnResult<BaseT> {
                         if constexpr (std::is_same_v<T, BaseT>)
                           return {{f(base), WalkResult::skip()}};
                         if (auto derived = dyn_cast<T>(base))
                           return {{f(derived), WalkResult::skip()}};
                         return {};
                       }),
                   domain);
  };

  /// Add a domain-agnostic replacement function.
  /// Since TypedAttr replacements only need to happen in the type domain, any
  /// replacement functions for TypedAttrs are only registered in one replacer.
  /// Other replacers must be registered in both.
  template <typename FnT,
            typename T = typename llvm::function_traits<
                std::decay_t<FnT>>::template arg_t<0>,
            typename BaseT = std::conditional_t<std::is_base_of_v<Attribute, T>,
                                                Attribute, Type>,
            typename ResultT = std::invoke_result_t<FnT, T>>
  std::enable_if_t<std::is_convertible_v<ResultT, BaseT>>
  addInferredDomainNonRecursiveReplacement(FnT &&callback) {
    addNonRecursiveReplacement(std::forward<FnT>(callback), TypeDomain::AsType);
    if constexpr (!std::is_same_v<BaseT, TypedAttr>)
      addNonRecursiveReplacement(std::forward<FnT>(callback),
                                 TypeDomain::AsValue);
  }

  /// Convenience helper for replacing parameters and returning parameters.
  TypedAttr replaceParameter(TypedAttr attr) {
    return cast_or_null<TypedAttr>(replace(attr, TypeDomain::AsType));
  };
};

using TypeDomain = LowerLITReplacer::TypeDomain;

} // namespace

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

/// Populate `replacer` with the lowering patterns for attributes and types
/// from the computed lowerings for each struct decl.
static void populateReplacer(StructDecls &decls, LowerLITReplacer &replacer,
                             MLIRContext *ctx) {
  auto typeType = TypeType::get(ctx);
  auto emptyStructType = KGEN::StructType::get(ctx, {});
  auto emptyStruct = StructAttr::get({}, emptyStructType);

  // TypeParamAttr dispatches replacing to different domains.
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](TypeParamAttr typeValue) {
        return TypeParamAttr::get(
            replacer.replace(typeValue.getTypeValue(), TypeDomain::AsValue),
            replacer.replace(typeValue.getMlirType(), TypeDomain::AsType),
            replacer.replace(typeValue.getType(), TypeDomain::AsType),
            cast<VTableAttr>(
                replacer.replace(typeValue.getVTable(), TypeDomain::AsType)));
      });

  // ParamRefTypes should be TypeValueType if in the value domain.
  replacer.addNonRecursiveReplacement(
      [&replacer](ParamType paramRef) {
        return TypeValueType::get(
            replacer.replaceParameter(paramRef.getParam()));
      },
      TypeDomain::AsValue);

  // The param types of a GeneratorType are always types, not values.
  for (TypeDomain domain : {TypeDomain::AsType, TypeDomain::AsValue}) {
    auto replaceAsType = [&replacer](Type type) {
      return replacer.replace(type, TypeDomain::AsType);
    };
    replacer.addNonRecursiveReplacement(
        [domain, replaceAsType, &replacer](GeneratorType gen) {
          SmallVector<Type> inputParamTypes(
              map_range(gen.getInputParamTypes(), replaceAsType));
          Attribute metadata = gen.getMetadata();
          if (metadata)
            metadata = replacer.replace(metadata, domain);
          return GeneratorType::get(inputParamTypes,
                                    replacer.replace(gen.getBody(), domain),
                                    metadata);
        },
        domain);
  }

  // All metatypes lower to `!kgen.type`.
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](StructMetaType) { return typeType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](AnyTraitType) { return typeType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](TraitType) { return typeType; });

  // #lit.ref.pack => #kgen.pack
  replacer.addInferredDomainNonRecursiveReplacement([&replacer](
                                                        RefPackAttr refPack) {
    SmallVector<TypedAttr> loweredElts;
    loweredElts.reserve(refPack.getValues().size());
    for (TypedAttr elt : refPack.getValues())
      loweredElts.push_back(replacer.replaceParameter(elt));
    auto type =
        cast<PackType>(replacer.replace(refPack.getType(), TypeDomain::AsType));
    return PackAttr::get(loweredElts, type);
  });

  // !lit.ref.pack<:variadic<!kgen.type> types, owned_in_mem, mut life, 42>
  // => !kgen.pack<variadic_ptr_map(types), 42>
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](RefPackType ref) {
        auto variadic = replacer.replaceParameter(ref.getVariadic());
        auto addrSpace = replacer.replaceParameter(ref.getAddressSpace());
        return PackType::get(
            ParamOperatorAttr::get(POC::VariadicPtrMap, variadic, addrSpace));
      });

  // !lit.ref -> !kgen.pointer
  for (TypeDomain domain : {TypeDomain::AsType, TypeDomain::AsValue})
    replacer.addNonRecursiveReplacement(
        [domain, &replacer](RefType ref) {
          return PointerType::get(
              replacer.replace(ref.getElementType(), domain),
              replacer.replaceParameter(ref.getAddressSpace()));
        },
        domain);

  // Replace all origin attributes with empty structs. These attributes are
  // all terminal.
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](AnyOriginAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](StaticOriginAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginUnionAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginMutCastAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](ImplicitOriginRefAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginSetAttr) { return emptyStruct; });

  // !lit.origin -> !kgen.struct<()>
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginType) { return emptyStructType; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](OriginSetType) { return emptyStructType; });

  auto noneType = KGEN::NoneType::get(ctx);

  // #lit.struct -> #kgen.struct
  replacer.addInferredDomainNonRecursiveReplacement(
      [&, noneType](LITStructAttr attr) -> Attribute {
        LIT::StructType ref = attr.getType();
        StructDecl &decl = decls.get(ref.getName());

        SmallVector<TypedAttr> values;
        values.reserve(attr.getValues().size());
        for (auto [entry, type] : llvm::zip(attr.getValues(), decl.fields)) {
          TypedAttr value = replacer.replaceParameter(std::get<1>(entry));
          if (!value)
            return nullptr;
          // We to check if this is a value for a struct field that is known to
          // be a pointer type, in which case we erase the element type.
          if (isa<PointerType>(type.second)) {
            auto type = cast<PointerType>(value.getType());
            auto ptrType = PointerType::get(noneType, type.getAddressSpace());
            value = ParamOperatorAttr::get(POC::PtrBitcast, value, ptrType);
          }
          values.push_back(value);
        }

        if (decl.isSingleElement())
          return values.front();
        if (auto type = cast_or_null<KGEN::StructType>(
                replacer.replace(ref, TypeDomain::AsType)))
          return StructAttr::get(values, type);
        return nullptr;
      });

  // #lit.struct.extract -> #kgen.struct.extract
  replacer.addInferredDomainNonRecursiveReplacement(
      [&](LIT::StructExtractAttr attr) -> Attribute {
        auto ref = cast<LIT::StructType>(attr.getStructValue().getType());
        int idx = decls.fieldIndices.at({ref.getName(), attr.getField()});
        auto value = replacer.replaceParameter(attr.getStructValue());
        if (!value)
          return nullptr;
        if (decls.get(ref.getName()).isSingleElement())
          return value;
        return KGEN::StructExtractAttr::get(value, idx);
      });

  // Since lowerings have been generated for all struct types, we just need to
  // lookup the lowered type and substitute the parameters.
  // - For the AsType type domain, convert into a StructType.
  // - For the AsValue type domain, convert into a symbol reference to the
  //   pre-created symbol generator op.
  replacer.addNonRecursiveReplacement(
      [&, noneType, ctx](LIT::StructType ref) -> Type {
        StructDecl &decl = decls.get(ref.getName());
        // Substitute the given parameters in.
        ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
        SmallVector<Type> fieldTypes;
        for (auto [idx, type] :
             llvm::enumerate(llvm::make_second_range(decl.fields))) {
          if (auto ptrType = dyn_cast<PointerType>(type)) {
            fieldTypes.push_back(PointerType::get(
                noneType,
                evaluator.getReboundAttribute(ptrType.getAddressSpace())));
          } else {
            fieldTypes.push_back(evaluator.getReboundType(type));
          }
        }
        if (decl.isSingleElement())
          return replacer.replace(fieldTypes.front(), TypeDomain::AsType);
        return replacer.replace(
            KGEN::StructType::get(ctx, fieldTypes, !decl.isRegisterPassable),
            TypeDomain::AsType);
      },
      TypeDomain::AsType);

  replacer.addNonRecursiveReplacement(
      [&](LIT::StructType ref) -> Type {
        StringAttr leafName = ref.getValue().getValue().getLeafReference();
        auto structDeclIter = decls.structDecls.find(leafName);
        StructDecl &decl = structDeclIter->second;
        SmallVector<TypedAttr> loweredParamValues(
            map_range(ref.getParamValues(), [&](TypedAttr value) {
              return replacer.replaceParameter(value);
            }));
        auto concreteSymRef = TypeGeneratorRefAttr::get(
            decl.symRef, loweredParamValues,
            replacer.replace(
                StructMetaType::get(LIT::StructType::get(
                    decl.symRef, loweredParamValues, ref.getSignature())),
                TypeDomain::AsType));
        return TypeValueType::get(concreteSymRef);
      },
      TypeDomain::AsValue);
}

// Check if there exists an illegal recursion among struct decls.
static LogicalResult detectIllegalStructDeclsRecursion(StructDecls &decls) {
  // DFS through the parametric types to see if there is recursion.
  mlir::AttrTypeReplacer dfs;
  auto computeLoweredType = [&](StructDecl &decl) -> LogicalResult {
    // If we have already seen this type, then there is recursion.
    if (decl.visited) {
      // TODO: Improve the error message. We could show the recursive path.
      mlir::emitError(decl.loc, "struct has recursive reference to itself");
      return failure();
    }

    // Set the visited flag. If there are no invalid types, we will never visit
    // this type again.
    decl.visited = true;

    // Now recurse on the field types.
    for (auto [idx, type] :
         llvm::enumerate(llvm::make_second_range(decl.fields))) {
      // Skip the ones that will become opaque pointers.
      if (!isa<PointerType>(type) && !dfs.replace(type))
        return failure();
    }
    // We know the type can be lowered.
    decl.done = true;
    return success();
  };

  dfs.addReplacement([&](LIT::StructType ref) -> std::pair<Type, WalkResult> {
    // Recurse into a the definition of a struct.
    StructDecl &decl = decls.get(ref.getName());
    if (!decl.done && failed(computeLoweredType(decl)))
      return {{}, WalkResult::interrupt()};
    return {ref, WalkResult::skip()};
  });

  // Start from any struct and make sure our DFS terminates.
  for (StructDecl &decl : llvm::make_second_range(decls.structDecls)) {
    if (decl.done)
      continue;
    if (failed(computeLoweredType(decl)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

namespace {
/// Struct operations need to refer to the struct declaration symbol.
struct LITTypeLowerer : public IRRewriter, LowerLITReplacer {
  explicit LITTypeLowerer(MLIRContext *ctx, StructDecls &structDecls);

  /// Get the index of the struct field.
  int getField(StringAttr name, LIT::StructType ref) {
    return structDecls.fieldIndices.lookup({ref.getName(), name});
  }
  /// Return true if the struct is single element.
  bool isSingleElement(LIT::StructType ref) {
    return structDecls.get(ref.getName()).isSingleElement();
  }
  Value getCastedToType(Location loc, Value value, Type type);

  /// Materialize destination conversions.
  template <typename OpT>
  LogicalResult materializeLowering(OpT op);

  /// The struct decl map.
  StructDecls &structDecls;
  /// Converter for debuginfo.
  DebugInfo::DebugInfoNonCyclicTypeConverter debugTypeConverter;
  /// Unrealized casts to resolve at the end of type lowering.
  SmallVector<mlir::UnrealizedConversionCastOp> unrealizedCasts;
};
} // namespace

static DebugInfo::DIType buildDebugInfoForStructRef(
    LIT::StructType ref, StructDecls &structDecls,
    DebugInfo::DebugInfoNonCyclicTypeConverter &converter) {
  // Substitute parameters into the field types.
  StructDecl &decl = structDecls.get(ref.getName());
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());

  auto getDebugInfoType = [&](const std::pair<StringAttr, Type> &nameAndType) {
    auto [name, type] = nameAndType;
    auto reboundType = evaluator.getReboundType(type);
    DebugInfo::DIType fieldDIType = converter.convertDebugType(reboundType);
    if (!fieldDIType) {
      fieldDIType = converter.convertDebugType(
          PointerType::get(KGEN::NoneType::get(type.getContext())));
    }
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
    sourceName =
        DebugInfo::SourceNameAttr::get(StringAttr::get(ref.getContext(), name),
                                       DebugInfo::SourceNameKind::Struct);
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

LITTypeLowerer::LITTypeLowerer(MLIRContext *ctx, StructDecls &structDecls)
    : IRRewriter(ctx), structDecls(structDecls) {
  populateReplacer(structDecls, *this, ctx);

  // Build a converter to handle updating converted types within debug info
  // constructs.
  debugTypeConverter.addConversion([&](Type type) -> std::optional<Type> {
    Type newType = replace(type, TypeDomain::AsType);
    if (newType != type)
      return debugTypeConverter.convertDebugType(newType);
    return std::nullopt;
  });
  debugTypeConverter.addConversion(
      [&](LIT::StructType type) -> DebugInfo::DIType {
        return buildDebugInfoForStructRef(type, structDecls,
                                          debugTypeConverter);
      });
  debugTypeConverter.addConversion([&](PointerType type) -> DebugInfo::DIType {
    DebugInfo::DIType elementType =
        debugTypeConverter.convertDebugType(type.getElementType());
    if (!elementType) {
      // If the type that we point to can't be converted into a
      // debuginfo type, make a None pointer debuginfo type.
      elementType = debugTypeConverter.convertDebugType(
          KGEN::NoneType::get(type.getContext()));
    }
    return DebugInfo::DITargetIndependentPointerType::get(elementType);
  });
  debugTypeConverter.addConversion([&](RefType type) -> DebugInfo::DIType {
    return debugTypeConverter.convertDebugType(type.getAsPointerType());
  });

  addInferredDomainNonRecursiveReplacement([&](DebugInfo::DIType type) {
    return debugTypeConverter.convertDebugType(type);
  });
}

static Value lowerOp(StructInsertOp op, StructInsertOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  LIT::StructType ref = op.getContainer().getType();
  if (b.isSingleElement(ref))
    return adaptor.getValue();

  int index = b.getField(op.getFieldAttr(), ref);
  return b.create<StructReplaceOp>(op.getLoc(), adaptor.getValue(),
                                   adaptor.getContainer(), index);
}

static Value lowerOp(LIT::StructExtractOp op,
                     LIT::StructExtractOpAdaptor adaptor, LITTypeLowerer &b) {
  LIT::StructType ref = op.getContainer().getType();
  if (b.isSingleElement(ref))
    return adaptor.getContainer();

  int index = b.getField(op.getFieldAttr(), ref);
  return b.create<KGEN::StructExtractOp>(op.getLoc(), adaptor.getContainer(),
                                         index);
}

static Value lowerOp(RefImmutOp op, RefImmutOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefToPointerOp op, RefToPointerOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getRef();
}

static Value lowerOp(RefFromPointerOp op, RefFromPointerOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return adaptor.getPtr();
}

static Value lowerOp(RefFromPointerREPLOp op,
                     RefFromPointerREPLOpAdaptor adaptor, LITTypeLowerer &b) {
  return adaptor.getPtr();
}

static Value lowerOp(RefLoadOp op, RefLoadOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return b.create<POP::LoadOp>(op.getLoc(), adaptor.getRef());
}

static Value lowerOp(RefStoreOp op, RefStoreOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  b.replaceOpWithNewOp<POP::StoreOp>(op, adaptor.getValue(), adaptor.getDest());
  return {};
}

static Value lowerOp(RefStructGEROp op, RefStructGEROpAdaptor adaptor,
                     LITTypeLowerer &b) {
  auto ref =
      cast<LIT::StructType>(op.getContainer().getType().getElementType());
  if (b.isSingleElement(ref))
    return adaptor.getContainer();

  int index = b.getField(op.getFieldAttr(), ref);
  return b.create<StructGEPOp>(op.getLoc(), adaptor.getContainer(), index);
}

/// Squash noop rebinds exposed by ref -> ptr lowering.
static Value lowerOp(RebindOp op, RebindOpAdaptor adaptor, LITTypeLowerer &b) {
  // If this is a noop after lowering, squish it
  if (adaptor.getInput().getType() ==
      b.replace(op.getType(), TypeDomain::AsType))
    return adaptor.getInput();
  // Otherwise just leave it and type replacement will form a valid rebind
  // in the new type domain.
  return op.getResult();
}

// lit.ref.pack.create => kgen.pack.create
static Value lowerOp(RefPackCreateOp op, RefPackCreateOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return b.create<PackCreateOp>(op.getLoc(),
                                b.replace(op.getType(), TypeDomain::AsType),
                                adaptor.getOperands());
}

// lit.ref.pack.extract => kgen.pack.extract
static Value lowerOp(RefPackExtractOp op, RefPackExtractOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  Value value = b.create<PackExtractOp>(op.getLoc(), adaptor.getOperands()[0],
                                        op.getIndex());
  // If the result didn't fold to a pointer type, we need to emit a rebind.
  Type expected = b.replace(op.getType(), TypeDomain::AsType);
  if (value.getType() != expected)
    value = b.create<RebindOp>(op.getLoc(), expected, value);
  return value;
}

Value LITTypeLowerer::getCastedToType(Location loc, Value value, Type type) {
  // If already casted, done.
  if (value.getType() == type)
    return value;

  // If coming from a cast, use input.
  if (auto castOp = value.getDefiningOp<mlir::UnrealizedConversionCastOp>())
    return getCastedToType(loc, castOp.getOperand(0), type);

  // Otherwise create a new cast.
  auto cast = create<mlir::UnrealizedConversionCastOp>(loc, type, value);
  unrealizedCasts.push_back(cast);
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
    Type newType = replace(value.getType(), TypeDomain::AsType);

    // When value is a function argument, location info's function scope is
    // different from the operations in the function body. Use op->getLoc()
    // for new cast op's location instead of using value.loc().
    castedOperands.push_back(getCastedToType(op->getLoc(), value, newType));
  }

  typename OpT::Adaptor adaptor(castedOperands, op->getAttrDictionary());
  if (op->getNumResults() == 1) {
    auto resultType = op->getResult(0).getType();
    Value result = lowerOp(op, adaptor, *this);
    if (result.getType() != resultType)
      result = getCastedToType(result.getLoc(), result, resultType);

    if (op->getResult(0) != result)
      replaceOp(op, {result});
  } else {
    assert(op->getNumResults() == 0);
    [[maybe_unused]] Value result = lowerOp(op, adaptor, *this);
    assert(!result && "nullary lowering shouldn't produce an op");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Entrypoint.
//===----------------------------------------------------------------------===//

LogicalResult LIT::lowerLITTypes(ModuleOp module, StructDecls &state) {
  if (failed(detectIllegalStructDeclsRecursion(state)))
    return failure();
  LITTypeLowerer b(module.getContext(), state);

  // Lower operations first.
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<StructInsertOp, LIT::StructExtractOp, RefImmutOp, RefToPointerOp,
              RefFromPointerOp, RefFromPointerREPLOp, RefStructGEROp, RefLoadOp,
              RefStoreOp, RebindOp, RefPackCreateOp, RefPackExtractOp>(
            [&](auto op) { return b.materializeLowering(op); })
        .Default([&](auto op) { return success(); });
  });
  if (result.wasInterrupted())
    return failure();

  module.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto structGen = dyn_cast<StructGeneratorOp>(op)) {
      // Make sure valueDomainType is translated in the value domain.
      Type valueDomainType =
          b.replace(structGen.getValueDomainType(), TypeDomain::AsValue);
      b.replaceElementsIn(structGen, TypeDomain::AsType, /*replaceAttrs=*/true,
                          /*replaceLocs=*/true,
                          /*replaceTypes=*/true);
      structGen.setValueDomainType(valueDomainType);
      return WalkResult::advance();
    }

    b.replaceElementsIn(op, TypeDomain::AsType, /*replaceAttrs=*/true,
                        /*replaceLocs=*/true,
                        /*replaceTypes=*/true);
    if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(op)) {
      b.setInsertionPoint(cast);
      Type inType = cast.getOperand(0).getType();
      Type outType = cast.getResult(0).getType();
      if (inType == outType) {
        b.replaceOp(cast, cast.getOperand(0));
        return WalkResult::skip();
      } else if (isa<PointerType>(inType) && isa<PointerType>(outType)) {
        b.replaceOpWithNewOp<POP::PointerBitcastOp>(cast, outType,
                                                    cast.getOperand(0));
        return WalkResult::skip();
      }
    }
    return WalkResult::advance();
  });
  return success();
}
