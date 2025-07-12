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
        [&](TypedAttr attr) -> FailureOr<Attribute> {
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
  std::enable_if_t<std::is_convertible_v<ResultT, FailureOr<BaseT>>>
  addNonRecursiveReplacement(FnT &&callback, DomainId domain) {
    addReplacement(mlir::AttrTypeReplacer::ReplaceFn<BaseT>(
                       [f = std::forward<FnT>(callback)](BaseT base)
                           -> mlir::AttrTypeReplacer::ReplaceFnResult<BaseT> {
                         if constexpr (std::is_same_v<T, BaseT>) {
                           FailureOr<BaseT> ret = f(base);
                           if (succeeded(ret))
                             return {{*ret, WalkResult::skip()}};
                           else
                             return {{nullptr, WalkResult::interrupt()}};
                         }
                         if (auto derived = dyn_cast<T>(base)) {
                           FailureOr<BaseT> ret = f(derived);
                           if (succeeded(ret))
                             return {{*ret, WalkResult::skip()}};
                           else
                             return {{nullptr, WalkResult::interrupt()}};
                         }
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
  std::enable_if_t<std::is_convertible_v<ResultT, FailureOr<BaseT>>>
  addInferredDomainNonRecursiveReplacement(FnT &&callback) {
    addNonRecursiveReplacement(std::forward<FnT>(callback), TypeDomain::AsType);
    if constexpr (!std::is_same_v<BaseT, TypedAttr>)
      addNonRecursiveReplacement(std::forward<FnT>(callback),
                                 TypeDomain::AsValue);
  }

  /// Convenience helper for replacing parameters and returning parameters.
  FailureOr<TypedAttr> replaceParameter(TypedAttr attr) {
    FailureOr<Attribute> attrOr = replace(attr, TypeDomain::AsType);
    if (failed(attrOr))
      return failure();

    return cast<TypedAttr>(*attrOr);
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
                             ParameterEvaluationContext &evalContext,
                             MLIRContext *ctx) {
  auto typeType = TypeType::get(ctx);
  auto emptyStructType = KGEN::StructType::get(ctx, {});
  auto emptyStruct = StructAttr::get({}, emptyStructType);

  // TypeParamAttr dispatches replacing to different domains.
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](TypeParamAttr typeValue) -> FailureOr<Attribute> {
        auto typeValueOr =
            replacer.replace(typeValue.getTypeValue(), TypeDomain::AsValue);
        auto mlirTypeOr =
            replacer.replace(typeValue.getMlirType(), TypeDomain::AsType);
        auto typeOr = replacer.replace(typeValue.getType(), TypeDomain::AsType);
        auto vtableOr =
            replacer.replace(typeValue.getVTable(), TypeDomain::AsType);

        if (failed(typeValueOr) || failed(mlirTypeOr) || failed(typeOr) ||
            failed(vtableOr))
          return failure();

        return TypeParamAttr::get(*typeValueOr, *mlirTypeOr, *typeOr,
                                  cast<VTableAttr>(*vtableOr));
      });

  // ParamRefTypes should be TypeValueType if in the value domain.
  replacer.addNonRecursiveReplacement(
      [&replacer](ParamType paramRef) -> FailureOr<Type> {
        auto paramRefOr = replacer.replaceParameter(paramRef.getParam());
        if (failed(paramRefOr))
          return failure();
        return TypeValueType::get(*paramRefOr);
      },
      TypeDomain::AsValue);

  // The param types of a GeneratorType are always types, not values.
  for (TypeDomain domain : {TypeDomain::AsType, TypeDomain::AsValue}) {
    // Simply report the error after cycle detected.
    replacer.addCycleBreaker(
        [&decls](Type t) -> std::optional<Type> {
          auto structTp = dyn_cast<LIT::StructType>(t);
          if (structTp) {
            // Simply return a nullptr to signal a error has occurs.
            mlir::emitError(decls.get(structTp.getName()).loc,
                            "struct has recursive reference to itself");
            return Type();
          }
          // Should be unreachable? must be a aggregated type in order to have
          // recursive reference.
          return std::nullopt;
        },
        domain);

    auto replaceAsType = [&replacer](Type type) {
      return replacer.replace(type, TypeDomain::AsType);
    };
    replacer.addNonRecursiveReplacement(
        [domain, replaceAsType,
         &replacer](GeneratorType gen) -> FailureOr<Type> {
          SmallVector<FailureOr<Type>> inputParamTypesOr(
              map_range(gen.getInputParamTypes(), replaceAsType));
          if (llvm::any_of(inputParamTypesOr, failed))
            return failure();

          SmallVector<Type> inputParamTypes(map_range(
              inputParamTypesOr, [](FailureOr<Type> t) { return *t; }));
          Attribute metadata = gen.getMetadata();
          if (metadata) {
            auto metadataOr = replacer.replace(metadata, domain);
            if (failed(metadataOr))
              return failure();
            metadata = *metadataOr;
          }
          auto bodyOr = replacer.replace(gen.getBody(), domain);
          if (failed(bodyOr))
            return failure();

          return GeneratorType::get(inputParamTypes, *bodyOr, metadata);
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
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](RefPackAttr refPack) -> FailureOr<Attribute> {
        SmallVector<TypedAttr> loweredElts;
        loweredElts.reserve(refPack.getValues().size());
        for (TypedAttr elt : refPack.getValues()) {
          auto eltOr = replacer.replaceParameter(elt);
          if (failed(eltOr))
            return failure();
          loweredElts.push_back(*eltOr);
        }
        FailureOr<Type> typeOr =
            replacer.replace(refPack.getType(), TypeDomain::AsType);
        if (failed(typeOr))
          return failure();
        auto type = cast<PackType>(*typeOr);
        return PackAttr::get(loweredElts, type);
      });

  // !lit.ref.pack<:variadic<!kgen.type> types, owned_in_mem, mut life, 42>
  // => !kgen.pack<variadic_ptr_map(types), 42>
  replacer.addInferredDomainNonRecursiveReplacement(
      [&replacer](RefPackType ref) -> FailureOr<Type> {
        auto variadicOr = replacer.replaceParameter(ref.getVariadic());
        auto addrSpaceOr = replacer.replaceParameter(ref.getAddressSpace());
        if (failed(variadicOr) || failed(addrSpaceOr))
          return failure();
        return PackType::get(ParamOperatorAttr::get(POC::VariadicPtrMap,
                                                    *variadicOr, *addrSpaceOr));
      });

  // !lit.ref -> !kgen.pointer
  for (TypeDomain domain : {TypeDomain::AsType, TypeDomain::AsValue})
    replacer.addNonRecursiveReplacement(
        [domain, &replacer](RefType ref) -> FailureOr<Type> {
          auto elemTpOr = replacer.replace(ref.getElementType(), domain);
          auto addrSpaceOr = replacer.replaceParameter(ref.getAddressSpace());
          if (failed(elemTpOr) || failed(addrSpaceOr))
            return failure();
          return PointerType::get(*elemTpOr, *addrSpaceOr);
        },
        domain);

  // Replace all origin attributes with empty structs. These attributes are
  // all terminal.
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](AnyOriginAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](StaticOriginAttr) { return emptyStruct; });
  replacer.addInferredDomainNonRecursiveReplacement(
      [=](ComptimeOriginAttr) { return emptyStruct; });
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
      [&, noneType](LITStructAttr attr) -> FailureOr<Attribute> {
        LIT::StructType ref = attr.getType();
        StructDecl &decl = decls.get(ref.getName());

        SmallVector<TypedAttr> values;
        values.reserve(attr.getValues().size());
        for (auto [entry, type] : llvm::zip(attr.getValues(), decl.fields)) {
          FailureOr<TypedAttr> valueOr =
              replacer.replaceParameter(std::get<1>(entry));
          if (failed(valueOr))
            return failure();

          TypedAttr value = *valueOr;
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

        auto refOr = replacer.replace(ref, TypeDomain::AsType);
        if (failed(refOr))
          return failure();
        if (auto type = cast_or_null<KGEN::StructType>(*refOr))
          return StructAttr::get(values, type);
        return failure();
      });

  // #lit.struct.extract -> #kgen.struct.extract
  replacer.addInferredDomainNonRecursiveReplacement(
      [&](LIT::StructExtractAttr attr) -> FailureOr<Attribute> {
        auto ref = cast<LIT::StructType>(attr.getStructValue().getType());
        int idx = decls.fieldIndices.at({ref.getName(), attr.getField()});
        auto valueOr = replacer.replaceParameter(attr.getStructValue());
        if (failed(valueOr))
          return failure();
        if (decls.get(ref.getName()).isSingleElement())
          return *valueOr;
        return KGEN::StructExtractAttr::get(*valueOr, idx);
      });

  // Since lowerings have been generated for all struct types, we just need to
  // lookup the lowered type and substitute the parameters.
  // - For the AsType type domain, convert into a StructType.
  // - For the AsValue type domain, convert into a symbol reference to the
  //   pre-created symbol generator op.
  replacer.addNonRecursiveReplacement(
      [&, noneType, ctx,
       evalCtxPtr = &evalContext](LIT::StructType ref) -> FailureOr<Type> {
        StructDecl &decl = decls.get(ref.getName());
        // Substitute the given parameters in.
        ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
        evaluator.setEvaluationContext(evalCtxPtr);
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
      [&](LIT::StructType ref) -> FailureOr<Type> {
        StringAttr leafName = ref.getValue().getValue().getLeafReference();
        auto structDeclIter = decls.structDecls.find(leafName);
        StructDecl &decl = structDeclIter->second;
        SmallVector<FailureOr<TypedAttr>> loweredParamValuesOr(
            map_range(ref.getParamValues(), [&](TypedAttr value) {
              return replacer.replaceParameter(value);
            }));
        if (llvm::any_of(loweredParamValuesOr, failed))
          return failure();

        SmallVector<TypedAttr> loweredParamValues(
            map_range(loweredParamValuesOr,
                      [&](FailureOr<TypedAttr> value) { return *value; }));
        auto structMetaOr = replacer.replace(
            StructMetaType::get(LIT::StructType::get(
                decl.symRef, loweredParamValues, ref.getSignature())),
            TypeDomain::AsType);
        if (failed(structMetaOr))
          return failure();

        auto concreteSymRef = TypeGeneratorRefAttr::get(
            decl.symRef, loweredParamValues, *structMetaOr);
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
  explicit LITTypeLowerer(ModuleOp module, StructDecls &structDecls,
                          mlir::LockedSymbolTableCollection &symtab);

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

  /// Evaluation context used for simplifying parameters.
  LITSymTabEvaluationContext evalContext;
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
    DebugInfo::DebugInfoNonCyclicTypeConverter &converter,
    ParameterEvaluationContext &evalContext) {
  // Substitute parameters into the field types.
  StructDecl &decl = structDecls.get(ref.getName());
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
  evaluator.setEvaluationContext(&evalContext);

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

LITTypeLowerer::LITTypeLowerer(ModuleOp module, StructDecls &structDecls,
                               mlir::LockedSymbolTableCollection &symtab)
    : IRRewriter(module.getContext()), evalContext(module, symtab),
      structDecls(structDecls) {
  populateReplacer(structDecls, *this, evalContext, module.getContext());

  // Build a converter to handle updating converted types within debug info
  // constructs.
  debugTypeConverter.addConversion([&](Type type) -> std::optional<Type> {
    FailureOr<Type> newTypeOr = replace(type, TypeDomain::AsType);
    if (succeeded(newTypeOr) && *newTypeOr != type)
      return debugTypeConverter.convertDebugType(*newTypeOr);
    return std::nullopt;
  });
  debugTypeConverter.addConversion(
      [&](LIT::StructType type) -> DebugInfo::DIType {
        return buildDebugInfoForStructRef(type, structDecls, debugTypeConverter,
                                          evalContext);
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
  auto typeOr = b.replace(op.getType(), TypeDomain::AsType);
  if (failed(typeOr))
    return nullptr;
  return b.create<PackCreateOp>(op.getLoc(), *typeOr, adaptor.getOperands());
}

// lit.ref.pack.extract => kgen.pack.extract
static Value lowerOp(RefPackExtractOp op, RefPackExtractOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  Value value = b.create<PackExtractOp>(op.getLoc(), adaptor.getOperands()[0],
                                        op.getIndex());
  // If the result didn't fold to a pointer type, we need to emit a rebind.
  FailureOr<Type> expectedOr = b.replace(op.getType(), TypeDomain::AsType);
  if (failed(expectedOr))
    return nullptr;
  if (value.getType() != *expectedOr)
    value = b.create<RebindOp>(op.getLoc(), *expectedOr, value);
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

    auto newTypeOr = replace(value.getType(), TypeDomain::AsType);
    if (failed(newTypeOr))
      return failure();
    // When value is a function argument, location info's function scope is
    // different from the operations in the function body. Use op->getLoc()
    // for new cast op's location instead of using value.loc().
    castedOperands.push_back(getCastedToType(op->getLoc(), value, *newTypeOr));
  }

  typename OpT::Adaptor adaptor(castedOperands, op->getAttrDictionary());
  if (op->getNumResults() == 1) {
    auto resultType = op->getResult(0).getType();
    Value result = lowerOp(op, adaptor, *this);
    if (!result)
      return failure();
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

LogicalResult LIT::lowerLITTypes(ModuleOp module, StructDecls &state,
                                 mlir::LockedSymbolTableCollection &symtab) {
  // Do a simple recursive type detection, this does not guarantees completeness
  // as it does not take parameter into account. Additional cycle detection will
  // be performed during lowering.
  if (failed(detectIllegalStructDeclsRecursion(state)))
    return failure();
  LITTypeLowerer b(module, state, symtab);

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

  result = module.walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    if (auto structGen = dyn_cast<StructGeneratorOp>(op)) {
      // Make sure valueDomainType is translated in the value domain.
      auto valueDomainTypeOr =
          b.replace(structGen.getValueDomainType(), TypeDomain::AsValue);
      if (failed(valueDomainTypeOr))
        return WalkResult::interrupt();

      LogicalResult res = b.replaceElementsIn(structGen, TypeDomain::AsType,
                                              /*replaceAttrs=*/true,
                                              /*replaceLocs=*/true,
                                              /*replaceTypes=*/true);
      if (failed(res))
        return WalkResult::interrupt();

      structGen.setValueDomainType(*valueDomainTypeOr);
      return WalkResult::skip();
    }

    LogicalResult res =
        b.replaceElementsIn(op, TypeDomain::AsType, /*replaceAttrs=*/true,
                            /*replaceLocs=*/true,
                            /*replaceTypes=*/true);

    if (failed(res))
      return WalkResult::interrupt();

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

  // Lower types in StructGeneratorOps last because we need witness entries to
  // keep using LIT types in order for ParameterEvaluator to work smoothly.
  for (StructGeneratorOp op : module.getOps<StructGeneratorOp>()) {
    op.getBody().walk([&](Operation *op) {
      LogicalResult res =
          b.replaceElementsIn(op, TypeDomain::AsType, /*replaceAttrs=*/true,
                              /*replaceLocs=*/true,
                              /*replaceTypes=*/true);
      if (failed(res))
        return WalkResult::interrupt();

      return WalkResult::advance();
    });
  }

  if (result.wasInterrupted())
    return failure();

  return success();
}
