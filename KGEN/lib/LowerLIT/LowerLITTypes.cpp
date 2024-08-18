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

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

namespace {
struct StructDecl {
  /// Return true if the struct should be flattened when lowered.
  bool isSingleElement() const {
    return isRegisterPassable && fields.size() == 1;
  }

  // Move-only type.
  StructDecl(const StructDecl &) = delete;
  StructDecl &operator=(const StructDecl &) = delete;
  StructDecl(StructDecl &&other) = default;
  StructDecl &operator=(StructDecl &&) = default;

  /// The un-parameterized SourceNameAttr for the struct decl.
  DebugInfo::SourceNameAttr sourceName;
  /// The struct input parameters.
  ParamDeclArrayAttr decls;
  /// True if the type is register-passable.
  bool isRegisterPassable;
  /// The location of the decl, for emitting errors.
  LocationAttr loc;
  /// The field names and types of the struct in order.
  SmallVector<std::pair<StringAttr, Type>> fields;

  /// Flags for tracking recursion during DFS.
  bool visited = false, done = false;
};

struct StructDecls {
  // Destructively process the module by collecting struct info and removing
  // trait and struct decls at the same time.
  LogicalResult process(ModuleOp module, SymbolTable &symtab);

  /// Lookup a struct decl.
  StructDecl &get(StringAttr name) { return structDecls.find(name)->second; }

  /// Populate `replacer` with the lowering patterns for attributes and types
  /// after computing the valid lowerings for each struct decl.
  void buildReplacer(mlir::AttrTypeReplacer &replacer, MLIRContext *ctx);

  /// A map from struct name and field name to index. Used for lowering `insert`
  /// and `extract` ops.
  DenseMap<std::pair<StringAttr, StringAttr>, int> fieldIndices;
  /// Map from struct name to the lowering info.
  llvm::MapVector<StringAttr, StructDecl> structDecls;
};
} // namespace

void StructDecls::buildReplacer(mlir::AttrTypeReplacer &replacer,
                                MLIRContext *ctx) {
  auto addReplacement = [&replacer](auto &&func) {
    // This is a legalization replacement, so we have to replace leaves first.
    // Don't rely on the replacer's recursion, which is post-order.
    using T = typename llvm::function_traits<
        std::decay_t<decltype(func)>>::template arg_t<0>;
    using BaseT =
        std::conditional_t<std::is_base_of_v<Attribute, T>, Attribute, Type>;
    replacer.addReplacement(mlir::AttrTypeReplacer::ReplaceFn<BaseT>(
        [f = std::forward<decltype(func)>(func)](BaseT base) mutable
        -> mlir::AttrTypeReplacer::ReplaceFnResult<BaseT> {
          if (auto derived = dyn_cast<T>(base))
            return {{f(derived), WalkResult::skip()}};
          return {};
        }));
  };

  auto typeType = TypeType::get(ctx);
  auto emptyStructType = KGEN::StructType::get(ctx, {});
  auto emptyStruct = StructAttr::get({}, emptyStructType);

  // Partially bound types never have any uses in KGEN. This attribute is
  // terminal.
  // TODO: Need to codegen here when Mojo has parametric traits.
  addReplacement([=, &replacer](BindTypeAttr bind) {
    AnyStructType metatype = bind.getType();
    auto ref = LIT::StructType::get(metatype.getSymbol(),
                                    metatype.getParamValues(), typeType);
    return TypeConstantAttr::get(replacer.replace(ref), typeType);
  });

  // All metatypes lower to `!kgen.type`.
  addReplacement([=](AnyStructType) { return typeType; });
  addReplacement([=](AnyTraitType) { return typeType; });
  addReplacement([=](TraitType) { return typeType; });

  // #lit.ref.pack => #kgen.pack
  addReplacement([&replacer](RefPackAttr refPack) {
    SmallVector<TypedAttr> loweredElts;
    loweredElts.reserve(refPack.getValues().size());
    for (TypedAttr elt : refPack.getValues())
      loweredElts.push_back(cast<TypedAttr>(replacer.replace(elt)));
    auto type = cast<PackType>(replacer.replace(refPack.getType()));
    return PackAttr::get(loweredElts, type);
  });

  // !lit.ref.pack<:variadic<!kgen.type> types, owned_in_mem, mut life, 42>
  // => !kgen.pack<variadic_ptr_map(types), 42>
  addReplacement([&](RefPackType ref) {
    auto variadic = cast<TypedAttr>(replacer.replace(ref.getVariadic()));
    auto addrSpace = cast<TypedAttr>(replacer.replace(ref.getAddressSpace()));
    return PackType::get(
        ParamOperatorAttr::get(POC::VariadicPtrMap, variadic, addrSpace));
  });

  // !lit.ref -> !kgen.pointer
  addReplacement([&replacer](RefType ref) {
    return PointerType::get(
        replacer.replace(ref.getElementType()),
        cast<TypedAttr>(replacer.replace(ref.getAddressSpace())));
  });

  // Replace all lifetime attributes with empty structs. These attributes are
  // all terminal.
  addReplacement([=](LifetimeAttr) { return emptyStruct; });
  addReplacement([=](LifetimeUnionAttr) { return emptyStruct; });
  addReplacement([=](LifetimeMutCastAttr) { return emptyStruct; });
  addReplacement([=](ImplicitLifetimeRefAttr) { return emptyStruct; });
  addReplacement([=](LifetimeSetAttr) { return emptyStruct; });

  // !lit.lifetime -> !kgen.struct<()>
  addReplacement([=](LifetimeType) { return emptyStructType; });
  addReplacement([=](LifetimeSetType) { return emptyStructType; });

  using AttrResult = std::pair<Attribute, WalkResult>;
  auto noneType = KGEN::NoneType::get(ctx);

  // #lit.struct -> #kgen.struct
  replacer.addReplacement([&, noneType](LITStructAttr attr) -> AttrResult {
    LIT::StructType ref = attr.getType();
    StructDecl &decl = get(ref.getName());

    SmallVector<TypedAttr> values;
    values.reserve(attr.getValues().size());
    for (auto [entry, type] : llvm::zip(attr.getValues(), decl.fields)) {
      TypedAttr value = cast<TypedAttr>(replacer.replace(std::get<1>(entry)));
      if (!value)
        return {{}, WalkResult::interrupt()};
      // We to check if this is a value for a struct field that is known to be
      // a pointer type, in which case we erase the element type.
      if (isa<PointerType>(type.second)) {
        auto type = cast<PointerType>(value.getType());
        auto ptrType = PointerType::get(noneType, type.getAddressSpace(),
                                        type.getExclusive());
        value = ParamOperatorAttr::get(POC::PtrBitcast, value, ptrType);
      }
      values.push_back(value);
    }

    if (decl.isSingleElement())
      return {values.front(), WalkResult::skip()};
    if (auto type = cast_or_null<KGEN::StructType>(replacer.replace(ref)))
      return {StructAttr::get(values, type), WalkResult::skip()};
    return {{}, WalkResult::interrupt()};
  });

  // #lit.struct.extract -> #kgen.struct.extract
  replacer.addReplacement([&](LIT::StructExtractAttr attr) -> AttrResult {
    auto ref = cast<LIT::StructType>(attr.getStructValue().getType());
    int idx = fieldIndices.at({ref.getName(), attr.getField()});
    auto value =
        cast_or_null<TypedAttr>(replacer.replace(attr.getStructValue()));
    if (!value)
      return {{}, WalkResult::interrupt()};
    if (get(ref.getName()).isSingleElement())
      return {value, WalkResult::skip()};
    return {KGEN::StructExtractAttr::get(value, idx), WalkResult::skip()};
  });
}

LogicalResult StructDecls::process(ModuleOp module, SymbolTable &symtab) {
  for (Operation &op : llvm::make_early_inc_range(module.getOps())) {
    if (isa<TraitDeclOp>(op)) {
      symtab.erase(&op);
      continue;
    }
    auto structOp = dyn_cast<StructDeclOp>(op);
    if (!structOp)
      continue;

    StructDecl info{};
    StringAttr structName = structOp.getSymNameAttr();
    info.sourceName = structOp.getSourceNameAttr();
    info.decls = structOp.getParamsAttr();
    info.isRegisterPassable = structOp.isRegisterPassable();
    info.loc = structOp.getLoc();

    // Collect the struct fields.
    for (auto [idx, field] : llvm::enumerate(structOp.getFieldDecls())) {
      info.fields.emplace_back(field.getNameAttr(), field.getType());
      fieldIndices.try_emplace({structName, field.getNameAttr()}, idx);
    }
    structDecls.try_emplace(structName, std::move(info));

    symtab.erase(&op);
  }

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
    StructDecl &decl = get(ref.getName());
    if (!decl.done && failed(computeLoweredType(decl)))
      return {{}, WalkResult::interrupt()};
    return {ref, WalkResult::skip()};
  });

  // Start from any struct and make sure our DFS terminates.
  for (StructDecl &decl : llvm::make_second_range(structDecls)) {
    if (decl.done)
      continue;
    if (failed(computeLoweredType(decl)))
      return failure();
  }
  return success();
}

namespace {} // namespace

//===----------------------------------------------------------------------===//
// Type Lowering
//===----------------------------------------------------------------------===//

namespace {
/// Struct operations need to refer to the struct declaration symbol.
struct LITTypeLowerer : public mlir::IRRewriter, mlir::AttrTypeReplacer {
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
  DebugInfo::DebugInfoTypeConverter debugTypeConverter;
  /// Unrealized casts to resolve at the end of type lowering.
  SmallVector<mlir::UnrealizedConversionCastOp> unrealizedCasts;
};
} // namespace

static DebugInfo::DIType
buildDebugInfoForStructRef(LIT::StructType ref, StructDecls &structDecls,
                           DebugInfo::DebugInfoTypeConverter &converter) {
  // Substitute parameters into the field types.
  StructDecl &decl = structDecls.get(ref.getName());
  ParameterEvaluator evaluator(decl.decls, ref.getParamValues());

  auto getDebugInfoType = [&](const std::pair<StringAttr, Type> &nameAndType) {
    auto [name, type] = nameAndType;
    auto reboundType = evaluator.getReboundType(type);
    DebugInfo::DIType fieldDIType = converter.convertDebugType(reboundType);
    if (!fieldDIType) {
      if (isa<KGEN::PointerType>(reboundType)) {
        fieldDIType = converter.convertDebugType(
            PointerType::get(KGEN::NoneType::get(type.getContext())));
      }
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
  structDecls.buildReplacer(*this, ctx);
  auto noneType = KGEN::NoneType::get(ctx);

  // Since lowerings have been generated for all struct types, we just need to
  // lookup the lowered type and substitute the parameters.
  addReplacement([&, noneType, ctx](LIT::StructType ref) -> Type {
    StructDecl &decl = this->structDecls.get(ref.getName());
    // Substitute the given parameters in.
    ParameterEvaluator evaluator(decl.decls, ref.getParamValues());
    SmallVector<Type> fieldTypes;
    for (auto [idx, type] :
         llvm::enumerate(llvm::make_second_range(decl.fields))) {
      if (auto ptrType = dyn_cast<PointerType>(type)) {
        fieldTypes.push_back(PointerType::get(
            noneType,
            cast<TypedAttr>(
                evaluator.getReboundAttribute(ptrType.getAddressSpace())),
            cast<TypedAttr>(
                evaluator.getReboundAttribute(ptrType.getExclusive()))));
      } else {
        fieldTypes.push_back(evaluator.getReboundType(type));
      }
    }
    if (decl.isSingleElement())
      return replace(fieldTypes.front());
    return replace(
        KGEN::StructType::get(ctx, fieldTypes, !decl.isRegisterPassable));
  });

  // Build a converter to handle updating converted types within debug info
  // constructs.
  debugTypeConverter.addConversion([&](Type type) -> std::optional<Type> {
    Type newType = replace(type);
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
      // If the type that we point to can't be converted into a debuginfo type,
      // make a None pointer debuginfo type.
      elementType = debugTypeConverter.convertDebugType(
          KGEN::NoneType::get(type.getContext()));
    }
    return DebugInfo::DITargetIndependentPointerType::get(elementType);
  });
  debugTypeConverter.addConversion([&](RefType type) -> DebugInfo::DIType {
    return debugTypeConverter.convertDebugType(type.getAsPointerType());
  });

  addReplacement([&](DebugInfo::DIType type) {
    return debugTypeConverter.convertDebugType(type);
  });
}

static Value lowerOp(LIT::StructCreateOp op, LIT::StructCreateOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  if (b.isSingleElement(op.getType()))
    return adaptor.getOperands().front();
  return b.create<KGEN::StructCreateOp>(op.getLoc(), b.replace(op.getType()),
                                        adaptor.getOperands());
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
  b.replaceOpWithNewOp<POP::StoreOp>(op, adaptor.getArg(), adaptor.getRef());
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
  if (adaptor.getInput().getType() == b.replace(op.getType()))
    return adaptor.getInput();
  // Otherwise just leave it and type replacement will form a valid rebind in
  // the new type domain.
  return op.getResult();
}

// lit.ref.pack.create => kgen.pack.create
static Value lowerOp(RefPackCreateOp op, RefPackCreateOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  return b.create<PackCreateOp>(op.getLoc(), b.replace(op.getType()),
                                adaptor.getOperands());
}

// lit.ref.pack.extract => kgen.pack.extract
static Value lowerOp(RefPackExtractOp op, RefPackExtractOpAdaptor adaptor,
                     LITTypeLowerer &b) {
  Value value = b.create<PackExtractOp>(op.getLoc(), adaptor.getOperands()[0],
                                        op.getIndex());
  // If the result didn't fold to a pointer type, we need to emit a rebind.
  Type expected = b.replace(op.getType());
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
    if (castOp.getOperand(0).getType() == type)
      return castOp.getOperand(0);

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
    Type newType = replace(value.getType());

    // When value is a function argument, location info's function scope is
    // different from the operations in the function body. Use op->getLoc() for
    // new cast op's location instead of using value.loc().
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
  StructDecls state;
  auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();
  if (failed(state.process(getOperation(), analysis.getTopLevelSymbolTable())))
    return signalPassFailure();
  LITTypeLowerer b(&getContext(), state);

  // Lower operations first.
  WalkResult result = getOperation().walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<LIT::StructCreateOp, StructInsertOp, LIT::StructExtractOp,
              RefImmutOp, RefToPointerOp, RefFromPointerOp,
              RefFromPointerREPLOp, RefStructGEROp, RefLoadOp, RefStoreOp,
              RebindOp, RefPackCreateOp, RefPackExtractOp>(
            [&](auto op) { return b.materializeLowering(op); })
        .Default([&](auto op) { return success(); });
  });
  if (result.wasInterrupted())
    return signalPassFailure();

  getOperation().walk<mlir::WalkOrder::PreOrder>([&](Operation *op) {
    b.replaceElementsIn(op, /*replaceAttrs=*/true, /*replaceLocs=*/true,
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
}
