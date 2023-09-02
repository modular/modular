//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
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
struct PointerLikeTypeTraits<POP::StructType>
    : public PointerLikeTypeTraits<mlir::Type> {
  static inline POP::StructType getFromVoidPointer(void *p) {
    return POP::StructType::getFromOpaquePointer(p);
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

  /// A map from struct name to field names and types. Used for type
  /// conversions.
  DenseMap<StringAttr, SmallVector<std::pair<StringAttr, Type>>> fields;
};

/// Struct operations need to refer to the struct declaration symbol.
struct StructOperationLowerer : public mlir::IRRewriter {
  explicit StructOperationLowerer(MLIRContext *ctx,
                                  StructDeclarations &structDecls);

  /// Get the index of the struct field.
  int64_t getField(StringAttr name, DeclRefType ref) const {
    return structDecls.fieldIndices.lookup({ref.getName(), name});
  }

  /// Replace a KGEN struct with a POP struct or an arbitrary type if it is was
  /// a single-element type that got flattened.
  /// This function returns PointerUnion<POP::StructType, Type> to
  /// distinguish between a flatten single-element struct and a struct that has
  /// multiple elements.
  /// Using PointerUnion<POP::StructType, Type> instead of Type because
  /// the single-element itself can also be a struct.
  /// PointerUnion doesn't know Type's RTTI.
  PointerUnion<POP::StructType, Type> substituteStructRef(DeclRefType ref);

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

  /// The empty `#pop.struct<>` attribute, which has empty struct type.
  POP::StructAttr emptyStructAttr;

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
};
} // namespace

StructOperationLowerer::StructOperationLowerer(MLIRContext *ctx,
                                               StructDeclarations &structDecls)
    : IRRewriter(ctx), structDecls(structDecls) {

  // Get the empty `#pop.struct<>` attribute, which has empty struct type.
  auto emptyStructType = POP::StructType::get(ctx, ArrayRef<Type>());
  emptyStructAttr = POP::StructAttr::get({}, emptyStructType);

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
  debugTypeConverter.addConversion(
      [&](LIT::NoneType type) -> std::optional<Type> {
        return DebugInfo::DIUnspecifiedType::get(type.getContext(), "void");
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

  auto processStructAttr = [&](LIT::StructAttr attr) -> Attribute {
    PointerUnion<POP::StructType, Type> newType =
        substituteStructRef(attr.getType());
    // Flatten single-element structs.
    if (auto type = dyn_cast<Type>(newType)) {
      ParameterEvaluator evaluator(attr.getType().getParamValues());
      Attribute value =
          evaluator.getReboundAttribute(std::get<1>(attr.getValues()[0]));

      return replace(value);
    }

    SmallVector<TypedAttr> values;
    for (auto [name, value] : attr.getValues())
      values.push_back(cast<TypedAttr>(replace(value)));
    return POP::StructAttr::get(values, cast<POP::StructType>(newType));
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

    return POP::StructExtractAttr::get(cast<TypedAttr>(structValue), fieldNo);
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
  if (auto sattr = dyn_cast<LIT::StructAttr>(attr)) {
    result = processStructAttr(sattr);
  } else if (auto sattr = dyn_cast<LIT::StructExtractAttr>(attr)) {
    result = processStructExtractAttr(sattr);
  } else if (auto symbolConstant = dyn_cast<SymbolConstantAttr>(attr)) {
    result = processSymbolConstantAttr(symbolConstant);
  } else if (auto paramOperator = dyn_cast<ParamOperatorAttr>(attr)) {
    result = processParamOperatorAttr(paramOperator);
  } else if (isa<LIT::LifetimeAttr>(attr)) {
    result = emptyStructAttr; // #lit.lifetime => #pop.struct<>
  } else if (auto paramDRE = dyn_cast<ParamDeclRefAttr>(attr)) {
    // References to parameters of lifetime type are folded to their singleton
    // value, completely eliminating any use of them, allowing us to just delete
    // them from signatures.
    if (isa<LIT::LifetimeType>(paramDRE.getType())) {
      // p => #pop.struct<>
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
    PointerUnion<POP::StructType, Type> result = substituteStructRef(ref);
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

    return cast<POP::StructType>(result);
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
          TypeArrayAttr::get(signature.getContext(), inputParamTypes),
          signature.getResultParamTypes(), signature.getValues(),
          signature.getMetadata());

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
  } else if (auto signature = dyn_cast<SignatureType>(type)) {
    result = processSignatureType(signature);
  } else if (isa<LIT::LifetimeType>(type)) {
    // !lit.lifetime => !pop.struct<>
    result = emptyStructAttr.getType();
  } else if (auto ref = dyn_cast<LIT::RefType>(type)) {
    // !lit.ref<@T, life> => !kgen.pointer<@T>
    result = PointerType::get(ref.getElementType());
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

PointerUnion<POP::StructType, Type>
StructOperationLowerer::substituteStructRef(DeclRefType ref) {
  auto it = structDecls.fields.find(ref.getName());
  if (LLVM_UNLIKELY(it == structDecls.fields.end())) {
    // This indicates that the type does not reference a struct.
    errDeclRef = ref;
    return Type(ref);
  }
  ParameterEvaluator evaluator(ref.getParamValues());
  SmallVector<Type> elementTypes;
  for (Type type : llvm::make_second_range(it->second)) {
    Type reboundType = evaluator.getReboundType(type);
    Type substituteReboundType = replace(reboundType);
    elementTypes.push_back(substituteReboundType);
  }

  // Flatten single-element structs.
  if (elementTypes.size() == 1)
    return elementTypes[0];

  return POP::StructType::get(ref.getContext(), elementTypes);
}

DebugInfo::DIType StructOperationLowerer::buildDebugInfoForStructRef(
    DeclRefType ref, DebugInfo::DebugInfoTypeConverter &converter) {
  auto it = structDecls.fields.find(ref.getName());
  if (it == structDecls.fields.end())
    return {};

  // Substitute parameters into the field types.
  ParameterEvaluator evaluator(ref.getParamValues());

  SmallVector<DebugInfo::DIMemberType> elementTypes;
  for (auto [name, type] : it->second) {
    elementTypes.push_back(DebugInfo::DIMemberType::get(
        name, converter.convertDebugType(evaluator.getReboundType(type))));
  }

  return DebugInfo::DIStructType::get(ref.getName(), elementTypes);
}

static Value lowerStructOp(StructCreateOp op, StructCreateOpAdaptor adaptor,
                           StructOperationLowerer &lowerer) {
  PointerUnion<POP::StructType, Type> newType =
      lowerer.substituteStructRef(op.getType());

  if (isa<Type>(newType)) {
    assert(adaptor.getOperands().size() == 1 &&
           "Flattening non-one element struct");
    return adaptor.getOperands()[0];
  }

  return lowerer.create<POP::StructCreateOp>(
      op.getLoc(), cast<POP::StructType>(newType), adaptor.getOperands());
}

static Value lowerStructOp(StructInsertOp op, StructInsertOpAdaptor adaptor,
                           StructOperationLowerer &lowerer) {
  int64_t index =
      lowerer.getField(op.getFieldAttr(), op.getContainer().getType());

  // Check to see if we need to flatten this.  Flattening an insert just
  // replaces the value.

  PointerUnion<POP::StructType, Type> resultStructType =
      lowerer.substituteStructRef(op.getType());
  if (index == 0) {
    if (isa<Type>(resultStructType))
      return adaptor.getValue();
  }

  auto result = lowerer.create<POP::StructReplaceOp>(
      op.getLoc(), adaptor.getValue(), adaptor.getContainer(),
      lowerer.getIndexAttr(index));

  auto structType = cast<POP::StructType>(result.getResult().getType());
  TypedAttr fieldTypedAttr = structType.getElementTypes()[index];

  if (auto attr = dyn_cast<TypeConstantAttr>(fieldTypedAttr)) {
    if (result->getOperand(0).getType() != attr.getValue()) {
      // If a Pointer type of the struct field is erased to NoneType
      // because of recursive nested type,
      // when inserting the new value to the field, a PointerBitcast is needed
      // here so that the created StructReplaceOp won't complain about the
      // types.
      OpBuilder builder(result);
      auto cast = builder.create<POP::PointerBitcastOp>(
          result.getLoc(), attr.getValue(), result->getOperand(0));
      result.setOperand(0, cast.getResult());
    }
  }

  return result;
}

static Value lowerStructOp(StructExtractOp op, StructExtractOpAdaptor adaptor,
                           StructOperationLowerer &lowerer) {
  int64_t index =
      lowerer.getField(op.getFieldAttr(), op.getContainer().getType());

  // Check to see if we need to flatten this.  Flattening an extract just
  // returns the value.
  if (index == 0) {
    if (isa<Type>(lowerer.substituteStructRef(op.getContainer().getType())))
      return adaptor.getContainer();
  }

  return lowerer.create<POP::StructExtractOp>(
      op.getLoc(), adaptor.getContainer(), lowerer.getIndexAttr(index));
}

static Value lowerStructOp(StructGEPOp op, StructGEPOpAdaptor adaptor,
                           StructOperationLowerer &lowerer) {
  auto structType =
      cast<DeclRefType>(op.getContainer().getType().getElementAsType());
  int64_t index = lowerer.getField(op.getFieldAttr(), structType);

  // Check to see if we need to flatten this.  A flattened gep is a noop.
  if (index == 0) {
    if (isa<Type>(lowerer.substituteStructRef(structType)))
      return adaptor.getContainer();
  }

  return lowerer.create<POP::StructGEPOp>(op.getLoc(), adaptor.getContainer(),
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
  assert(op->getNumResults() == 1);
  auto resultType = op->getResult(0).getType();

  Value result = lowerStructOp(op, adaptor, *this);

  if (result.getType() != resultType)
    result = getCastedToType(result, resultType, *this);

  replaceOp(op, {result});

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
    // Our IR will only have single-block regions.
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

//===----------------------------------------------------------------------===//
// LowerStructsPass
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERSTRUCTS
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerStructsPass
    : public KGEN::impl::LowerStructsBase<LowerStructsPass> {
  using LowerStructsBase::LowerStructsBase;

  void runOnOperation() override;
};
} // namespace

void LowerStructsPass::runOnOperation() {
  // Collect all struct declarations and erase them.
  auto &analysis = getAnalysis<mlir::SymbolTableAnalysis>();

  StructDeclarations structDecls;

  for (auto decl :
       llvm::make_early_inc_range(getOperation().getOps<StructDeclOp>())) {
    SmallVector<std::pair<StringAttr, Type>> fields;
    for (auto [idx, field] : llvm::enumerate(decl.getFieldDecls())) {
      fields.emplace_back(field.getNameAttr(), field.getType());
      structDecls.fieldIndices.try_emplace(
          {decl.getNameAttr(), field.getNameAttr()}, idx);
    }

    structDecls.fields.try_emplace(decl.getNameAttr(), std::move(fields));
    analysis.getTopLevelSymbolTable().erase(decl);
  }

  StructOperationLowerer structLowerer(&getContext(), structDecls);

  // Lower KGEN struct operations.
  structLowerer.eraseRecursivePointerField = true;
  WalkResult result = getOperation()->walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<StructCreateOp, StructInsertOp, StructExtractOp, StructGEPOp>(
            [&](auto op) { return structLowerer.materializeLowering(op); })
        .Case<GeneratorOp>([&](auto op) { return lowerFuncOp(op); })
        .Default([](auto) { return success(); });
  });
  structLowerer.eraseRecursivePointerField = false;
  if (result.wasInterrupted())
    return signalPassFailure();

  // update converted types with debug info
  structLowerer.runDebugTypeConversion = true;

  // Type references can be used in nested types. Walk through all the types and
  // rewrite them in-place to use the lowered types. Walk pre-order, and while
  // doing so, erase any trivial casts left over from the type conversion.
  std::function<LogicalResult(Operation *)> replaceTypes =
      [&](Operation *op) -> LogicalResult {
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
    for (Region &region : op->getRegions())
      for (Operation &op : llvm::make_early_inc_range(region.getOps()))
        if (failed(replaceTypes(&op)))
          return failure();
    return success();
  };
  if (failed(replaceTypes(getOperation())))
    return signalPassFailure();
}
