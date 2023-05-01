//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/LITDialect/LITOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
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
  PointerUnion<POP::StructType, Type> substituteStructRef(DeclRefType ref);

  /// Try to build debug information for the given struct ref.
  DebugInfo::DIType
  buildDebugInfoForStructRef(DeclRefType ref,
                             DebugInfo::DebugInfoTypeConverter &converter);

  /// Recursively substitute types.
  Type substituteTypes(Type type) { return replacer.replace(type); }

  /// Materialize destination conversions.
  template <typename OpT>
  LogicalResult materializeLowering(OpT op);

  /// The struct decl map.
  StructDeclarations &structDecls;

  /// The type converter.
  mlir::AttrTypeReplacer replacer;

  /// Set to the value of an invalid DeclRefType.
  DeclRefType errDeclRef;
};
} // namespace

StructOperationLowerer::StructOperationLowerer(MLIRContext *ctx,
                                               StructDeclarations &structDecls)
    : IRRewriter(ctx), structDecls(structDecls) {

  replacer.addReplacement([&](DeclRefType type) -> Type {
    auto result = substituteStructRef(type);
    if (auto type = dyn_cast<Type>(result))
      return type;
    return cast<POP::StructType>(result);
  });

  replacer.addReplacement([&](LIT::StructAttr attr) -> Attribute {
    auto newType = substituteStructRef(attr.getType());
    // Flatten single-element structs.
    if (auto type = dyn_cast<Type>(newType)) {
      ParameterEvaluator evaluator(attr.getType().getParamValues());
      auto value =
          evaluator.getReboundAttribute(std::get<1>(attr.getValues()[0]));
      return replacer.replace(value);
    }

    SmallVector<TypedAttr> values;
    for (auto [name, value] : attr.getValues())
      values.push_back(cast<TypedAttr>(replacer.replace(value)));
    return POP::StructAttr::get(values, cast<POP::StructType>(newType));
  });

  replacer.addReplacement([&](LIT::StructExtractAttr attr) -> Attribute {
    auto litStructType = cast<DeclRefType>(attr.getStructValue().getType());
    int64_t fieldNo = getField(attr.getField(), litStructType);
    auto structValue = replacer.replace(attr.getStructValue());

    // If this is an extract of element 0, check to see if it
    // is a flattened struct.
    if (fieldNo == 0) {
      if (isa<Type>(substituteStructRef(litStructType)))
        return structValue;
    }

    return POP::StructExtractAttr::get(
        cast<TypedAttr>(structValue),
        IntegerAttr::get(IndexType::get(attr.getContext()), fieldNo));
  });
}

PointerUnion<POP::StructType, Type>
StructOperationLowerer::substituteStructRef(DeclRefType ref) {
  auto it = structDecls.fields.find(ref.getName());
  if (LLVM_UNLIKELY(it == structDecls.fields.end())) {
    // This indicates that the type does not reference a struct.
    errDeclRef = ref;
    return Type(ref);
  }

  // Substitute parameters into the field types.
  ParameterEvaluator evaluator(ref.getParamValues());
  SmallVector<Type> elementTypes;
  for (Type type : llvm::make_second_range(it->second))
    elementTypes.push_back(replacer.replace(evaluator.getReboundType(type)));

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
  auto newType = lowerer.substituteStructRef(op.getType());
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
  if (index == 0) {
    if (isa<Type>(lowerer.substituteStructRef(op.getType())))
      return adaptor.getValue();
  }

  return lowerer.create<POP::StructReplaceOp>(op.getLoc(), adaptor.getValue(),
                                              adaptor.getContainer(),
                                              lowerer.getIndexAttr(index));
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
      cast<DeclRefType>(op.getContainer().getType().getResolvedElementType());
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
  for (Value value : op->getOperands()) {
    auto newType = substituteTypes(value.getType());
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
    decl->erase();
  }
  StructOperationLowerer structLowerer(&getContext(), structDecls);

  // Lower KGEN struct operations.
  WalkResult result = getOperation()->walk([&](Operation *op) -> WalkResult {
    return llvm::TypeSwitch<Operation *, LogicalResult>(op)
        .Case<StructCreateOp, StructInsertOp, StructExtractOp, StructGEPOp>(
            [&](auto op) { return structLowerer.materializeLowering(op); })
        .Default([](auto) { return success(); });
  });
  if (result.wasInterrupted())
    return signalPassFailure();

  // Build a converter to handle updating converted types within debug info
  // constructs.
  DebugInfo::DebugInfoTypeConverter debugTypeConverter;
  debugTypeConverter.addConversion([&](Type type) -> std::optional<Type> {
    Type newType = structLowerer.substituteTypes(type);
    if (newType != type)
      return debugTypeConverter.convertDebugType(newType);
    return std::nullopt;
  });
  debugTypeConverter.addConversion([&](DeclRefType type) -> DebugInfo::DIType {
    return structLowerer.buildDebugInfoForStructRef(type, debugTypeConverter);
  });
  debugTypeConverter.addConversion(
      [&](LIT::NoneType type) -> std::optional<Type> {
        return DebugInfo::DIUnspecifiedType::get(type.getContext(), "void");
      });

  structLowerer.replacer.addReplacement([&](DebugInfo::DIType type) -> Type {
    return debugTypeConverter.convertDebugType(type);
  });
  // Type references can be used in nested types. Walk through all the types and
  // rewrite them in-place to use the lowered types. Walk pre-order, and while
  // doing so, erase any trivial casts left over from the type conversion.
  std::function<LogicalResult(Operation *)> replaceTypes =
      [&](Operation *op) -> LogicalResult {
    structLowerer.replacer.replaceElementsIn(
        op, /*replaceAttrs=*/true, /*replaceLocs=*/true, /*replaceTypes=*/true);
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
