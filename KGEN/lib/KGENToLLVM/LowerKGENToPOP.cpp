//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/KGENDialect/KGENTypes.h"
#include "KGEN/KGENDialect/ParameterEvaluator.h"
#include "KGEN/KGENPasses.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/IR/DebugInfoTypes.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

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
class StructOperationLowerer : public mlir::IRRewriter {
public:
  explicit StructOperationLowerer(MLIRContext *ctx,
                                  StructDeclarations &structDecls)
      : IRRewriter(ctx), structDecls(structDecls) {}

  /// Get the index of the struct field.
  int64_t getField(StringAttr name, DeclRefType ref) const {
    return structDecls.fieldIndices.lookup({ref.getName(), name});
  }

  /// Replace a KGEN struct with a POP struct.
  Type substituteStructRef(DeclRefType ref);

  /// Try to build debug informatino for the given struct ref.
  DebugInfo::DIType
  buildDebugInfoForStructRef(DeclRefType ref,
                             DebugInfo::DebugInfoTypeConverter &converter);

  /// Recursively substitute types.
  Type substituteTypes(Type type);

  /// Materialize source conversions.
  void replaceOp(Operation *op, ValueRange values) override;

  /// Materialize destination conversions.
  template <typename OpT>
  void materializeLowering(OpT op);

private:
  StructDeclarations &structDecls;
};
} // namespace

Type StructOperationLowerer::substituteStructRef(DeclRefType ref) {
  auto it = structDecls.fields.find(ref.getName());
  assert(it != structDecls.fields.end());

  // Substitute parameters into the field types.
  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : ref.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());

  SmallVector<Type> elementTypes;
  for (Type type : llvm::make_second_range(it->second))
    elementTypes.push_back(evaluator.getReboundType(type));
  return POP::StructType::get(ref.getContext(), elementTypes);
}

DebugInfo::DIType StructOperationLowerer::buildDebugInfoForStructRef(
    DeclRefType ref, DebugInfo::DebugInfoTypeConverter &converter) {
  auto it = structDecls.fields.find(ref.getName());
  if (it == structDecls.fields.end())
    return {};

  // Substitute parameters into the field types.
  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : ref.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());

  SmallVector<DebugInfo::DIMemberType> elementTypes;
  for (auto [name, type] : it->second) {
    elementTypes.push_back(DebugInfo::DIMemberType::get(
        name, converter.convertDebugType(evaluator.getReboundType(type))));
  }
  return DebugInfo::DIStructType::get(ref.getName(), elementTypes);
}

Type StructOperationLowerer::substituteTypes(Type type) {
  if (auto ref = dyn_cast<DeclRefType>(type))
    type = substituteStructRef(ref);
  auto itf = dyn_cast<mlir::SubElementTypeInterface>(type);
  if (!itf)
    return type;
  return itf.replaceSubElements([&](Type type) -> Type {
    if (auto ref = dyn_cast<DeclRefType>(type))
      return substituteStructRef(ref);
    return type;
  });
}

void StructOperationLowerer::replaceOp(Operation *op, ValueRange values) {
  auto type = op->getResultTypes().front();
  if (!isa<DeclRefType>(type))
    return IRRewriter::replaceOp(op, values);
  auto source = create<mlir::UnrealizedConversionCastOp>(op->getLoc(), type,
                                                         values.front());
  IRRewriter::replaceOp(op, source.getResult(0));
}

static void lowerStructOp(StructCreateOp op, StructCreateOpAdaptor adaptor,
                          StructOperationLowerer &lowerer) {
  lowerer.replaceOpWithNewOp<POP::StructConstructOp>(
      op, lowerer.substituteStructRef(op.getType()), op.getOperands());
}

static void lowerStructOp(StructInsertOp op, StructInsertOpAdaptor adaptor,
                          StructOperationLowerer &lowerer) {
  int64_t index =
      lowerer.getField(op.getFieldAttr(), op.getContainer().getType());
  lowerer.replaceOpWithNewOp<POP::StructReplaceOp>(op, adaptor.getValue(),
                                                   adaptor.getContainer(),
                                                   lowerer.getIndexAttr(index));
}

static void lowerStructOp(StructExtractOp op, StructExtractOpAdaptor adaptor,
                          StructOperationLowerer &lowerer) {
  int64_t index =
      lowerer.getField(op.getFieldAttr(), op.getContainer().getType());
  lowerer.replaceOpWithNewOp<POP::StructGetOp>(op, adaptor.getContainer(),
                                               lowerer.getIndexAttr(index));
}

static void lowerStructOp(StructGEPOp op, StructGEPOpAdaptor adaptor,
                          StructOperationLowerer &lowerer) {
  Type structType = op.getContainer().getType().getResolvedElementType();
  int64_t index =
      lowerer.getField(op.getFieldAttr(), cast<DeclRefType>(structType));
  lowerer.replaceOpWithNewOp<POP::StructGEPOp>(op, adaptor.getContainer(),
                                               lowerer.getIndexAttr(index));
}

template <typename OpT>
void StructOperationLowerer::materializeLowering(OpT op) {
  setInsertionPoint(op);
  SmallVector<Value> values;
  values.reserve(op->getNumOperands());
  for (Value value : op->getOperands()) {
    auto dest = create<mlir::UnrealizedConversionCastOp>(
        op->getLoc(), substituteTypes(value.getType()), value);
    values.push_back(dest.getResult(0));
  }
  typename OpT::Adaptor adaptor(values, op->getAttrDictionary());
  lowerStructOp(op, adaptor, *this);
}

//===----------------------------------------------------------------------===//
// List Lowering
//===----------------------------------------------------------------------===//

/// Lists can only be expanded as element types of POP structs and function
/// types. Expand list types in-place: `struct<list<T[2]>>` becomes `struct<T,
/// T>`. Return the modified index of a struct element. Anywhere else, we
/// implicitly wrap the list types in a struct. For example,
/// `pointer<list<T[N]>>` becomes `pointer<array<N, T>>`.
static Type expandListsInType(Type type);

/// Expand lists in a struct type.
static std::pair<POP::StructType, int64_t>
expandListsInStruct(POP::StructType structType, int64_t index = 0) {
  SmallVector<Type> elementTypes, newTypes;
  (void)structType.resolveElementTypes(elementTypes);
  int64_t newIndex = index;
  for (auto [idx, elementType] : llvm::enumerate(elementTypes)) {
    if (auto list = dyn_cast<ListType>(elementType)) {
      int64_t length = *list.getResolvedLength();
      newTypes.append(length, list.getResolvedElementType());
      if (index > static_cast<int64_t>(idx))
        newIndex += length - 1;
    } else {
      newTypes.push_back(elementType);
    }
  }
  return {POP::StructType::get(structType.getContext(), newTypes), newIndex};
}

/// Expand lists in a function type.
static FunctionType expandListsInFunc(FunctionType funcType) {
  auto expandInList = [](TypeRange types) {
    SmallVector<Type> results;
    for (Type type : types) {
      if (auto list = dyn_cast<ListType>(type))
        results.append(*list.getResolvedLength(),
                       list.getResolvedElementType());
      else
        results.push_back(type);
    }
    return results;
  };
  return FunctionType::get(funcType.getContext(),
                           expandInList(funcType.getInputs()),
                           expandInList(funcType.getResults()));
}

/// Convert a list type to an array type of the same length.
static POP::ArrayType convertListToArrayType(ListType list) {
  return POP::ArrayType::get(list.getLength(), list.getElementType());
}

/// Expand or rewrite list element types.
static Type expandListsInType(Type type) {
  auto flattenFirst = [](Type type) {
    Type nextType;
    while (true) {
      if (auto structType = dyn_cast<POP::StructType>(type))
        nextType = expandListsInStruct(structType).first;
      else if (auto funcType = dyn_cast<FunctionType>(type))
        nextType = expandListsInFunc(funcType);
      else
        return type;
      if (nextType == type)
        break;
      type = nextType;
    }
    return type;
  };
  type = flattenFirst(type);

  if (auto list = dyn_cast<ListType>(type))
    return convertListToArrayType(list);

  auto itf = dyn_cast<mlir::SubElementTypeInterface>(type);
  if (!itf)
    return type;
  return itf.replaceSubElements([&](Type type) -> Type {
    if (isa<POP::StructType, FunctionType>(type))
      return flattenFirst(type);
    if (auto list = dyn_cast<ListType>(type))
      return convertListToArrayType(list);
    return type;
  });
}

/// Materialize a 1-to-N destination conversion for lists.
static ValueRange
materializeListDestConversion(mlir::RewriterBase &b,
                              mlir::TypedValue<ListType> list) {
  SmallVector<Type> resultTypes(*list.getType().getResolvedLength(),
                                list.getType().getResolvedElementType());
  return b
      .create<mlir::UnrealizedConversionCastOp>(list.getLoc(), resultTypes,
                                                list)
      .getResults();
}

/// Materialize a N-to-1 source conversion for lists.
static Value materializeListSourceConversion(mlir::RewriterBase &b,
                                             Location loc, ValueRange values,
                                             ListType list) {
  return b.create<mlir::UnrealizedConversionCastOp>(loc, list, values)
      .getResult(0);
}

/// Materialize a 1-to-1 conversion.
static Value materializeConversion(PatternRewriter &b, Value value, Type type) {
  return b.create<mlir::UnrealizedConversionCastOp>(value.getLoc(), type, value)
      .getResult(0);
}

static bool isListType(Type type) { return isa<ListType>(type); }

static bool structHasListElement(POP::StructType structType) {
  return llvm::any_of(structType.getElementTypes(), [](TypedAttr type) {
    return isListType(cast<ConcreteTypeConstantAttr>(type).getValue());
  });
}

/// ```
/// %list = kgen.param.constant: list<index[2]> = <[1, 2]>
/// ```
///
/// becomes
///
/// ```
/// %l0 = kgen.param.constant = <1>
/// %l1 = kgen.param.constant = <2>
/// ```
struct ExpandListConstantOp : public mlir::OpRewritePattern<ParamConstantOp> {
  ExpandListConstantOp(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(ParamConstantOp op,
                                PatternRewriter &b) const override {
    auto list = dyn_cast<ListType>(op.getType());
    if (!list)
      return failure();
    SmallVector<Value> values;
    values.reserve(*list.getResolvedLength());
    for (TypedAttr value : cast<ListAttr>(op.getValue()).getValues())
      values.push_back(b.create<ParamConstantOp>(op.getLoc(), value));
    b.replaceOp(op,
                materializeListSourceConversion(b, op.getLoc(), values, list));
    return success();
  }
};

/// get(%list[i]) -> %li
struct ExpandListGetOp : public mlir::OpRewritePattern<ListGetOp> {
  ExpandListGetOp(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(ListGetOp op,
                                PatternRewriter &b) const override {
    b.replaceOp(
        op, materializeListDestConversion(
                b, op.getList())[cast<IntegerAttr>(op.getIndex()).getInt()]);
    return success();
  }
};

/// create(%l0, %l1) -> %l0, %l1
struct ExpandListCreateOp : public mlir::OpRewritePattern<ListCreateOp> {
  ExpandListCreateOp(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(ListCreateOp op,
                                PatternRewriter &b) const override {
    b.replaceOp(op, materializeListSourceConversion(
                        b, op.getLoc(), op.getOperands(), op.getType()));
    return success();
  }
};

/// ```
/// for (%e0, ...) in dot(map^i(I0), %list) do IV(i+1) = f(%e0, ..., IV(i))
/// ```
///
/// becomes
///
/// ```
/// IV1 = f(%list[map^0(I0)], IV0),
/// IV2 = f(%list[map^1(I0)], IV1),
/// ...
/// ```
struct ExpandListIterateOp : public mlir::OpRewritePattern<ListIterateOp> {
  ExpandListIterateOp(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}

  /// Hash a compound index.
  struct HashLoopIndex : public llvm::DenseMapInfo<SmallVector<int64_t>> {
    static SmallVector<int64_t> getEmptyKey() { return {-1}; }
    static SmallVector<int64_t> getTombstoneKey() { return {-2}; }
    static unsigned getHashValue(ArrayRef<int64_t> value) {
      return llvm::hash_combine_range(value.begin(), value.end());
    }
    static bool isEqual(ArrayRef<int64_t> lhs, ArrayRef<int64_t> rhs) {
      return lhs == rhs;
    }
  };

  LogicalResult matchAndRewrite(ListIterateOp op,
                                PatternRewriter &b) const override {
    ValueRange elements = materializeListDestConversion(b, op.getList());
    SmallVector<Value> inductionVars = llvm::to_vector(op.getArguments());
    SmallVector<int64_t> index;
    index.reserve(op.getInit().size());
    for (TypedAttr init : op.getInit())
      index.push_back(cast<IntegerAttr>(init).getInt());
    mlir::AffineMap map = op.getMap();
    int64_t length = *op.getList().getType().getResolvedLength();

    // Keep a set of the visited indices that are in-bounds. An index indicates
    // an exit condition if it was already seen (which means the sequence
    // loops), or if any of the values are out-of-bounds.
    DenseSet<SmallVector<int64_t>, HashLoopIndex> seenIndices;
    auto indexIndicatesExit = [&seenIndices,
                               length](const SmallVector<int64_t> &index) {
      if (llvm::any_of(index,
                       [length](int64_t i) { return i < 0 || i >= length; }))
        return true;
      return !seenIndices.insert(index).second;
    };

    // Start unrolling the loop body.
    ArrayRef<BlockArgument> listArgs =
        op.getBody().getArguments().slice(0, index.size());
    ArrayRef<BlockArgument> ivArgs =
        op.getBody().getArguments().slice(listArgs.size());
    while (!indexIndicatesExit(index)) {
      BlockAndValueMapping bv;
      // Map the list values.
      for (auto [i, listArg] : llvm::zip(index, listArgs))
        bv.map(listArg, elements[i]);
      for (auto [iv, ivArg] : llvm::zip(inductionVars, ivArgs))
        bv.map(ivArg, iv);

      // Clone the body with the remapped arguments.
      for (Operation &loopOp : op.getBody().front().without_terminator())
        b.clone(loopOp, bv);

      // Assign the next induction variables and update the list.
      auto yield = cast<ListYieldOp>(op.getBody().front().getTerminator());
      inductionVars.clear();
      for (Value value : yield.getOperands())
        inductionVars.push_back(bv.lookup(value));
      index = map.compose(index);
    }

    // Replace the results of the operation with the last values of the
    // induction variables.
    b.replaceOp(op, inductionVars);
    return success();
  }
};

/// construct(%a, %list, %b) -> construct(%a, %l0, %l1, %l2, %b).
struct ExpandStructConstructOp
    : public mlir::OpRewritePattern<POP::StructConstructOp> {
  ExpandStructConstructOp(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(POP::StructConstructOp op,
                                PatternRewriter &b) const override {
    if (llvm::none_of(op.getOperandTypes(), isListType))
      return failure();

    SmallVector<Value> operands;
    for (Value value : op.getElements()) {
      if (auto list = dyn_cast<ListType>(value.getType())) {
        ValueRange elements = materializeListDestConversion(b, value);
        operands.append(elements.begin(), elements.end());
      } else {
        operands.push_back(value);
      }
    }
    auto construct = b.create<POP::StructConstructOp>(
        op.getLoc(), expandListsInStruct(op.getType()).first, operands);
    b.replaceOp(op,
                materializeConversion(b, construct.getResult(), op.getType()));
    return success();
  }
};

/// %list = get %struct[2] -> %li = get %struct[2 + i] for each i.
struct ExpandStructGetOp : public mlir::OpRewritePattern<POP::StructGetOp> {
  ExpandStructGetOp(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(POP::StructGetOp op,
                                PatternRewriter &b) const override {
    if (!structHasListElement(op.getContainer().getType()))
      return failure();

    auto [type, index] = expandListsInStruct(op.getContainer().getType(),
                                             op.getIndexAttr().getInt());
    Value container = materializeConversion(b, op.getContainer(), type);
    if (auto list = dyn_cast<ListType>(op.getType())) {
      SmallVector<Value> results;
      int64_t length = *list.getResolvedLength();
      results.reserve(length);
      for (int64_t i = 0; i < length; ++i)
        results.push_back(
            b.create<POP::StructGetOp>(op.getLoc(), container, i + index));
      b.replaceOp(
          op, materializeListSourceConversion(b, op.getLoc(), results, list));
    } else {
      b.startRootUpdate(op);
      op.setOperand(container);
      op.setIndexAttr(b.getIndexAttr(index));
      b.finalizeRootUpdate(op);
    }
    return success();
  }
};

/// %s = replace(%s0, %list) -> replace(replace(replace(%s0, %l0), %l1), %l2).
struct ExpandStructReplaceOp
    : public mlir::OpRewritePattern<POP::StructReplaceOp> {
  ExpandStructReplaceOp(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(POP::StructReplaceOp op,
                                PatternRewriter &b) const override {
    if (!structHasListElement(op.getType()))
      return failure();

    auto [type, index] = expandListsInStruct(op.getContainer().getType(),
                                             op.getIndexAttr().getInt());
    Value container = materializeConversion(b, op.getContainer(), type);
    if (isa<ListType>(op.getValue().getType())) {
      for (auto [i, value] :
           llvm::enumerate(materializeListDestConversion(b, op.getValue())))
        container = b.create<POP::StructReplaceOp>(op.getLoc(), value,
                                                   container, i + index);
    } else {
      container = b.create<POP::StructReplaceOp>(op.getLoc(), op.getValue(),
                                                 container, index);
    }
    b.replaceOp(op, materializeConversion(b, container, op.getType()));
    return success();
  }
};

/// gep %s[i] -> bitcast(gep %s[i])
struct ExpandStructGEPOp : public mlir::OpRewritePattern<POP::StructGEPOp> {
  ExpandStructGEPOp(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(POP::StructGEPOp op,
                                PatternRewriter &b) const override {
    auto structType = cast<POP::StructType>(
        op.getContainer().getType().getResolvedElementType());
    if (!structHasListElement(structType))
      return failure();

    auto [type, index] =
        expandListsInStruct(structType, op.getIndexAttr().getInt());
    Value containerPtr = materializeConversion(b, op.getContainer(),
                                               POP::PointerType::get(type));
    if (auto list = dyn_cast<ListType>(op.getType().getResolvedElementType())) {
      // If the list is empty, then we point to nothing. Generate a nullptr...
      if (!*list.getResolvedLength()) {
        Value zero = b.create<mlir::index::ConstantOp>(op.getLoc(), 0);
        b.replaceOpWithNewOp<POP::IndexToPointerOp>(op, op.getType(), zero);
        return success();
      }

      // Take the address of the first element in the expanded list.
      Value startPtr = b.create<POP::StructGEPOp>(op.getLoc(), containerPtr,
                                                  b.getIndexAttr(index));
      // Bitcast it to ptr<array<N, T>>.
      Value listPtr = b.create<POP::PointerBitcastOp>(
          op.getLoc(), POP::PointerType::get(convertListToArrayType(list)),
          startPtr);
      b.replaceOp(op, materializeConversion(b, listPtr, op.getType()));
    } else {
      b.startRootUpdate(op);
      op.setOperand(containerPtr);
      op.setIndexAttr(b.getIndexAttr(index));
      b.finalizeRootUpdate(op);
    }
    return success();
  }
};

/// %list = load(%listPtr) -> %0 = load(%listPtr[0]), %1 = load(%listPtr[1])
struct ExpandListLoad : public mlir::OpRewritePattern<POP::LoadOp> {
  ExpandListLoad(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(POP::LoadOp op,
                                PatternRewriter &b) const override {
    auto list =
        dyn_cast<ListType>(op.getPtr().getType().getResolvedElementType());
    if (!list)
      return failure();

    Type arrPtrType = POP::PointerType::get(convertListToArrayType(list));
    Value arrPtr = materializeConversion(b, op.getPtr(), arrPtrType);
    Value elPtr = b.create<POP::PointerBitcastOp>(
        op.getLoc(), POP::PointerType::get(list.getElementType()), arrPtr);
    SmallVector<Value> results;
    int64_t length = *list.getResolvedLength();
    results.reserve(length);
    for (int64_t i = 0; i < length; ++i) {
      Value offset = b.create<mlir::index::ConstantOp>(op.getLoc(), i);
      Value curPtr = b.create<POP::OffsetOp>(op.getLoc(), elPtr, offset);
      results.push_back(
          b.create<POP::LoadOp>(op.getLoc(), curPtr, op.getAlignmentAttr()));
    }
    Value newList =
        materializeListSourceConversion(b, op.getLoc(), results, list);
    b.replaceOp(op, newList);
    return success();
  }
};

/// store(%list, %listPtr) -> store(%l0, %listPtr[0]), ...
struct ExpandListStore : public mlir::OpRewritePattern<POP::StoreOp> {
  ExpandListStore(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(POP::StoreOp op,
                                PatternRewriter &b) const override {
    auto list =
        dyn_cast<ListType>(op.getPtr().getType().getResolvedElementType());
    if (!list)
      return failure();

    Type arrPtrType = POP::PointerType::get(convertListToArrayType(list));
    Value arrPtr = materializeConversion(b, op.getPtr(), arrPtrType);
    Value elPtr = b.create<POP::PointerBitcastOp>(
        op.getLoc(), POP::PointerType::get(list.getElementType()), arrPtr);
    ValueRange elements = materializeListDestConversion(b, op.getArg());
    for (auto [idx, element] : llvm::enumerate(elements)) {
      Value offset = b.create<mlir::index::ConstantOp>(op.getLoc(), idx);
      Value curPtr = b.create<POP::OffsetOp>(op.getLoc(), elPtr, offset);
      b.create<POP::StoreOp>(op.getLoc(), element, curPtr,
                             op.getAlignmentAttr());
    }
    b.eraseOp(op);
    return success();
  }
};

/// Convert lists elements of a variant to arrays.
static POP::VariantType convertVariantType(POP::VariantType variant) {
  SmallVector<TypedAttr> types;
  for (TypedAttr type : variant.getTypes()) {
    if (auto list =
            dyn_cast<ListType>(cast<ConcreteTypeConstantAttr>(type).getValue()))
      types.push_back(
          ConcreteTypeConstantAttr::get(convertListToArrayType(list)));
    else
      types.push_back(type);
  }
  return POP::VariantType::get(variant.getContext(), types);
}

/// Expand `pop.variant.get` to a series of `pop.array.get`.
struct ExpandVariantGet : public mlir::OpRewritePattern<POP::VariantGetOp> {
  ExpandVariantGet(MLIRContext *ctx) : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(POP::VariantGetOp op,
                                PatternRewriter &b) const override {
    auto list = dyn_cast<ListType>(op.getType());
    if (!list)
      return failure();
    Value variant = materializeConversion(
        b, op.getVariant(), convertVariantType(op.getVariant().getType()));
    Value arr = b.create<POP::VariantGetOp>(
        op.getLoc(), convertListToArrayType(list), variant);
    SmallVector<Value> results;
    int64_t length = *list.getResolvedLength();
    results.reserve(length);
    for (int64_t i = 0; i < length; ++i)
      results.push_back(b.create<POP::ArrayGetOp>(op.getLoc(), arr, i));
    b.replaceOp(op,
                materializeListSourceConversion(b, op.getLoc(), results, list));
    return success();
  }
};

/// Expand `pop.variant.create` to `pop.array.create`.
struct ExpandVariantCreate
    : public mlir::OpRewritePattern<POP::VariantCreateOp> {
  ExpandVariantCreate(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(POP::VariantCreateOp op,
                                PatternRewriter &b) const override {
    auto list = dyn_cast<ListType>(op.getOperand().getType());
    if (!list)
      return failure();

    ValueRange elements = materializeListDestConversion(b, op.getOperand());
    Value arr = b.create<POP::ArrayCreateOp>(
        op.getLoc(), convertListToArrayType(list), elements);
    Value variant = b.create<POP::VariantCreateOp>(
        op.getLoc(), convertVariantType(op.getType()), arr);
    b.replaceOp(op, materializeConversion(b, variant, op.getType()));
    return success();
  }
};

/// Expand lists in a debuginfo.value. Lists don't have a runtime
/// representation, but we use an array for the purposes of showing the
/// debuginfo.
struct ExpandListDebugValue
    : public mlir::OpRewritePattern<DebugInfo::ValueOp> {
  ExpandListDebugValue(MLIRContext *ctx)
      : OpRewritePattern(ctx, /*benefit=*/2) {}

  LogicalResult matchAndRewrite(DebugInfo::ValueOp op,
                                PatternRewriter &b) const override {
    auto list = dyn_cast<ListType>(op.getValue().getType());
    if (!list)
      return failure();

    ValueRange elements = materializeListDestConversion(
        b, mlir::TypedValue<ListType>(op.getValue()));
    b.updateRootInPlace(op, [&] {
      op.setOperand(b.create<POP::ArrayCreateOp>(op.getLoc(), elements));
    });
    return success();
  }
};

/// Expand lists in a generic operation by expanding operands, results, and
/// block arguments in-place.
struct ExpandGenericOperation : public mlir::RewritePattern {
  ExpandGenericOperation(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &b) const override {
    // Don't recurse on the unrealized casts. They will be removed by DCE.
    if (isa<mlir::UnrealizedConversionCastOp>(op))
      return failure();

    // Expand the operands.
    SmallVector<Value> operands;
    bool hasListOperand = false;
    for (Value operand : op->getOperands()) {
      if (isa<ListType>(operand.getType())) {
        hasListOperand = true;
        ValueRange expanded = materializeListDestConversion(b, operand);
        operands.append(expanded.begin(), expanded.end());
      } else {
        operands.push_back(operand);
      }
    }
    if (hasListOperand) {
      b.updateRootInPlace(op, [&] { op->setOperands(operands); });
      return success();
    }

    // Expand block arguments.
    bool changed = false;
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        // The number of arguments changes and iterators get invalidated.
        b.setInsertionPointToStart(&block);
        for (unsigned i = 0; i < block.getNumArguments();) {
          BlockArgument arg = block.getArgument(i);
          auto list = dyn_cast<ListType>(arg.getType());
          if (!list) {
            ++i;
            continue;
          }

          if (!changed) {
            changed = true;
            b.startRootUpdate(op);
          }
          int64_t length = *list.getResolvedLength();
          SmallVector<Value> expanded;
          expanded.reserve(length);
          for (int64_t j = 0; j < length; ++j)
            expanded.push_back(block.insertArgument(
                i + j + 1, list.getResolvedElementType(), arg.getLoc()));
          arg.replaceAllUsesWith(
              materializeListSourceConversion(b, arg.getLoc(), expanded, list));
          block.eraseArgument(i);
          i += length;
        }
      }
    }
    if (changed) {
      b.finalizeRootUpdate(op);
      return success();
    }

    // Expand the results. Results are immutable so we have to rebuild the op.
    SmallVector<Type> results;
    b.setInsertionPoint(op);
    bool hasListResult = false;
    for (Value result : op->getResults()) {
      if (auto list = dyn_cast<ListType>(result.getType())) {
        hasListResult = true;
        results.append(*list.getResolvedLength(),
                       list.getResolvedElementType());
      } else {
        results.push_back(result.getType());
      }
    }
    if (hasListResult) {
      // Copy everything over.
      OperationState state(op->getLoc(), op->getName(), op->getOperands(),
                           results, {}, op->getSuccessors());
      state.attributes = op->getAttrDictionary();
      for (Region &region : op->getRegions()) {
        state.addRegion(std::make_unique<Region>());
        state.regions.back()->takeBody(region);
      }
      Operation *newOp = b.create(state);
      SmallVector<Value> results;
      results.reserve(op->getNumResults());
      // Slice out the expanded results and materialize conversions for them.
      for (int64_t i = 0, j = 0, e = op->getNumResults(); i < e; ++i) {
        if (auto list = dyn_cast<ListType>(op->getResult(i).getType())) {
          int64_t length = *list.getResolvedLength();
          ValueRange listValues = newOp->getResults().slice(j, length);
          j += length;
          results.push_back(materializeListSourceConversion(b, op->getLoc(),
                                                            listValues, list));
        } else {
          results.push_back(newOp->getResult(j++));
        }
      }
      b.replaceOp(op, results);
      return success();
    }

    // Nothing changed.
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENTOPOP
#include "KGEN/KGENPasses.h.inc"
} // namespace M::KGEN

namespace {
struct LowerKGENToPOPPass
    : public KGEN::impl::LowerKGENToPOPBase<LowerKGENToPOPPass> {
  using LowerKGENToPOPBase::LowerKGENToPOPBase;

  void runOnOperation() override;
};
} // namespace

void LowerKGENToPOPPass::runOnOperation() {
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
  getOperation()->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<StructCreateOp, StructInsertOp, StructExtractOp, StructGEPOp>(
            [&](auto op) { structLowerer.materializeLowering(op); });
  });

  // Build a converter to handle updating converted types within debug info
  // constructs.
  DebugInfo::DebugInfoTypeConverter debugTypeConverter;
  debugTypeConverter.addConversion([&](Type type) -> Optional<Type> {
    Type newType = structLowerer.substituteTypes(type);
    if (newType != type)
      return debugTypeConverter.convertDebugType(newType);
    return llvm::None;
  });
  debugTypeConverter.addConversion([&](DeclRefType type) -> DebugInfo::DIType {
    return structLowerer.buildDebugInfoForStructRef(type, debugTypeConverter);
  });
  debugTypeConverter.addConversion([&](ListType type) -> Optional<Type> {
    Type elementType = type.getResolvedElementType();
    if (!elementType)
      return llvm::None;

    // Treat a list as an array for the sake of debugging.
    return DebugInfo::DIArrayType::get(
        debugTypeConverter.convertDebugType(elementType),
        *type.getResolvedLength());
  });

  // Type references can be used in nested types. Walk through all the types and
  // rewrite them in-place to use the lowered types.
  mlir::AttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](Type type) -> Type { return structLowerer.substituteTypes(type); });
  replacer.addReplacement([&](DebugInfo::DIType type) -> Type {
    return debugTypeConverter.convertDebugType(type);
  });
  replacer.recursivelyReplaceElementsIn(getOperation(), /*replaceAttrs=*/true,
                                        /*replaceLocs=*/true,
                                        /*replaceTypes=*/true);

  // Transpose boundary operations involving lists. We have to do this after
  // KGEN structs are lowered so that we don't have to iterate between both. Use
  // the greedy rewrite driver to apply the lowerings iteratively.
  mlir::GreedyRewriteConfig config;
  config.maxIterations = mlir::GreedyRewriteConfig::kNoIterationLimit;
  RewritePatternSet patterns(&getContext());
  patterns
      .insert<ExpandListConstantOp, ExpandListGetOp, ExpandListCreateOp,
              ExpandListIterateOp, ExpandStructConstructOp, ExpandStructGetOp,
              ExpandStructReplaceOp, ExpandStructGEPOp, ExpandListStore,
              ExpandListLoad, ExpandVariantGet, ExpandVariantCreate,
              ExpandListDebugValue, ExpandGenericOperation>(&getContext());
  (void)mlir::applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                           config);

  // Okay, there should be no list operations at the boundary between nested
  // types and values anymore. Expand lists in anything left.
  std::vector<mlir::UnrealizedConversionCastOp> leftoverCasts;
  getOperation()->walk([&](Operation *op) {
    if (auto cast = dyn_cast<mlir::UnrealizedConversionCastOp>(op))
      leftoverCasts.push_back(cast);

    // Expand any lists in attributes.
    op->setAttrs(op->getAttrDictionary()
                     .replaceSubElements(expandListsInType)
                     .cast<DictionaryAttr>());

    // Expand the result types.
    for (OpResult result : op->getOpResults())
      result.setType(expandListsInType(result.getType()));

    // Expand the block argument types.
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          arg.setType(expandListsInType(arg.getType()));
  });
  // Clean up any leftover casts after substituting types.
  for (mlir::UnrealizedConversionCastOp cast : leftoverCasts) {
    if (cast.getOperandTypes() == cast.getResultTypes()) {
      cast.replaceAllUsesWith(cast.getOperands());
      cast->erase();
    }
  }
}
