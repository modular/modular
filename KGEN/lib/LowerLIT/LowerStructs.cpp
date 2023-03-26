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
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;
using namespace LIT;

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

  /// Replace a KGEN struct with a POP struct.
  POP::StructType substituteStructRef(DeclRefType ref);

  /// Try to build debug information for the given struct ref.
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

  /// The struct decl map.
  StructDeclarations &structDecls;

  /// The type converter.
  mlir::AttrTypeReplacer replacer;
};
} // namespace

StructOperationLowerer::StructOperationLowerer(MLIRContext *ctx,
                                               StructDeclarations &structDecls)
    : IRRewriter(ctx), structDecls(structDecls) {
  replacer.addReplacement(
      [&](DeclRefType type) -> Type { return substituteStructRef(type); });
  replacer.addReplacement([&](LIT::StructAttr attr) {
    SmallVector<TypedAttr> values;
    for (auto [name, value] : attr.getValues())
      values.push_back(value);
    return POP::StructAttr::get(values, substituteStructRef(attr.getType()));
  });
  replacer.addReplacement([&](LIT::StructExtractAttr attr) {
    auto litStructType = cast<DeclRefType>(attr.getStructValue().getType());
    int64_t fieldNo = getField(attr.getField(), litStructType);
    return POP::StructExtractAttr::get(
        replacer.replace(attr.getStructValue()),
        IntegerAttr::get(IndexType::get(attr.getContext()), fieldNo));
  });
}

POP::StructType StructOperationLowerer::substituteStructRef(DeclRefType ref) {
  auto it = structDecls.fields.find(ref.getName());
  assert(it != structDecls.fields.end());

  // Substitute parameters into the field types.
  ParameterEvaluator evaluator(ref.getParamValues());
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
  ParameterEvaluator evaluator(ref.getParamValues());

  SmallVector<DebugInfo::DIMemberType> elementTypes;
  for (auto [name, type] : it->second) {
    elementTypes.push_back(DebugInfo::DIMemberType::get(
        name, converter.convertDebugType(evaluator.getReboundType(type))));
  }
  return DebugInfo::DIStructType::get(ref.getName(), elementTypes);
}

Type StructOperationLowerer::substituteTypes(Type type) {
  return replacer.replace(type);
}

void StructOperationLowerer::replaceOp(Operation *op, ValueRange values) {
  auto type = op->getResultTypes().front();
  if (!isa<DeclRefType>(type))
    return IRRewriter::replaceOp(op, values);
  auto source = create<mlir::UnrealizedConversionCastOp>(op->getLoc(), type,
                                                         values.front());
  IRRewriter::replaceOp(op, source.getResult(0));
}

static Operation *lowerStructOp(StructCreateOp op,
                                StructCreateOpAdaptor adaptor,
                                StructOperationLowerer &lowerer) {
  return lowerer.create<POP::StructCreateOp>(
      op.getLoc(), lowerer.substituteStructRef(op.getType()), op.getOperands());
}

static Operation *lowerStructOp(StructInsertOp op,
                                StructInsertOpAdaptor adaptor,
                                StructOperationLowerer &lowerer) {
  int64_t index =
      lowerer.getField(op.getFieldAttr(), op.getContainer().getType());
  return lowerer.create<POP::StructReplaceOp>(op.getLoc(), adaptor.getValue(),
                                              adaptor.getContainer(),
                                              lowerer.getIndexAttr(index));
}

static Operation *lowerStructOp(StructExtractOp op,
                                StructExtractOpAdaptor adaptor,
                                StructOperationLowerer &lowerer) {
  int64_t index =
      lowerer.getField(op.getFieldAttr(), op.getContainer().getType());
  return lowerer.create<POP::StructExtractOp>(
      op.getLoc(), adaptor.getContainer(), lowerer.getIndexAttr(index));
}

static Operation *lowerStructOp(StructGEPOp op, StructGEPOpAdaptor adaptor,
                                StructOperationLowerer &lowerer) {
  Type structType = op.getContainer().getType().getResolvedElementType();
  int64_t index =
      lowerer.getField(op.getFieldAttr(), cast<DeclRefType>(structType));
  return lowerer.create<POP::StructGEPOp>(op.getLoc(), adaptor.getContainer(),
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
  Operation *newOp = lowerStructOp(op, adaptor, *this);
  values.clear();
  for (auto [result, type] :
       llvm::zip(newOp->getResults(), op->getResultTypes())) {
    auto src =
        create<mlir::UnrealizedConversionCastOp>(op->getLoc(), type, result);
    values.push_back(src.getResult(0));
  }
  replaceOp(op, values);
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
  getOperation()->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        .Case<StructCreateOp, StructInsertOp, StructExtractOp, StructGEPOp>(
            [&](auto op) { structLowerer.materializeLowering(op); });
  });

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
  debugTypeConverter.addConversion([&](ListType type) -> std::optional<Type> {
    Type elementType = type.getResolvedElementType();
    if (!elementType)
      return std::nullopt;

    // Treat a list as an array for the sake of debugging.
    return DebugInfo::DIArrayType::get(
        debugTypeConverter.convertDebugType(elementType),
        *type.getResolvedLength());
  });

  // Type references can be used in nested types. Walk through all the types and
  // rewrite them in-place to use the lowered types.
  structLowerer.replacer.addReplacement([&](DebugInfo::DIType type) -> Type {
    return debugTypeConverter.convertDebugType(type);
  });
  structLowerer.replacer.recursivelyReplaceElementsIn(
      getOperation(), /*replaceAttrs=*/true, /*replaceLocs=*/true,
      /*replaceTypes=*/true);
}
