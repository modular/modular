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
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace M;
using namespace KGEN;

//===----------------------------------------------------------------------===//
// ConvertKGENStructOp
//===----------------------------------------------------------------------===//

namespace {
/// Information about a struct declaration.
struct StructDeclarations {
  /// A map from struct name and field name to index. Used for lowering `insert`
  /// and `extract` ops.
  DenseMap<std::pair<StringAttr, StringAttr>, int64_t> fields;

  /// A map from struct name to field types. Used for type conversions.
  DenseMap<StringAttr, SmallVector<Type>> fieldTypes;
};

/// Struct operations need to refer to the struct declaration symbol.
class StructOperationLowerer : public mlir::IRRewriter {
public:
  explicit StructOperationLowerer(MLIRContext *ctx,
                                  StructDeclarations &structDecls)
      : IRRewriter(ctx), structDecls(structDecls) {}

  ~StructOperationLowerer() {
    for (mlir::UnrealizedConversionCastOp cast : conversions)
      replaceOp(cast, cast.getOperands());
  }

  /// Get the index of the struct field.
  int64_t getField(StringAttr name, RefType ref) const {
    return structDecls.fields.lookup({ref.getName(), name});
  }

  /// Replace a KGEN struct with a POP struct.
  Type substituteStructRef(RefType ref);

  /// Recursively substitute types.
  Type substituteTypes(Type type);

  /// Materialize source conversions.
  void replaceOp(Operation *op, ValueRange values) override;

  /// Materialize destination conversions.
  template <typename OpT>
  void materializeLowering(OpT op);

private:
  StructDeclarations &structDecls;
  std::vector<mlir::UnrealizedConversionCastOp> conversions;
};
} // namespace

Type StructOperationLowerer::substituteStructRef(RefType ref) {
  auto it = structDecls.fieldTypes.find(ref.getName());
  assert(it != structDecls.fieldTypes.end());
  // Substitute parameters into the field types.
  ParameterEvaluator evaluator;
  for (ParamBindAttr bind : ref.getParamValues())
    evaluator.setParameterValue(bind.getDecl(), bind.getValue());

  SmallVector<Type> elementTypes;
  for (Type type : it->second) {
    Type elementType = substituteTypes(evaluator.getReboundType(type));
    elementTypes.push_back(elementType);
  }
  return POP::StructType::get(ref.getContext(), elementTypes);
}

Type StructOperationLowerer::substituteTypes(Type type) {
  if (auto ref = dyn_cast<RefType>(type))
    return substituteStructRef(ref);
  auto itf = dyn_cast<mlir::SubElementTypeInterface>(type);
  if (!itf)
    return type;
  return itf.replaceSubElements([&](Type type) -> Type {
    if (auto ref = dyn_cast<RefType>(type))
      return substituteStructRef(ref);
    return type;
  });
}

void StructOperationLowerer::replaceOp(Operation *op, ValueRange values) {
  auto type = op->getResultTypes().front();
  if (!isa<RefType>(type))
    return IRRewriter::replaceOp(op, values);
  auto source = create<mlir::UnrealizedConversionCastOp>(op->getLoc(), type,
                                                         values.front());
  conversions.push_back(source);
  IRRewriter::replaceOp(op, source.getResult(0));
}

template <typename OpT>
void StructOperationLowerer::materializeLowering(OpT op) {
  SmallVector<Value> values;
  values.reserve(op->getNumOperands());
  for (Value value : op->getOperands()) {
    auto dest = create<mlir::UnrealizedConversionCastOp>(
        op->getLoc(), substituteTypes(value.getType()), value);
    conversions.push_back(dest);
    values.push_back(dest.getResult(0));
  }
  typename OpT::Adaptor adaptor(values, op->getAttrDictionary());
  lowerStructOp(op, adaptor, *this);
}

//===----------------------------------------------------------------------===//
// Struct Operation Lowerings
//===----------------------------------------------------------------------===//

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
      lowerer.getField(op.getFieldAttr(), cast<RefType>(structType));
  lowerer.replaceOpWithNewOp<POP::StructGEPOp>(op, adaptor.getContainer(),
                                               lowerer.getIndexAttr(index));
}

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
    SmallVector<Type> fieldTypes;
    for (auto [idx, field] : llvm::enumerate(decl.getFieldDecls())) {
      fieldTypes.push_back(field.getType());
      structDecls.fields.try_emplace({decl.getNameAttr(), field.getNameAttr()},
                                     idx);
    }
    structDecls.fieldTypes.try_emplace(decl.getNameAttr(),
                                       std::move(fieldTypes));
    decl->erase();
  }
  StructOperationLowerer lowerer(&getContext(), structDecls);

  // Lower operations.
  getOperation()->walk([&](Operation *op) {
    lowerer.setInsertionPoint(op);
    llvm::TypeSwitch<Operation *>(op)
        .Case<StructInsertOp, StructExtractOp, StructCreateOp, StructGEPOp>(
            [&](auto op) { lowerer.materializeLowering(op); });
  });

  // Type references can be used in nested types. Walk through all the types and
  // rewrite them in-place to use the lowered types.
  auto substituteTypes = [&](Type type) -> Type {
    return lowerer.substituteTypes(type);
  };
  getOperation()->walk([&](Operation *op) {
    // Substitute any references in attributes.
    op->setAttrs(op->getAttrDictionary()
                     .replaceSubElements(substituteTypes)
                     .cast<DictionaryAttr>());

    // Substitute the result types.
    for (OpResult result : op->getOpResults())
      result.setType(substituteTypes(result.getType()));

    // Substitute the block argument types.
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          arg.setType(substituteTypes(arg.getType()));
  });
}
