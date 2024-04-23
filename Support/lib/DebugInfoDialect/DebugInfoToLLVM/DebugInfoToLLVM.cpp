//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/IR/DebugInfoAttrs.h"
#include "Support/DebugInfoDialect/IR/DebugInfoDialect.h"
#include "Support/DebugInfoDialect/IR/DebugInfoOps.h"
#include "Support/DebugInfoDialect/Transforms/Conversion.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/DebugStringHelper.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"

using namespace M;
using namespace M::DebugInfo;

namespace LLVM = mlir::LLVM;

//===----------------------------------------------------------------------===//
// MetadataConversion
//===----------------------------------------------------------------------===//

namespace {
/// This class handles the conversion of DebugInfo metadata to the LLVM IR
/// metadata equivalent.
struct MetadataConverter {
  MetadataConverter(DebugInfo::DebugInfoTypeConverter &typeConverter)
      : typeConverter(typeConverter) {}

  /// Convert the given derived debug info attribute to LLVM.
  template <typename T>
  auto convertAttr(T attr) {
    // Infer the LLVM type from the attribute kind.
    using LLVMTypeT = std::remove_pointer_t<decltype(convertAttrImpl(attr))>;
    return cast_or_null<LLVMTypeT>(convertAttrImpl(DIAttr(attr)));
  }

  /// Convert the given derived debug info type to LLVM.
  template <typename T>
  auto convertType(T type) {
    // Infer the LLVM type from the Type kind.
    using LLVMTypeT = std::remove_pointer_t<decltype(convertTypeImpl(type))>;
    return cast_or_null<LLVMTypeT>(convertTypeImpl(DIType(type)));
  }

private:
  Attribute convertAttrImpl(DIAttr attr);
  LLVM::DICompileUnitAttr convertAttrImpl(DICompileUnitAttr attr);
  LLVM::DIFileAttr convertAttrImpl(DIFileAttr attr);
  LLVM::DILexicalBlockAttr convertAttrImpl(DILexicalBlockAttr attr);
  LLVM::DILocalVariableAttr convertAttrImpl(DILocalVariableAttr attr);
  LLVM::DIScopeAttr convertAttrImpl(DIScopeAttr attr);
  LLVM::DISubprogramAttr convertAttrImpl(DISubprogramAttr attr);

  LLVM::DIExpressionAttr convertAttrImpl(DIAggregatesIntoExprAttr attr);
  LLVM::DIExpressionAttr convertAttrImpl(DIRefOfExprAttr attr);
  LLVM::DIExpressionAttr convertAttrImpl(DIDerefExprAttr attr);
  LLVM::DIExpressionAttr convertAttrImpl(DIIRValueExprAttr attr);

  LLVM::DITypeAttr convertTypeImpl(DIType type);
  LLVM::DITypeAttr convertTypeImpl(DIArrayType type);
  LLVM::DIBasicTypeAttr convertTypeImpl(DIBasicType type);
  LLVM::DITypeAttr convertTypeImpl(DIPointerType type);
  LLVM::DITypeAttr convertTypeImpl(DIStructType type);
  LLVM::DISubroutineTypeAttr convertTypeImpl(DISubroutineType type);
  LLVM::DITypeAttr convertTypeImpl(DIUnresolvedMLIRType type);
  LLVM::DIBasicTypeAttr convertTypeImpl(DIUnspecifiedType type);
  LLVM::DITypeAttr convertTypeImpl(DIVariantType type);
  LLVM::DITypeAttr convertTypeImpl(DIVectorType type);

  DebugInfo::DebugInfoTypeConverter &typeConverter;
  DenseMap<Attribute, Attribute> convertedAttrs;
  DenseMap<Type, LLVM::DITypeAttr> convertedTypes;
};
} // namespace

//===----------------------------------------------------------------------===//
// Attributes

Attribute MetadataConverter::convertAttrImpl(DIAttr attr) {
  if (!attr)
    return nullptr;
  if (Attribute converted = convertedAttrs.lookup(attr))
    return converted;

  Attribute result =
      TypeSwitch<DIAttr, Attribute>(attr)
          .Case<DIAggregatesIntoExprAttr, DICompileUnitAttr, DIDerefExprAttr,
                DIFileAttr, DIIRValueExprAttr, DILexicalBlockAttr,
                DILocalVariableAttr, DIRefOfExprAttr, DISubprogramAttr>(
              [&](auto attr) { return convertAttrImpl(attr); });
  return convertedAttrs[attr] = result;
}

LLVM::DICompileUnitAttr
MetadataConverter::convertAttrImpl(DICompileUnitAttr attr) {
  return LLVM::DICompileUnitAttr::get(
      attr.getContext(),
      mlir::DistinctAttr::create(mlir::UnitAttr::get(attr.getContext())),
      attr.getSourceLanguage(), convertAttr(attr.getFile()), attr.getProducer(),
      attr.getIsOptimized(),
      static_cast<LLVM::DIEmissionKind>(attr.getEmissionKind()),
      static_cast<LLVM::DINameTableKind>(attr.getNameTableKind()));
}

LLVM::DIFileAttr MetadataConverter::convertAttrImpl(DIFileAttr attr) {
  return LLVM::DIFileAttr::get(attr.getContext(), attr.getName(),
                               attr.getDirectory());
}

LLVM::DILexicalBlockAttr
MetadataConverter::convertAttrImpl(DILexicalBlockAttr attr) {
  return LLVM::DILexicalBlockAttr::get(convertAttr(attr.getScope()),
                                       convertAttr(attr.getFile()),
                                       attr.getLine(), attr.getColumn());
}

LLVM::DILocalVariableAttr
MetadataConverter::convertAttrImpl(DILocalVariableAttr attr) {
  return LLVM::DILocalVariableAttr::get(
      convertAttr(attr.getScope()), attr.getName(), convertAttr(attr.getFile()),
      attr.getLine(), attr.getArg(), attr.getAlignInBits(),
      convertType(attr.getType()));
}

LLVM::DISubprogramAttr
MetadataConverter::convertAttrImpl(DISubprogramAttr attr) {
  return LLVM::DISubprogramAttr::get(
      attr.getContext(),
      mlir::DistinctAttr::create(mlir::UnitAttr::get(attr.getContext())),
      convertAttr(attr.getCompileUnit()), convertAttr(attr.getScope()),
      attr.getName().encode(), attr.getLinkageName(),
      convertAttr(attr.getFile()), attr.getLine(), attr.getScopeLine(),
      static_cast<LLVM::DISubprogramFlags>(attr.getSubprogramFlags()),
      convertType(attr.getType()));
}

//===----------------------------------------------------------------------===//
// DIExpression Attributes

LLVM::DIExpressionAttr
MetadataConverter::convertAttrImpl(DIAggregatesIntoExprAttr attr) {
  auto prefix = llvm::dyn_cast_or_null<LLVM::DIExpressionAttr>(
      convertAttr(cast<DIAttr>(attr.getFieldExpr())));
  if (!prefix)
    return {};

  auto llvmStructType =
      cast<LLVM::DICompositeTypeAttr>(convertType(attr.getDIType()));

  // Fragments for single-element structs are elided in the LLVM representation.
  if (llvmStructType.getElements().size() == 1)
    return prefix;

  auto targetMember = cast<LLVM::DIDerivedTypeAttr>(
      llvmStructType.getElements()[attr.getFieldIndex()]);
  uint64_t fieldSize = targetMember.getSizeInBits();

  // Fragments that cover the entire size of the struct are elided in the LLVM
  // representation. This can happen if the other elements of this struct are
  // 0-sized.
  if (fieldSize == llvmStructType.getSizeInBits())
    return prefix;

  uint64_t prefixSize = 0;
  for (LLVM::DINodeAttr member :
       llvmStructType.getElements().take_front(attr.getFieldIndex())) {
    auto memberType = cast<LLVM::DIDerivedTypeAttr>(member);
    uint64_t sizeInBits = memberType.getSizeInBits();
    uint32_t alignInBits = memberType.getAlignInBits();
    prefixSize =
        llvm::alignTo(prefixSize, std::max(1u, alignInBits)) + sizeInBits;
  }

  if (uint32_t fieldAlignment = targetMember.getAlignInBits())
    prefixSize = llvm::alignTo(prefixSize, fieldAlignment);

  SmallVector<LLVM::DIExpressionElemAttr> expr(prefix.getOperations());
  expr.push_back(LLVM::DIExpressionElemAttr::get(
      attr.getContext(), llvm::dwarf::DW_OP_LLVM_fragment,
      {prefixSize, fieldSize}));
  return LLVM::DIExpressionAttr::get(attr.getContext(), expr);
}

LLVM::DIExpressionAttr
MetadataConverter::convertAttrImpl(DIRefOfExprAttr attr) {
  auto prefix = dyn_cast_or_null<LLVM::DIExpressionAttr>(
      convertAttr(cast<DIAttr>(attr.getValueExpr())));
  if (!prefix)
    return {};

  SmallVector<LLVM::DIExpressionElemAttr> expr(prefix.getOperations());
  expr.push_back(LLVM::DIExpressionElemAttr::get(
      attr.getContext(), llvm::dwarf::DW_OP_LLVM_implicit_pointer, {}));
  return LLVM::DIExpressionAttr::get(attr.getContext(), expr);
}

LLVM::DIExpressionAttr
MetadataConverter::convertAttrImpl(DIDerefExprAttr attr) {
  auto prefix = dyn_cast_or_null<LLVM::DIExpressionAttr>(
      convertAttr(cast<DIAttr>(attr.getPtrExpr())));
  if (!prefix)
    return {};

  SmallVector<LLVM::DIExpressionElemAttr> expr(prefix.getOperations());
  expr.push_back(LLVM::DIExpressionElemAttr::get(attr.getContext(),
                                                 llvm::dwarf::DW_OP_deref, {}));
  return LLVM::DIExpressionAttr::get(attr.getContext(), expr);
}

LLVM::DIExpressionAttr
MetadataConverter::convertAttrImpl(DIIRValueExprAttr attr) {
  // The base case is just an empty/trivial location list.
  return LLVM::DIExpressionAttr::get(attr.getContext(), {});
}

//===----------------------------------------------------------------------===//
// Types

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIType type) {
  if (!type)
    return {};
  if (LLVM::DITypeAttr converted = convertedTypes.lookup(type))
    return converted;

  // Run the type through the type converter to resolve any lingering types.
  type = typeConverter.convertDebugType(type);

  // Dispatch to the right metadata converter.
  LLVM::DITypeAttr result =
      TypeSwitch<DIType, LLVM::DITypeAttr>(type)
          .Case<DIArrayType, DIBasicType, DIPointerType, DIStructType,
                DISubroutineType, DIUnresolvedMLIRType, DIUnspecifiedType,
                DIVariantType, DIVectorType>(
              [&](auto type) { return convertTypeImpl(type); });
  return convertedTypes[type] = result;
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIArrayType type) {
  Builder builder(type.getContext());
  auto element = LLVM::DISubrangeAttr::get(
      type.getContext(), builder.getI64IntegerAttr(type.getElementCount()),
      /*lowerBound=*/nullptr, /*upperBound=*/nullptr, /*stride=*/nullptr);
  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_array_type, {},
      StringAttr::get(type.getContext()), nullptr, /*line=*/0,
      /*scope=*/nullptr, convertType(type.getElementType()),
      LLVM::DIFlags::Zero, type.getSizeInBits(),
      /*alignInBits=*/0, element);
}

LLVM::DIBasicTypeAttr MetadataConverter::convertTypeImpl(DIBasicType type) {
  return LLVM::DIBasicTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_base_type, type.getName(),
      type.getSizeInBits(), type.getEncoding());
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIPointerType type) {
  return LLVM::DIDerivedTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_pointer_type,
      /*name=*/nullptr, convertType(type.getElementType()),
      type.getSizeInBits(), type.getAlignInBits(), /*offsetInBits=*/0, {});
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIStructType type) {
  SmallVector<LLVM::DINodeAttr> elementTypes;

  // Convert each of the members.
  uint64_t structSize = 0;
  uint32_t structAlign = 0;
  for (DIMemberType member : type.getMembers()) {
    // Compute the offset/align of the element.
    uint64_t sizeInBits = member.getSizeInBits();
    uint32_t alignInBits = member.getAlignInBits();
    uint64_t offsetInBits =
        llvm::alignTo(structSize, std::max(1u, alignInBits));
    structSize = offsetInBits + sizeInBits;
    structAlign = std::max(structAlign, alignInBits);

    elementTypes.push_back(LLVM::DIDerivedTypeAttr::get(
        member.getContext(), llvm::dwarf::DW_TAG_member, member.getName(),
        convertType(member.getType()), sizeInBits, alignInBits, offsetInBits,
        {}));
  }

  // Pad the struct size to the largest element alignment.
  if (structAlign)
    structSize = llvm::alignTo(structSize, structAlign);

  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_structure_type, {}, type.getName(),
      /*file=*/nullptr, /*line=*/0, /*scope=*/nullptr, /*baseType=*/nullptr,
      LLVM::DIFlags::Zero, structSize, structAlign, elementTypes);
}

LLVM::DISubroutineTypeAttr
MetadataConverter::convertTypeImpl(DISubroutineType type) {
  // Grab the result type if we have one.
  SmallVector<LLVM::DITypeAttr> convertedTypes;
  if (type.getResultTypes().size() == 1)
    convertedTypes.push_back(convertType(type.getResultTypes()[0]));
  else
    convertedTypes.push_back(LLVM::DINullTypeAttr::get(type.getContext()));

  for (auto argType : type.getArgumentTypes())
    convertedTypes.push_back(convertType(argType));
  return LLVM::DISubroutineTypeAttr::get(
      type.getContext(), type.getCallingConvention(), convertedTypes);
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIUnresolvedMLIRType type) {
  // TODO: We could choose to fail here if we get an unresolved type, as opposed
  // to what's described here (i.e. just replace with an unspecified type and
  // use the string representation of the type).
  return LLVM::DIBasicTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_unspecified_type,
      mlir::debugString(type.getType()), /*sizeInBits=*/0, /*encoding=*/0);
}

LLVM::DIBasicTypeAttr
MetadataConverter::convertTypeImpl(DIUnspecifiedType type) {
  return LLVM::DIBasicTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_unspecified_type, type.getName(),
      /*sizeInBits=*/0, /*encoding=*/0);
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIVariantType type) {
  MLIRContext *context = type.getContext();
  SmallVector<LLVM::DINodeAttr> variantTypes;

  // Convert each of the members.
  for (DIMemberType member : type.getVariants()) {
    // TODO(#30619): add discriminator value to the DW_TAG_variant entry once
    // upstream is ready.
    LLVM::DITypeAttr memberType = LLVM::DIDerivedTypeAttr::get(
        context, llvm::dwarf::DW_TAG_member, member.getName(),
        convertType(member.getType()), member.getSizeInBits(),
        member.getAlignInBits(), 0, {});
    variantTypes.push_back(memberType);
  }

  // TODO(#30619): add discriminator field to the DW_TAG_variant_part entry once
  // upstream is ready.
  return LLVM::DICompositeTypeAttr::get(
      context, llvm::dwarf::DW_TAG_variant_part, {}, StringAttr::get(context),
      nullptr, 0, nullptr, nullptr, LLVM::DIFlags::Zero, type.getSizeInBits(),
      type.getAlignInBits(), variantTypes);
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIVectorType type) {
  Builder builder(type.getContext());
  auto element = LLVM::DISubrangeAttr::get(
      type.getContext(), builder.getI64IntegerAttr(type.getElementCount()),
      /*lowerBound=*/nullptr, /*upperBound=*/nullptr, /*stride=*/nullptr);
  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_array_type, {}, type.getName(),
      nullptr, /*line=*/0, /*scope=*/nullptr,
      convertType(type.getElementType()), LLVM::DIFlags::Vector,
      type.getSizeInBits(), /*alignInBits=*/0, element);
}

//===----------------------------------------------------------------------===//
// KillOp
//===----------------------------------------------------------------------===//

namespace {
struct ConvertKillOp : public OpRewritePattern<KillOp> {
  ConvertKillOp(MLIRContext *ctx, DIAttrTypeReplacer &replacer)
      : OpRewritePattern<KillOp>(ctx), replacer(replacer) {}

  LogicalResult matchAndRewrite(KillOp op,
                                PatternRewriter &rewriter) const override {
    auto undef = rewriter.create<LLVM::UndefOp>(
        op.getLoc(), LLVM::LLVMStructType::getLiteral(getContext(), {}));
    rewriter.create<LLVM::DbgValueOp>(
        replacer.replace<LocationAttr>(op.getLoc()), undef,
        replacer.replace<LLVM::DILocalVariableAttr>(op.getValueInfo()));
    rewriter.eraseOp(op);
    return success();
  }

  /// The replacer used to update attributes.
  DIAttrTypeReplacer &replacer;
};
} // namespace

//===----------------------------------------------------------------------===//
// ValueOp
//===----------------------------------------------------------------------===//

namespace {
struct ConvertValueOp : public OpRewritePattern<ValueOp> {
  ConvertValueOp(MLIRContext *ctx, DIAttrTypeReplacer &replacer)
      : OpRewritePattern<ValueOp>(ctx), replacer(replacer) {}

  LogicalResult matchAndRewrite(ValueOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<LLVM::DbgValueOp>(
        replacer.replace<LocationAttr>(op.getLoc()), op.getValue(),
        replacer.replace<LLVM::DILocalVariableAttr>(op.getValueInfo()),
        replacer.replace<LLVM::DIExpressionAttr>(op.getConversionExpr()));
    rewriter.eraseOp(op);
    return success();
  }

  /// The replacer used to update attributes.
  DIAttrTypeReplacer &replacer;
};
} // namespace

//===----------------------------------------------------------------------===//
// OpLocations
//===----------------------------------------------------------------------===//

namespace {
/// This pattern handles converting the debug information for non-debuginfo
/// operations.
struct ConvertOpLocations : public mlir::RewritePattern {
  ConvertOpLocations(MLIRContext *ctx, DIAttrTypeReplacer &replacer)
      : mlir::RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        replacer(replacer) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op, [&] {
      // Update the debug info attributes within the locations of this operation
      // to use the LLVM equivalent.
      replacer.replaceElementsIn(op);
    });
    return success();
  }

  /// The replacer used to update attributes.
  DIAttrTypeReplacer &replacer;
};
} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population
//===----------------------------------------------------------------------===//

static void populateDebugInfoToLLVMPatterns(DIAttrTypeReplacer &replacer,
                                            RewritePatternSet &patterns) {
  patterns.add<ConvertKillOp, ConvertValueOp, ConvertOpLocations>(
      patterns.getContext(), replacer);
}

//===----------------------------------------------------------------------===//
// DebugInfoToLLVMTypeConverter
//===----------------------------------------------------------------------===//

namespace {
struct DebugInfoToLLVMTypeConverter : public DebugInfo::DebugInfoTypeConverter {
  DebugInfoToLLVMTypeConverter(mlir::LLVMTypeConverter &typeConverter) {
    addUnresolvedConverter(typeConverter);

    // TODO: Cover more LLVM types here as needed.
    const llvm::DataLayout &dataLayout = typeConverter.getDataLayout();
    addConversion([&](LLVM::LLVMPointerType type) -> DebugInfo::DIType {
      // Convert the pointer element type.
      DIType diEltType =
          DebugInfo::DIUnspecifiedType::get(type.getContext(), "opaque");

      size_t size = dataLayout.getPointerSizeInBits();
      llvm::Align align =
          dataLayout.getPointerPrefAlignment(type.getAddressSpace());
      return DebugInfo::DIPointerType::get(diEltType, size,
                                           align.value() * CHAR_BIT);
    });
    addConversion([&](LLVM::LLVMStructType structType) {
      MLIRContext *ctx = structType.getContext();

      SmallVector<DebugInfo::DIMemberType> elementTypes;
      for (auto [index, type] : llvm::enumerate(structType.getBody())) {
        // Build the member using a somewhat reasonable name given we don't have
        // a better one here.
        elementTypes.push_back(DebugInfo::DIMemberType::get(
            StringAttr::get(ctx, "field_" + Twine(index)),
            convertDebugType(type)));
      }

      StringRef name = structType.isIdentified() ? structType.getName() : "";
      return DebugInfo::DIStructType::get(StringAttr::get(ctx, name),
                                          elementTypes);
    });
    addConversion([&](LLVM::LLVMArrayType arrayType) {
      return DebugInfo::DIArrayType::get(
          convertDebugType(arrayType.getElementType()),
          arrayType.getNumElements());
    });
    addConversion([&](LLVM::LLVMFixedVectorType vecType) {
      return DebugInfo::DIVectorType::get(
          convertDebugType(vecType.getElementType()), vecType.getNumElements());
    });
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

namespace M::DebugInfo {
#define GEN_PASS_DEF_DEBUGINFOTOLLVM
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h.inc"
} // namespace M::DebugInfo

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct DebugInfoToLLVMPass
    : public impl::DebugInfoToLLVMBase<DebugInfoToLLVMPass> {
  using Base::Base;

  void runOnOperation() override;
};

/// Summary of all the variables that are tracked with debug info in a function.
///
/// - A "variable" refers to a source variable (described by the
///   DILocalVariableAttr of a DbgValueOp)
/// - A "value" refers to an IR Value that is used to define the value of a
///   variable at some program location (as an operand of a DbgValueOp).
struct DebugVariableSummary {
public:
  /// Debug info for variables that have at most one non-undef definition.
  struct SingleDefVariable {
    SingleDefVariable(LLVM::DbgValueOp op, bool undefs)
        : debugValue(op), hasAdditionalUndefs(undefs) {}

    /// If all definitions of this variable is undef, this is the first debug
    /// value op for this variable. Otherwise, this debug value tracks the
    /// single non-undef definition of this variable.
    LLVM::DbgValueOp debugValue;
    /// Whether there exists additional undef values for this variable.
    /// This affects whether the definition can be easily allocated to stack.
    bool hasAdditionalUndefs;
  };

  llvm::MapVector<Value, SmallVector<SingleDefVariable>> trackers;
};
} // namespace

/// Filter out the debug values that are not needed, and summarize the rest in
/// a DebugVariableSummary.
static DebugVariableSummary
filterAndSummarizeDebugVariables(mlir::FunctionOpInterface func) {
  // Summarize debug values by variable.
  llvm::MapVector<LLVM::DILocalVariableAttr,
                  DebugVariableSummary::SingleDefVariable>
      uniqueDebugValues;
  func->walk([&](LLVM::DbgValueOp op) {
    Value value = op.getValue();
    // Don't build debug info for token values.
    if (isa<LLVM::LLVMTokenType>(value.getType())) {
      op->erase();
      return;
    }

    if (value.getDefiningOp<LLVM::UndefOp>()) {
      auto [iter, inserted] =
          uniqueDebugValues.try_emplace(op.getVarInfo(), op, false);
      if (!inserted)
        iter->second.hasAdditionalUndefs = true;
      return;
    }

    // Not undef.
    auto [iter, inserted] =
        uniqueDebugValues.try_emplace(op.getVarInfo(), op, false);
    // If this variable was seen before, either override initial undef, or
    // invalidate this variable altogether.
    if (!inserted && iter->second.debugValue) {
      if (iter->second.debugValue.getValue().getDefiningOp<LLVM::UndefOp>()) {
        iter->second.debugValue = op;
        iter->second.hasAdditionalUndefs = true;
      } else {
        iter->second.debugValue = nullptr;
      }
    }
  });

  DebugVariableSummary summary;
  // Re-categorize debug values by operand Value, skipping those that have
  // been invalidated due to having more than one non-undef definition.
  for (auto &[_, dbgValueInfo] : uniqueDebugValues) {
    if (!dbgValueInfo.debugValue)
      continue;

    summary.trackers[dbgValueInfo.debugValue.getValue()].push_back(
        dbgValueInfo);
  }

  return summary;
}

/// This function converts instances of llvm.dbg.value to llvm.dbg.declare when
/// desirable. LLVM optimizations and codegen often muck up the use of
/// llvm.dbg.value (and other debug intrinsics), which creates subpar debugging
/// experiences. Converting to llvm.dbg.declare provides a more stable debugging
/// environment, and more closely matches what a traditional frontend would
/// provide in O0 modes.
///
/// The current conversion policy considers two separate axes:
/// - The number of dbg.values for a variable (regardless of whether the value
/// is undef or not) determines whether dbg.value or dbg.declare is used.
/// - The number of non-undef dbg.values for a variable determines whether we
/// allocate to stack or not.
///
/// Or, listing out the possible combinations:
/// - =1 dbg.value: use dbg.declare, allocate to stack (i.e. var is alive for
/// its entire scope)
/// - >1 dbg.value, =1 non-undef: use dbg.value, allocate to stack (i.e. var is
/// stationary for its entire lifetime)
/// - >1 dbg.value, >1 non-undef: use dbg.value, don't allocate to stack (i.e.
/// var moves around, or exists as fragments)
///
/// TODO: As we grow support we may want to consider making this optional
/// depending on the debug mode.
static void convertDbgValueToDeclare(ModuleOp module) {
  // A lot more logic is required to make this reverse-mem2reg work when
  // multiple DbgValueOps for one variable exists. Going with the simplest
  // solution for now until we decide to retire this altogether.
  for (auto func : module.getOps<mlir::FunctionOpInterface>()) {
    DebugVariableSummary debugVariableSummary =
        filterAndSummarizeDebugVariables(func);

    for (auto &[value, trackers] : debugVariableSummary.trackers) {
      // Don't build debug information for simple constants.
      if (value.getDefiningOp<LLVM::ConstantOp>() &&
          isa<IntegerType, FloatType>(value.getType()))
        continue;

      // The converted alloca op that will hold the value if it is converted to
      // a stack allocation.
      LLVM::AllocaOp allocaOp;

      // Get the allocaOp for this value. If one has not already been created,
      // create one and save it for the next invocation.
      auto getAllocaOp = [&, value = value](LLVM::DbgValueOp op) {
        if (allocaOp)
          return allocaOp;

        // Build a new allocation to store the intermediate value.
        OpBuilder allocBuilder = OpBuilder::atBlockBegin(&func.front());
        Location erasedLoc = UnknownLoc::get(op->getContext());
        auto allocSize = allocBuilder.create<LLVM::ConstantOp>(
            erasedLoc, allocBuilder.getI32Type(), 1);

        allocaOp = allocBuilder.create<LLVM::AllocaOp>(
            erasedLoc, LLVM::LLVMPointerType::get(value.getContext()),
            value.getType(), allocSize, 0);
        return allocaOp;
      };

      // Converter for a single DbgValueOp.
      // `hasLimitedScope` controls whether the DbgValueOp is converted into a
      // DbgDeclareOp or is kept, because the scope of a DbgDeclareOp cannot be
      // limited to a subset of the parent scope.
      auto convertDbgValue = [&, value = value](LLVM::DbgValueOp op,
                                                bool hasLimitedScope) {
        if (hasLimitedScope) {
          ArrayRef<LLVM::DIExpressionElemAttr> location =
              op.getLocationExpr().getOperations();
          SmallVector<LLVM::DIExpressionElemAttr> newLocations = {
              LLVM::DIExpressionElemAttr::get(op.getContext(),
                                              llvm::dwarf::DW_OP_deref, {})};
          newLocations.append(location.begin(), location.end());
          op.setOperand(getAllocaOp(op));
          op.setLocationExprAttr(
              LLVM::DIExpressionAttr::get(op->getContext(), newLocations));
        } else {
          ArrayRef<LLVM::DIExpressionElemAttr> location =
              op.getLocationExpr().getOperations();
          if (!isa<BlockArgument>(op.getValue()) && !location.empty() &&
              location.front().getOpcode() == llvm::dwarf::DW_OP_deref) {
            // For cases where the locationExpr begins with a deref, just
            // pop off the initial deref and convert directly into a
            // DbgDeclareOp. In this case no alloca needs to be created.
            // Block args however are not compatible directly with DbgDeclare.
            auto refLocation = LLVM::DIExpressionAttr::get(
                op->getContext(), location.drop_front());
            OpBuilder(op).create<LLVM::DbgDeclareOp>(
                op.getLoc(), value, op.getVarInfo(), refLocation);
          } else {
            // For all other cases, create alloca and use dbg declare with it.
            OpBuilder(op).create<LLVM::DbgDeclareOp>(
                op.getLoc(), getAllocaOp(op), op.getVarInfo(),
                op.getLocationExpr());
          }
          op->erase();
        }
      };

      // Run converter on all single-def variables.
      for (auto &singleDefInfo : trackers) {
        LLVM::DbgValueOp op = singleDefInfo.debugValue;
        const bool hasUndefs = singleDefInfo.hasAdditionalUndefs;

        // If additional undef dbg.values exist for this variable, cannot create
        // a dbg.declare for it as they don't allow undef dbg.values to limit
        // their live ranges. Keep the dbg.value ops but reference the stack
        // allocation instead.
        // Otherwise, if it only has one dbg.value, replace the old dbg.value
        // with a dbg.declare.
        convertDbgValue(op, hasUndefs);
      }

      // If no alloca was created for this value, nothing else is needed.
      // Otherwise, all users of the original value need to go thru the alloca.
      if (!allocaOp)
        continue;

      // Update all of the old value uses to route through the alloca instead
      // of using the value directly.
      for (auto it = value.user_begin(), e = value.user_end(); it != e;) {
        // Grab the next unique user.
        auto *user = *it;
        while (++it != e && *it == user)
          continue;

        // If the user is another dbg.value, it must be for a variable that has
        // multiple non-undef definitions. Cannot convert to dbg.declare as it
        // has a limited scope.
        if (auto dbgUser = dyn_cast<LLVM::DbgValueOp>(user)) {
          convertDbgValue(dbgUser, true);
          continue;
        }

        // If the user was already converted to a declare, skip it.
        if (isa<LLVM::DbgDeclareOp>(user))
          continue;

        // Otherwise, this is a normal use, replace it with a load from an
        // alloca.
        OpBuilder loadBuilder(user);
        user->replaceUsesOfWith(
            value, loadBuilder.create<LLVM::LoadOp>(user->getLoc(),
                                                    value.getType(), allocaOp));
      }

      // Store into the alloca at the place where the value was defined.
      if (auto *valueOp = value.getDefiningOp()) {
        OpBuilder storeBuilder(valueOp->getNextNode());
        storeBuilder.create<LLVM::StoreOp>(value.getLoc(), value, allocaOp);
      } else {
        // If the value is a block argument, we need to search for an insertion
        // point after the start of the block.
        auto insertPt = value.getParentBlock()->begin();
        while (isa<LLVM::DbgValueOp, LLVM::DbgDeclareOp, LLVM::AllocaOp,
                   LLVM::ConstantOp>(*insertPt))
          ++insertPt;

        // Block arguments might not contain debuginfo scope (which can trip up
        // verifiers later), so to keep it simple, we also use erasedLoc.
        OpBuilder storeBuilder(&*insertPt);
        Location erasedLoc = UnknownLoc::get(value.getContext());
        storeBuilder.create<LLVM::StoreOp>(erasedLoc, value, allocaOp);
      }
    }
  }
}

/// Returns whether this debug value is unsupported by LLVM.
static bool isUnsupportedDebugValue(DebugInfo::ValueOp op) {
  /// LLVM does not yet support emitting DW_OP_LLVM_implicit_pointer to asm. If
  /// it is not yet optimized out by the time we emit to LLVM, it has to be
  /// removed.
  auto walkResult = op.getConversionExprAttr().walk(
      [](DebugInfo::DIRefOfExprAttr refof) { return WalkResult::interrupt(); });
  return walkResult.wasInterrupted();
}

namespace {
/// A linearized canonical representation of an inline call stack (as opposed to
/// a binary-tree-based representation used by CallSiteLoc) that allows easy
/// ancestor-child comparison.
class CallStack {
public:
  CallStack() = default;
  CallStack(Location overallLoc) {
    walkLocation(overallLoc, LocWalkPolicy::CallerPriority, [&](Location loc) {
      if (auto fusedLoc = dyn_cast<mlir::FusedLocWith<DIScopeAttr>>(loc)) {
        if (fusedLoc.getLocations().size() != 1)
          return WalkResult::advance();

        if (auto fileLineCol =
                dyn_cast<FileLineColLoc>(fusedLoc.getLocations().front()))
          frames.emplace_back(fusedLoc.getMetadata(), fileLineCol.getLine());
      }
      return WalkResult::advance();
    });
  }

  /// The call stack ordered from caller to callee.
  /// Each frame encodes the scope of the location & the line number.
  using Frame = std::pair<DIScopeAttr, unsigned>;
  SmallVector<Frame> frames;
};

/// Maps each frame of a CallStack to some user-defined data `T`.
template <typename T>
class CallStackWith {
public:
  bool empty() const { return dataFrames.empty(); }

  /// Reference to the data value mapped to the last (innermost) frame.
  T &backData() { return dataFrames.back().second; }

  /// Update the internal call stack to represent `newStack` instead.
  /// Any stack frame that will no longer exist is considered invalidated, and
  /// will be returned in the order of their positions in the call stack.
  /// Each newly added stack frame will come with a default-constructed `T`.
  ///
  /// For example, calling with
  ///   dataFrames = [(L0, T0), (L1, T1), (L2, T2), (L3, T3)]
  ///   newStack   = [L0, L1, L4, L5]
  /// results in
  ///   dataFrames = [(L0, T0), (L1, T1), (L4, T()), (L5, T())]
  /// and returns [T2, T3].
  SmallVector<T> updateTo(const CallStack &newStack) {
    // Walk until `newStack.frames` & `dataFrames` diverge.
    auto thisIter = dataFrames.begin();
    auto newIter = newStack.frames.begin();
    for (; thisIter != dataFrames.end() && newIter != newStack.frames.end();
         ++thisIter, ++newIter)
      if (thisIter->first != *newIter)
        break;

    SmallVector<T> invalidated;
    // Diverged in the middle of `dataFrames`. Invalidate everything afterwards.
    if (thisIter != dataFrames.end()) {
      std::transform(thisIter, dataFrames.end(),
                     std::back_inserter(invalidated),
                     [](auto it) { return it.second; });
      dataFrames.truncate(dataFrames.size() - invalidated.size());
    }

    // Append anything at or after `newIter` to `dataFrames`.
    for (; newIter != newStack.frames.end(); ++newIter)
      dataFrames.emplace_back(*newIter, T());

    return invalidated;
  }

private:
  /// The call stack ordered from caller to callee.
  /// Each frame encodes both a location and a custom data `T`.
  SmallVector<std::pair<CallStack::Frame, T>> dataFrames;
};
} // namespace

/// Pre-Massaging to bridge the semantics between DebugInfo's DebugValues &
/// LLVM's counterpart. Also removes unsupported cases.
///
/// - Sink kill Debug Value ops so that they are the last instructions from
/// their source line. This way variables are guaranteed to be killed only at
/// the end of the line.
/// - Remove Debug Value ops that contain unsupported components by LLVM.
static void preAdaptDebugValuesToLLVM(Operation *op) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      // The kill Debug Value Ops corresponding to the current line at each
      // inlined scope.
      CallStackWith<SmallVector<Operation *>> pendingKillsByLoc;
      for (Operation &op : llvm::make_early_inc_range(block.getOperations())) {
        // Ops without a location follow the location of the previous op.
        const CallStack callStack(op.getLoc());
        if (!callStack.frames.empty()) {
          SmallVector<SmallVector<Operation *>> invalidated =
              pendingKillsByLoc.updateTo(callStack);
          // This is the start of a new line. Move all pending kill debug values
          // before this op.
          for (SmallVector<Operation *> &kills : invalidated)
            for (Operation *kill : kills)
              kill->moveBefore(&op);
        }

        if (ValueOp dbgValue = dyn_cast<ValueOp>(op)) {
          if (isUnsupportedDebugValue(dbgValue)) {
            dbgValue->erase();
            continue;
          }
        } else if (!pendingKillsByLoc.empty() && isa<KillOp>(op)) {
          pendingKillsByLoc.backData().push_back(&op);
        }

        preAdaptDebugValuesToLLVM(&op);
      }

      // Any still pending kills can be moved before the last op of the block.
      if (!pendingKillsByLoc.empty()) {
        Operation *lastOp = &block.back();
        SmallVector<SmallVector<Operation *>> invalidated =
            pendingKillsByLoc.updateTo({});
        for (SmallVector<Operation *> &kills : invalidated)
          for (Operation *kill : kills)
            kill->moveBefore(lastOp);
      }
    }
  }
}

void DebugInfoToLLVMPass::runOnOperation() {
  // Configure dialect conversion.
  mlir::ConversionTarget target(getContext());
  target.addIllegalDialect<DebugInfoDialect>();
  target.addLegalOp<LLVM::DbgValueOp>();

  // Unknown operations are legal if they don't have debug info attached.
  target.markUnknownOpDynamicallyLegal([](Operation *op) -> bool {
    auto hasDIAttr = [](Location loc) -> bool {
      return !!loc->findInstanceOf<mlir::FusedLocWith<DebugInfo::DIAttr>>();
    };
    if (hasDIAttr(op->getLoc()))
      return false;
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (BlockArgument arg : block.getArguments())
          if (hasDIAttr(arg.getLoc()))
            return false;
    return true;
  });

  // Set LLVM lowering options.
  mlir::LowerToLLVMOptions options(&getContext());
  mlir::LLVMTypeConverter typeConverter(&getContext(), options);

  // Configure the metadata converter.
  DebugInfoToLLVMTypeConverter debugTypeConverter(typeConverter);
  MetadataConverter metadataConverter(debugTypeConverter);
  DIAttrTypeReplacer replacer;
  replacer.addReplacement(
      [&](DIAttr attr) { return metadataConverter.convertAttr(attr); });

  // Populate patterns and run the conversion.
  mlir::RewritePatternSet patterns(&getContext());
  populateDebugInfoToLLVMPatterns(replacer, patterns);

  // Massage DebugValues before conversion.
  preAdaptDebugValuesToLLVM(getOperation());

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();

  // Clean up the generated LLVM.
  convertDbgValueToDeclare(getOperation());
}
