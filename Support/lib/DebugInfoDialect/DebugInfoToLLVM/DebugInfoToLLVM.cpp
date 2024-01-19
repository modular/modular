//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/DebugInfoDialect/DebugInfoToLLVM/DebugInfoToLLVM.h"
#include "Support/DebugInfoDialect/DebugInfoToLLVM/DIExpressionSimplifier.h"
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
      static_cast<LLVM::DIEmissionKind>(attr.getEmissionKind()));
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

  uint64_t prefixSize = 0;
  for (LLVM::DINodeAttr member :
       llvmStructType.getElements().take_front(attr.getFieldIndex())) {
    auto memberType = cast<LLVM::DIDerivedTypeAttr>(member);
    uint64_t sizeInBits = memberType.getSizeInBits();
    uint32_t alignInBits = memberType.getAlignInBits();
    prefixSize =
        llvm::alignTo(prefixSize, std::max(1u, alignInBits)) + sizeInBits;
  }

  auto targetMember = cast<LLVM::DIDerivedTypeAttr>(
      llvmStructType.getElements()[attr.getFieldIndex()]);
  uint64_t fieldSize = targetMember.getSizeInBits();
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
                DIVectorType>([&](auto type) { return convertTypeImpl(type); });
  return convertedTypes[type] = result;
}

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIArrayType type) {
  Builder builder(type.getContext());
  auto element = LLVM::DISubrangeAttr::get(
      type.getContext(), builder.getI64IntegerAttr(type.getElementCount()),
      /*lowerBound=*/nullptr, /*upperBound=*/nullptr, /*stride=*/nullptr);
  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_array_type,
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
      type.getSizeInBits(), type.getAlignInBits(), /*offsetInBits=*/0);
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
        convertType(member.getType()), sizeInBits, alignInBits, offsetInBits));
  }

  // Pad the struct size to the largest element alignment.
  if (structAlign)
    structSize = llvm::alignTo(structSize, structAlign);

  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_structure_type, type.getName(),
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

LLVM::DITypeAttr MetadataConverter::convertTypeImpl(DIVectorType type) {
  Builder builder(type.getContext());
  auto element = LLVM::DISubrangeAttr::get(
      type.getContext(), builder.getI64IntegerAttr(type.getElementCount()),
      /*lowerBound=*/nullptr, /*upperBound=*/nullptr, /*stride=*/nullptr);
  return LLVM::DICompositeTypeAttr::get(
      type.getContext(), llvm::dwarf::DW_TAG_array_type, type.getName(),
      nullptr, /*line=*/0, /*scope=*/nullptr,
      convertType(type.getElementType()), LLVM::DIFlags::Vector,
      type.getSizeInBits(), /*alignInBits=*/0, element);
}

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
        replacer.replace<mlir::LocationAttr>(op.getLoc()), op.getValue(),
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
    rewriter.updateRootInPlace(op, [&] {
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
  patterns.add<ConvertValueOp, ConvertOpLocations>(patterns.getContext(),
                                                   replacer);
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
} // namespace

/// This function converts instances of llvm.dbg.value to llvm.dbg.addr when
/// desirable. LLVM optimizations and codegen often muck up the use of
/// llvm.dbg.value (and other debug intrinsics), which creates subpar debugging
/// experiences. Converting to llvm.dbg.addr provides a more stable debugging
/// environment, and more closely matches what a traditional frontend would
/// provide in O0 modes.
///
/// TODO: As we grow support we may want to consider making this optional
/// depending on the debug mode.
static void convertDbgValueToAddr(ModuleOp module) {
  // A lot more logic is required to make this reverse-mem2reg work when
  // multiple DbgValueOps for one variable exists. Going with the simplest
  // solution for now until we decide to retire this altogether.
  //
  // We perform variable-uniqueness check per-function to reduce the memory
  // footprint of uniqueness tracking.
  for (auto func : module.getOps<mlir::FunctionOpInterface>()) {
    // A map from each LocalVariable to a unique DbgValueOp.
    // If a LocalVariable maps to a nullptr DbgValueOp, that means this variable
    // had more than one DbgValueOp for it, and cannot be trivially converted
    // into a declare.
    llvm::MapVector<LLVM::DILocalVariableAttr, LLVM::DbgValueOp> dbgValueCount;
    func->walk([&](LLVM::DbgValueOp op) {
      auto [iter, inserted] =
          dbgValueCount.try_emplace(op.getVarInfoAttr(), op);
      // If this variable was seen before, invalidate the current op.
      if (!inserted)
        iter->second = nullptr;
    });

    for (auto [varInfo, op] : dbgValueCount) {
      // Skip conversion for variables with multiple DbgValueOps.
      if (!op)
        continue;

      Value value = op.getValue();

      // Don't build debug information for simple constants.
      if (value.getDefiningOp<LLVM::ConstantOp>() &&
          isa<IntegerType, FloatType>(value.getType()))
        continue;

      // Don't build debug info for token values.
      if (isa<LLVM::LLVMTokenType>(value.getType())) {
        op->erase();
        continue;
      }

      // If the locationExpr begins with a deref, just pop off the deref and
      // convert directly into a DbgDeclareOp.
      ArrayRef<LLVM::DIExpressionElemAttr> location =
          op.getLocationExpr().getOperations();
      if (!location.empty() &&
          location.front().getOpcode() == llvm::dwarf::DW_OP_deref) {
        auto refLocation = LLVM::DIExpressionAttr::get(op->getContext(),
                                                       location.drop_front());
        OpBuilder(op).create<LLVM::DbgDeclareOp>(op.getLoc(), value,
                                                 op.getVarInfo(), refLocation);
        op->erase();
        continue;
      }

      // Build a new allocation to store the intermediate value.
      OpBuilder allocBuilder = OpBuilder::atBlockBegin(&func.front());
      Location erasedLoc = UnknownLoc::get(op->getContext());
      auto allocSize = allocBuilder.create<LLVM::ConstantOp>(
          erasedLoc, allocBuilder.getI32Type(), 1);

      auto allocaOp = allocBuilder.create<LLVM::AllocaOp>(
          erasedLoc, LLVM::LLVMPointerType::get(value.getContext()),
          value.getType(), allocSize, 0);

      // Replace the old dbg.value with a dbg.declare.
      OpBuilder(op).create<LLVM::DbgDeclareOp>(
          op.getLoc(), allocaOp, op.getVarInfo(), op.getLocationExpr());
      op->erase();

      // Update all of the old value uses to route through the alloca instead of
      // using the value directly.
      while (!value.use_empty()) {
        auto *user = *value.user_begin();
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
        storeBuilder.create<LLVM::StoreOp>(erasedLoc, value, allocaOp);
      }
    }
  }
}

/// LLVM does not yet support emitting DW_OP_LLVM_implicit_pointer to asm. If it
/// is not yet optimized out by the time we emit to LLVM, it has to be removed.
static void removeImplicitPointerDIExpr(Operation *op) {
  op->walk([](DebugInfo::ValueOp op) {
    auto walkResult =
        op.getConversionExprAttr().walk([](DebugInfo::DIRefOfExprAttr refof) {
          return WalkResult::interrupt();
        });
    if (walkResult.wasInterrupted())
      op->erase();
  });
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

  // Remove unsupported cases.
  removeImplicitPointerDIExpr(getOperation());

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();

  // Clean up the generated LLVM.
  convertDbgValueToAddr(getOperation());
  simplifyLLVMDIExpressionRecursively(getOperation());
}
