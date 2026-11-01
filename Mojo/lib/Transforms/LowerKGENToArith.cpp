//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions:
// https://llvm.org/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//
//
// Lower scalar KGEN/POP arithmetic to Arith/Index. SIMD forms
// (`!pop.simd<N, T>`, N > 1) are left for a target lowering.
//
//===----------------------------------------------------------------------===//

#include "Mojo/KGENDialect/KGENAttrs.h"
#include "Mojo/KGENDialect/KGENOps.h"
#include "Mojo/KGENDialect/KGENTypes.h"
#include "Mojo/POPDialect/POPDialect.h"
#include "Mojo/POPDialect/POPEnums.h"
#include "Mojo/POPDialect/POPOps.h"
#include "Mojo/POPDialect/POPTypes.h"
#include "Mojo/ToolCommon/KGENPasses.h"
#include "Support/Compiler/DirectConversion.h"
#include "Support/Compiler/MLIRDType.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Index/IR/IndexDialect.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace M;
using namespace KGEN;

namespace M::KGEN {
#define GEN_PASS_DEF_LOWERKGENTOARITH
#include "Mojo/KGENPasses.h.inc"
} // namespace M::KGEN

//===----------------------------------------------------------------------===//
// Type converter
//===----------------------------------------------------------------------===//

std::optional<Type> KGEN::dtypeToBuiltin(MLIRContext *ctx, KGENDType dtype) {
  if (dtype.isBool())
    return IntegerType::get(ctx, 1);
  if (dtype.isInt())
    return IntegerType::get(ctx, dtype.getIntegerWidthInBits());
  if (dtype.isIndex() || dtype.isUIndex())
    return IndexType::get(ctx);
  // Width alone cannot tell `bf16` from `f16`, and picking the wrong one
  // silently changes the arithmetic.
  if (dtype.isFloat()) {
    if (FloatType fp = getEquivalentFloatType(ctx, dtype))
      return fp;
  }
  return {};
}

POPScalarTypeConverter::POPScalarTypeConverter(MLIRContext *ctx) {
  addConversion([](Type t) { return t; });
  addConversion([ctx](SIMDType simd) -> std::optional<Type> {
    std::optional<KGENDType> dtype = simd.getResolvedDType();
    std::optional<uint64_t> size = simd.getResolvedSize();
    if (!dtype || !size || *size != 1)
      return {};
    return dtypeToBuiltin(ctx, *dtype);
  });

  auto cast = [](OpBuilder &b, Type type, ValueRange inputs,
                 Location loc) -> Value {
    return UnrealizedConversionCastOp::create(b, loc, TypeRange{type}, inputs)
        .getResult(0);
  };
  addSourceMaterialization(cast);
  addTargetMaterialization(cast);
}

namespace {

static bool isFloatType(Type t) { return isa<FloatType>(t); }

/// Defined below. Every pattern here gates on it: the direct conversion
/// driver has no `ConversionTarget`, so it applies a pattern to every op that
/// pattern is registered for, and declining the SIMD forms is the pattern's
/// own job. They belong to the target lowering.
static bool isScalarForm(Operation *op);

//===----------------------------------------------------------------------===//
// kgen.param.constant → arith.constant (scalar only)
//===----------------------------------------------------------------------===//

struct ConvertKGENParamConstant
    : public DirectOpConversionPattern<ParamConstantOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(ParamConstantOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType || isa<MemRefType>(resultType))
      return mlir::failure();

    Attribute val = op.getValue();
    if (auto intAttr = dyn_cast<IntegerAttr>(val)) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, resultType, IntegerAttr::get(resultType, intAttr.getValue()));
      return mlir::success();
    }
    if (auto fpAttr = dyn_cast<FloatAttr>(val)) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, resultType, FloatAttr::get(resultType, fpAttr.getValue()));
      return mlir::success();
    }
    // `!pop.scalar<T>`: the SIMDAttr stores N copies of one value.
    if (auto simdAttr = dyn_cast<SIMDAttr>(val)) {
      ArrayRef<DTypeValue> values = simdAttr.getValues();
      if (values.empty())
        return mlir::failure();
      const DTypeValue &dv = values.front();
      KGENDType dtype = dv.getDType();
      if (dtype.isFloat()) {
        APFloat fp = dv.getFloatVal();
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            op, resultType, FloatAttr::get(resultType, fp));
        return mlir::success();
      }
      if (dtype.isBool()) {
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            op, resultType,
            IntegerAttr::get(resultType, dv.getBoolVal() ? 1 : 0));
        return mlir::success();
      }
      if (dtype.isInt() || dtype.isIndex() || dtype.isUIndex()) {
        APInt iv = dv.getIntVal();
        rewriter.replaceOpWithNewOp<arith::ConstantOp>(
            op, resultType, IntegerAttr::get(resultType, iv));
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

//===----------------------------------------------------------------------===//
// Scalar pop.{add,sub,mul,floordiv,select,max,min} → arith / index
//===----------------------------------------------------------------------===//

struct ConvertPOPAdd : public DirectOpConversionPattern<POP::AddOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::AddOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType || isa<MemRefType>(resultType))
      return mlir::failure();
    if (isFloatType(resultType))
      rewriter.replaceOpWithNewOp<arith::AddFOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
    return mlir::success();
  }
};

// Needed for the `sub(arg, step)` left in the body of a raised `hlcf.for` by
// mojo's downcounting `_ZeroStartingRange` iterator (the
// `select(eq(arg, 0), arg, arg - 1)` no-progress-on-exit shape); the value is
// dead once the loop is raised, but it must still legalize out of `pop`.
struct ConvertPOPSub : public DirectOpConversionPattern<POP::SubOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::SubOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType || isa<MemRefType>(resultType))
      return mlir::failure();
    if (isFloatType(resultType))
      rewriter.replaceOpWithNewOp<arith::SubFOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<arith::SubIOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
    return mlir::success();
  }
};

// Runtime tile loops scale the loop index by the tile extent, leaving integer
// arithmetic on `index` in the body.
struct ConvertPOPMul : public DirectOpConversionPattern<POP::MulOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::MulOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType || isa<MemRefType>(resultType))
      return mlir::failure();
    if (isFloatType(resultType))
      rewriter.replaceOpWithNewOp<arith::MulFOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
    else
      rewriter.replaceOpWithNewOp<arith::MulIOp>(op, adaptor.getLhs(),
                                                 adaptor.getRhs());
    return mlir::success();
  }
};

// `pop.floordiv` rounds toward negative infinity; only the signed integer
// form maps onto one arith op. Float and unsigned operands stay illegal rather
// than lowering to the wrong division.
struct ConvertPOPFloorDiv : public DirectOpConversionPattern<POP::FloorDivOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::FloorDivOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType || isa<MemRefType>(resultType) || isFloatType(resultType))
      return mlir::failure();
    std::optional<KGENDType> dtype = op.getType().getResolvedDType();
    if (!dtype || dtype->isUInt())
      return mlir::failure();
    rewriter.replaceOpWithNewOp<arith::FloorDivSIOp>(op, adaptor.getLhs(),
                                                     adaptor.getRhs());
    return mlir::success();
  }
};

// Needed for `pop.select` on `index` left in the body of a raised `hlcf.for`
// by mojo's upcounting iterator (the `select(eq(arg, ub), arg, arg+step)`
// no-progress-on-exit shape).
struct ConvertPOPSelect : public DirectOpConversionPattern<POP::SelectOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::SelectOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, adaptor.getCondition(),
                                                 adaptor.getTrueValue(),
                                                 adaptor.getFalseValue());
    return mlir::success();
  }
};

// Scalar `pop.simd.select` → `arith.select`; the SIMD form is the target's.
struct ConvertPOPSIMDSelect
    : public DirectOpConversionPattern<POP::SIMDSelectOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::SIMDSelectOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    if (isa<MemRefType>(adaptor.getTrueValue().getType()))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, adaptor.getCondition(),
                                                 adaptor.getTrueValue(),
                                                 adaptor.getFalseValue());
    return mlir::success();
  }
};

template <typename OpTy, typename ArithFOp, typename ArithSIOp,
          typename ArithUIOp, typename IndexSOp, typename IndexUOp>
struct ConvertPOPScalarMinMax : public DirectOpConversionPattern<OpTy> {
  using DirectOpConversionPattern<OpTy>::DirectOpConversionPattern;
  using OpAdaptor = typename OpTy::Adaptor;

  LogicalResult matchAndRewrite(OpTy op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type resultType = this->getTypeConverter()->convertType(op.getType());
    if (!resultType || isa<MemRefType>(resultType))
      return mlir::failure();
    if (isFloatType(resultType)) {
      rewriter.replaceOpWithNewOp<ArithFOp>(op, adaptor.getLhs(),
                                            adaptor.getRhs());
      return mlir::success();
    }
    // The builtin type is signless, so the op must carry the signedness the
    // dtype holds, as the LLVM lowering does.
    std::optional<KGENDType> dtype = op.getType().getResolvedDType();
    if (!dtype)
      return mlir::failure();
    bool isSigned = dtype->isSInt();
    // Index stays in `index` dialect (BIR/loop-bound matcher want index.maxs).
    if (isa<IndexType>(adaptor.getLhs().getType())) {
      if (isSigned)
        rewriter.replaceOpWithNewOp<IndexSOp>(op, adaptor.getLhs(),
                                              adaptor.getRhs());
      else
        rewriter.replaceOpWithNewOp<IndexUOp>(op, adaptor.getLhs(),
                                              adaptor.getRhs());
    } else if (isSigned) {
      rewriter.replaceOpWithNewOp<ArithSIOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
    } else {
      rewriter.replaceOpWithNewOp<ArithUIOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
    }
    return mlir::success();
  }
};
using ConvertPOPMax =
    ConvertPOPScalarMinMax<POP::MaxOp, arith::MaxNumFOp, arith::MaxSIOp,
                           arith::MaxUIOp, mlir::index::MaxSOp,
                           mlir::index::MaxUOp>;
using ConvertPOPMin =
    ConvertPOPScalarMinMax<POP::MinOp, arith::MinNumFOp, arith::MinSIOp,
                           arith::MinUIOp, mlir::index::MinSOp,
                           mlir::index::MinUOp>;

//===----------------------------------------------------------------------===//
// pop.cmp → arith.cmpi / arith.cmpf (scalar only)
//===----------------------------------------------------------------------===//

static arith::CmpIPredicate toArithIntPred(CmpPredicate pred, bool isSigned) {
  switch (pred) {
  case CmpPredicate::EQ:
    return arith::CmpIPredicate::eq;
  case CmpPredicate::NE:
    return arith::CmpIPredicate::ne;
  case CmpPredicate::LT:
    return isSigned ? arith::CmpIPredicate::slt : arith::CmpIPredicate::ult;
  case CmpPredicate::GT:
    return isSigned ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::ugt;
  case CmpPredicate::LE:
    return isSigned ? arith::CmpIPredicate::sle : arith::CmpIPredicate::ule;
  case CmpPredicate::GE:
    return isSigned ? arith::CmpIPredicate::sge : arith::CmpIPredicate::uge;
  }
  llvm_unreachable("unhandled CmpPredicate");
}

static arith::CmpFPredicate toArithFloatPred(CmpPredicate pred) {
  switch (pred) {
  case CmpPredicate::EQ:
    return arith::CmpFPredicate::OEQ;
  case CmpPredicate::NE:
    return arith::CmpFPredicate::ONE;
  case CmpPredicate::LT:
    return arith::CmpFPredicate::OLT;
  case CmpPredicate::GT:
    return arith::CmpFPredicate::OGT;
  case CmpPredicate::LE:
    return arith::CmpFPredicate::OLE;
  case CmpPredicate::GE:
    return arith::CmpFPredicate::OGE;
  }
  llvm_unreachable("unhandled CmpPredicate");
}

struct ConvertPOPCmp : public DirectOpConversionPattern<POP::CmpOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::CmpOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type operandType = getTypeConverter()->convertType(op.getLhs().getType());
    if (!operandType || isa<MemRefType>(operandType))
      return mlir::failure();

    CmpPredicate pred = op.getPred();
    if (isFloatType(operandType)) {
      rewriter.replaceOpWithNewOp<arith::CmpFOp>(
          op, toArithFloatPred(pred), adaptor.getLhs(), adaptor.getRhs());
      return mlir::success();
    }
    // The builtin type is signless, so the predicate must carry the
    // signedness the dtype holds, as the LLVM lowering does.
    std::optional<KGENDType> dtype = op.getLhs().getType().getResolvedDType();
    if (!dtype)
      return mlir::failure();
    rewriter.replaceOpWithNewOp<arith::CmpIOp>(
        op, toArithIntPred(pred, dtype->isSInt()), adaptor.getLhs(),
        adaptor.getRhs());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// pop.cast_{from,to}_builtin: identity once `!kgen.scalar<T>` is converted
// to its builtin type. Mojo's `bool.mojo` emits these around `index.cmp`
// results.
//===----------------------------------------------------------------------===//

struct ConvertPOPCastFromBuiltin
    : public DirectOpConversionPattern<POP::CastFromBuiltinOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::CastFromBuiltinOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type convertedResultTy = getTypeConverter()->convertType(op.getType());
    if (!convertedResultTy || convertedResultTy != adaptor.getInput().getType())
      return mlir::failure();
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

struct ConvertPOPCastToBuiltin
    : public DirectOpConversionPattern<POP::CastToBuiltinOp> {
  using DirectOpConversionPattern::DirectOpConversionPattern;

  LogicalResult matchAndRewrite(POP::CastToBuiltinOp op, OpAdaptor adaptor,
                                DirectRewriter &rewriter) const override {
    if (!isScalarForm(op))
      return mlir::failure();
    Type convertedResultTy = getTypeConverter()->convertType(op.getType());
    if (!convertedResultTy || convertedResultTy != adaptor.getInput().getType())
      return mlir::failure();
    rewriter.replaceOp(op, adaptor.getInput());
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

/// Whether every SIMD-typed operand and result of `op` is a scalar
/// (`!pop.scalar<T>`). SIMD forms (N > 1) belong to a target lowering.
static bool isScalarForm(Operation *op) {
  auto isScalar = [](Type t) {
    auto simd = dyn_cast<SIMDType>(t);
    if (!simd)
      return true;
    std::optional<uint64_t> size = simd.getResolvedSize();
    return size && *size == 1;
  };
  return llvm::all_of(op->getOperandTypes(), isScalar) &&
         llvm::all_of(op->getResultTypes(), isScalar);
}

struct LowerKGENToArithPass
    : public M::KGEN::impl::LowerKGENToArithBase<LowerKGENToArithPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();

    POPScalarTypeConverter typeConverter(ctx);

    RewritePatternSet patterns(ctx);
    populateKGENToArithConversionPatterns(typeConverter, patterns);

    // An aggregate constant (a struct, say) is scalar by that test but has
    // no `arith.constant` form, so claiming it would strand it.
    auto mustConvert = [](Operation *op) {
      if (!isScalarForm(op))
        return false;
      if (auto constOp = dyn_cast<ParamConstantOp>(op))
        return isa<IntegerAttr, FloatAttr, SIMDAttr>(constOp.getValue());
      return isa<POP::AddOp, POP::SubOp, POP::MulOp, POP::FloorDivOp,
                 POP::CmpOp, POP::MaxOp, POP::MinOp, POP::SelectOp,
                 POP::SIMDSelectOp, POP::CastFromBuiltinOp,
                 POP::CastToBuiltinOp>(op);
    };

    if (failed(M::applyDirectConversion(getOperation(), &typeConverter,
                                        std::move(patterns),
                                        /*skipOp=*/nullptr, mustConvert)))
      signalPassFailure();
  }
};

} // namespace

void M::KGEN::populateKGENToArithConversionPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns) {
  patterns
      .add<ConvertKGENParamConstant, ConvertPOPAdd, ConvertPOPSub,
           ConvertPOPMul, ConvertPOPFloorDiv, ConvertPOPSelect,
           ConvertPOPSIMDSelect, ConvertPOPMax, ConvertPOPMin, ConvertPOPCmp,
           ConvertPOPCastFromBuiltin, ConvertPOPCastToBuiltin>(
          converter, patterns.getContext());
}
