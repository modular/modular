//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPOps.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/PatternMatch.h"

using namespace M;
using namespace KGEN;
using namespace POP;

//===----------------------------------------------------------------------===//
// POPDialect
//===----------------------------------------------------------------------===//

Operation *POPDialect::materializeConstant(OpBuilder &b, Attribute value,
                                           Type type, Location loc) {
  return b.create<ParamConstantOp>(loc, type, cast<TypedAttr>(value));
}

//===----------------------------------------------------------------------===//
// SIMD Folder Helpers
//===----------------------------------------------------------------------===//

namespace detail {
/// Detector for whether `T` posseses a `has_value` method.
template <typename T>
using IsOptionalType = decltype(std::declval<T>().has_value());

/// `std::optional<T>` -> `T`
template <typename T>
struct remove_optional {
  using type = T;
};
template <typename T>
struct remove_optional<std::optional<T>> {
  using type = T;
};

/// Perform folding of an n-ary SIMD vector operation of a given dtype by
/// applying the operation `op` to each vector element. `getValue` transforms a
/// `DTypeValue` to the value used to represent the dtype: `APSInt` for
/// integers, `APFloat` for floats, and `bool` for bools.
template <size_t... I, typename OpFn, typename GetValueFn>
static SIMDAttr foldSIMDOpImpl(std::index_sequence<I...>,
                               ArrayRef<Attribute> operands, OpFn op,
                               KGENDType dtype, GetValueFn getValue) {
  SmallVector<DTypeValue> results;
  auto firstArg = cast<SIMDAttr>(operands.front());
  for (unsigned i = 0, e = firstArg.getValues().size(); i != e; ++i) {
    auto result =
        std::apply(op, std::make_tuple(getValue(
                           cast<SIMDAttr>(operands[I]).getValues()[i])...));
    // Allow folders to return failure. This indicates undefined behaviour,
    // which we do not fold.
    if constexpr (llvm::is_detected<IsOptionalType, decltype(result)>::value) {
      if (!result)
        return {};
      results.emplace_back(*result, dtype);
    } else {
      results.emplace_back(result, dtype);
    }
  }
  auto type = cast<SIMDType>(cast<TypedAttr>(operands.front()).getType());
  return SIMDAttr::get(results, SIMDType::get(type.getContext(),
                                              *type.getResolvedSize(), dtype));
}

/// Return true if the function type `OpFn` is a function whose first argument
/// type is `TestType`, which can be an integer, float, index, or bool type.
template <typename OpFn, typename TestType>
static constexpr bool testOpFnType() {
  return std::is_same_v<
      TestType,
      std::decay_t<typename llvm::function_traits<OpFn>::template arg_t<0>>>;
}

/// Base case for getting an op function of a given type. This one returns none.
template <typename TestType>
static constexpr auto getOpFnOfType() {
  return std::nullopt;
}

/// This function selects a function which can be applied to `TestType` from
/// `OpFns`. If the head op function is of the given type, return it. Otherwise,
/// check the rest of the functions.
template <typename TestType, typename OpFn, typename... OpFns>
static constexpr auto getOpFnOfType(OpFn op, OpFns &&...fns) {
  if constexpr (testOpFnType<OpFn, TestType>())
    return op;
  else
    return getOpFnOfType<TestType>(std::forward<OpFns>(fns)...);
}

/// Try to fold the operation using one of the provided fold functions for a
/// given dtype. If a fold function for that dtype is not provided, if such a
/// dtype is encountered by the folder, it will assert; a folder must be
/// provided for each dtype for which the operation is valid.
template <typename TestType, typename GetValueFn, typename... OpFns>
static SIMDAttr foldSIMDOpDType(GetValueFn getValue,
                                ArrayRef<Attribute> operands, KGENDType dtype,
                                OpFns &&...ops) {
  auto op = getOpFnOfType<TestType>(std::forward<OpFns>(ops)...);
  if constexpr (std::is_same_v<decltype(op), std::nullopt_t>) {
    llvm_unreachable("unhandled dtype");
  } else {
    return foldSIMDOpImpl(std::make_index_sequence<
                              llvm::function_traits<decltype(op)>::num_args>(),
                          operands, op, dtype, getValue);
  }
}

/// This enum indicates how index folding should be done.
enum IndexFold {
  kNoIndex,     // no index folding allowed
  kIndexResult, // index operation creates an index
  kOtherResult  // index operation does not create an index
};

/// Try to fold an operation with index dtype using one of the provided fold
/// functions. Index folds are performed using the same function as integer
/// dtype folds. An index fold is performed by computing the result in 64-bit
/// and 32-bit arithmetic. If the results match, then the operation can fold.
/// See the MLIR `index` dialect for more details.
template <IndexFold foldType, typename... OpFns>
static SIMDAttr foldSIMDOpIndex(ArrayRef<Attribute> operands, KGENDType dtype,
                                OpFns &&...ops) {
  auto op = getOpFnOfType<APSInt>(std::forward<OpFns>(ops)...);
  if constexpr (std::is_same_v<decltype(op), std::nullopt_t>) {
    llvm_unreachable("unhandled dtype");
  } else {
    // Define the index fold function using the integer fold function. Detect a
    // bool function. Return a bool instead of an index value in that case.
    using OpResultT = typename llvm::function_traits<decltype(op)>::result_t;
    // Check if the fold function is failable. If the fold function can fail,
    // make sure to propagate the failure in both 64-bit and 32-bit arithmetic.
    constexpr bool isOptional =
        llvm::is_detected<IsOptionalType, OpResultT>::value;
    using ResultT = typename remove_optional<OpResultT>::type;
    constexpr bool isIndexResult = foldType == kIndexResult;
    auto indexOp = [&op](auto... args)
        -> std::optional<std::conditional_t<isIndexResult, int64_t, ResultT>> {
      OpResultT result64 = op(args...);
      if constexpr (isOptional)
        if (!result64.has_value())
          return {};
      OpResultT result32 = op(args.trunc(32)...);
      if constexpr (isOptional)
        if (!result32.has_value())
          return {};
      auto unwrap = [](OpResultT value) {
        if constexpr (isOptional)
          return *value;
        else
          return value;
      };

      // Compare the results. Return the index value if the fold results match.
      // If the result type isn't an index represented as an APSInt, just
      // compare the results directly.
      if constexpr (isIndexResult) {
        if (unwrap(result64).trunc(32) == unwrap(result32))
          return unwrap(result64).getSExtValue();
        return {};
      } else {
        if (unwrap(result64) == unwrap(result32))
          return unwrap(result64);
        return {};
      }
    };
    return foldSIMDOpImpl(std::make_index_sequence<
                              llvm::function_traits<decltype(op)>::num_args>(),
                          operands, indexOp, dtype, [](DTypeValue val) {
                            return APSInt(APInt(64, val.getIndexVal()),
                                          /*isUnsigned=*/false);
                          });
  }
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <IndexFold indexFoldType, typename... OpFns>
static SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands, KGENDType inputDType,
                           KGENDType resultDType, OpFns &&...ops) {
  if (inputDType.isInt())
    return ::detail::foldSIMDOpDType<APSInt>(
        [](const DTypeValue &val) { return val.getIntVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  // FIXME: Should we even do floating point folds? Results don't match hardware
  // and not all float semantics are supported.
  if (inputDType.isFloat())
    return ::detail::foldSIMDOpDType<APFloat>(
        [](const DTypeValue &val) { return val.getFloatVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  if (inputDType.isBool())
    return ::detail::foldSIMDOpDType<bool>(
        [](const DTypeValue &val) { return val.getBoolVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  if constexpr (indexFoldType != kNoIndex) {
    if (inputDType.isIndex())
      return ::detail::foldSIMDOpIndex<indexFoldType>(
          operands, resultDType, std::forward<OpFns>(ops)...);
  }
  llvm_unreachable("unhandled dtype");
}
} // namespace detail

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <::detail::IndexFold indexFoldType, typename... OpFns>
static SIMDAttr foldSIMDOpResult(ArrayRef<Attribute> operands,
                                 KGENDType resultDType, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  return ::detail::foldSIMDOp<indexFoldType>(
      operands, *cast<SIMDAttr>(operands.front()).getType().getResolvedDType(),
      resultDType, std::forward<OpFns>(ops)...);
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype, assuming the result dtype is the
/// same as the operands' dtypes.
template <typename... OpFns>
static SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  KGENDType dtype =
      *cast<SIMDAttr>(operands.front()).getType().getResolvedDType();
  return ::detail::foldSIMDOp<::detail::kIndexResult>(
      operands, dtype, dtype, std::forward<OpFns>(ops)...);
}

//===----------------------------------------------------------------------===//
// Arithmetic Operation Folders
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Unary Operations

OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt val) { return -val; },
      [](APFloat val) { return llvm::neg(val); });
}

//===----------------------------------------------------------------------===//
// Binary Operations

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs + rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs + rhs; });
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs - rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs - rhs; });
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs * rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs * rhs; });
}

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands,
      [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
        if (rhs.isZero())
          return std::nullopt;
        return lhs / rhs;
      },
      [](APFloat lhs, APFloat rhs) -> std::optional<APFloat> {
        if (rhs.isZero())
          return std::nullopt;
        return lhs / rhs;
      });
}

OpFoldResult RemOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands,
      [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
        if (rhs.isZero())
          return std::nullopt;
        return lhs % rhs;
      },
      [](APFloat lhs, APFloat rhs) -> std::optional<APFloat> {
        if (rhs.isZero())
          return std::nullopt;
        (void)lhs.remainder(rhs);
        return lhs;
      });
}

OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs > rhs ? lhs : rhs; },
      [](APFloat lhs, APFloat rhs) { return llvm::maximum(lhs, rhs); });
}

OpFoldResult MinOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs < rhs ? lhs : rhs; },
      [](APFloat lhs, APFloat rhs) { return llvm::minimum(lhs, rhs); });
}

OpFoldResult ShlOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(operands,
                    [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
                      if (rhs.uge(lhs.getBitWidth()))
                        return std::nullopt;
                      return APSInt(lhs.shl(rhs), lhs.isSigned());
                    });
}

OpFoldResult ShrOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
        if (rhs.uge(lhs.getBitWidth()))
          return std::nullopt;
        return APSInt(lhs.isSigned() ? lhs.ashr(rhs) : lhs.lshr(rhs),
                      lhs.isSigned());
      });
}

//===----------------------------------------------------------------------===//
// Ternary Operations

OpFoldResult FMAOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt a, APSInt b, APSInt c) { return a * b + c; },
      [](APFloat a, APFloat b, APFloat c) { return a * b + c; });
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

ErrorTreeOr<SuccessType> LoadOp::interpret(ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]);
  if (!ptr)
    return ErrorTree(getLoc(), Error("non-constant inputs"));

  ErrorOr<TypedAttr> result =
      state.readAttributeFromMemory(ptr.getAddr(), getType());
  if (result.isError())
    return ErrorTree(getLoc(), result.takeError());
  state.mapResults(result.takeValue());
  return success();
}

//===----------------------------------------------------------------------===//
// CmpOp
//===----------------------------------------------------------------------===//

template <typename ArgT>
static bool compareConstants(CmpPredicate pred, ArgT lhs, ArgT rhs) {
  switch (pred) {
  case CmpPredicate::EQ:
    return lhs == rhs;
  case CmpPredicate::NE:
    return lhs != rhs;
  case CmpPredicate::LT:
    return lhs < rhs;
  case CmpPredicate::GT:
    return lhs > rhs;
  case CmpPredicate::LE:
    return lhs <= rhs;
  case CmpPredicate::GE:
    return lhs >= rhs;
  }
}

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();

  std::optional<KGENDType> operandTy = getLhs().getType().getResolvedDType();
  std::optional<int64_t> size = getType().getResolvedSize();

  // Handle the case of inputs being the same but non-constant. Avoid floats as
  // they could be NAN.
  if (operandTy && size && operandTy->isIntLike() && getLhs() == getRhs()) {
    // Create a SIMD constant of all trues or all false.
    SmallVector<DTypeValue> allTrues(*size, {true, KGENDType::kBool});
    SmallVector<DTypeValue> allFalse(*size, {false, KGENDType::kBool});

    switch (getPred()) {
    case CmpPredicate::EQ:
      return SIMDAttr::get(allTrues, getType());
    case CmpPredicate::NE:
      return SIMDAttr::get(allFalse, getType());
    case CmpPredicate::LT:
      return SIMDAttr::get(allFalse, getType());
    case CmpPredicate::GT:
      return SIMDAttr::get(allFalse, getType());
    case CmpPredicate::LE:
      return SIMDAttr::get(allTrues, getType());
    case CmpPredicate::GE:
      return SIMDAttr::get(allTrues, getType());
    }
  }

  // Handle the case of applying the operation at compile time on the constant
  // values.
  return foldSIMDOpResult<::detail::kOtherResult>(
      operands, KGENDType::kBool,
      [&](APSInt lhs, APSInt rhs) {
        return compareConstants(getPred(), lhs, rhs);
      },
      [&](APFloat lhs, APFloat rhs) {
        return compareConstants(getPred(), lhs, rhs);
      },
      [&](bool lhs, bool rhs) {
        return compareConstants(getPred(), lhs, rhs);
      });
}

//===----------------------------------------------------------------------===//
// Bitwise Operation Folders
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs & rhs; },
      [](bool lhs, bool rhs) { return lhs && rhs; });
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs | rhs; },
      [](bool lhs, bool rhs) { return lhs || rhs; });
}

OpFoldResult XOrOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs ^ rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs ^ rhs; });
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto condVals = dyn_cast_or_null<SIMDAttr>(operands[0]);
  auto trueVals = dyn_cast_or_null<SIMDAttr>(operands[1]);
  auto falseVals = dyn_cast_or_null<SIMDAttr>(operands[2]);
  if (!condVals || !trueVals || !falseVals)
    return {};
  SmallVector<DTypeValue> results;
  for (auto [cond, trueVal, falseVal] : llvm::zip(
           condVals.getValues(), trueVals.getValues(), falseVals.getValues()))
    results.push_back(cond.getBoolVal() ? trueVal : falseVal);
  return SIMDAttr::get(results, getType());
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // Don't fold if the size changes. This requires knowing the endianness of the
  // target.
  std::optional<KGENDType> dtype = getType().getResolvedDType();
  if (!dtype || !getType().getResolvedSize() ||
      getInput().getType().getResolvedSize() != getType().getResolvedSize())
    return {};
  if (dtype->isInt()) {
    return foldSIMDOpResult<::detail::kNoIndex>(
        operands, *dtype,
        [&](APSInt in) { return APSInt(in, dtype->isUInt()); },
        [&](APFloat in) {
          return APSInt(in.bitcastToAPInt(), dtype->isUInt());
        });
  }
  assert(dtype->isFloat());
  // Check to make sure we have a supported float dtype.
  if (!DTypeValue::isValidFloatDType(*dtype))
    return {};
  const llvm::fltSemantics &sem = DTypeValue::getFloatSemantics(*dtype);
  return foldSIMDOpResult<::detail::kNoIndex>(
      operands, *dtype, [&](APSInt in) { return APFloat(sem, in); },
      [&](APFloat in) { return APFloat(sem, in.bitcastToAPInt()); });
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerBitcastOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]))
    return PointerAttr::get(ptr.getAddr(), getType());

  auto cast = getInput().getDefiningOp<PointerBitcastOp>();
  if (cast && cast.getInput().getType() == getType())
    return cast.getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto in = dyn_cast_if_present<SIMDAttr>(operands[0]);
  std::optional<KGENDType> dtype = getType().getResolvedDType();
  if (!in || !dtype) {
    if (getInput().getType() == getOutput().getType())
      return getInput();
    return {};
  }

  // Exit early if the input and output dtypes are the same.
  if (*dtype == *in.getType().getResolvedDType())
    return in;

  if (dtype->isFloat()) {
    // Cannot fold cast to unsupported float dtype.
    if (!DTypeValue::isValidFloatDType(*dtype))
      return {};
    const llvm::fltSemantics &sem = DTypeValue::getFloatSemantics(*dtype);
    return foldSIMDOpResult<::detail::kOtherResult>(
        operands, *dtype,
        [&](APSInt in) {
          APFloat fp(sem);
          fp.convertFromAPInt(in, in.isSigned(), APFloat::rmNearestTiesToEven);
          return fp;
        },
        [&](APFloat in) {
          bool ignored;
          in.convert(sem, APFloat::rmNearestTiesToEven, &ignored);
          return in;
        },
        [&](bool in) { return APFloat(sem, in); });
  }
  if (dtype->isInt()) {
    // Note that float to integer casts are undefined if the float value is
    // too large to fit in the integer dtype.
    unsigned width = dtype->getIntegerWidthInBits();
    return foldSIMDOpResult<::detail::kOtherResult>(
        operands, *dtype, [&](APSInt in) { return in.extOrTrunc(width); },
        [&](APFloat in) -> std::optional<APSInt> {
          APSInt iv(width, dtype->isUInt());
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv;
        },
        [&](bool in) { return APSInt(APInt(width, in), dtype->isUInt()); });
  }
  if (dtype->isIndex()) {
    // Cast to index like it's a 64-bit integer. Index-to-index cast is handled
    // by the early exit above.
    return foldSIMDOpResult<::detail::kNoIndex>(
        operands, *dtype, [](APSInt in) { return in.getSExtValue(); },
        [](APFloat in) -> std::optional<int64_t> {
          APSInt iv(64, /*isUnsigned=*/false);
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv.getSExtValue();
        },
        [](bool in) { return static_cast<int64_t>(in); });
  }
  assert(dtype->isBool());
  return foldSIMDOpResult<::detail::kOtherResult>(
      operands, *dtype, [](APSInt in) { return !in.isZero(); },
      [](APFloat in) { return !in.isZero(); });
}

//===----------------------------------------------------------------------===//
// SIMDExtractElementOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDExtractElementOp::fold(FoldAdaptor adaptor) {
  // Extracting from a scalar is always going to return the scalar.
  if (getVector().getType().isScalar())
    return getVector();

  auto operands = adaptor.getOperands();
  auto vec = dyn_cast_if_present<SIMDAttr>(operands[0]);
  auto idx = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!vec || !idx)
    return {};
  return SIMDAttr::get(vec.getValues()[idx.getInt()], getType());
}

//===----------------------------------------------------------------------===//
// SIMDInsertElementOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDInsertElementOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto vec = dyn_cast_if_present<SIMDAttr>(operands[0]);
  auto val = dyn_cast_if_present<SIMDAttr>(operands[1]);
  auto idx = dyn_cast_if_present<IntegerAttr>(operands[2]);
  if (!vec || !val || !idx)
    return {};
  SmallVector<DTypeValue> values(vec.getValues());
  values[idx.getInt()] = val.getValues().front();
  return SIMDAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// SIMDShuffleOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDShuffleOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  std::optional<int64_t> size = getType().getResolvedSize();
  auto lhs = dyn_cast_if_present<SIMDAttr>(operands[0]);
  auto rhs = dyn_cast_if_present<SIMDAttr>(operands[1]);
  auto mask = dyn_cast_if_present<VariadicAttr>(adaptor.getMaskAttr());
  if (!size || !lhs || !rhs || !mask)
    return {};

  // Is the mask a known constant.
  if (llvm::any_of(mask.getValues(), [](Attribute operand) {
        return !isa_and_nonnull<IntegerAttr>(operand);
      }))
    return {};

  // Concatenate the input simd vectors.
  SmallVector<DTypeValue> args(lhs.getValues());
  llvm::append_range(args, rhs.getValues());

  // Perform the permutation based on the mask.
  SmallVector<DTypeValue> result;
  result.reserve(mask.getValues().size());
  for (TypedAttr maskVal : mask.getValues())
    result.emplace_back(args[cast<IntegerAttr>(maskVal).getInt()]);

  return SIMDAttr::get(result, getType());
}

//===----------------------------------------------------------------------===//
// SIMDSplatOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDSplatOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  std::optional<int64_t> size = getType().getResolvedSize();
  auto scalar = dyn_cast_if_present<SIMDAttr>(operands[0]);
  if (!size || !scalar)
    return {};
  SmallVector<DTypeValue> values(*size, scalar.getValues().front());
  return SIMDAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

ErrorTreeOr<SuccessType> StoreOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  auto value = llvm::cast_if_present<TypedAttr>(operands[0]);
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[1]);
  if (!value || !ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  ErrorOrSuccess result = state.writeAttributeToMemory(ptr.getAddr(), value);
  if (result.isError())
    return ErrorTree(getLoc(), result.takeError());
  return success();
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

ErrorTreeOr<SuccessType> OffsetOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]);
  auto offset = dyn_cast_or_null<IntegerAttr>(operands[1]);
  if (!ptr || !offset)
    return ErrorTree(getLoc(), "non-constant inputs");
  std::optional<int64_t> elSize = DataLayoutInterface::getTypeAllocSize(
      state.getTarget(), cast<PointerType>(ptr.getType()).getElementAsType());
  if (!elSize)
    return ErrorTree(getLoc(), "could not query pointer element size");
  state.mapResults(PointerAttr::get(ptr.getAddr() + *elSize * offset.getInt(),
                                    ptr.getType()));
  return success();
}

OpFoldResult OffsetOp::fold(FoldAdaptor adaptor) {
  IntegerAttr offset = dyn_cast_or_null<IntegerAttr>(adaptor.getIndex());
  if (!offset)
    return {};

  if (offset.getInt() != 0)
    return {};

  return getPtr();
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

ErrorTreeOr<SuccessType>
StackAllocationOp::interpret(ArrayRef<Attribute> operands,
                             InterpreterState &state) {
  auto count = dyn_cast<IntegerAttr>(getCount());
  if (!count)
    return ErrorTree(getLoc(), "not concrete");
  Type type = cast<PointerType>(getType()).getElementAsType();
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeAllocSize(state.getTarget(), type);
  if (!size)
    return ErrorTree(getLoc(), "could not query type size");
  int64_t addr = state.allocateMemory(count.getInt() * *size);
  state.mapResults(PointerAttr::get(addr, getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// StructCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult StructCreateOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  SmallVector<TypedAttr> values;
  values.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return StructAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// StructExtractOp
//===----------------------------------------------------------------------===//

OpFoldResult StructExtractOp::fold(FoldAdaptor adaptor) {
  if (auto container = adaptor.getContainer())
    return StructExtractAttr::get(cast<TypedAttr>(container), getIndexAttr());
  if (auto structCreate = getOperand().getDefiningOp<StructCreateOp>())
    return structCreate.getOperand(adaptor.getIndex().getSExtValue());
  return {};
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

OpFoldResult StructReplaceOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto value = llvm::cast_if_present<TypedAttr>(operands[0]);
  auto container = dyn_cast_if_present<StructAttr>(operands[1]);
  if (!value || !container)
    return {};
  SmallVector<TypedAttr> values(container.getValues());
  values[getIndexAttr().getInt()] = value;
  return StructAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// StructGEPOp
//===----------------------------------------------------------------------===//

ErrorTreeOr<SuccessType>
POP::StructGEPOp::interpret(ArrayRef<Attribute> operands,
                            InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  if (!ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  int64_t offset = 0;
  auto structType = getContainer().getType().getElementAs<StructType>();

  // Move the address over the elements before the one we are reading.
  unsigned index = getIndexAttr().getInt();
  for (unsigned i = 0; i != index; ++i) {
    auto dl = cast<DataLayoutInterface>(structType.getConcreteElementType(i));
    offset = llvm::alignTo(offset, *dl.getTypeAlign(state.getTarget()));
    offset += *dl.getTypeSize(state.getTarget());
  }

  // Align the address to the target element.
  Type targetType = structType.getConcreteElementType(index);
  offset = llvm::alignTo(
      offset,
      *cast<DataLayoutInterface>(targetType).getTypeAlign(state.getTarget()));
  state.mapResults(
      PointerAttr::get(ptr.getAddr() + offset, PointerType::get(targetType)));
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayCreateOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  SmallVector<TypedAttr> values;
  values.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return POP::ArrayAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// ArrayRepeatOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayRepeatOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  std::optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return {};
  SmallVector<TypedAttr> args;
  args.reserve(operands.size());
  for (Attribute operand : operands) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    args.push_back(value);
  }
  SmallVector<TypedAttr> values;
  values.reserve(*size);
  while (static_cast<int64_t>(values.size()) < *size)
    values.append(args);
  return POP::ArrayAttr::get(ArrayRef(values).take_front(*size), getType());
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayGetOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto array = dyn_cast_if_present<POP::ArrayAttr>(operands[0]);
  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (!array || !index)
    return {};
  return array.getValues()[index.getInt()];
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayReplaceOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto value = llvm::cast_if_present<TypedAttr>(operands[0]);
  auto array = dyn_cast_if_present<POP::ArrayAttr>(operands[1]);
  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (!value || !array || !index)
    return {};
  SmallVector<TypedAttr> values(array.getValues());
  values[index.getInt()] = value;
  return POP::ArrayAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// ArrayGEPOp
//===----------------------------------------------------------------------===//

ErrorTreeOr<SuccessType> ArrayGEPOp::interpret(ArrayRef<Attribute> operands,
                                               InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands[0]);
  auto index = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!ptr || !index)
    return ErrorTree(getLoc(), "non-constant inputs");

  auto arrayType = getArray().getType().getElementAs<POP::ArrayType>();
  auto dl = cast<DataLayoutInterface>(arrayType.getElementAsType());
  int64_t addr =
      ptr.getAddr() +
      index.getInt() * (llvm::alignTo(*dl.getTypeSize(state.getTarget()),
                                      *dl.getTypeAlign(state.getTarget())));
  state.mapResults(PointerAttr::get(addr, PointerType::get(dl)));
  return success();
}

LogicalResult ArrayGEPOp::canonicalize(ArrayGEPOp op,
                                       PatternRewriter &rewriter) {
  PointerType ptrToArray = op.getArray().getType();
  std::optional<int64_t> size =
      ptrToArray.getElementAs<ArrayType>().getResolvedSize();

  // We are only going to canonicalize scalars.
  if (!size || *size != 1)
    return rewriter.notifyMatchFailure(op, "Size is not known to be scalar");

  // Don't repeatedly canonicalize already constant values.
  if (op.getIndex().getDefiningOp() &&
      op.getIndex().getDefiningOp()->hasTrait<OpTrait::ConstantLike>())
    return rewriter.notifyMatchFailure(op,
                                       "ArrayGEP index is already constant.");

  // Otherwise we have gep into a array of one element with a dynamic value. It
  // is undefined behaviour for that to be anything but `0` so we can replace it
  // with the constant `0`. This frees the use to be DCE'd and unblocks other
  // optimizations.
  auto zero = rewriter.create<KGEN::ParamConstantOp>(op.getLoc(),
                                                     rewriter.getIndexAttr(0));
  rewriter.replaceOpWithNewOp<ArrayGEPOp>(op, op.getType(), op.getArray(),
                                          zero);
  return success();
}

//===----------------------------------------------------------------------===//
// PackCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult PackCreateOp::fold(FoldAdaptor adaptor) {
  SmallVector<TypedAttr> values;
  values.reserve(adaptor.getOperands().size());
  for (Attribute operand : adaptor.getOperands()) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return PackAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// PackGetOp
//===----------------------------------------------------------------------===//

OpFoldResult PackGetOp::fold(FoldAdaptor adaptor) {
  auto index = dyn_cast_or_null<IntegerAttr>(adaptor.getIndexAttr());
  if (!index)
    return {};

  if (auto pack = dyn_cast_or_null<PackAttr>(adaptor.getPack()))
    return pack.getValues()[index.getInt()];

  // Canonicalize `get(create(x)) -> x`.
  if (auto create = getPack().getDefiningOp<PackCreateOp>())
    return create.getOperands()[index.getInt()];

  return {};
}

//===----------------------------------------------------------------------===//
// PackSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult PackSizeOp::fold(FoldAdaptor adaptor) {
  if (auto pack = dyn_cast_if_present<PackAttr>(adaptor.getOperand()))
    return IntegerAttr::get(IndexType::get(getContext()),
                            pack.getValues().size());
  return {};
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantCreateOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto value = llvm::cast_if_present<TypedAttr>(operands[0]);
  if (!value)
    return {};
  return VariantAttr::get(value, getType());
}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantIsOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto variant = dyn_cast_if_present<VariantAttr>(operands[0]);
  if (!variant)
    return {};
  return BoolAttr::get(getContext(),
                       variant.getValue().getType() == getTestType());
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantGetOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  if (auto variant = dyn_cast_if_present<VariantAttr>(operands[0])) {
    // If the variant value type is not equal to the result type, this is
    // undefined behaviour.
    if (variant.getValue().getType() != getType())
      return {};
    return variant.getValue();
  }

  // Canonicalize `pop.variant.get(pop.variant.create(x)) -> x`.
  auto create = getVariant().getDefiningOp<VariantCreateOp>();
  if (!create || create.getOperand().getType() != getType())
    return {};
  return create.getOperand();
}

//===----------------------------------------------------------------------===//
// IndexToPointerOp
//===----------------------------------------------------------------------===//

OpFoldResult IndexToPointerOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto index = dyn_cast_if_present<SIMDAttr>(operands[0]);
  if (!index)
    return {};
  // Check for a pointer type. Create a pointer constant attribute.
  if (isa<PointerType>(getType()))
    return PointerAttr::get(index.getValues().front().getIndexVal(), getType());
  // Otherwise, this is converting to an address dtype vector. The DTypeValue
  // storage is the same, but the type is different.
  SmallVector<DTypeValue> values;
  for (const DTypeValue &value : index.getValues())
    values.emplace_back(value.getIndexVal(), KGENDType::address);
  return SIMDAttr::get(values, cast<SIMDType>(getType()));
}

//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerToIndexOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  // Check for a pointer input. The result must be a scalar index.
  if (auto ptr = dyn_cast_if_present<PointerAttr>(operands[0])) {
    DTypeValue index(static_cast<int64_t>(ptr.getAddr()), KGENDType::index);
    return SIMDAttr::get(index, getType());
  }
  // Otherwise, the input might be an address vector.
  if (auto simd = dyn_cast_if_present<SIMDAttr>(operands[0])) {
    SmallVector<DTypeValue> values;
    for (const DTypeValue &value : simd.getValues())
      values.emplace_back(value.getIndexVal(), KGENDType::index);
    return SIMDAttr::get(values, getType());
  }
  return {};
}

//===----------------------------------------------------------------------===//
// CastToBuiltinOp
//===----------------------------------------------------------------------===//

/// Convert a SIMD attribute to a vector-typed attribute.
template <typename AttrT, typename TransformFn>
static ArrayElementsAttr convertSIMDToVectorAttr(SIMDAttr simd, VectorType type,
                                                 TransformFn fn) {
  SmallVector<decltype(fn(std::declval<DTypeValue>()))> values;
  for (const DTypeValue &value : simd.getValues())
    values.push_back(fn(value));
  return AttrT::get(type, values);
}

OpFoldResult CastToBuiltinOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto simd = dyn_cast_if_present<SIMDAttr>(operands[0]);
  if (!simd) {
    // Fold A->B->A cast.
    if (auto parent = getInput().getDefiningOp<CastFromBuiltinOp>();
        parent && parent.getInput().getType() == getType())
      return parent.getInput();
    return {};
  }

  // Conversion to a 1D vector type.
  std::optional<KGENDType> dtype = simd.getType().getResolvedDType();
  if (!dtype)
    return {};
  if (auto vector = dyn_cast<VectorType>(getType())) {
    if (dtype->isBool())
      return convertSIMDToVectorAttr<IntArrayElementsAttr>(
          simd, vector,
          [](DTypeValue val) { return APInt(1, val.getBoolVal()); });
    if (dtype->isIndex())
      return convertSIMDToVectorAttr<IndexArrayElementsAttr>(
          simd, vector, [](DTypeValue val) { return val.getIndexVal(); });
    if (dtype->isInt())
      return convertSIMDToVectorAttr<IntArrayElementsAttr>(
          simd, vector, [](DTypeValue val) { return val.getIntVal(); });
    assert(dtype->isFloat() && "unexpected dtype");
    return convertSIMDToVectorAttr<FloatArrayElementsAttr>(
        simd, vector, [](DTypeValue val) { return val.getFloatVal(); });
  }

  assert(simd.getValues().size() == 1 && "expected a scalar constant");
  const DTypeValue &value = simd.getValues().front();

  // Convert to a scalar attribute.
  Builder b(simd.getContext());
  if (dtype->isBool())
    return b.getBoolAttr(value.getBoolVal());
  if (dtype->isIndex())
    return b.getIndexAttr(value.getIndexVal());
  if (dtype->isInt())
    return b.getIntegerAttr(cast<IntegerType>(getType()), value.getIntVal());
  assert(dtype->isFloat() && "unexpected dtype");
  return b.getFloatAttr(cast<FloatType>(getType()), value.getFloatVal());
}

//===----------------------------------------------------------------------===//
// CastFromBuiltinOp
//===----------------------------------------------------------------------===//

OpFoldResult CastFromBuiltinOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  auto val = llvm::cast_if_present<TypedAttr>(operands[0]);
  if (!val) {
    // Fold A->B->A cast.
    if (auto parent = getInput().getDefiningOp<CastToBuiltinOp>();
        parent && parent.getInput().getType() == getType())
      return parent.getInput();
    return {};
  }

  // Ensure the incoming value is an expected constant kind.
  if (!isa<IntArrayElementsAttr, FloatArrayElementsAttr, IndexArrayElementsAttr,
           IntegerAttr, FloatAttr>(val))
    return {};

  // Conversion from vector constant.
  std::optional<KGENDType> dtype = getType().getResolvedDType();
  if (!dtype)
    return {};
  if (auto vector = dyn_cast<VectorType>(val.getType())) {
    SmallVector<DTypeValue> values;
    if (dtype->isBool())
      for (APInt value : cast<IntArrayElementsAttr>(val).getValues())
        values.emplace_back(!value.isZero(), *dtype);
    else if (dtype->isIndex())
      for (int64_t value : cast<IndexArrayElementsAttr>(val))
        values.emplace_back(value, *dtype);
    else if (dtype->isInt())
      for (APInt value : cast<IntArrayElementsAttr>(val).getValues())
        values.emplace_back(value, *dtype);
    else
      for (APFloat value : cast<FloatArrayElementsAttr>(val).getValues())
        values.emplace_back(value, *dtype);
    return SIMDAttr::get(values, getType());
  }

  // Handle scalar constants.
  if (dtype->isBool())
    return SIMDAttr::get({cast<BoolAttr>(val).getValue(), *dtype}, getType());
  if (dtype->isIndex())
    return SIMDAttr::get({cast<IntegerAttr>(val).getInt(), *dtype}, getType());
  if (dtype->isInt())
    return SIMDAttr::get({cast<IntegerAttr>(val).getValue(), *dtype},
                         getType());
  assert(dtype->isFloat() && "unexpected dtype");
  return SIMDAttr::get({cast<FloatAttr>(val).getValue(), *dtype}, getType());
}

//===----------------------------------------------------------------------===//
// VariadicCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult VariadicCreateOp::fold(FoldAdaptor adaptor) {
  SmallVector<TypedAttr> values;
  values.reserve(adaptor.getOperands().size());
  for (Attribute operand : adaptor.getOperands()) {
    auto value = llvm::cast_if_present<TypedAttr>(operand);
    if (!value)
      return {};
    values.push_back(value);
  }
  return KGEN::VariadicAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// VariadicGetOp
//===----------------------------------------------------------------------===//

OpFoldResult VariadicGetOp::fold(FoldAdaptor adaptor) {
  auto indexAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getIndex());
  if (!indexAttr)
    return {};
  auto index = static_cast<size_t>(indexAttr.getInt());

  if (auto variadic = dyn_cast_or_null<VariadicAttr>(adaptor.getVariadic())) {
    if (index >= variadic.getValues().size())
      return {};
    return variadic.getValues()[index];
  }

  // Canonicalize `get(create(x)) -> x`.
  if (auto create = getVariadic().getDefiningOp<VariadicCreateOp>()) {
    if (index >= create.getOperands().size())
      return {};
    return create.getOperands()[index];
  }

  return {};
}

//===----------------------------------------------------------------------===//
// VariadicAppendOp
//===----------------------------------------------------------------------===//

OpFoldResult VariadicAppendOp::fold(FoldAdaptor adaptor) {
  auto value = llvm::cast_if_present<TypedAttr>(adaptor.getValue());
  if (!value)
    return {};
  auto variadic = dyn_cast_or_null<VariadicAttr>(adaptor.getVariadic());
  if (!variadic)
    return {};

  SmallVector<TypedAttr> values;
  values.reserve(variadic.getValues().size() + 1);
  for (Attribute varVal : variadic.getValues())
    values.push_back(cast<TypedAttr>(varVal));
  values.push_back(value);
  return VariadicAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// VariadicSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult VariadicSizeOp::fold(FoldAdaptor adaptor) {
  auto indexType = IndexType::get(getContext());
  if (auto variadic =
          dyn_cast_if_present<KGEN::VariadicAttr>(adaptor.getOperand()))
    return IntegerAttr::get(indexType, variadic.getValues().size());

  if (auto create = getOperand().getDefiningOp<VariadicCreateOp>())
    return IntegerAttr::get(indexType, create.getOperands().size());

  return {};
}

//===----------------------------------------------------------------------===//
// StringSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult StringSizeOp::fold(FoldAdaptor adaptor) {
  if (auto str = dyn_cast_or_null<StringAttr>(adaptor.getStr()))
    return Builder(getContext()).getIndexAttr(str.getValue().size());
  return {};
}

//===----------------------------------------------------------------------===//
// StringConcatOp
//===----------------------------------------------------------------------===//

OpFoldResult StringConcatOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<StringAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<StringAttr>(adaptor.getRhs());
  if (!lhs || !rhs)
    return {};
  return StringAttr::get(lhs.getValue() + rhs.getValue(),
                         StringType::get(getContext()));
}

//===----------------------------------------------------------------------===//
// DTypeToUI8
//===----------------------------------------------------------------------===//

OpFoldResult DTypeToUI8::fold(FoldAdaptor adaptor) {
  auto ui8Type = IntegerType::get(getContext(), 8,
                                  IntegerType::SignednessSemantics::Unsigned);
  if (auto dtype =
          dyn_cast_if_present<KGEN::DTypeConstantAttr>(adaptor.getDType()))
    return IntegerAttr::get(ui8Type, dtype.getDType().getValue());

  return {};
}

//===----------------------------------------------------------------------===//
// DTypeFromUI8
//===----------------------------------------------------------------------===//

OpFoldResult DTypeFromUI8::fold(FoldAdaptor adaptor) {
  if (auto val = dyn_cast_if_present<IntegerAttr>(adaptor.getValue()))
    return KGEN::DTypeConstantAttr::get(getContext(), KGENDType(val.getUInt()));

  return {};
}
