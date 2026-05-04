//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file declares utility functions primarily for parsing, printing and
// verifying POP related operations and types.
//
//===----------------------------------------------------------------------===//

#ifndef KGEN_POPDIALECT_POPUTILS_H
#define KGEN_POPDIALECT_POPUTILS_H

#include "KGEN/KGENDialect/FoldUtils.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M::KGEN::POP {

/// Get the value of a scalar index-like parameter value.
/// This is a temporary helper utility during the Int->SIMD unification project.
/// After it's done, we should remove the IntegerAttr case.
ErrorOr<int64_t> getScalarIndexValue(TypedAttr value);

/// Fold a cast between two SIMD types.
OpFoldResult foldCast(TypedAttr operand, SIMDType resultType,
                      SIMDType inputType, SIMDType outputType,
                      std::optional<int64_t> indexBitWidth = std::nullopt);

/// Fold a SIMD splat operation.
OpFoldResult foldSIMDSplat(Value scalarVal, Attribute scalarAttr,
                           SIMDType resultType);

/// Fold a SIMD Or-reduction operation.
OpFoldResult foldSIMDReduceOr(Value vectorVal, Attribute vectorAttr,
                              SIMDType vectorType);
/// Fold a SIMD And-reduction operation.
OpFoldResult foldSIMDReduceAnd(Value vectorVal, Attribute vectorAttr,
                               SIMDType vectorType);

/// Convert a NormalizedCmpPredicate to the full CmpPredicate.
inline CmpPredicate toCmpPredicate(NormalizedCmpPredicate cc) {
  switch (cc) {
  case NormalizedCmpPredicate::EQ:
    return CmpPredicate::EQ;
  case NormalizedCmpPredicate::LT:
    return CmpPredicate::LT;
  case NormalizedCmpPredicate::LE:
    return CmpPredicate::LE;
  }
  llvm_unreachable("invalid NormalizedCmpPredicate");
}

/// Fold a SIMD comparison operation. Handles constant folding, bool identity
/// folds (eq(true, x) -> x), and unsigned comparisons with zero. Returns null
/// if no fold applies.
FoldValue foldSIMDCmp(CmpPredicate cc, FoldValues operands,
                      TargetInfoAttr target = {});

/// Fold a SIMD left-shift operation.
FoldValue foldSIMDShl(FoldValues operands, TargetInfoAttr target = {});

/// Fold a SIMD right-shift operation.
FoldValue foldSIMDShr(FoldValues operands, TargetInfoAttr target = {});

/// Fold a SIMD abs operation.
FoldValue foldSIMDAbs(FoldValues operands, TargetInfoAttr target = {});

/// Fold a SIMD round operation.
OpFoldResult foldSIMDRound(Attribute val, TargetInfoAttr targetInfo);

/// Fold a SIMD div operation.
FoldValue foldSIMDDiv(FoldValues operands, TargetInfoAttr target = {});

/// Fold a SIMD floordiv operation.
FoldValue foldSIMDFloorDiv(FoldValues operands, TargetInfoAttr target = {});

//===----------------------------------------------------------------------===//
// SIMD Folder Helpers
//===----------------------------------------------------------------------===//

/// This enum indicates how index folding should be done.
enum IndexFold : uint8_t {
  kNoIndex,     // no index folding allowed
  kIndexResult, // index operation creates an index
  kOtherResult, // index operation does not create an index
};

namespace Detail {
/// Detector for whether `T` possesses a `has_value` method.
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

/// Perform the folding of a SIMD vector reduction of a given dtype by
/// accumulatively applying the binary operation `op` to each vector
/// element, in order, starting with the first. `getValue` transforms a
/// `DTypeValue` to the value used to represent the dtype: `APSInt` for
/// integers, `APFloat` for floats, and `bool` for bools.
template <typename OpFn, typename GetValueFn>
static SIMDAttr foldSIMDReduceOpImpl(Attribute operand, OpFn op,
                                     KGENDType dtype, GetValueFn getValue) {
  auto firstArg = cast<SIMDAttr>(operand);
  auto values = firstArg.getValues();
  auto accumResult = getValue(values[0]);
  for (unsigned i = 1, e = values.size(); i < e; ++i) {
    auto res = std::apply(op, std::make_pair(accumResult, getValue(values[i])));
    // Allow folders to return failure. This indicates undefined behaviour,
    // which we do not fold.
    if constexpr (llvm::is_detected<IsOptionalType, decltype(res)>::value) {
      if (!res)
        return {};
      accumResult = *res;
    } else {
      accumResult = res;
    }
  }
  return SIMDAttr::get(DTypeValue(accumResult, dtype),
                       SIMDType::get(operand.getContext(), 1, dtype));
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
static constexpr auto getOpFnOfType([[maybe_unused]] OpFn op, OpFns &&...fns) {
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
static SIMDAttr foldSIMDOpDType([[maybe_unused]] GetValueFn getValue,
                                [[maybe_unused]] ArrayRef<Attribute> operands,
                                [[maybe_unused]] KGENDType dtype,
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

/// Try to fold the operation using one of the provided fold functions for a
/// given dtype. If a fold function for that dtype is not provided, if such a
/// dtype is encountered by the folder, it will assert; a folder must be
/// provided for each dtype for which the operation is valid.
template <typename TestType, typename GetValueFn, typename... OpFns>
static SIMDAttr foldSIMDReduceOpDType([[maybe_unused]] GetValueFn getValue,
                                      [[maybe_unused]] Attribute operand,
                                      [[maybe_unused]] KGENDType dtype,
                                      OpFns &&...ops) {
  auto op = getOpFnOfType<TestType>(std::forward<OpFns>(ops)...);
  if constexpr (std::is_same_v<decltype(op), std::nullopt_t>) {
    llvm_unreachable("unhandled dtype");
  } else {
    return foldSIMDReduceOpImpl(operand, op, dtype, getValue);
  }
}

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
      auto unwrap = [](OpResultT value) {
        if constexpr (isOptional)
          return *value;
        else
          return value;
      };

      OpResultT result64 = op(args...);
      if constexpr (isOptional)
        if (!result64.has_value())
          return {};

      OpResultT result32 = op(args.trunc(32)...);
      if constexpr (isOptional)
        if (!result32.has_value())
          return {};
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
                            return APSInt(
                                APInt(64, val.getIndexVal()),
                                /*isUnsigned=*/!val.getDType().isIndex());
                          });
  }
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <IndexFold indexFoldType, typename... OpFns>
SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands, KGENDType inputDType,
                    KGENDType resultDType, std::optional<int64_t> indexBitWidth,
                    OpFns &&...ops) {
  if (inputDType.isInt())
    return Detail::foldSIMDOpDType<APSInt>(
        [](const DTypeValue &val) { return val.getIntVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  // FIXME: Should we even do floating point folds? Results don't match hardware
  // and not all float semantics are supported.
  if (inputDType.isFloat())
    return Detail::foldSIMDOpDType<APFloat>(
        [](const DTypeValue &val) { return val.getFloatVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  if (inputDType.isBool())
    return Detail::foldSIMDOpDType<bool>(
        [](const DTypeValue &val) { return val.getBoolVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  if (inputDType.isIndex() || inputDType.isUIndex() || inputDType.isAddress()) {
    // If we know the index type's bit width, treat it as if it were an integer
    // type of that same bit width. This avoids the complexities of dealing with
    // index types.
    if (indexBitWidth) {
      int64_t bitWidth = *indexBitWidth;
      bool isUnsigned = !inputDType.isIndex();
      return Detail::foldSIMDOpDType<APSInt>(
          [bitWidth, isUnsigned](const DTypeValue &val) {
            auto indexAPInt = APInt(64, val.getIndexVal());
            return APSInt(indexAPInt, isUnsigned).extOrTrunc(bitWidth);
          },
          operands, resultDType, std::forward<OpFns>(ops)...);
    }
    return Detail::foldSIMDOpIndex<indexFoldType>(operands, resultDType,
                                                  std::forward<OpFns>(ops)...);
  }
  llvm_unreachable("unhandled dtype");
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <typename... OpFns>
SIMDAttr foldBitwiseSIMDReduceOp(Attribute operand, KGENDType inputDType,
                                 KGENDType resultDType, OpFns &&...ops) {
  if (inputDType.isBool())
    return Detail::foldSIMDReduceOpDType<bool>(
        [](const DTypeValue &val) { return val.getBoolVal(); }, operand,
        resultDType, std::forward<OpFns>(ops)...);
  // For bitwise reductions we can treat index-like types as ints. The result
  // would be the same no matter the eventual index bitwidth, whether it was
  // extended/truncated before or after the fold.
  if (inputDType.isIntLike())
    return Detail::foldSIMDReduceOpDType<APSInt>(
        [](const DTypeValue &val) { return val.getIntVal(); }, operand,
        resultDType, std::forward<OpFns>(ops)...);
  llvm_unreachable("unhandled dtype");
}
} // namespace Detail

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <IndexFold indexFoldType, typename... OpFns>
SIMDAttr foldSIMDOpResult(ArrayRef<Attribute> operands, KGENDType resultDType,
                          std::optional<int64_t> indexBitWidth,
                          OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  return Detail::foldSIMDOp<indexFoldType>(
      operands, *cast<SIMDAttr>(operands.front()).getType().getResolvedDType(),
      resultDType, indexBitWidth, std::forward<OpFns>(ops)...);
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <IndexFold indexFoldType, typename... OpFns>
SIMDAttr foldSIMDOpResult(ArrayRef<Attribute> operands, KGENDType resultDType,
                          OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  return Detail::foldSIMDOp<indexFoldType>(
      operands, *cast<SIMDAttr>(operands.front()).getType().getResolvedDType(),
      resultDType, /*indexBitWidth=*/std::nullopt, std::forward<OpFns>(ops)...);
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype, assuming the result dtype is the
/// same as the operands' dtypes.
template <typename... OpFns>
SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands,
                    std::optional<int64_t> indexBitWidth, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  KGENDType dtype =
      *cast<SIMDAttr>(operands.front()).getType().getResolvedDType();
  return Detail::foldSIMDOp<kIndexResult>(operands, dtype, dtype, indexBitWidth,
                                          std::forward<OpFns>(ops)...);
}

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype, assuming the result dtype is the
/// same as the operands' dtypes.
template <typename... OpFns>
SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  KGENDType dtype =
      *cast<SIMDAttr>(operands.front()).getType().getResolvedDType();
  return Detail::foldSIMDOp<kIndexResult>(operands, dtype, dtype,
                                          /*indexBitWidth=*/std::nullopt,
                                          std::forward<OpFns>(ops)...);
}

/// Try to fold a SIMD vector reduction operation using one of the provided
/// functions for each possible operand dtype, assuming the result dtype is the
/// same as the operands' dtypes.
template <typename... OpFns>
SIMDAttr foldBitwiseSIMDReduceOp(Attribute operand, OpFns &&...ops) {
  if (!isa_and_nonnull<SIMDAttr>(operand))
    return {};
  KGENDType dtype = *cast<SIMDAttr>(operand).getType().getResolvedDType();
  return Detail::foldBitwiseSIMDReduceOp(operand, dtype, dtype,
                                         std::forward<OpFns>(ops)...);
}

/// Interpret a memcpy operation.
ErrorTreeOrSuccess interpretMemcpy(Attribute dst, Attribute src, Attribute len,
                                   Location loc, InterpreterState &state);
/// Interpret a memcpy operation.
ErrorTreeOrSuccess interpretMemcpy(Attribute dst, Attribute src, Attribute len,
                                   Location loc,
                                   ParametricInterpreterState &state);

} // namespace M::KGEN::POP

#endif // KGEN_POPDIALECT_POPUTILS_H
