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

/// Try to fold an operation with index dtype using one of the provided fold
/// functions. Index folds are performed using the same function as integer
/// dtype folds. An index fold is performed by computing the result in 64-bit
/// and 32-bit arithmetic. If the results match, then the operation can fold.
/// See the MLIR `index` dialect for more details.
template <typename... OpFns>
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
    constexpr bool isIndexResult = std::is_same_v<APSInt, ResultT>;
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
template <bool allowIndex, typename... OpFns>
static SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands, KGENDType inputDType,
                           KGENDType resultDType, OpFns &&...ops) {
  if (inputDType.isInt())
    return ::detail::foldSIMDOpDType<APSInt>(
        [](DTypeValue val) { return val.getIntVal(); }, operands, resultDType,
        std::forward<OpFns>(ops)...);
  // FIXME: Should we even do floating point folds? Results don't match hardware
  // and not all float semantics are supported.
  if (inputDType.isFloat())
    return ::detail::foldSIMDOpDType<APFloat>(
        [](DTypeValue val) { return val.getFloatVal(); }, operands, resultDType,
        std::forward<OpFns>(ops)...);
  if (inputDType.isBool())
    return ::detail::foldSIMDOpDType<bool>(
        [](DTypeValue val) { return val.getBoolVal(); }, operands, resultDType,
        std::forward<OpFns>(ops)...);
  if constexpr (allowIndex) {
    if (inputDType.isIndex())
      return ::detail::foldSIMDOpIndex(operands, resultDType,
                                       std::forward<OpFns>(ops)...);
  }
  llvm_unreachable("unhandled dtype");
}
} // namespace detail

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <bool AllowIndex, typename... OpFns>
static SIMDAttr foldSIMDOpResult(ArrayRef<Attribute> operands,
                                 KGENDType resultDType, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  return ::detail::foldSIMDOp<AllowIndex>(
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
  return ::detail::foldSIMDOp</*AllowIndex=*/true>(operands, dtype, dtype,
                                                   std::forward<OpFns>(ops)...);
}

//===----------------------------------------------------------------------===//
// Arithmetic Operation Folders
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Unary Operations

OpFoldResult AbsOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt val) { return val.abs(); },
      [](APFloat val) { return llvm::abs(val); });
}

OpFoldResult NegOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt val) { return -val; },
      [](APFloat val) { return llvm::neg(val); });
}

//===----------------------------------------------------------------------===//
// Binary Operations

OpFoldResult AddOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs + rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs + rhs; });
}

OpFoldResult SubOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs - rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs - rhs; });
}

OpFoldResult MulOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs * rhs; },
      [](APFloat lhs, APFloat rhs) { return lhs * rhs; });
}

OpFoldResult DivOp::fold(ArrayRef<Attribute> operands) {
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

OpFoldResult RemOp::fold(ArrayRef<Attribute> operands) {
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

OpFoldResult MaxOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs > rhs ? lhs : rhs; },
      [](APFloat lhs, APFloat rhs) { return llvm::maximum(lhs, rhs); });
}

OpFoldResult MinOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs < rhs ? lhs : rhs; },
      [](APFloat lhs, APFloat rhs) { return llvm::minimum(lhs, rhs); });
}

OpFoldResult ShlOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(operands,
                    [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
                      if (rhs.uge(lhs.getBitWidth()))
                        return std::nullopt;
                      return APSInt(lhs.shl(rhs), lhs.isSigned());
                    });
}

OpFoldResult ShrOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
        if (rhs.uge(lhs.getBitWidth()))
          return std::nullopt;
        return APSInt(lhs.isSigned() ? lhs.ashr(rhs) : lhs.lshr(rhs),
                      lhs.isSigned());
      });
}

OpFoldResult CopySignOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(operands, [](APFloat lhs, APFloat rhs) {
    return rhs.isNegative() ? -llvm::abs(lhs) : llvm::abs(lhs);
  });
}

//===----------------------------------------------------------------------===//
// Ternary Operations

OpFoldResult FMAOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt a, APSInt b, APSInt c) { return a * b + c; },
      [](APFloat a, APFloat b, APFloat c) { return a * b + c; });
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess LoadOp::interpret(ArrayRef<Attribute> operands,
                                 InterpreterState &state,
                                 SmallVectorImpl<OpFoldResult> &results) {
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]);
  if (!ptr)
    return Error("non-constant inputs");

  ErrorOr<TypedAttr> result =
      state.readAttributeFromMemory(ptr.getAddr(), getType());
  if (result.isError())
    return result.takeError();
  results.push_back(result.takeValue());
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

OpFoldResult CmpOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOpResult</*AllowIndex=*/true>(
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

OpFoldResult AndOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs & rhs; },
      [](bool lhs, bool rhs) { return lhs && rhs; });
}

OpFoldResult OrOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs | rhs; },
      [](bool lhs, bool rhs) { return lhs || rhs; });
}

OpFoldResult XOrOp::fold(ArrayRef<Attribute> operands) {
  return foldSIMDOp(
      operands, [](APSInt lhs, APSInt rhs) { return lhs ^ rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs ^ rhs; });
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(ArrayRef<Attribute> operands) {
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

OpFoldResult BitcastOp::fold(ArrayRef<Attribute> operands) {
  // Don't fold if the size changes. This requires knowing the endianness of the
  // target.
  Optional<KGENDType> dtype = getType().getResolvedDType();
  if (!dtype || !getType().getResolvedSize() ||
      getInput().getType().getResolvedSize() != getType().getResolvedSize())
    return {};
  if (dtype->isInt()) {
    return foldSIMDOpResult</*AllowIndex=*/false>(
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
  return foldSIMDOpResult</*AllowIndex=*/false>(
      operands, *dtype, [&](APSInt in) { return APFloat(sem, in); },
      [&](APFloat in) { return APFloat(sem, in.bitcastToAPInt()); });
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerBitcastOp::fold(ArrayRef<Attribute> operands) {
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

OpFoldResult CastOp::fold(ArrayRef<Attribute> operands) {
  auto in = dyn_cast_if_present<SIMDAttr>(operands[0]);
  Optional<KGENDType> dtype = getType().getResolvedDType();
  if (in && dtype) {
    if (dtype->isFloat()) {
      // Cannot fold cast to unsupported float dtype.
      if (!DTypeValue::isValidFloatDType(*dtype))
        return {};
      const llvm::fltSemantics &sem = DTypeValue::getFloatSemantics(*dtype);
      return foldSIMDOpResult</*AllowIndex=*/true>(
          operands, *dtype,
          [&](APSInt in) {
            APFloat fp(sem);
            fp.convertFromAPInt(in, in.isSigned(),
                                APFloat::rmNearestTiesToEven);
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
      return foldSIMDOpResult</*AllowIndex=*/true>(
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
      // Cast to index like it's a 64-bit integer.
      return foldSIMDOpResult</*AllowIndex=*/true>(
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
    return foldSIMDOpResult</*AllowIndex=*/true>(
        operands, *dtype, [&](APSInt in) { return !in.isZero(); },
        [&](APFloat in) { return !in.isZero(); }, [&](bool in) { return in; });
  }

  if (getInput().getType() == getOutput().getType())
    return getInput();
  return {};
}

//===----------------------------------------------------------------------===//
// SIMDExtractElementOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDExtractElementOp::fold(ArrayRef<Attribute> operands) {
  auto vec = dyn_cast_if_present<SIMDAttr>(operands[0]);
  auto idx = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!vec || !idx)
    return {};
  return SIMDAttr::get(vec.getValues()[idx.getInt()], getType());
}

//===----------------------------------------------------------------------===//
// SIMDInsertElementOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDInsertElementOp::fold(ArrayRef<Attribute> operands) {
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
// SIMDSplatOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDSplatOp::fold(ArrayRef<Attribute> operands) {
  Optional<int64_t> size = getType().getResolvedSize();
  auto scalar = dyn_cast_if_present<SIMDAttr>(operands[0]);
  if (!size || !scalar)
    return {};
  SmallVector<DTypeValue> values(*size, scalar.getValues().front());
  return SIMDAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess StoreOp::interpret(ArrayRef<Attribute> operands,
                                  InterpreterState &state,
                                  SmallVectorImpl<OpFoldResult> &results) {
  auto value = llvm::cast_if_present<TypedAttr>(operands[0]);
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[1]);
  if (!value || !ptr)
    return Error("non-constant inputs");

  return state.writeAttributeToMemory(ptr.getAddr(), value);
}

//===----------------------------------------------------------------------===//
// OffsetOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess OffsetOp::interpret(ArrayRef<Attribute> operands,
                                   InterpreterState &state,
                                   SmallVectorImpl<OpFoldResult> &results) {
  auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]);
  auto offset = dyn_cast_or_null<IntegerAttr>(operands[1]);
  if (!ptr || !offset)
    return Error("non-constant inputs");
  Optional<int64_t> elSize =
      DataLayoutInterface::getTypeSizeInBytes(state.getTarget(), ptr.getType());
  if (!elSize)
    return Error("could not query pointer element size");
  results.push_back(PointerAttr::get(ptr.getAddr() + *elSize * offset.getInt(),
                                     ptr.getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess
StackAllocationOp::interpret(ArrayRef<Attribute> operands,
                             InterpreterState &state,
                             SmallVectorImpl<OpFoldResult> &results) {
  auto count = dyn_cast<IntegerAttr>(getCount());
  Type type = cast<PointerType>(getType()).getResolvedElementType();
  if (!count || !type)
    return Error("not concrete");
  Optional<int64_t> size =
      DataLayoutInterface::getTypeSizeInBytes(state.getTarget(), type);
  Optional<int64_t> align =
      DataLayoutInterface::getTypeAlignInBytes(state.getTarget(), type);
  if (!size || !align)
    return Error("could not query type size");
  size_t addr =
      state.allocateMemory(count.getInt() * llvm::alignTo(*size, *align));
  results.push_back(PointerAttr::get(addr, getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// StructConstructOp
//===----------------------------------------------------------------------===//

OpFoldResult StructConstructOp::fold(ArrayRef<Attribute> operands) {
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
// StructGetOp
//===----------------------------------------------------------------------===//

OpFoldResult StructGetOp::fold(ArrayRef<Attribute> operands) {
  auto container = dyn_cast_or_null<StructAttr>(operands[0]);
  if (!container)
    return {};
  return container.getValues()[getIndexAttr().getInt()];
}

//===----------------------------------------------------------------------===//
// StructReplaceOp
//===----------------------------------------------------------------------===//

OpFoldResult StructReplaceOp::fold(ArrayRef<Attribute> operands) {
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

ErrorOrSuccess
POP::StructGEPOp::interpret(ArrayRef<Attribute> operands,
                            InterpreterState &state,
                            SmallVectorImpl<OpFoldResult> &results) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  if (!ptr)
    return Error("non-constant inputs");

  size_t offset = 0;
  auto structType = getContainer().getType().getElementTypeAs<StructType>();

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
  results.push_back(
      PointerAttr::get(ptr.getAddr() + offset, PointerType::get(targetType)));
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayCreateOp::fold(ArrayRef<Attribute> operands) {
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

OpFoldResult ArrayRepeatOp::fold(ArrayRef<Attribute> operands) {
  Optional<int64_t> size = getType().getResolvedSize();
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
  return POP::ArrayAttr::get(llvm::makeArrayRef(values).take_front(*size),
                             getType());
}

//===----------------------------------------------------------------------===//
// ArrayGetOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayGetOp::fold(ArrayRef<Attribute> operands) {
  auto array = dyn_cast_if_present<POP::ArrayAttr>(operands[0]);
  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (!array || !index)
    return {};
  return array.getValues()[index.getInt()];
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayReplaceOp::fold(ArrayRef<Attribute> operands) {
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

ErrorOrSuccess ArrayGEPOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state,
                                     SmallVectorImpl<OpFoldResult> &results) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands[0]);
  auto index = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!ptr || !index)
    return Error("non-constant inputs");

  auto arrayType = getArray().getType().getElementTypeAs<POP::ArrayType>();
  auto dl = cast<DataLayoutInterface>(arrayType.getResolvedElementType());
  size_t addr =
      ptr.getAddr() +
      index.getInt() * (llvm::alignTo(*dl.getTypeSize(state.getTarget()),
                                      *dl.getTypeAlign(state.getTarget())));
  results.push_back(PointerAttr::get(addr, PointerType::get(dl)));
  return success();
}

//===----------------------------------------------------------------------===//
// VariantCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantCreateOp::fold(ArrayRef<Attribute> operands) {
  auto value = llvm::cast_if_present<TypedAttr>(operands[0]);
  if (!value)
    return {};
  return VariantAttr::get(value, getType());
}

//===----------------------------------------------------------------------===//
// VariantIsOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantIsOp::fold(ArrayRef<Attribute> operands) {
  auto variant = dyn_cast_if_present<VariantAttr>(operands[0]);
  if (!variant)
    return {};
  return BoolAttr::get(getContext(),
                       variant.getValue().getType() == getTestType());
}

//===----------------------------------------------------------------------===//
// VariantGetOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantGetOp::fold(ArrayRef<Attribute> operands) {
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
// ListGetOp
//===----------------------------------------------------------------------===//

OpFoldResult ListGetOp::fold(ArrayRef<Attribute> operands) {
  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (!index)
    return {};

  if (auto list = dyn_cast_or_null<ListAttr>(operands[0]))
    return list.getValues()[index.getInt()];

  // Canonicalize `get(create(x)) -> x`.
  if (auto create = getList().getDefiningOp<ListCreateOp>())
    return create.getOperands()[index.getInt()];

  return {};
}

//===----------------------------------------------------------------------===//
// ListCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult ListCreateOp::fold(ArrayRef<Attribute> operands) {
  SmallVector<TypedAttr> values;
  for (Attribute operand : operands) {
    if (!operand)
      return {};
    values.push_back(cast<TypedAttr>(operand));
  }
  return ListAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// IndexToPointerOp
//===----------------------------------------------------------------------===//

OpFoldResult IndexToPointerOp::fold(ArrayRef<Attribute> operands) { return {}; }

//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerToIndexOp::fold(ArrayRef<Attribute> operands) { return {}; }
