//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "KGEN/Interpreter/InterpreterState.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/POPDialect/POPAttrs.h"
#include "KGEN/POPDialect/POPDialect.h"
#include "KGEN/POPDialect/POPEnums.h"
#include "KGEN/POPDialect/POPOps.h"
#include "KGEN/POPDialect/POPTypes.h"
#include "Support/MDialect/MAttrs.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/Compression.h"
#include <unistd.h>

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

/// This enum indicates how index folding should be done.
enum IndexFold {
  kNoIndex,     // no index folding allowed
  kIndexResult, // index operation creates an index
  kOtherResult, // index operation does not create an index
  k64BitResult, // index operation does not create an index and produces 64-bit
                // result
  k32BitResult, // index operation does not create an index and produces 32-bit
                // result
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
      auto unwrap = [](OpResultT value) {
        if constexpr (isOptional)
          return *value;
        else
          return value;
      };

      // For a k32BitResult don't try to fold as 64bit index, as it won't always
      // correct for the result type.
      if constexpr (foldType == k32BitResult) {
        OpResultT result32 = op(args.trunc(32)...);
        if constexpr (isOptional)
          if (!result32.has_value())
            return {};
        return unwrap(result32);
      }

      OpResultT result64 = op(args...);
      if constexpr (isOptional)
        if (!result64.has_value())
          return {};
      if constexpr (foldType == k64BitResult) {
        // Return value that matches the result type
        return unwrap(result64);
      }

      OpResultT result32 = op(args.trunc(32)...);
      if constexpr (isOptional)
        if (!result32.has_value())
          return {};
      if constexpr (foldType == k32BitResult)
        return unwrap(result32);
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
    return ::Detail::foldSIMDOpDType<APSInt>(
        [](const DTypeValue &val) { return val.getIntVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  // FIXME: Should we even do floating point folds? Results don't match hardware
  // and not all float semantics are supported.
  if (inputDType.isFloat())
    return ::Detail::foldSIMDOpDType<APFloat>(
        [](const DTypeValue &val) { return val.getFloatVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  if (inputDType.isBool())
    return ::Detail::foldSIMDOpDType<bool>(
        [](const DTypeValue &val) { return val.getBoolVal(); }, operands,
        resultDType, std::forward<OpFns>(ops)...);
  if (inputDType.isIndex() || inputDType.isAddress())
    return ::Detail::foldSIMDOpIndex<indexFoldType>(
        operands, resultDType, std::forward<OpFns>(ops)...);
  llvm_unreachable("unhandled dtype");
}
} // namespace Detail

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible operand dtype given a result dtype.
template <::Detail::IndexFold indexFoldType, typename... OpFns>
static SIMDAttr foldSIMDOpResult(ArrayRef<Attribute> operands,
                                 KGENDType resultDType, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  return ::Detail::foldSIMDOp<indexFoldType>(
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
  return ::Detail::foldSIMDOp<::Detail::kIndexResult>(
      operands, dtype, dtype, std::forward<OpFns>(ops)...);
}

//===----------------------------------------------------------------------===//
// Arithmetic Operation Folders
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Unary Operations

OpFoldResult NegOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt val) { return -val; },
      [](APFloat val) { return llvm::neg(val); });
}

//===----------------------------------------------------------------------===//
// Binary Operations

// Check if the input is an integer constant and return it.
// In case of a SIMD input, check that all values are equal.
static std::optional<APSInt> getIntVal(Value val) {
  SIMDAttr constAttr;
  if (!mlir::matchPattern(val, mlir::m_Constant(&constAttr)))
    return std::nullopt;
  const APSInt &constVal = constAttr.getValues().front().getIntVal();
  if (!llvm::all_of(constAttr.getValues(), [&](const DTypeValue &val) {
        return val.getIntVal() == constVal;
      }))
    return std::nullopt;
  return constVal;
}

template <typename OpT>
static bool hasIntLikeType(OpT op) {
  std::optional<KGENDType> dtype = op->getType().getResolvedDType();
  return dtype && dtype->isIntLike();
}

static bool isIntZero(Value val) {
  std::optional<APSInt> maybeConst = getIntVal(val);
  return maybeConst && maybeConst->isZero();
};

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  if (SIMDAttr const &res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) { return lhs + rhs; },
          [](APFloat lhs, APFloat rhs) { return lhs + rhs; })) {
    return res;
  }
  // integer X+0 or 0+X -> X
  //
  // for floating-point types that have negative zero, this optimization is
  // not valid because -0 + 0 = 0
  // TODO: this optimization can be done for fp types
  // if we add a 'fast fp math' or 'ignore negative 0' config parameter.
  if (hasIntLikeType(this)) {
    if (isIntZero(getLhs()))
      return getRhs();
    if (isIntZero(getRhs()))
      return getLhs();
  }
  return {};
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  if (SIMDAttr const &res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) { return lhs - rhs; },
          [](APFloat lhs, APFloat rhs) { return lhs - rhs; })) {
    return res;
  }
  // X-0 -> X
  // Note that unlike the 'add' case above, this optimization
  // is valid for floating-point types as well, because -0 - 0 = -0
  // TODO: generalize to support floating-point types.
  if (hasIntLikeType(this)) {
    if (isIntZero(getRhs()))
      return getLhs();
  }
  return {};
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  if (auto res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) { return lhs * rhs; },
          [](APFloat lhs, APFloat rhs) { return lhs * rhs; }))
    return res;

  if (!hasIntLikeType(this))
    return {};

  // Pattern-match trivial cases, such as 0*x or 1*x. Support both scalar and
  // SIMD types.
  auto foldTrivialMultiplication = [&](Value lhs, Value rhs) -> OpFoldResult {
    if (auto maybeVal = getIntVal(lhs)) {
      auto constVal = maybeVal.value();
      if (constVal.isZero())
        return lhs;
      if (constVal.isOne())
        return rhs;
    }
    return {};
  };

  // Try to fold trivial multiplication expecting a constant operand in lhs.
  // For example, 0*x = 0
  if (auto res = foldTrivialMultiplication(getLhs(), getRhs()))
    return res;

  // Otherwise, swap operands and try again. This will help to fold trivial
  // multiplication such as x*0 = 0
  if (auto res = foldTrivialMultiplication(getRhs(), getLhs()))
    return res;

  return {};
}

LogicalResult DivOp::canonicalize(DivOp op, PatternRewriter &b) {
  std::optional<KGEN::KGENDType> dtype = op.getType().getResolvedDType();
  if (!dtype)
    return b.notifyMatchFailure(op, "result type isn't resolved");

  if (!dtype->isIntLike())
    return b.notifyMatchFailure(op, "result type isn't int-like");

  std::optional<size_t> size = op.getType().getResolvedSize();
  if (!size)
    return b.notifyMatchFailure(op, "result type size isn't resolved");

  // Canonicalize "x / 2^n" into "x >> n"
  SIMDAttr rhsAttr;
  if (!mlir::matchPattern(op.getRhs(), mlir::m_Constant(&rhsAttr)))
    return b.notifyMatchFailure(op, "rhs is not a constant");

  if (!llvm::all_of(rhsAttr.getValues(), [&](const DTypeValue &val) {
        APInt intVal = val.getIntVal();
        return intVal.isStrictlyPositive() && intVal.isPowerOf2();
      })) {
    return b.notifyMatchFailure(op, "rhs values are not positive power of 2");
  }

  ssize_t intWidth = dtype->getWidthInBits();
  if (dtype->isIndex()) {
    TargetInfoAttr target = lookupTargetInfo(op);
    intWidth = target ? target.resolveIndexBitWidth() : 64;
  }
  assert(intWidth > 0 && "Could not determine size of an integer");

  SmallVector<DTypeValue> values;
  values.reserve(*size);
  for (size_t i = 0, e = *size; i < e; ++i) {
    APInt intVal = rhsAttr.getValues()[i].getIntVal();
    values.push_back(DTypeValue(
        APInt(intWidth, intVal.logBase2(), dtype->isSInt()), *dtype));
  }

  b.replaceOpWithNewOp<ShrOp>(
      op, op.getType(), op.getLhs(),
      b.create<ParamConstantOp>(op.getLoc(),
                                SIMDAttr::get(values, op.getType())));

  return success();
}

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(),
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
  return foldSIMDOp(
      adaptor.getOperands(),
      [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
        if (rhs.isZero())
          return std::nullopt;
        return lhs % rhs;
      },
      [](APFloat lhs, APFloat rhs) -> std::optional<APFloat> {
        if (rhs.isZero())
          return std::nullopt;
        (void)lhs.mod(rhs);
        return lhs;
      });
}

template <typename OpT>
static bool hasEqualOperands(OpT op) {
  return op->getLhs() == op->getRhs();
}

OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  if (SIMDAttr const &res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) -> APSInt {
            return lhs > rhs ? lhs : rhs;
          },
          [](APFloat lhs, APFloat rhs) -> APFloat {
            return llvm::maxnum(lhs, rhs);
          },
          [](bool lhs, bool rhs) -> bool { return lhs | rhs; })) {
    return res;
  }
  if (hasEqualOperands(this)) {
    return getLhs();
  }
  return {};
}

OpFoldResult MinOp::fold(FoldAdaptor adaptor) {
  if (SIMDAttr const &res = foldSIMDOp(
          adaptor.getOperands(),
          [](APSInt lhs, APSInt rhs) -> APSInt {
            return lhs < rhs ? lhs : rhs;
          },
          [](APFloat lhs, APFloat rhs) -> APFloat {
            return llvm::minnum(lhs, rhs);
          },
          [](bool lhs, bool rhs) -> bool { return lhs & rhs; })) {
    return res;
  }
  if (hasEqualOperands(this)) {
    return getLhs();
  }
  return {};
}

OpFoldResult ShlOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(adaptor.getOperands(),
                    [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
                      if (rhs.uge(lhs.getBitWidth()))
                        return std::nullopt;
                      return APSInt(lhs.shl(rhs), lhs.isSigned());
                    });
}

OpFoldResult ShrOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(adaptor.getOperands(),
                    [](APSInt lhs, APSInt rhs) -> std::optional<APSInt> {
                      if (rhs.uge(lhs.getBitWidth()))
                        return std::nullopt;
                      return APSInt(lhs.isSigned() ? lhs.ashr(rhs)
                                                   : lhs.lshr(rhs),
                                    lhs.isSigned());
                    });
}

//===----------------------------------------------------------------------===//
// Ternary Operations

OpFoldResult FMAOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(),
      [](APSInt a, APSInt b, APSInt c) { return a * b + c; },
      [](APFloat a, APFloat b, APFloat c) {
        (void)a.fusedMultiplyAdd(b, c, APFloat::rmNearestTiesToEven);
        return a;
      });
}

//===----------------------------------------------------------------------===//
// LoadOp
//===----------------------------------------------------------------------===//

/// We can fold loads of `pop.global_constant` ops.
OpFoldResult LoadOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getOrdering() != AtomicOrdering::NOT_ATOMIC) {
    // Don't fold atomic loads.
    return {};
  }

  Operation *parent = getPtr().getDefiningOp();
  if (!parent)
    return {};

  // `load(global_constant())` is a load of the whole value.
  if (auto cst = dyn_cast<GlobalConstantOp>(parent))
    return cst.getValue();

  auto findValueAt = [&](GlobalConstantOp cst, uint64_t idx) -> OpFoldResult {
    auto attr = dyn_cast<POP::ArrayAttr>(cst.getValue());
    if (!attr || idx >= attr.getValues().size() ||
        attr.getType().getElementType() != getType())
      return {};
    return attr.getValues()[idx];
  };

  auto findOffsetValueAt = [&](GlobalConstantOp cst,
                               Value offset) -> OpFoldResult {
    APInt idx;
    if (!mlir::matchPattern(offset, mlir::m_ConstantInt(&idx)) ||
        idx.isNegative())
      return {};
    return findValueAt(cst, idx.getLimitedValue());
  };

  // `load(gep(global_constant()))` is a load of a specific element, if the gep
  // index is a constant.
  if (auto gep = dyn_cast<ArrayGEPOp>(parent)) {
    if (auto cst = gep.getArray().getDefiningOp<GlobalConstantOp>())
      return findOffsetValueAt(cst, gep.getIndex());
    return {};
  }

  // `load(offset(bitcast(global_constant())))` where the offset index is known.
  if (auto offset = dyn_cast<OffsetOp>(parent)) {
    if (auto bitcast = offset.getPtr().getDefiningOp<PointerBitcastOp>())
      if (auto cst = bitcast.getInput().getDefiningOp<GlobalConstantOp>())
        return findOffsetValueAt(cst, offset.getIndex());
    return {};
  }

  // `load(bitcast(global_constant())` where the element types are equal is a
  // load of the first element.
  if (auto bitcast = dyn_cast<PointerBitcastOp>(parent)) {
    if (auto cst = bitcast.getInput().getDefiningOp<GlobalConstantOp>())
      return findValueAt(cst, 0);
    return {};
  }

  return {};
}

LogicalResult LoadOp::canonicalize(LoadOp op, PatternRewriter &b) {
  if (op.getOrdering() != AtomicOrdering::NOT_ATOMIC) {
    // Don't canonicalize atomic loads.
    return failure();
  }

  // Canonicalize "store x -> ptr; tmp = load ptr" into "store; tmp = x".
  if (auto store = dyn_cast_if_present<StoreOp>(op->getPrevNode())) {
    if ((store.getPtr() == op.getPtr()) &&
        (store.getOrdering() == AtomicOrdering::NOT_ATOMIC)) {
      b.replaceOp(op, store.getArg());
      return success();
    }
  }

  // Canonicalize `load(bitcast(ptr)) -> bitcast(load(ptr))` if the element type
  // is also a pointer.
  if (!isa<PointerType>(op.getType()))
    return b.notifyMatchFailure(op.getLoc(), "element type is not a pointer");
  auto bitcast = op.getPtr().getDefiningOp<PointerBitcastOp>();
  if (!bitcast || !bitcast->hasOneUse())
    return b.notifyMatchFailure(op.getLoc(), "pointer is not a bitcast");
  Value ptr = bitcast.getInput();
  auto ptrType = dyn_cast<PointerType>(ptr.getType());
  if (!ptrType || !isa<PointerType>(ptrType.getElementType()))
    return b.notifyMatchFailure(op.getLoc(), "bitcast input is not a pointer");

  // Rewrite the load in-place.
  b.setInsertionPointAfter(op);
  auto newBitcast = b.create<PointerBitcastOp>(op.getLoc(), op.getType(), op);
  b.modifyOpInPlace(op, [&] {
    op.setOperand(ptr);
    Value(op).setType(ptrType.getElementType());
  });
  b.replaceAllUsesExcept(op, newBitcast, newBitcast);
  return success();
}

ErrorTreeOrSuccess LoadOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state) {
  ErrorOr<Attribute> result =
      state.readAttributeFromPointer(operands[0], getType());
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
  llvm_unreachable("invalid cmp predicate");
}

OpFoldResult CmpOp::fold(FoldAdaptor adaptor) {
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
  if (OpFoldResult result = foldSIMDOpResult<::Detail::kOtherResult>(
          adaptor.getOperands(), KGENDType::kBool,
          [&](APSInt lhs, APSInt rhs) {
            return compareConstants(getPred(), lhs, rhs);
          },
          [&](APFloat lhs, APFloat rhs) {
            return compareConstants(getPred(), lhs, rhs);
          },
          [&](bool lhs, bool rhs) {
            return compareConstants(getPred(), lhs, rhs);
          }))
    return result;

  // Fold `eq(true, x) -> x` and `ne(false, x) -> x`.
  if (operandTy && operandTy == DType::kBool &&
      llvm::is_contained({CmpPredicate::EQ, CmpPredicate::NE}, getPred())) {
    // Only one input will be constant.
    auto lhs = dyn_cast_or_null<SIMDAttr>(adaptor.getLhs());
    auto rhs = dyn_cast_or_null<SIMDAttr>(adaptor.getRhs());
    assert(!(lhs && rhs) && "constant case should be handled");
    // If `rhs` contains the constant, move it to `lhs`.
    if (rhs)
      lhs = rhs;
    // Check that the constant is either all true or all false elements, then
    // match `eq` with `true` or `ne` with `false`.
    if (lhs && llvm::all_equal(lhs.getValues()) &&
        (getPred() == CmpPredicate::EQ) == lhs.getValues().front().getBoolVal())
      return rhs ? getLhs() : getRhs();
  }

  // Fold
  // * `gt(0, unsigned_val)` into false
  // * `le(0, unsigned_val)` into true
  // * `ge(unsigned_val, 0)` into true
  // * `lt(unsigned_val, 0)` into false
  if (operandTy && operandTy->isUInt()) {
    auto foldUnsignedCmp = [&](CmpPredicate foldIntoTrue,
                               CmpPredicate foldIntoFalse, CmpPredicate pred,
                               Attribute op1, Attribute op2) -> OpFoldResult {
      if (!llvm::is_contained({foldIntoTrue, foldIntoFalse}, pred))
        return {};

      // Only one input will be constant.
      [[maybe_unused]] auto op1SimdAttr = dyn_cast_or_null<SIMDAttr>(op1);
      auto op2SimdAttr = dyn_cast_or_null<SIMDAttr>(op2);
      assert(!(op1SimdAttr && op2SimdAttr) &&
             "constant case should be handled");

      if (!op2SimdAttr) {
        // Always expect constant value in op2
        return {};
      }

      // If the `op2SimdAttr` is constant and zero, then we can simplify the
      // comparison.
      if (llvm::all_equal(op2SimdAttr.getValues()) &&
          op2SimdAttr.getValues()[0].getData().isZero()) {

        SmallVector<DTypeValue> values(
            *size, DTypeValue(pred == foldIntoTrue, KGENDType::kBool));
        return SIMDAttr::get(values, getType());
      }
      return {};
    };

    if (OpFoldResult res =
            foldUnsignedCmp(CmpPredicate::LE, CmpPredicate::GT, getPred(),
                            adaptor.getRhs(), adaptor.getLhs()))
      return res;
    if (OpFoldResult res =
            foldUnsignedCmp(CmpPredicate::GE, CmpPredicate::LT, getPred(),
                            adaptor.getLhs(), adaptor.getRhs()))
      return res;
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Bool Operation Folders
//===----------------------------------------------------------------------===//

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<BoolAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<BoolAttr>(adaptor.getRhs());
  if (lhs && rhs)
    return BoolAttr::get(getContext(), lhs.getValue() && rhs.getValue());

  // Commutative operation, constant operands are pushed to the end.
  if (rhs) {
    // lhs && true == lhs
    if (rhs.getValue())
      return getLhs();

    // lhs && false == false
    return BoolAttr::get(getContext(), false);
  }
  return {};
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<BoolAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<BoolAttr>(adaptor.getRhs());
  if (lhs && rhs)
    return BoolAttr::get(getContext(), lhs.getValue() || rhs.getValue());

  // Commutative operation, constant operands are pushed to the end.
  if (rhs) {
    // lhs || false == lhs
    if (!rhs.getValue())
      return getLhs();

    // lhs || true == true
    return BoolAttr::get(getContext(), true);
  }
  return {};
}

OpFoldResult XOrOp::fold(FoldAdaptor adaptor) {
  auto lhs = dyn_cast_or_null<BoolAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_or_null<BoolAttr>(adaptor.getRhs());

  if (lhs && rhs)
    return BoolAttr::get(getContext(), lhs.getValue() ^ rhs.getValue());

  if (rhs) {
    // `xor(x, 0)` -> `x`.
    if (!rhs.getValue())
      return getLhs();

    // `xor(xor(x, 1), 1) -> x`.
    auto xorOp = getLhs().getDefiningOp<XOrOp>();
    if (xorOp && mlir::matchPattern(xorOp.getRhs(), mlir::m_One()))
      return xorOp.getLhs();
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Bitwise Operation Folders
//===----------------------------------------------------------------------===//

OpFoldResult SIMDAndOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt lhs, APSInt rhs) { return lhs & rhs; },
      [](bool lhs, bool rhs) { return lhs && rhs; });
}

OpFoldResult SIMDOrOp::fold(FoldAdaptor adaptor) {
  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt lhs, APSInt rhs) { return lhs | rhs; },
      [](bool lhs, bool rhs) { return lhs || rhs; });
}

OpFoldResult SIMDXOrOp::fold(FoldAdaptor adaptor) {
  SIMDAttr attr;
  if (mlir::matchPattern(getRhs(), mlir::m_Constant(&attr))) {
    // `xor(x, 0)` -> `x`.
    if (llvm::all_of(attr.getValues(), [](const DTypeValue &value) {
          return value.getData().isZero();
        }))
      return getLhs();

    // `xor(xor(x, 1), 1) -> x`.
    auto pred =
        getType().getResolvedDType() == DType::kBool
            ? [](const DTypeValue &value) { return value.getBoolVal(); }
            : [](const DTypeValue &value) {
                return value.getData().isMask(value.getData().getBitWidth());
              };
    if (llvm::all_of(attr.getValues(), pred)) {
      auto xorOp = getLhs().getDefiningOp<SIMDXOrOp>();
      if (xorOp && xorOp.getRhs() == getRhs())
        return xorOp.getLhs();
    }
  }

  return foldSIMDOp(
      adaptor.getOperands(), [](APSInt lhs, APSInt rhs) { return lhs ^ rhs; },
      [](bool lhs, bool rhs) -> bool { return lhs ^ rhs; });
}

//===----------------------------------------------------------------------===//
// BitcastOp
//===----------------------------------------------------------------------===//

// Unlike other places, invoking foldSIMDOpResult with kOtherResult will
// require that 32 and 64 bit representation are the same, which is not needed
// to bitcast index to some other type.
// The helper function simply uses appropriate IndexFold type depending on a
// index's size within AS or calls foldSIMDOpResult if input type is not an
// index.
template <typename... OpsFns>
static OpFoldResult bitcastSIMDIndex(ArrayRef<Attribute> operands,
                                     KGENDType inputDType,
                                     KGENDType outputDType,
                                     TargetInfoAttr target, OpsFns &&...ops) {
  if (inputDType.isIndex()) {
    ssize_t indexWidth = target ? target.resolveIndexBitWidth() : 64;
    if (indexWidth == 64) {
      return foldSIMDOpResult<::Detail::k64BitResult>(
          operands, outputDType, std::forward<OpsFns>(ops)...);
    }
    if (indexWidth == 32) {
      return foldSIMDOpResult<::Detail::k32BitResult>(
          operands, outputDType, std::forward<OpsFns>(ops)...);
    }
    return {};
  }
  return foldSIMDOpResult<::Detail::kNoIndex>(operands, outputDType,
                                              std::forward<OpsFns>(ops)...);
}

OpFoldResult BitcastOp::fold(FoldAdaptor adaptor) {
  // Don't fold if the size changes. This requires knowing the endianness of the
  // target.
  std::optional<KGENDType> dtype = getType().getResolvedDType();
  std::optional<KGENDType> inputDType = getInput().getType().getResolvedDType();
  if (!dtype || !inputDType || !getType().getResolvedSize() ||
      getInput().getType().getResolvedSize() != getType().getResolvedSize())
    return {};
  if (inputDType->isBool() ||
      dtype->isBool()) // Modeling bool bitcast requires packing.
    return {};

  TargetInfoAttr target = lookupTargetInfo(*this);
  if (dtype->isInt()) {
    return bitcastSIMDIndex(
        adaptor.getOperands(), *inputDType, *dtype, target,
        [&](const APSInt &in) { return APSInt(in, dtype->isUInt()); },
        [&](const APFloat &in) {
          return APSInt(in.bitcastToAPInt(), dtype->isUInt());
        });
  }
  if (dtype->isIndex()) {
    return foldSIMDOpResult<::Detail::kOtherResult>(
        adaptor.getOperands(), *dtype,
        [&](const APSInt &in) -> APSInt {
          // Must zero extend to 64bit, otherwise there will be segfault during
          // SIMDAttr construction as it expects 64bit index by default.
          return APSInt(in, /*isUnsigned=*/true).extend(64);
        },
        [&](const APFloat &in) -> APSInt {
          // Must zero extend to 64bit, otherwise there will be segfault during
          // SIMDAttr construction as it expects 64bit index by default.
          return APSInt(in.bitcastToAPInt(), /*isUnsigned=*/true).extend(64);
        });
  }
  assert(dtype->isFloat());
  // Check to make sure we have a supported float dtype.
  const llvm::fltSemantics *sem = dtype->getFloatSemantics();
  if (!sem)
    return {};
  return bitcastSIMDIndex(
      adaptor.getOperands(), *inputDType, *dtype, target,
      [&](const APSInt &in) { return APFloat(*sem, in); },
      [&](const APFloat &in) { return APFloat(*sem, in.bitcastToAPInt()); });
}

//===----------------------------------------------------------------------===//
// PointerBitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerBitcastOp::fold(FoldAdaptor adaptor) {
  if (auto ptr = dyn_cast_or_null<PointerAttr>(adaptor.getInput()))
    return PointerAttr::get(ptr.getAddr(), getType());

  auto cast = getInput().getDefiningOp<PointerBitcastOp>();
  if (cast && cast.getInput().getType() == getType())
    return cast.getInput();
  return {};
}

/// Canonicalize `bitcast(bitcast(x)) -> bitcast(x)`, only if the intermediate
/// bitcast has one use.
LogicalResult PointerBitcastOp::canonicalize(PointerBitcastOp op,
                                             PatternRewriter &b) {
  auto cast = op.getInput().getDefiningOp<PointerBitcastOp>();
  if (!cast)
    return b.notifyMatchFailure(op.getLoc(), "not a bitcast of a bitcast");
  if (!cast->hasOneUse())
    return b.notifyMatchFailure(op.getLoc(),
                                "intermediate cast has multiple uses");
  b.replaceOpWithNewOp<PointerBitcastOp>(op, op.getType(), cast.getInput());
  // Erase the intermediate cast -- its only use has been removed.
  b.eraseOp(cast);
  return success();
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  auto in = dyn_cast_if_present<SIMDAttr>(adaptor.getInput());
  std::optional<KGENDType> dtype = getType().getResolvedDType();
  if (!in || !dtype) {
    if (getInput().getType() == getOutput().getType())
      return getInput();
    return {};
  }

  std::optional<KGENDType> inType = in.getType().getResolvedDType();

  // Exit early if the input and output dtypes are the same.
  if (*dtype == *inType)
    return in;

  if (dtype->isFloat()) {
    // Cannot fold cast to unsupported float dtype.
    const llvm::fltSemantics *sem = dtype->getFloatSemantics();
    if (!sem)
      return {};
    return foldSIMDOpResult<::Detail::kOtherResult>(
        adaptor.getOperands(), *dtype,
        [&](const APSInt &in) -> APFloat {
          APFloat fp(*sem);
          fp.convertFromAPInt(in, in.isSigned(), APFloat::rmNearestTiesToEven);
          return fp;
        },
        [&](APFloat in) {
          bool ignored;
          in.convert(*sem, APFloat::rmNearestTiesToEven, &ignored);
          return in;
        },
        [&](bool in) { return APFloat(*sem, in); });
  }
  if (dtype->isInt()) {
    // Note that float to integer casts are undefined if the float value is
    // too large to fit in the integer dtype.
    unsigned width = dtype->getIntegerWidthInBits();
    return foldSIMDOpResult<::Detail::kOtherResult>(
        adaptor.getOperands(), *dtype,
        [&](const APSInt &in) -> APSInt { return in.extOrTrunc(width); },
        [&](const APFloat &in) -> std::optional<APSInt> {
          APSInt iv(width, dtype->isUInt());
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv;
        },
        [&](bool in) { return APSInt(APInt(width, in), dtype->isUInt()); });
  }
  if (dtype->isIndex() || dtype->isAddress()) {
    // Cast to index like it's a 64-bit integer. Address is handled like index.
    return foldSIMDOpResult<::Detail::kOtherResult>(
        adaptor.getOperands(), *dtype,
        [inType](const APSInt &in) -> int64_t {
          return inType->isSInt() ? in.getSExtValue() : in.getZExtValue();
        },
        [](const APFloat &in) -> std::optional<int64_t> {
          APSInt iv(64, /*isUnsigned=*/false);
          bool ignored;
          if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
              APFloat::opInvalidOp)
            return {};
          return iv.getSExtValue();
        },
        [](bool in) { return static_cast<int64_t>(in); });
  }
  if (dtype->isInvalid()) {
    // Invalid is not inhabitable.
    return {};
  }
  assert(dtype->isBool());
  return foldSIMDOpResult<::Detail::kOtherResult>(
      adaptor.getOperands(), *dtype,
      [](const APSInt &in) -> bool { return !in.isZero(); },
      [](const APFloat &in) -> bool { return !in.isZero(); });
}

/// Canonicalize integer type `cast(cast(x : T1 to T2) : T3) -> cast(T1 to T3)`,
/// when second cast discards the result of the first cast.
LogicalResult CastOp::canonicalize(CastOp op, PatternRewriter &b) {
  auto cast = op.getInput().getDefiningOp<CastOp>();
  if (!cast)
    return b.notifyMatchFailure(op.getLoc(), "not a cast of a cast");
  if (!cast->hasOneUse())
    return b.notifyMatchFailure(op.getLoc(),
                                "intermediate cast has multiple uses");

  auto inType = cast.getType().getResolvedDType();
  auto outType = op.getType().getResolvedDType();
  auto intermediateType = cast.getInput().getType().getResolvedDType();

  auto isUnsupportedType = [](auto t) {
    return !t || (!t->isIntLike() && (t->isComplex() || !t->isFloat()));
  };

  Location loc = op.getLoc();
  // Both cast should convert to/from integer-like or floating point types.
  if (isUnsupportedType(inType) || isUnsupportedType(outType) ||
      isUnsupportedType(intermediateType) ||
      inType->isIntLike() != outType->isIntLike() ||
      inType->isIntLike() != intermediateType->isIntLike())
    return b.notifyMatchFailure(loc, "not all types are known or supported");

  auto getWidthInBits = [&](KGENDType type) -> ssize_t {
    if (ssize_t width = type.getWidthInBits(); width != -1)
      return width;
    if (!type.isIndex())
      return -1;
    TargetInfoAttr targetInfo = lookupTargetInfo(op);
    if (!targetInfo)
      return -1;
    return targetInfo.resolveIndexBitWidth();
  };

  ssize_t inWidth = getWidthInBits(*inType);
  ssize_t outWidth = getWidthInBits(*outType);
  ssize_t intermediateWidth = getWidthInBits(*intermediateType);

  if (inWidth == -1 || outWidth == -1 || intermediateWidth == -1)
    return b.notifyMatchFailure(loc, "bitwidths of types are unknown");

  if (outWidth < inWidth) {
    // Except for floating point, intermediate cast is redundant if the final
    // cast truncates its result.
    // For the floating point allows fptrunc(fpext)
    if (outWidth > intermediateWidth && intermediateType->isIntLike()) {
      return b.notifyMatchFailure(loc,
                                  "intermediate truncation affects result");
    }
  } else if (outWidth > inWidth) {
    // Final cast converts input to wider type. Possible to optimize:
    //  - zext(zext)
    //  - sext(sext)
    //  - fpext(fpext)
    if (inWidth < intermediateWidth ||
        (inType->isIntLike() && inType->isSInt() != outType->isSInt())) {
      return b.notifyMatchFailure(loc, "intermediate extension affects result");
    }
  } else {
    // Final cast converts either index to/from integer or index/integer to/from
    // floating point of the same width. Possible to optimize:
    // - fptosi(fpext)
    // - fptoui(fpext)
    // - uitofp(zext)
    // - sitofp(sext)
    if (inWidth < intermediateWidth ||
        intermediateType->isIntLike() != inType->isIntLike()) {
      return b.notifyMatchFailure(loc, "intermediate extension affects result");
    }

    // Final cast converts integer to/from integer of a different sign.
    if (inType->isInt() && !outType->isIndex() &&
        inType->isSInt() != outType->isSInt()) {
      return b.notifyMatchFailure(loc, "intermediate extension affects result");
    }
  }

  b.replaceOpWithNewOp<CastOp>(op, op.getType(), cast.getInput());
  // Erase the intermediate cast -- its only use has been removed.
  b.eraseOp(cast);
  return success();
}

ErrorTreeOrSuccess CastOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state) {
  // First try to fold the cast. If that fails, fallback to special cases.
  if (auto result = fold(operands)) {
    state.mapResults(cast<Attribute>(result));
    return success();
  }

  auto in = dyn_cast_if_present<SIMDAttr>(operands[0]);
  std::optional<KGENDType> dtype = getType().getResolvedDType();
  if (!in || !dtype)
    return ErrorTree(getLoc(), "types must be known at this point");

  if (!in.getType().getResolvedDType()->isIndex() ||
      dtype->getIntegerWidthInBits() != 64)
    return ErrorTree(getLoc(), "not implemented");

  // A special case when the input is index type and output is 64-bit integer.
  // Currently, it's only one known case when folder can fail that makes
  // interpreter unhappy.
  unsigned ptrWidth =
      state.getTarget() ? state.getTarget().resolveIndexBitWidth() : 64;
  unsigned width = 64;
  auto res = foldSIMDOpResult<::Detail::k64BitResult>(
      operands, *dtype,
      [&](const APSInt &in) -> APSInt {
        // First extend or truncate to pointer width and only after that to
        // 64-bit integer
        return in.extOrTrunc(ptrWidth).extOrTrunc(width);
      },
      [&](const APFloat &in) -> std::optional<APSInt> {
        APSInt iv(width, dtype->isUInt());
        bool ignored;
        if (in.convertToInteger(iv, APFloat::rmTowardZero, &ignored) ==
            APFloat::opInvalidOp)
          return {};
        return iv;
      },
      [&](bool in) { return APSInt(APInt(width, in), dtype->isUInt()); });
  state.mapResults(res);
  return success();
}

//===----------------------------------------------------------------------===//
// SIMDExtractElementOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDExtractElementOp::fold(FoldAdaptor adaptor) {
  // Extracting from a scalar is always going to return the scalar.
  if (getVector().getType().isScalar()) {
    if (Attribute attr = adaptor.getVector())
      return attr;
    return getVector();
  }

  auto vec = dyn_cast_if_present<SIMDAttr>(adaptor.getVector());
  auto idx = dyn_cast_if_present<IntegerAttr>(adaptor.getPosition());
  if (!vec || !idx)
    return {};
  return SIMDAttr::get(vec.getValues()[idx.getInt()], getType());
}

//===----------------------------------------------------------------------===//
// SIMDInsertElementOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDInsertElementOp::fold(FoldAdaptor adaptor) {
  auto vec = dyn_cast_if_present<SIMDAttr>(adaptor.getVector());

  // Treat insert into undef as being an insert into zero.
  if (!vec) {
    if (auto vecCst = adaptor.getVector())
      if (isa<UninitMemAttr, UnknownAttr>(vecCst))
        vec = SIMDAttr::getZeroValue(getType());
  }

  auto val = dyn_cast_if_present<SIMDAttr>(adaptor.getValue());
  auto idx = dyn_cast_if_present<IntegerAttr>(adaptor.getPosition());
  if (!vec || !val || !idx)
    return {};
  SmallVector<DTypeValue> values(vec.getValues());
  values[idx.getInt()] = val.getValues().front();
  return SIMDAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// SIMDSelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDSelectOp::fold(FoldAdaptor adaptor) {
  auto condVals = dyn_cast_or_null<SIMDAttr>(adaptor.getCondition());
  auto trueVals = dyn_cast_or_null<SIMDAttr>(adaptor.getTrueValue());
  auto falseVals = dyn_cast_or_null<SIMDAttr>(adaptor.getFalseValue());
  if (condVals && trueVals && falseVals) {
    SmallVector<DTypeValue> results;
    for (auto [cond, trueVal, falseVal] : llvm::zip(
             condVals.getValues(), trueVals.getValues(), falseVals.getValues()))
      results.push_back(cond.getBoolVal() ? trueVal : falseVal);
    return SIMDAttr::get(results, getType());
  }

  // Fold `select(x, y, y) -> y`.
  if (getTrueValue() == getFalseValue())
    return getTrueValue();

  // Check if all the values are true or false then fold to either of the
  // operands in that case.
  if (condVals) {
    bool allTrue = true, allFalse = true;
    for (auto cond : condVals.getValues()) {
      if (cond.getBoolVal())
        allFalse = false;
      else
        allTrue = false;
    }

    // Fold `select(true, x, y) -> x`
    if (allTrue)
      return getTrueValue();

    // Fold `select(false, x, y) -> y`
    if (allFalse)
      return getFalseValue();
  }

  // Fold `select(x, true, false) -> x`.
  if (getType().getResolvedDType() == KGENDType::kBool && trueVals &&
      falseVals) {
    if (llvm::all_of(
            trueVals.getValues(),
            [](const DTypeValue &value) { return value.getBoolVal(); }) &&
        llvm::all_of(falseVals.getValues(), [](const DTypeValue &value) {
          return !value.getBoolVal();
        }))
      return getCondition();
  }

  return {};
}

/// Canonicalize `select(x, false, true) -> not(x)`.
LogicalResult SIMDSelectOp::canonicalize(SIMDSelectOp op, PatternRewriter &b) {
  if (op.getType().getResolvedDType() != KGENDType::kBool)
    return b.notifyMatchFailure(op.getLoc(), "not bool dtype");

  SIMDAttr trueVals, falseVals;
  if (!mlir::matchPattern(op.getTrueValue(), mlir::m_Constant(&trueVals)) ||
      !mlir::matchPattern(op.getFalseValue(), mlir::m_Constant(&falseVals)))
    return b.notifyMatchFailure(op.getLoc(), "values are not constants");

  if (!llvm::all_of(
          trueVals.getValues(),
          [](const DTypeValue &value) { return !value.getBoolVal(); }) ||
      !llvm::all_of(falseVals.getValues(),
                    [](const DTypeValue &value) { return value.getBoolVal(); }))
    return b.notifyMatchFailure(
        op.getLoc(), "values are not 'false' and 'true' respectively");

  // The pattern has matched. Re-use the 'true' constant.
  b.replaceOpWithNewOp<SIMDXOrOp>(op, op.getCondition(), op.getFalseValue());
  return success();
}

//===----------------------------------------------------------------------===//
// SIMDShuffleOp
//===----------------------------------------------------------------------===//

OpFoldResult SIMDShuffleOp::fold(FoldAdaptor adaptor) {
  std::optional<int64_t> size = getType().getResolvedSize();
  auto lhs = dyn_cast_if_present<SIMDAttr>(adaptor.getLhs());
  auto rhs = dyn_cast_if_present<SIMDAttr>(adaptor.getRhs());
  auto mask = dyn_cast_if_present<ArrayAttr>(getMaskAttr());
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
  std::optional<int64_t> size = getType().getResolvedSize();

  if (size == 1) {
    if (Attribute scalar = adaptor.getScalar())
      return scalar;
    return getScalar();
  }

  auto scalar = dyn_cast_if_present<SIMDAttr>(adaptor.getScalar());
  if (!size || !scalar)
    return {};
  SmallVector<DTypeValue> values(*size, scalar.getValues().front());
  return SIMDAttr::get(values, getType());
}

//===----------------------------------------------------------------------===//
// StoreOp
//===----------------------------------------------------------------------===//

LogicalResult StoreOp::canonicalize(StoreOp op, PatternRewriter &b) {
  if (op.getOrdering() != AtomicOrdering::NOT_ATOMIC) {
    // Don't canonicalize atomic stores.
    return failure();
  }

  // Storing an unknown to a pointer is a nop as it is legal to assume the
  // memory is already the same value.
  if (auto cst = dyn_cast_or_null<KGEN::ParamConstantOp>(
          op.getArg().getDefiningOp())) {
    if (isa<UninitMemAttr>(cst.getValue())) {
      b.eraseOp(op);
      return success();
    }
  }

  // Canonicalize `store x, bitcast(ptr) -> store bitcast(x), ptr` if the
  // element type is a pointer type.
  if (!isa<PointerType>(op.getArg().getType()))
    return b.notifyMatchFailure(op.getLoc(), "arg is not a pointer");
  auto bitcast = op.getPtr().getDefiningOp<PointerBitcastOp>();
  if (!bitcast)
    return b.notifyMatchFailure(op.getLoc(), "ptr is not a bitcast");
  auto ptrType = dyn_cast<PointerType>(bitcast.getInput().getType());
  if (!ptrType || !isa<PointerType>(ptrType.getElementType()))
    return b.notifyMatchFailure(op.getLoc(), "bitcast input is not a pointer");

  // Rewrite the store in-place.
  auto newBitcast = b.create<PointerBitcastOp>(
      op.getLoc(), ptrType.getElementType(), op.getArg());
  b.modifyOpInPlace(op, [&] {
    op.getPtrMutable().set(bitcast.getInput());
    op.getArgMutable().set(newBitcast);
  });
  return success();
}

ErrorTreeOrSuccess StoreOp::interpret(ArrayRef<Attribute> operands,
                                      InterpreterState &state) {
  auto value = cast_or_null<TypedAttr>(operands[0]);
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

ErrorTreeOrSuccess OffsetOp::interpret(ArrayRef<Attribute> operands,
                                       InterpreterState &state) {
  if (!state.getTarget())
    return ErrorTree(getLoc(), "operation requires a target model");

  auto ptr = dyn_cast_or_null<PointerAttr>(operands[0]);
  auto offset = dyn_cast_or_null<IntegerAttr>(operands[1]);
  if (!ptr || !offset)
    return ErrorTree(getLoc(), "non-constant inputs");
  std::optional<int64_t> elSize = DataLayoutInterface::getTypeAllocSize(
      state.getTarget(), cast<PointerType>(ptr.getType()).getElementType());
  if (!elSize)
    return ErrorTree(getLoc(), "could not query pointer element size");
  state.mapResults(PointerAttr::get(ptr.getAddr() + *elSize * offset.getInt(),
                                    ptr.getType()));
  return success();
}

OpFoldResult OffsetOp::fold(FoldAdaptor adaptor) {
  auto offset = dyn_cast_or_null<IntegerAttr>(adaptor.getIndex());
  if (!offset)
    return {};

  if (offset.getInt() != 0)
    return {};

  return getPtr();
}

LogicalResult OffsetOp::canonicalize(OffsetOp op, PatternRewriter &b) {
  // Canonicalize `%ptr[%c0][%c1] -> %ptr[%c0 + %c1]`, where the indices are
  // constants.
  APInt c1;
  if (!mlir::matchPattern(op.getIndex(), mlir::m_ConstantInt(&c1)))
    return b.notifyMatchFailure(op, "not a constant offset");
  auto parent = op.getPtr().getDefiningOp<OffsetOp>();
  if (!parent)
    return b.notifyMatchFailure(op, "parent is not an offset");
  APInt c0;
  if (!mlir::matchPattern(parent.getIndex(), mlir::m_ConstantInt(&c0)))
    return b.notifyMatchFailure(op, "parent is not a constant offset");

  // However unlikely, don't canonicalize if the arithmetic overflows. Note that
  // it is always valid to fold addition of index values, regardless of width.
  bool ov = false;
  APInt newOffset = c0.sadd_ov(c1, ov);
  if (ov)
    return b.notifyMatchFailure(op, "offset addition overflows");

  b.replaceOpWithNewOp<OffsetOp>(
      op, parent.getPtr(),
      b.create<ParamConstantOp>(op.getLoc(),
                                b.getIndexAttr(newOffset.getSExtValue())));
  return success();
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

OpFoldResult SelectOp::fold(FoldAdaptor adaptor) {
  // Narrow to one of the conditional values.
  if (auto cond = dyn_cast_if_present<BoolAttr>(adaptor.getCondition())) {
    if (cond.getValue()) {
      if (Attribute attr = adaptor.getTrueValue())
        return attr;
      return getTrueValue();
    }
    if (Attribute attr = adaptor.getFalseValue())
      return attr;
    return getFalseValue();
  }

  // Fold `select x, true, false -> x`.
  auto trueAttr = dyn_cast_if_present<BoolAttr>(adaptor.getTrueValue());
  auto falseAttr = dyn_cast_if_present<BoolAttr>(adaptor.getFalseValue());
  if (trueAttr && falseAttr && trueAttr.getValue() == true &&
      falseAttr.getValue() == false)
    return getCondition();

  // Fold `select x, undef, y -> y` and `select x, y, undef -> y`.
  if (isa_and_nonnull<UnknownAttr, UninitMemAttr>(adaptor.getTrueValue()))
    return getFalseValue();
  if (isa_and_nonnull<UnknownAttr, UninitMemAttr>(adaptor.getFalseValue()))
    return getTrueValue();

  // `x ? y : y -> y`.
  if (getTrueValue() == getFalseValue())
    return getTrueValue();

  return {};
}

namespace {
/// Fold the following pattern
///   %cond: i1
///   %true  = kgen.param.constant: scalar<bool> = <true>
///   %false = kgen.param.constant: scalar<bool> = <false>
///   %res   = pop.select %cond, %true, %false : !pop.scalar<bool>
/// Into
///   %res   = pop.cast_from_builtin %cond i1 to !pop.scalar<bool>
struct SelectTrueFalseScalarBool : OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &b) const override {
    auto simdTy = dyn_cast<POP::SIMDType>(op.getType());
    if (!simdTy || !simdTy.isScalar() ||
        simdTy.getResolvedDType() != KGENDType::kBool) {
      return b.notifyMatchFailure(op, "result type isn't !pop.scalar<bool>");
    }

    SIMDAttr trueVal, falseVal;
    if (!mlir::matchPattern(op.getTrueValue(), mlir::m_Constant(&trueVal)) ||
        !mlir::matchPattern(op.getFalseValue(), mlir::m_Constant(&falseVal)))
      return b.notifyMatchFailure(op, "True/False value isn't constant");

    auto isBoolAttr = [](SIMDAttr attr, bool value) {
      return attr.getValues().front().getBoolVal() == value;
    };

    if (isBoolAttr(trueVal, true) && isBoolAttr(falseVal, false)) {
      b.replaceOpWithNewOp<POP::CastFromBuiltinOp>(op, op.getType(),
                                                   op.getCondition());
      return success();
    }

    return b.notifyMatchFailure(op, "failed to match true/false constants");
  }
};

/// Canonicalize `select x, (select x, a, b), c` into `select x, a, c` or
/// `select x, a, (select x, b, c)` into `select x, a, c`.
struct SelectOfSelect : OpRewritePattern<SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectOp op,
                                PatternRewriter &b) const override {
    bool knownCondition;
    SelectOp branchSelect;
    if ((branchSelect = op.getTrueValue().getDefiningOp<SelectOp>())) {
      knownCondition = true;
    } else if ((branchSelect = op.getFalseValue().getDefiningOp<SelectOp>())) {
      knownCondition = false;
    } else {
      return b.notifyMatchFailure(
          op.getLoc(), "true or false value not defined by a select");
    }
    if (branchSelect.getCondition() != op.getCondition()) {
      return b.notifyMatchFailure(op.getLoc(),
                                  "branch select condition is not the same");
    }
    Value foldedValue = knownCondition ? branchSelect.getTrueValue()
                                       : branchSelect.getFalseValue();
    b.modifyOpInPlace(
        op, [&] { op->setOperand(1 + (knownCondition ? 0 : 1), foldedValue); });
    return success();
  }
};
} // namespace

void SelectOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<SelectTrueFalseScalarBool, SelectOfSelect>(context);
}

//===----------------------------------------------------------------------===//
// StackAllocationOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess StackAllocationOp::compile(Payload &payload,
                                          TargetInfoAttr target) {
  auto countAttr = dyn_cast<IntegerAttr>(getCount());
  if (!countAttr)
    return Error("array size is not a constant");
  int64_t count = countAttr.getInt();

  if (!target) {
    if (count != 1)
      return Error("array allocation requires a target model");
    return success();
  }

  // Determine the allocation size.
  Type type = cast<PointerType>(getType()).getElementType();
  std::optional<int64_t> size =
      DataLayoutInterface::getTypeAllocSize(target, type);
  if (!size)
    return Error("could not query type size");

  // Determine the alignment. If the alignment is unspecified or zero, query
  // the natural alignment of the type.
  int64_t align = 0;
  if (TypedAttr alignAttr = getAlignmentAttr())
    align = cast<IntegerAttr>(alignAttr).getInt();
  if (align < 0)
    return Error("invalid alignment value: " + Twine(align));
  if (align == 0) {
    std::optional<int64_t> typeAlign =
        DataLayoutInterface::getTypeABIAlign(target, type);
    if (!typeAlign)
      return Error("could not query type alignment");
    align = *typeAlign;
  }

  payload.size = count * *size;
  payload.align = align;
  return success();
}

ErrorTreeOrSuccess StackAllocationOp::interpret(ArrayRef<Attribute> operands,
                                                const Payload &payload,
                                                InterpreterState &state) {
  // If there is no target model, we know it is a count 1 alloc.
  if (!state.getTarget())
    return ErrorTree(getLoc(), "stack allocation requires a target model");

  ErrorOr<int64_t> addr =
      state.allocateStackMemory(payload.size, payload.align);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());
  state.mapResults(PointerAttr::get(addr.takeValue(), getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeStartOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
StackAllocLifetimeStartOp::interpret(ArrayRef<Attribute> operands,
                                     InterpreterState &state) {
  return success();
}

//===----------------------------------------------------------------------===//
// StackAllocLifetimeEndOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
StackAllocLifetimeEndOp::interpret(ArrayRef<Attribute> operands,
                                   InterpreterState &state) {
  return success();
}

//===----------------------------------------------------------------------===//
// AlignedFreeOp
//===----------------------------------------------------------------------===//

LogicalResult AlignedFreeOp::canonicalize(AlignedFreeOp op,
                                          PatternRewriter &b) {
  auto bitcast = op.getPtr().getDefiningOp<PointerBitcastOp>();
  if (!bitcast)
    return failure();
  b.modifyOpInPlace(op, [&] { op.getPtrMutable().set(bitcast.getInput()); });
  return success();
}

//===----------------------------------------------------------------------===//
// AlignedAllocOp
//===----------------------------------------------------------------------===//

/// Interpret an aligned allocation.
static ErrorTreeOrSuccess interpretAllocation(int64_t size, int64_t align,
                                              Location loc, Type type,
                                              InterpreterState &state) {
  // The default "system" alignment technically has no guarantees and varies
  // depending on the underlying allocator implementation. Just use 64 for
  // consistency.
  if (align <= 0)
    align = 64;

  ErrorOr<int64_t> addr = state.allocateHeapMemory(size, align);
  if (addr.isError())
    return ErrorTree(loc, addr.takeError());
  state.mapResults(PointerAttr::get(addr.takeValue(), type));
  return success();
}

ErrorTreeOrSuccess AlignedAllocOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  auto alignAttr = dyn_cast_or_null<IntegerAttr>(operands.front());
  auto sizeAttr = dyn_cast_or_null<IntegerAttr>(operands.back());
  if (!alignAttr || !sizeAttr)
    return ErrorTree(getLoc(), "non-concrete inputs");
  return interpretAllocation(sizeAttr.getInt(), alignAttr.getInt(), getLoc(),
                             getType(), state);
}

//===----------------------------------------------------------------------===//
// AlignedFreeOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess AlignedFreeOp::interpret(ArrayRef<Attribute> operands,
                                            InterpreterState &state) {
  auto ptr = cast<PointerAttr>(operands.front());
  if (ErrorOrSuccess err = state.freeHeapMemory(ptr.getAddr()); err.isError())
    return ErrorTree(getLoc(), err.takeError());
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayCreateOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayCreateOp::fold(FoldAdaptor adaptor) {
  ArrayRef<Attribute> operands = adaptor.getOperands();
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
  ArrayRef<Attribute> operands = adaptor.getOperands();
  std::optional<int64_t> size = getType().getResolvedSize();
  if (!size)
    return {};
  assert(size >= 0 && "size is non-negative");
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

  // If the array comes from an undef constant then the result is also undef,
  // irrespective of the index.
  if (auto cst =
          dyn_cast_or_null<KGEN::ParamConstantOp>(getArray().getDefiningOp())) {
    if (isa<UninitMemAttr>(cst.getValue()))
      return UninitMemAttr::get(getType());
    if (isa<UnknownAttr>(cst.getValue()))
      return UnknownAttr::get(getType());
  }

  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (!index)
    return {};

  std::optional<int64_t> size = getArray().getType().getResolvedSize();
  if (!size)
    return {};

  // Bounds check the array access.
  int64_t idx = index.getInt();
  if (idx < 0 || idx >= *size)
    return {};

  // Try fold if the array is a constant.
  auto array = dyn_cast_if_present<POP::ArrayAttr>(adaptor.getArray());
  if (array)
    return array.getValues()[idx];

  // If we directly come from an `ArrayCreate` we can just fold to the operand
  // of that.
  if (auto arrayCreate = getArray().getDefiningOp<POP::ArrayCreateOp>())
    return arrayCreate.getOperand(idx);

  // If we come from a repeat we can work out which operand we are.
  if (auto repeat = getArray().getDefiningOp<POP::ArrayRepeatOp>())
    return repeat.getOperand(idx % repeat.getNumOperands());

  return {};
}

//===----------------------------------------------------------------------===//
// ArrayReplaceOp
//===----------------------------------------------------------------------===//

OpFoldResult ArrayReplaceOp::fold(FoldAdaptor adaptor) {
  auto value = llvm::cast_if_present<TypedAttr>(adaptor.getValue());
  auto array = dyn_cast_if_present<POP::ArrayAttr>(adaptor.getArray());
  auto index = dyn_cast<IntegerAttr>(getIndex());
  if (!value || !array || !index)
    return {};
  SmallVector<TypedAttr> values(array.getValues());
  values[index.getInt()] = value;
  return POP::ArrayAttr::get(values, getType());
}

LogicalResult ArrayReplaceOp::canonicalize(ArrayReplaceOp op,
                                           PatternRewriter &rewriter) {
  auto indexAttr = dyn_cast<IntegerAttr>(op.getIndex());
  if (!indexAttr)
    return rewriter.notifyMatchFailure(op, "dynamic index not supported");
  unsigned index = indexAttr.getInt();

  if (auto arrayCreateOp = op.getArray().getDefiningOp<ArrayCreateOp>()) {
    SmallVector<Value> newOperands = arrayCreateOp.getOperands();
    newOperands[index] = op.getValue();
    rewriter.replaceOpWithNewOp<ArrayCreateOp>(op, newOperands);
    return success();
  }

  if (auto paramConstantOp =
          op.getArray().getDefiningOp<KGEN::ParamConstantOp>()) {
    auto constArr = cast<POP::ArrayAttr>(paramConstantOp.getValue());
    SmallVector<Value> newOperands;
    newOperands.reserve(constArr.getValues().size());
    for (unsigned i = 0, e = constArr.getValues().size(); i < e; ++i) {
      if (i == index)
        newOperands.push_back(op.getValue());
      else
        newOperands.push_back(rewriter.create<KGEN::ParamConstantOp>(
            paramConstantOp.getLoc(), constArr.getValues()[i]));
    }
    rewriter.replaceOpWithNewOp<ArrayCreateOp>(op, newOperands);
    return success();
  }

  return rewriter.notifyMatchFailure(
      op, "array must be a constant or an ArrayCreateOp");
}

//===----------------------------------------------------------------------===//
// ArrayGEPOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess ArrayGEPOp::interpret(ArrayRef<Attribute> operands,
                                         InterpreterState &state) {
  if (!state.getTarget())
    return ErrorTree(getLoc(), "operation requires a target model");

  auto ptr = dyn_cast_if_present<PointerAttr>(operands[0]);
  auto index = dyn_cast_if_present<IntegerAttr>(operands[1]);
  if (!ptr || !index)
    return ErrorTree(getLoc(), "non-constant inputs");

  auto arrayType = getArray().getType().getElementAs<POP::ArrayType>();
  Type elementType = arrayType.getElementType();
  auto dl = cast<DataLayoutInterface>(elementType);
  int64_t addr =
      ptr.getAddr() +
      index.getInt() * (llvm::alignTo(*dl.getTypeSize(state.getTarget()),
                                      *dl.getTypeAlign(state.getTarget())));
  state.mapResults(PointerAttr::get(addr, PointerType::get(elementType)));
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
  if (auto indexOp = op.getIndex().getDefiningOp();
      indexOp && indexOp->hasTrait<OpTrait::ConstantLike>())
    return rewriter.notifyMatchFailure(op,
                                       "ArrayGEP index is already constant.");

  // Otherwise we have gep into a array of one element with a dynamic value. It
  // is undefined behaviour for that to be anything but `0` so we can replace it
  // with the constant `0`. This frees the use to be DCE'd and unblocks other
  // optimizations.
  auto zero =
      rewriter.create<ParamConstantOp>(op.getLoc(), rewriter.getIndexAttr(0));
  rewriter.replaceOpWithNewOp<ArrayGEPOp>(op, op.getType(), op.getArray(),
                                          zero);
  return success();
}

//===----------------------------------------------------------------------===//
// PointerToIndexOp
//===----------------------------------------------------------------------===//

OpFoldResult PointerToIndexOp::fold(FoldAdaptor adaptor) {
  // Check for a pointer input. The result must be a scalar index.
  if (auto ptr = dyn_cast_if_present<PointerAttr>(adaptor.getValue()))
    return Builder(getContext()).getIndexAttr(ptr.getAddr());
  return {};
}

LogicalResult PointerToIndexOp::canonicalize(PointerToIndexOp op,
                                             PatternRewriter &b) {
  auto bitcast = op.getValue().getDefiningOp<PointerBitcastOp>();
  if (!bitcast)
    return failure();
  b.modifyOpInPlace(op, [&] { op.getValueMutable().set(bitcast.getInput()); });
  return success();
}

//===----------------------------------------------------------------------===//
// CompilerGlobalLoadOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess CompilerGlobalLoadOp::interpret(ArrayRef<Attribute> operands,
                                                   InterpreterState &state) {
  Attribute value = state.getNamedGlobal(getNameAttr());
  if (!value)
    return ErrorTree(
        getLoc(),
        "cannot evaluate standalone capturing closure at compile time");
  state.mapResults(value);
  return success();
}

//===----------------------------------------------------------------------===//
// CompilerGlobalStoreOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess
CompilerGlobalStoreOp::interpret(ArrayRef<Attribute> operands,
                                 InterpreterState &state) {
  state.setNamedGlobal(getNameAttr(), operands.front());
  return success();
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
  auto simd = dyn_cast_if_present<SIMDAttr>(adaptor.getInput());
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
  auto val = llvm::cast_if_present<TypedAttr>(adaptor.getInput());
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
  // Check to see if all the inputs are constant, if so, convert to
  // VariadicAttr.
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

/// Canonicalize `pop.variadic.create(x,x,x) -> `pop.variadic.splat(x)`. This
/// notably turns all 1 element creates into a splat.
LogicalResult VariadicCreateOp::canonicalize(VariadicCreateOp op,
                                             PatternRewriter &b) {
  // Canonicalize a 1+ operand create into a splat if we can.
  if (size_t numElements = op.getNumOperands()) {
    Value splatValue = op.getOperand(0);
    if (llvm::all_of(op.getOperands().drop_front(),
                     [&](Value operand) { return operand == splatValue; })) {
      b.replaceOpWithNewOp<VariadicSplatOp>(op, op.getType(), splatValue,
                                            numElements);
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// VariadicSplatOp
//===----------------------------------------------------------------------===//

OpFoldResult VariadicSplatOp::fold(FoldAdaptor adaptor) {
  // We can poke at this only if the result #elts is a known constant.
  auto numEltsCst = dyn_cast<IntegerAttr>(getNumElements());
  if (!numEltsCst)
    return {};

  // If the input is constant, splat to a VariadicAttr.
  if (Attribute cst = adaptor.getOperand()) {
    SmallVector<TypedAttr> values(numEltsCst.getInt(), cast<TypedAttr>(cst));
    return KGEN::VariadicAttr::get(values, getType());
  }

  // Fold a splat to zero values to a constant.
  if (numEltsCst.getValue().isZero())
    return KGEN::VariadicAttr::get(ArrayRef<TypedAttr>(), getType());

  return {};
}

//===----------------------------------------------------------------------===//
// VariadicGetOp
//===----------------------------------------------------------------------===//

OpFoldResult VariadicGetOp::fold(FoldAdaptor adaptor) {
  // Canonicalize `get(splat(x)) -> x`.
  if (auto splat = getVariadic().getDefiningOp<VariadicSplatOp>())
    return splat.getOperand();

  auto indexAttr = dyn_cast_or_null<IntegerAttr>(adaptor.getIndex());
  if (!indexAttr)
    return {};
  unsigned index = indexAttr.getInt();

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
// VariadicSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult VariadicSizeOp::fold(FoldAdaptor adaptor) {
  auto indexType = IndexType::get(getContext());
  if (auto variadic =
          dyn_cast_if_present<KGEN::VariadicAttr>(adaptor.getOperand()))
    return IntegerAttr::get(indexType, variadic.getValues().size());

  if (auto create = getOperand().getDefiningOp<VariadicCreateOp>())
    return IntegerAttr::get(indexType, create.getOperands().size());

  if (auto splat = getOperand().getDefiningOp<VariadicSplatOp>())
    return splat.getNumElements();

  return {};
}

//===----------------------------------------------------------------------===//
// StringAddressOp
//===----------------------------------------------------------------------===//

ErrorTreeOrSuccess StringAddressOp::interpret(ArrayRef<Attribute> operands,
                                              InterpreterState &state) {
  // Ensure the string is null-terminated. This is safe because `StringAttr`
  // always stores a null terminator.
  auto value = dyn_cast<StringAttr>(operands.front());
  if (!value)
    return ErrorTree(getLoc(), Error("argument is not a concrete string"));
  StringRef str(value.data(), value.size() + 1);
  if (value.getValue().empty())
    str = "\0";

  MemoryHandleAttr hdl = MemoryHandleAttr::get(getContext(), str);
  ErrorOr<int64_t> addr = state.mapConstGlobalMemory(hdl);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());
  state.mapResults(PointerAttr::get(getContext(), addr.takeValue(), getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// StringSizeOp
//===----------------------------------------------------------------------===//

OpFoldResult StringSizeOp::fold(FoldAdaptor adaptor) {
  if (auto str = dyn_cast_or_null<TypedAttr>(adaptor.getStr()))
    return StringSizeAttr::get(getContext(), str);
  return {};
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

//===----------------------------------------------------------------------===//
// VariantBitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult VariantBitcastOp::fold(FoldAdaptor adaptor) {
  if (auto ptr = dyn_cast_or_null<PointerAttr>(adaptor.getVariant()))
    return PointerAttr::get(ptr.getAddr(), getType());
  return {};
}

//===----------------------------------------------------------------------===//
// VariantDiscrGEPOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess VariantDiscrGEPOp::compile(Payload &payload,
                                          TargetInfoAttr target) {
  if (!target)
    return Error("requires a target model");

  auto variantType = getVariant().getType().getElementAs<VariantType>();
  std::optional<int64_t> size = variantType.getContentSize(target);
  if (!size)
    return Error("failed to compute size");
  payload.offset = *size;
  return success();
}

ErrorTreeOrSuccess VariantDiscrGEPOp::interpret(ArrayRef<Attribute> operands,
                                                const Payload &payload,
                                                InterpreterState &state) {
  auto ptr = dyn_cast_if_present<PointerAttr>(operands.front());
  if (!ptr)
    return ErrorTree(getLoc(), "non-constant inputs");

  state.mapResults(PointerAttr::get(ptr.getAddr() + payload.offset, getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// GlobalAllocOp
//===----------------------------------------------------------------------===//

ErrorOrSuccess GlobalAllocOp::compile(Payload &payload, TargetInfoAttr target) {
  if (!target)
    return Error("global alloc requires a target");

  auto countAttr = dyn_cast<IntegerAttr>(getCount());
  if (!countAttr)
    return Error("count is not concrete");

  Type type = getType().getElementType();
  payload.size =
      countAttr.getInt() * *DataLayoutInterface::getTypeAllocSize(target, type);

  if (auto alignAttr = dyn_cast_or_null<IntegerAttr>(getAlignmentAttr()))
    payload.align = alignAttr.getInt();
  else
    payload.align = *DataLayoutInterface::getTypeABIAlign(target, type);

  payload.addressSpace =
      cast<IntegerAttr>(getType().getAddressSpace()).getInt();
  return success();
}

ErrorTreeOrSuccess GlobalAllocOp::interpret(ArrayRef<Attribute> operands,
                                            const Payload &payload,
                                            InterpreterState &state) {
  ErrorOr<int64_t> addr = state.allocatePersistentMemory(
      payload.size, payload.align, payload.addressSpace);
  if (addr.isError())
    return ErrorTree(getLoc(), addr.takeError());

  state.mapResults(PointerAttr::get(addr.takeValue(), getType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ExternalCallOp
//===----------------------------------------------------------------------===//

static ErrorTreeOrSuccess interpretMalloc(ExternalCallOp op,
                                          ArrayRef<Attribute> operands,
                                          InterpreterState &state) {
  if (operands.size() != 1 || op->getNumResults() != 1) {
    return ErrorTree(op.getLoc(), "unable to interpret call to 'malloc', "
                                  "expected 1 operand and 1 result");
  }
  size_t size = dyn_cast_or_null<IntegerAttr>(operands.front()).getInt();
  return interpretAllocation(size, /*align=*/0, op.getLoc(),
                             op->getResultTypes()[0], state);
}

static ErrorTreeOrSuccess interpretFree(ExternalCallOp op,
                                        ArrayRef<Attribute> operands,
                                        InterpreterState &state) {
  if (operands.size() != 1 || op.getNumResults() != 0) {
    return ErrorTree(
        op.getLoc(),
        "unable to interpret call to 'free', expected 1 operand and 0 results");
  }

  auto ptr = cast<PointerAttr>(operands.front());
  if (ErrorOrSuccess err = state.freeHeapMemory(ptr.getAddr()); err.isError())
    return ErrorTree(op.getLoc(), err.takeError());
  return success();
}

static ErrorTreeOrSuccess interpreterWrite(ExternalCallOp op,
                                           ArrayRef<Attribute> operands,
                                           InterpreterState &state) {
  if (!(operands.size() == 3 && op.getNumResults() == 1))
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected 3 operands and 1 results");
  Type resultType = op.getResultTypes().front();
  if (!resultType.isIntOrIndex() && !isa<POP::SIMDType>(resultType))
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected integer result type");
  IntegerAttr fileDescriptor = dyn_cast<IntegerAttr>(operands[0]);
  if (!fileDescriptor)
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected integer typed first operand");

  PointerAttr buffer = cast<PointerAttr>(operands[1]);
  if (!buffer)
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected pointer typed second operand");

  IntegerAttr nbytes = cast<IntegerAttr>(operands[2]);
  if (!nbytes)
    return ErrorTree(op.getLoc(), "unable to interpret call to 'write', "
                                  "expected integer typed third operand");
  unsigned ptrSize = state.getTarget().getDataLayout().getPointerSize();
  ErrorOr<const void *> mem =
      state.getReadableMemory(buffer.getAddr(), ptrSize);
  if (mem)
    return ErrorTree(op.getLoc(), mem.takeError());
  int size = nbytes.getValue().getZExtValue();
  int numWritten =
      write(fileDescriptor.getValue().getZExtValue(), (const void *)*mem, size);
  if (auto simdType = dyn_cast<POP::SIMDType>(resultType)) {
    auto simdAttr = SIMDAttr::get(numWritten, simdType);
    state.mapResults(simdAttr);
  } else {
    state.mapResults(IntegerAttr::get(resultType, numWritten));
  }
  return success();
}

/// FIXME(#26342): We shouldn't implement interpreter support for external_call,
/// this bakes assumptions about the functions. This is a temporary workaround
/// because of the fact that the gpu path does not use the dedicated pop memory
/// operations.
ErrorTreeOrSuccess ExternalCallOp::interpret(ArrayRef<Attribute> operands,
                                             InterpreterState &state) {
  // external_call can take things through a !kgen.pack.  Expand that out before
  // we try to interpret it.
  SmallVector<Attribute> expandedOperands;
  expandedOperands.reserve(operands.size());
  for (auto attr : operands) {
    if (auto pack = dyn_cast<PackAttr>(attr)) {
      expandedOperands.append(pack.getValues().begin(), pack.getValues().end());
    } else {
      expandedOperands.push_back(attr);
    }
  }

  StringRef callee = getCallee();
  if (callee == "malloc")
    return interpretMalloc(*this, expandedOperands, state);
  if (callee == "free")
    return interpretFree(*this, expandedOperands, state);
  if (callee == "write")
    return interpreterWrite(*this, expandedOperands, state);

  return ErrorTree(
      getLoc(),
      Twine("unable to interpret call to unknown external function: " + callee)
          .str());
}

//===----------------------------------------------------------------------===//
// UnionBitcastOp
//===----------------------------------------------------------------------===//

OpFoldResult UnionBitcastOp::fold(FoldAdaptor adaptor) {
  auto ptr = dyn_cast_or_null<PointerAttr>(adaptor.getValue());
  if (!ptr)
    return {};
  return PointerAttr::get(ptr.getAddr(), getType());
}

//===----------------------------------------------------------------------===//
// UnionWrapOp
//===----------------------------------------------------------------------===//

OpFoldResult UnionWrapOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<TypedAttr>(adaptor.getValue()))
    return UnionAttr::get(attr, getType());

  // Fold `wrap(unwrap(x)) -> x` if the types are the same.
  if (auto unwrap = getValue().getDefiningOp<UnionUnwrapOp>();
      unwrap && unwrap.getValue().getType() == getType())
    return unwrap.getValue();

  return {};
}

//===----------------------------------------------------------------------===//
// UnionUnwrapOp
//===----------------------------------------------------------------------===//

OpFoldResult UnionUnwrapOp::fold(FoldAdaptor adaptor) {
  if (auto attr = dyn_cast_or_null<UnionAttr>(adaptor.getValue()))
    if (attr.getValue().getType() == getType())
      return attr.getValue();

  // Fold `unwrap(wrap(x)) -> x`.
  if (auto wrap = getValue().getDefiningOp<UnionWrapOp>();
      wrap && wrap.getValue().getType() == getType())
    return wrap.getValue();

  return {};
}
