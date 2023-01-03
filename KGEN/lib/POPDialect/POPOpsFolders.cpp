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
// Arithmetic Operation Folders
//===----------------------------------------------------------------------===//

namespace detail {
/// Detector for whether `T` posseses a `has_value` method.
template <typename T>
using IsOptionalType = decltype(std::declval<T>().has_value());

/// Perform folding of an n-ary SIMD vector operation of a given dtype by
/// applying the operation `op` to each vector element. `getValue` transforms a
/// `DTypeValue` to the value used to represent the dtype: `APSInt` for
/// integers, `APFloat` for floats, and `bool` for bools.
template <size_t... I, typename OpFn, typename GetValueFn>
static SIMDAttr foldSIMDOpImpl(std::index_sequence<I...>,
                               ArrayRef<Attribute> operands, OpFn op,
                               GetValueFn getValue) {
  auto type = cast<SIMDType>(cast<TypedAttr>(operands.front()).getType());
  SmallVector<DTypeValue> results;
  auto firstArg = cast<SIMDAttr>(operands.front());
  DType dtype = *firstArg.getType().getResolvedDType();
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
  return SIMDAttr::get(results, type);
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
                                ArrayRef<Attribute> operands, OpFns &&...ops) {
  auto op = getOpFnOfType<TestType>(std::forward<OpFns>(ops)...);
  if constexpr (std::is_same_v<decltype(op), std::nullopt_t>) {
    llvm_unreachable("unhandled dtype");
  } else {
    return foldSIMDOpImpl(std::make_index_sequence<
                              llvm::function_traits<decltype(op)>::num_args>(),
                          operands, op, getValue);
  }
}
} // namespace detail

/// Try to fold an n-ary SIMD vector operation using one of the provided
/// functions for each possible dtype.
template <typename... OpFns>
static SIMDAttr foldSIMDOp(ArrayRef<Attribute> operands, OpFns &&...ops) {
  if (llvm::any_of(operands, [](Attribute operand) {
        return !isa_and_nonnull<SIMDAttr>(operand);
      }))
    return {};
  DType dtype = *cast<SIMDAttr>(operands.front()).getType().getResolvedDType();
  if (dtype.isInt())
    return ::detail::foldSIMDOpDType<APSInt>(
        [](DTypeValue val) { return val.getIntVal(); }, operands,
        std::forward<OpFns>(ops)...);
  // FIXME: Should we even do floating point folds? Results don't match hardware
  // and not all float semantics are supported.
  if (dtype.isFloat())
    return ::detail::foldSIMDOpDType<APFloat>(
        [](DTypeValue val) { return val.getFloatVal(); }, operands,
        std::forward<OpFns>(ops)...);
  if (dtype.isBool())
    return ::detail::foldSIMDOpDType<bool>(
        [](DTypeValue val) { return val.getBoolVal(); }, operands,
        std::forward<OpFns>(ops)...);
  llvm_unreachable("unhandled dtype");
}

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
