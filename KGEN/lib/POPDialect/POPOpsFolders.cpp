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
