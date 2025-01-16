//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MTypeInterfaces.h"
#include "Support/MDialect/MDialect.h"
#include "Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace M;

//===----------------------------------------------------------------------===//
// DataLayoutInterface
//===----------------------------------------------------------------------===//

namespace {
struct IntegerLayout
    : public DataLayoutInterface::ExternalModel<IntegerLayout, IntegerType> {
  /// The size of an integer type is its width rounded up to the nearest byte.
  std::optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return llvm::divideCeil(cast<IntegerType>(type).getWidth(), CHAR_BIT);
  }

  /// The alignment of an integer type is its width in bytes rounded up to the
  /// nearest power of 2, but capped at the pointer width.
  std::optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getDataLayout().getIntegerABIAlign(
        cast<IntegerType>(type).getWidth());
  }
};

struct FloatLayout
    : public DataLayoutInterface::ExternalModel<FloatLayout, Type> {
  /// The size of an integer type is its width in bytes.
  std::optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return cast<FloatType>(type).getWidth() / CHAR_BIT;
  }

  /// The alignment of a float type is its width in bytes rounded up to the
  /// nearest power of 2, but capped at the pointer width.
  std::optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getDataLayout().getFloatABIAlign(
        cast<FloatType>(type).getWidth());
  }
};

struct FunctionLayout
    : public DataLayoutInterface::ExternalModel<FunctionLayout, FunctionType> {
  /// The size of a function type is the pointer width.
  std::optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return llvm::divideCeil(target.getDataLayout().getPointerBitWidth(),
                            CHAR_BIT);
  }

  /// The align of a function type is the pointer width.
  std::optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getDataLayout().getPointerABIAlign();
  }
};

struct IndexLayout
    : public DataLayoutInterface::ExternalModel<IndexLayout, IndexType> {
  /// The size of an index type is the one found in the TargetInfoAttr.
  std::optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return llvm::divideCeil(target.resolveIndexBitWidth(), CHAR_BIT);
  }

  /// The align of an index type is the pointer width.
  std::optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getDataLayout().getPointerABIAlign();
  }
};
} // namespace

void MDialect::injectTypeInterfaces() {
  IntegerType::attachInterface<IntegerLayout>(*getContext());
  BFloat16Type::attachInterface<FloatLayout>(*getContext());
  Float16Type::attachInterface<FloatLayout>(*getContext());
  Float32Type::attachInterface<FloatLayout>(*getContext());
  Float64Type::attachInterface<FloatLayout>(*getContext());
  Float80Type::attachInterface<FloatLayout>(*getContext());
  Float128Type::attachInterface<FloatLayout>(*getContext());
  FunctionType::attachInterface<FunctionLayout>(*getContext());
  IndexType::attachInterface<IndexLayout>(*getContext());
}

std::optional<int64_t>
DataLayoutInterface::getTypeStoreSize(TargetInfoAttr target, Type type) {
  if (auto iface = llvm::dyn_cast<DataLayoutInterface>(type))
    return iface.getTypeSize(target);
  return {};
}

std::optional<int64_t>
DataLayoutInterface::getTypeAllocSize(TargetInfoAttr target, Type type) {
  std::optional<int64_t> typeSize = getTypeStoreSize(target, type);
  std::optional<int64_t> typeABIAlign = getTypeABIAlign(target, type);
  if (!typeSize || !typeABIAlign)
    return {};
  return llvm::alignTo(*typeSize, *typeABIAlign);
}

std::optional<int64_t>
DataLayoutInterface::getTypeABIAlign(TargetInfoAttr target, Type type) {
  if (auto iface = llvm::dyn_cast<DataLayoutInterface>(type))
    return iface.getTypeAlign(target);
  return {};
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MTypeInterfaces.cpp.inc"
