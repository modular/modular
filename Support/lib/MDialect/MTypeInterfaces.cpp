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
  Optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return llvm::divideCeil(cast<IntegerType>(type).getWidth(), CHAR_BIT);
  }

  /// The alignment of an integer type is its width in bytes rounded up to the
  /// nearest power of 2, but capped at the pointer width.
  Optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return std::min<int64_t>(llvm::PowerOf2Ceil(*getTypeSize(type, target)),
                             target.getPointerSize());
  }
};

struct FloatLayout
    : public DataLayoutInterface::ExternalModel<FloatLayout, FloatType> {
  /// The size of an integer type is its width in bytes.
  Optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return cast<FloatType>(type).getWidth() / CHAR_BIT;
  }

  /// The alignment of a float type is its width in bytes rounded up to the
  /// nearest power of 2, but capped at the pointer width.
  Optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return std::min<int64_t>(llvm::PowerOf2Ceil(*getTypeSize(type, target)),
                             target.getPointerSize());
  }
};

struct FunctionLayout
    : public DataLayoutInterface::ExternalModel<FunctionLayout, FunctionType> {
  /// The size of a function type is the pointer width.
  Optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return target.getPointerSize();
  }

  /// The align of a function type is the pointer width.
  Optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getPointerSize();
  }
};

struct IndexLayout
    : public DataLayoutInterface::ExternalModel<IndexLayout, IndexType> {
  /// The size of an index type is the pointer width.
  Optional<int64_t> getTypeSize(Type type, TargetInfoAttr target) const {
    return target.getPointerSize();
  }

  /// The align of an index type is the pointer width.
  Optional<int64_t> getTypeAlign(Type type, TargetInfoAttr target) const {
    return target.getPointerSize();
  }
};
} // namespace

void MDialect::injectAttrInterfaces() {
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

Optional<int64_t>
M::DataLayoutInterface::getTypeSizeInBytes(TargetInfoAttr target, Type type) {
  if (auto iface = llvm::dyn_cast<DataLayoutInterface>(type))
    return iface.getTypeSize(target);
  return {};
}

Optional<int64_t>
M::DataLayoutInterface::getTypeAlignInBytes(TargetInfoAttr target, Type type) {
  if (auto iface = llvm::dyn_cast<DataLayoutInterface>(type))
    return iface.getTypeAlign(target);
  return {};
}

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "Support/MDialect/MTypeInterfaces.cpp.inc"
