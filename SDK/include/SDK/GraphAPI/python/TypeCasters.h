//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_TYPECASTERS_H
#define SDK_GRAPHAPI_PYTHON_TYPECASTERS_H

#include "Support/ML/DType.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinDialect.h"
#include "nanobind/make_iterator.h"
#include "nanobind/nanobind.h"
#include <Support/AssertStream.h>
#include <exception>
#include <type_traits>
#include <vector>

namespace nb = nanobind;

namespace M::Graph::Python {
template <typename T>
class NanobindWrapper {
  // This exists because of typeinfo madness.
  // MLIR doesn't generate typeinfo, and _most_ of its types don't have
  // a key function, so we can generate them ourselves.
  // For a few types, such as mlir::RewriterBase, they _do_ have key functions,
  // and as a result we can't use such types as inputs to our bindings.
  //
  // Instead, we create this wrapper for which we know we can generate typeinfo,
  // and then we codegen usages of it.
public:
  T value;
  NanobindWrapper(T &&value) : value(value) {}
  operator T() { return value; }
};
} // namespace M::Graph::Python

namespace NB_NAMESPACE {
namespace detail {

/// Casts object <-> MLIRContext.
template <>
struct type_caster<::mlir::MLIRContext> {
protected:
  ::mlir::MLIRContext *value;

public:
  static constexpr auto Name = const_name("Context");

  template <typename T>
  using Cast = ::mlir::MLIRContext *;

  operator ::mlir::MLIRContext *() { return value; }
  operator ::mlir::MLIRContext *&() { return value; }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (src.is_none()) {
      src = nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Context")
                .attr("current");
    }
    nb::object capsule = mlirApiObjectToCapsule(src);
    value = unwrap(mlirPythonCapsuleToContext(capsule.ptr()));
    return !mlirContextIsNull(wrap(value));
  }
};

/// Casts object <-> MlirType.
template <>
struct type_caster<::mlir::Type> {
  NB_TYPE_CASTER(::mlir::Type, const_name("Type"))
  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) {
    nb::object capsule = mlirApiObjectToCapsule(src);
    value = unwrap(mlirPythonCapsuleToType(capsule.ptr()));
    return !mlirTypeIsNull(wrap(value));
  }
  static handle from_cpp(::mlir::Type t, rv_policy policy,
                         cleanup_list *cleanup) {
    nb::object capsule =
        nb::steal<nb::object>(mlirPythonTypeToCapsule(wrap(t)));
    return nb::module_::import_(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Type")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  }
};

/// Casts object <-> DType.
template <>
struct type_caster<::M::DType> {
  NB_TYPE_CASTER(::M::DType, const_name("max._core.dtype.DType"))
  using DTypeCaster = make_caster<M::DType::Cases>;
  DTypeCaster dtypeCaster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (dtypeCaster.from_python(src, flags, cleanup)) {
      value = M::DType(dtypeCaster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(::M::DType t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return DTypeCaster::from_cpp(static_cast<M::DType::Cases>(t.getValue()),
                                 policy, cleanup);
  }
};

/// Casts sequence <-> ArrayRef.
template <typename Entry>
struct type_caster<::llvm::ArrayRef<Entry>> {
  using Caster = make_caster<Entry>;
  using VecCaster = make_caster<std::vector<Entry>>;
  NB_TYPE_CASTER(::llvm::ArrayRef<Entry>,
                 const_name("collections.abc.Sequence[") + Caster::Name +
                     const_name("]"))

  VecCaster caster;
  bool used = false;

  bool from_python(handle_t<nb::sequence> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    // Nanobind's built in typecasters for collections re-use
    // their internal typecasters, which isn't safe for array refs.
    // We can do something smart like specialize the caster for
    // std::vector<ArrayRef<T>> to hold a vector of typecaster instances.
    ASSERT_STREAM(!used, "ArrayRef typecasters cannot be reused.");
    if (!caster.from_python(src, flags, cleanup))
      return false;
    used = true;
    const Entry *start = caster.value.data();
    value = ::llvm::ArrayRef<Entry>(start, caster.value.size());
    return true;
  }
  static handle from_cpp(::llvm::ArrayRef<Entry> ar, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    // TODO(MAXPLAT-123): Make these views.
    return VecCaster::from_cpp(ar.vec(), policy, cleanup);
  }
};

/// Casts sequence[bool] <-> ArrayRef<bool>.
/// The default implementation doesn't work because std::vector<bool>
/// is specialized as a bit vector, which we can't take a reference to.
template <>
struct type_caster<::llvm::ArrayRef<bool>> {
  using Caster = make_caster<bool>;
  NB_TYPE_CASTER(::llvm::ArrayRef<bool>,
                 const_name("collections.abc.Sequence[bool]"))

  Caster caster;
  ::llvm::SmallVector<bool> storage;
  bool used = false;

  bool from_python(handle_t<nb::sequence> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    // Nanobind's built in typecasters for collections re-use
    // their internal typecasters, which isn't safe for array refs.
    // We can do something smart like specialize the caster for
    // std::vector<ArrayRef<T>> to hold a vector of typecaster instances.
    ASSERT_STREAM(!used, "ArrayRef typecasters cannot be reused.");
    for (auto entry : src) {
      if (!caster.from_python(entry, flags, cleanup)) {
        storage.clear();
        return false;
      }
      storage.push_back(caster.value);
    }
    used = true;
    const bool *start = storage.data();
    value = ::llvm::ArrayRef<bool>(start, storage.size());
    return true;
  }

  static handle from_cpp(::llvm::ArrayRef<bool> ar, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    // TODO(MAXPLAT-123): Make these views.
    return make_caster<std::vector<bool>>::from_cpp(ar.vec(), policy, cleanup);
  }
};

} // namespace detail
} // namespace NB_NAMESPACE

#endif // SDK_GRAPHAPI_PYTHON_TYPECASTERS_H
