//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_TYPECASTERS_H
#define SDK_GRAPHAPI_PYTHON_TYPECASTERS_H

#include "Support/ML/DType.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/BuiltinDialect.h"
#include "pybind11/pybind11.h"

namespace M::Graph::Python {
template <typename T>
class PybindWrapper {
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
  PybindWrapper(T &&value) : value(value) {}
  operator T() { return value; }
};
} // namespace M::Graph::Python

namespace PYBIND11_NAMESPACE {
namespace detail {

/// Casts object <-> MLIRContext.
template <>
struct type_caster<::mlir::MLIRContext> {
protected:
  ::mlir::MLIRContext *value;

public:
  static constexpr auto name = const_name("Context");

  operator ::mlir::MLIRContext *() { return value; }
  operator ::mlir::MLIRContext *&() { return value; }
  template <typename T>
  using cast_op_type =
      conditional_t<std::is_pointer<remove_reference_t<T>>::value,
                    typename std::add_pointer<intrinsic_t<T>>::type,
                    typename std::add_lvalue_reference<intrinsic_t<T>>::type>;

  bool load(handle src, bool should_implicit_convert) {
    if (src.is_none()) {
      src = py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
                .attr("Context")
                .attr("current");
    }
    py::object capsule = mlirApiObjectToCapsule(src);
    value = unwrap(mlirPythonCapsuleToContext(capsule.ptr()));
    return !mlirContextIsNull(wrap(value));
  }
};

/// Casts object <-> MlirType.
template <>
struct type_caster<::mlir::Type> {
  PYBIND11_TYPE_CASTER(::mlir::Type, const_name("Type"));
  bool load(handle src, bool) {
    py::object capsule = mlirApiObjectToCapsule(src);
    value = unwrap(mlirPythonCapsuleToType(capsule.ptr()));
    return !mlirTypeIsNull(wrap(value));
  }
  static handle cast(::mlir::Type t, return_value_policy, handle) {
    py::object capsule =
        py::reinterpret_steal<py::object>(mlirPythonTypeToCapsule(wrap(t)));
    return py::module::import(MAKE_MLIR_PYTHON_QUALNAME("ir"))
        .attr("Type")
        .attr(MLIR_PYTHON_CAPI_FACTORY_ATTR)(capsule)
        .attr(MLIR_PYTHON_MAYBE_DOWNCAST_ATTR)()
        .release();
  }
};

/// Casts object <-> DType.
template <>
struct type_caster<::M::DType> {
  PYBIND11_TYPE_CASTER(::M::DType, const_name("DType"));
  bool load(handle src, bool) {
    value = M::DType(src.cast<uint8_t>());
    return true;
  }
  static handle cast(::M::DType t, return_value_policy, handle) {
    return py::cast(t.getValue());
  }
};

} // namespace detail
} // namespace PYBIND11_NAMESPACE

#endif // SDK_GRAPHAPI_PYTHON_TYPECASTERS_H
