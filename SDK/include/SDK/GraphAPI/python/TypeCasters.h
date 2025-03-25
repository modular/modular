//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_GRAPHAPI_PYTHON_TYPECASTERS_H
#define SDK_GRAPHAPI_PYTHON_TYPECASTERS_H

#include "Support/ML/DType.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/Location.h"
#include "nanobind/nanobind.h"
#include "nanobind/stl/string_view.h"
#include <SDK/GraphAPI/python/Bindings.h>
#include <Support/AssertStream.h>
#include <mlir-c/Bindings/Python/Interop.h>
#include <mlir-c/IR.h>
#include <nanobind/stl/unique_ptr.h>

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

namespace {
template <std::size_t N>
struct StringLiteral {
  char data[N + 1];
  constexpr StringLiteral(const llvm::StringRef str) {
    for (size_t i = 0; i < N; ++i)
      data[i] = str.data()[i];
    data[N] = '\0';
  }
};

template <typename T>
constexpr auto type_name() {
  constexpr auto name = llvm::getTypeName<T>();
  return nb::detail::const_name(StringLiteral<name.size()>(name).data);
}

template <typename T>
struct is_attribute_interface_base {
private:
  template <typename C, typename Concrete, typename... Traits>
  static std::true_type
  test(const mlir::AttributeInterface<Concrete, Traits...> *);

  template <typename C>
  static std::false_type test(...);

public:
  using type = decltype(test<T>(std::declval<T *>()));
  static constexpr bool value = type::value;
};

template <typename T>
constexpr bool is_attribute_interface() {
  return is_attribute_interface_base<T>::value;
}
} // namespace

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
    nb::object capsule;
    try {
      capsule = mlirApiObjectToCapsule(src);
    } catch (nb::builtin_exception) {
      return false;
    }
    value = unwrap(mlirPythonCapsuleToContext(capsule.ptr()));
    return !mlirContextIsNull(wrap(value));
  }
};

/// Casts MlirLocation <-> mlir::Location.
template <>
struct type_caster<::mlir::Location> {
  static constexpr auto Name = const_name("Location");
  template <typename T>
  using Cast = movable_cast_t<::mlir::Location>;
  using Caster = make_caster<MlirLocation>;
  Caster caster;

  std::optional<::mlir::Location> value;

  explicit operator ::mlir::Location *() { return &*value; }
  explicit operator ::mlir::Location &() { return *value; }
  explicit operator ::mlir::Location &&() { return std::move(*value); }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = unwrap(caster.value);
    return true;
  }

  template <typename T>
  static constexpr bool can_cast() {
    return Caster::can_cast<T>();
  }

  static handle from_cpp(::mlir::Location t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(wrap(t), policy, cleanup);
  }
};

/// Casts str <-> llvm::StringRef.
template <>
struct type_caster<::llvm::StringRef> {
  NB_TYPE_CASTER(::llvm::StringRef, const_name("str"))
  using Caster = make_caster<std::string_view>;
  Caster caster;

  bool from_python(handle_t<nb::str> src, uint8_t flags,
                   cleanup_list *cleanup) noexcept {
    if (!caster.from_python(src, flags, cleanup)) {
      return false;
    }
    value = caster.value;
    return true;
  }

  static handle from_cpp(::llvm::StringRef t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::from_cpp(t, policy, cleanup);
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

/// Casts AttributeInterface <-> python.
template <typename AttributeInterface>
struct type_caster<
    AttributeInterface,
    std::enable_if_t<is_attribute_interface<AttributeInterface>(), int>> {
  NB_TYPE_CASTER(AttributeInterface, const_name("\"") +
                                         type_name<AttributeInterface>() +
                                         const_name("\""))

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    ::mlir::Attribute attr;
    try {
      attr = nb::cast<::mlir::Attribute>(src);
    } catch (nb::cast_error) {
      return false;
    }
    if (!::mlir::isa<AttributeInterface>(attr))
      return false;
    value = ::mlir::dyn_cast<AttributeInterface>(attr);
    return true;
  }

  static handle from_cpp(AttributeInterface ar, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return make_caster<std::unique_ptr<::mlir::Attribute>>::from_cpp(
        std::make_unique<::mlir::Attribute>(ar), policy, cleanup);
  }
};

template <>
struct type_hook<::mlir::Attribute> {
  static const std::type_info *get(::mlir::Attribute *attr) {
    if (attr) {
      if (auto info = M::Graph::Python::lookupTypeID(attr->getTypeID()))
        return info;
    }
    return &typeid(::mlir::Attribute);
  }
};

template <>
struct type_hook<::mlir::Type> {
  static const std::type_info *get(::mlir::Type *attr) {
    if (attr) {
      if (auto info = M::Graph::Python::lookupTypeID(attr->getTypeID()))
        return info;
    }
    return &typeid(::mlir::Type);
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

/// Casts object <-> mlir::TypeRange.
template <>
struct type_caster<::mlir::TypeRange> {
  using Caster = make_caster<llvm::ArrayRef<mlir::Type>>;
  NB_TYPE_CASTER(::mlir::TypeRange, Caster::Name)
  Caster caster;

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    if (caster.from_python(src, flags, cleanup)) {
      value = mlir::TypeRange(caster.value);
      return true;
    }
    return false;
  }
  static handle from_cpp(::mlir::TypeRange t, rv_policy policy,
                         cleanup_list *cleanup) noexcept {
    return Caster::VecCaster::from_cpp(t, policy, cleanup);
  }
};

} // namespace detail
} // namespace NB_NAMESPACE

#endif // SDK_GRAPHAPI_PYTHON_TYPECASTERS_H
