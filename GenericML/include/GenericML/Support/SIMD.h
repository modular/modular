//===- GenericML/Support/SIMD.h -------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements an architecture independent SIMD wrapper.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_SIMD_H
#define GENERICML_SUPPORT_SIMD_H

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <type_traits>

namespace M {
// TODO: Add the kPreferredSIMDBitWidth by detecting the SIMD width at compile
// time.
#ifdef __AVX2__
static constexpr size_t kPreferredSIMDBitWidth = 256;
#else  // __AVX2__
static constexpr size_t kPreferredSIMDBitWidth = 128;
#endif // __AVX2__

/// The SIMDVector class is an architecture independent wrapper for operating on
/// SIMD types. It is designed to emulate the SIMD operations if the do not
/// exist on the target architecture.
template <typename ElemTy, size_t SIMDBitWidth = kPreferredSIMDBitWidth>
class SIMDVector {

  static_assert(SIMDBitWidth > 0, "SIMDBitWidth must be positive");
  static_assert(SIMDBitWidth % 2 == 0, "SIMDBitWidth must be even");
  static_assert(SIMDBitWidth % sizeof(ElemTy) == 0,
                "SIMDBitWidth must be a multiple of the size of the ElemTy");
  static_assert(std::is_arithmetic_v<ElemTy>, "ElemTy must be arithmetic");

public:
  using element_type = ElemTy;
#ifdef __GNUC__
  using vector_type = element_type __attribute__((
      vector_size(sizeof(element_type) * (SIMDBitWidth / (8 * sizeof(ElemTy)))),
      aligned(1)));
#else  // __GNUC__
  // We are just going to emulate the SIMD type using an array.
  using vector_type = std::array<element_type, SIMDBitWidth>;
#endif // __GNUC__

  SIMDVector() = default;

  /// Initializes the vector with the given values. If a single input is
  /// provided, then the value is broadcasted.
  template <typename Arg, typename... Args>
  SIMDVector(Arg arg, Args... args) {
    if constexpr ((sizeof...(Args)) == 0) {
      static_assert(std::is_trivially_assignable_v<Arg, element_type>,
                    "broadcasted value must be trivially assignable to the "
                    "element type of the SIMD vector.");
      if constexpr (isEmulated)
        std::fill(data(), data() + width, (element_type)arg);
      else
        vectorData = element_type(arg) - vector_type{};
    } else {
      static_assert(sizeof...(Args) == width - 1, "Wrong number of arguments");
      static_assert(
          std::is_trivially_assignable_v<Arg, element_type> &&
              (std::is_trivially_assignable_v<Args, element_type> && ...),
          "Args must be trivially assignable to element_type");
      std::initializer_list<element_type> initList{arg, args...};
      memcpy(data(), &initList, width * sizeof(element_type));
    }
  }

  /// Gets the number of elements in the vector.
  static constexpr size_t size() { return width; }

  /// Gets the size of the vector in bytes.
  static constexpr size_t getSizeInBytes() {
    return size() * sizeof(element_type);
  }

  /// Loads the vector from the given memory location.
  static SIMDVector loadFrom(const element_type *mem) {
    SIMDVector<ElemTy, SIMDBitWidth> result;
    result.assign(mem, mem + size());
    return result;
  }

  /// Loads the vector from the given memory range.
  void assign(const element_type *begin, const element_type *end) {
    assert(std::distance(begin, end) == width &&
           "Wrong number of elements when assigning the vector");
    memcpy(data(), begin, getSizeInBytes());
  }

  /// Stores the vector to the given memory location.
  void storeTo(element_type *mem) { memcpy(mem, data(), getSizeInBytes()); }

  /// Returns the element at the given index.
  element_type &operator[](size_t index) {
    assert(index < size() && "Index out of bounds");
    return vectorData[index];
  }

  /// Adds the given vector to this vector.
  SIMDVector operator+(SIMDVector other) const {
    SIMDVector result(*this);
    result += other;
    return result;
  }

  /// Inplace adds the given vector to this vector.
  SIMDVector &operator+=(SIMDVector other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a + b; });
    else
      vectorData += other.value();
    return *this;
  }

  /// Subtracts the given vector to this vector.
  SIMDVector operator-(SIMDVector other) const {
    SIMDVector result(*this);
    result -= other;
    return result;
  }

  /// Inplace subtracts the given vector to this vector.
  SIMDVector &operator-=(SIMDVector other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a - b; });
    else
      vectorData -= other.value();
    return *this;
  }

  /// Subtracts the given vector to this vector.
  SIMDVector operator*(SIMDVector other) const {
    SIMDVector result(*this);
    result *= other;
    return result;
  }

  /// Inplace multiplies the given vector to this vector.
  SIMDVector &operator*=(SIMDVector other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a * b; });
    else
      vectorData *= other.value();
    return *this;
  }

private:
#ifdef __GNUC__
  static constexpr bool isEmulated = false;
#else  // __GNUC__
  static constexpr bool isEmulated = true;
#endif // __GNUC__
  vector_type vectorData;
  static constexpr size_t width = SIMDBitWidth / (8 * sizeof(ElemTy));

  const vector_type &value() { return vectorData; }

  element_type *data() {
    if constexpr (std::is_array_v<vector_type>)
      return vectorData.data();
    else
      return (element_type *)&vectorData;
  }
};
} // namespace M

#endif // GENERICML_SUPPORT_SIMD_H
