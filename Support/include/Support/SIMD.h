//===- Support/SIMD.h -----------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// This file implements an architecture independent SIMD wrapper.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_SIMD_H
#define SUPPORT_SIMD_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Compiler.h" // for __has_builtin
#include "llvm/Support/TypeName.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <functional>
#include <numeric>
#include <type_traits>

namespace M {
#if defined(__AVX512__)
static constexpr size_t kPreferredSIMDBitWidth = 512;
#elif defined(__AVX2__) || defined(__AVX__)
static constexpr size_t kPreferredSIMDBitWidth = 256;
#elif defined(__ARM_NEON__) || defined(__ARM_NEON)
static constexpr size_t kPreferredSIMDBitWidth = 128;
#else
static constexpr size_t kPreferredSIMDBitWidth = 128;
#endif

/// For our optimized representation of SIMDVector, we use Clang's extended
/// vector type.  For compatibility, we use std::array.
#ifdef __GNUC__
#define LLCL_SIMD_EMULATED 0
#else
#define LLCL_SIMD_EMULATED 1
#endif

template <typename T, size_t Width>
class SIMDVector;

//===----------------------------------------------------------------------===//
// is_simd_vector
//===----------------------------------------------------------------------===//

template <typename...>
struct is_simd_vector : std::false_type {};
template <typename T, size_t Width>
struct is_simd_vector<SIMDVector<T, Width>> : std::true_type {};

template <typename... T>
inline constexpr bool is_simd_vector_v = is_simd_vector<T...>::value;

//===----------------------------------------------------------------------===//
// SIMDVector
//===----------------------------------------------------------------------===//

/// The SIMDVector class is an architecture independent wrapper for operating on
/// SIMD types. Width is the number of elements in the SIMDVector (and not the
/// bytecount nor the bitcount). The SIMDVector is designed to emulate the SIMD
/// operations if they do not exist on the target architecture or host compiler.
template <typename ElemTy,
          size_t Width = kPreferredSIMDBitWidth / (CHAR_BIT * sizeof(ElemTy))>
class SIMDVector {

  static_assert(Width != 0, "Width must be positive");
  static_assert(std::is_arithmetic_v<ElemTy>, "ElemTy must be arithmetic");

public:
  using element_type = ElemTy;
  static constexpr size_t byte_count = Width * sizeof(ElemTy);
  static constexpr size_t bit_count = CHAR_BIT * byte_count;
  static constexpr size_t width = Width;

#if LLCL_SIMD_EMULATED
  // We are just going to emulate the SIMD type using an array.
  using vector_type = std::array<element_type, width>;
#else  // LLCL_SIMD_EMULATED
  using vector_type __attribute__((vector_size(byte_count))) = element_type;
#endif // LLCL_SIMD_EMULATED

  SIMDVector() = default;

  /// Initialize the vector with the vector_type.
  SIMDVector(vector_type v) {
    if constexpr (isEmulated)
      memcpy(data(), v.data(), byte_count);
    else
      vectorData = v;
  }

  /// Initializes the vector with the given values. If a single scalar is
  /// provided, then the value is broadcasted.
  template <typename Arg>
  SIMDVector(Arg arg) {
    // If the input is a simd vector, then copy the data.
    if constexpr (is_simd_vector_v<Arg>) {
      static_assert(Arg::width == width, "width mismatch");
      if constexpr (std::is_same_v<typename Arg::element_type, element_type>) {
        if constexpr (isEmulated)
          memcpy(data(), arg.data(), byte_count);
        else
          vectorData = arg.value();
      } else {
        std::copy(arg.data(), arg.data() + arg.size(), data());
      }
    } else {
      // Otherwise, we are going to splat the arithmetic value into the vector.
      static_assert(std::is_arithmetic_v<Arg>,
                    "broadcasted value must be an arithmetic type.");
      if constexpr (isEmulated)
        std::fill(data(), data() + width, (element_type)arg);
      else
        vectorData = element_type(arg) - vector_type{};
    }
  }

  /// Initializes the vector with the given values. The number of input values
  /// must be equal to the width of the vector.
  template <typename Arg, typename... Args>
  SIMDVector(Arg arg0, Arg arg1, Args... args) {
    static_assert(sizeof...(Args) == width - 2, "Wrong number of arguments");
    static_assert(std::is_arithmetic_v<Arg> &&
                      (std::is_arithmetic_v<Args> && ...),
                  "Args must be an arithmetic type.");
    std::initializer_list<element_type> initList{arg0, arg1, args...};
    memcpy(data(), &initList, byte_count);
  }

  /// Gets the number of elements in the vector.
  static constexpr size_t size() { return width; }

  /// Gets the size of the vector in bytes.
  static constexpr size_t getSizeInBytes() { return byte_count; }

  /// Loads the vector from the given memory location.
  static SIMDVector loadFrom(const element_type *mem) {
    SIMDVector result;
    result.assign(mem, mem + size());
    return result;
  }

  /// Bitcasts the vector into another SIMD vector with the same bytecount.
  template <typename ResultElementType, size_t TargetWidth = Width>
  SIMDVector<ResultElementType, TargetWidth> bitCast() const {
    if constexpr (std::is_same_v<ResultElementType, ElemTy> &&
                  TargetWidth == Width) {
      return *this;
    } else {
      SIMDVector<ResultElementType, TargetWidth> result;
      static_assert(result.getSizeInBytes() == getSizeInBytes(),
                    "bytecount mismatch");
      memcpy(result.data(), data(), getSizeInBytes());
      return result;
    }
  }

  /// Loads the vector from the given memory range.
  void assign(const element_type *begin, const element_type *end) {
    assert(std::distance(begin, end) == size() &&
           "Wrong number of elements when assigning the vector");
    memcpy(data(), begin, getSizeInBytes());
  }

  /// Stores the vector to the given memory location.
  void storeTo(element_type *mem) const {
    memcpy(mem, data(), getSizeInBytes());
  }

  /// Returns the element at the given index.
  const element_type &operator[](size_t index) const {
    assert(index < size() && "Index out of bounds");
    return vectorData[index];
  }
  element_type &operator[](size_t index) {
    assert(index < size() && "Index out of bounds");
    return vectorData[index];
  }

  /// Returns the simd vector where each element is added to the input scalar
  /// value.
  SIMDVector operator+(ElemTy other) {
    SIMDVector otherVector(other);
    return *this + otherVector;
  }

  /// Adds the given vector to this vector.
  SIMDVector operator+(const SIMDVector &other) const {
    SIMDVector result(*this);
    result += other;
    return result;
  }

  /// Inplace adds the given vector to this vector.
  SIMDVector &operator+=(const SIMDVector &other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a + b; });
    else
      vectorData += other.value();
    return *this;
  }

  /// Negates the given vector.
  SIMDVector operator-() const {
    SIMDVector result(0);
    result -= *this;
    return result;
  }

  /// Subtracts the given vector to this vector.
  SIMDVector operator-(const SIMDVector &other) const {
    SIMDVector result(*this);
    result -= other;
    return result;
  }

  /// Inplace subtracts the given vector to this vector.
  SIMDVector &operator-=(const SIMDVector &other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a - b; });
    else
      vectorData -= other.value();
    return *this;
  }

  /// Multiplies the given vector to this vector.
  SIMDVector operator*(const SIMDVector &other) const {
    SIMDVector result(*this);
    result *= other;
    return result;
  }

  /// Inplace multiplies the given vector to this vector.
  SIMDVector &operator*=(const SIMDVector &other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a * b; });
    else
      vectorData *= other.value();
    return *this;
  }

  /// Divide the given vector to this vector.
  SIMDVector operator/(const SIMDVector &other) const {
    SIMDVector result(*this);
    result /= other;
    return result;
  }

  /// Inplace divides the given vector to this vector.
  SIMDVector &operator/=(const SIMDVector &other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a / b; });
    else
      vectorData /= other.value();
    return *this;
  }

  /// Performs a bitwise shift left between the given vector and this vector.
  SIMDVector operator<<(const SIMDVector<int32_t, Width> &other) const {
    SIMDVector result(*this);
    result <<= other;
    return result;
  }

  /// Performs inplace bitwise shift left between the given vector and this
  /// vector.
  SIMDVector operator<<=(const SIMDVector<int32_t, Width> &other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a << b; });
    else
      vectorData <<= other.value();
    return *this;
  }

  /// Performs a bitwise shift right between the given vector and this vector.
  SIMDVector operator>>(const SIMDVector<int32_t, Width> &other) const {
    SIMDVector result(*this);
    result >>= other;
    return result;
  }

  /// Performs inplace bitwise shift right between the given vector and this
  /// vector.
  SIMDVector operator>>=(const SIMDVector<int32_t, Width> &other) {
    if constexpr (isEmulated)
      std::transform(data(), data() + size(), other.data(), data(),
                     [](auto a, auto b) { return a >> b; });
    else
      vectorData >>= other.value();
    return *this;
  }

  /// Performs element-wise equality comparison between two simd vectors.
  SIMDVector<int32_t, Width> operator==(const SIMDVector &other) const {
    if constexpr (isEmulated) {
      SIMDVector<int32_t, Width> result;
      std::transform(data(), data() + size(), other.data(), result.data(),
                     [](auto a, auto b) { return a == b; });
      return result;
    }
    return this->value() == other.value();
  }

  /// Performs element-wise less-than comparison between two simd vectors.
  SIMDVector<int32_t, Width> operator<(const SIMDVector &other) const {
    if constexpr (isEmulated) {
      SIMDVector<int32_t, Width> result;
      std::transform(data(), data() + size(), other.data(), result.data(),
                     [](auto a, auto b) { return a < b; });
      return result;
    }
    return this->value() < other.value();
  }

  /// Performs element-wise greater-than comparison between two simd vectors.
  SIMDVector<int32_t, Width> operator>(const SIMDVector &other) const {
    if constexpr (isEmulated) {
      SIMDVector<int32_t, Width> result;
      std::transform(data(), data() + size(), other.data(), result.data(),
                     [](auto a, auto b) { return a > b; });
      return result;
    }
    return this->value() > other.value();
  }

  /// Returns a new SIMDVector where the function is applied to each element of
  /// the vector. Note that because this takes an arbitrary function, we cannot
  /// perform this operation in a SIMD vectorized manner. The resulting SIMD
  /// vector will have the same element type as the return type of the function.
  template <typename FuncTy>
  SIMDVector<std::invoke_result_t<FuncTy, element_type>, width>
  transform(FuncTy func) const {
    SIMDVector result;
    std::transform(data(), data() + size(), result.data(), func);
    return result;
  }

  /// Computes the elementwise absolute value of the simd vector.
  SIMDVector abs() const {
    if constexpr (std::is_unsigned_v<element_type>) {
      return *this;
    } else {
#if __has_builtin(__builtin_elementwise_abs)
      static_assert(!isEmulated,
                    "Expect that the SIMD vector is not emulated if "
                    "__builtin_elementwise_abs is available.");
      return __builtin_elementwise_abs(value());
#else  // __has_builtin(__builtin_elementwise_abs)
      return transform([](auto a) { return std::abs(a); });
#endif // __has_builtin(__builtin_elementwise_abs)
    }
  }

  /// Computes the elementwise min value of the two simd vectors.
  SIMDVector elementwiseMin(const SIMDVector &other) const {
#if __has_builtin(__builtin_elementwise_min)
    static_assert(!isEmulated, "Expect that the SIMD vector is not emulated if "
                               "__builtin_elementwise_min is available.");
    return __builtin_elementwise_min(value(), other.value());
#else  // __has_builtin(__builtin_elementwise_min)
    SIMDVector result;
    std::transform(data(), data() + size(), other.data(), result.data(),
                   [](auto a, auto b) { return std::min(a, b); });
    return result;
#endif // __has_builtin(__builtin_elementwise_min)
  }

  /// Computes the elementwise max value of the two simd vectors.
  SIMDVector elementwiseMax(const SIMDVector &other) const {
#if __has_builtin(__builtin_elementwise_max)
    static_assert(!isEmulated, "Expect that the SIMD vector is not emulated if "
                               "__builtin_elementwise_max is available.");
    return __builtin_elementwise_max(value(), other.value());
#else  // __has_builtin(__builtin_elementwise_max)
    SIMDVector result;
    std::transform(data(), data() + size(), other.data(), result.data(),
                   [](auto a, auto b) { return std::max(a, b); });
    return result;
#endif // __has_builtin(__builtin_elementwise_max)
  }

  /// Uses the current simd vector as a selector, takes two vectors, and returns
  /// a new simd vector where the values are selected from the lhs or rhs
  /// depending on the selected value. The behavior is the same as the ternary
  /// if operator. The selector vector must contain either 0 or 1.
  template <typename T, int N>
  SIMDVector<T, N> select(const SIMDVector<T, N> &lhs,
                          const SIMDVector<T, N> &rhs) const {
    static_assert(
        std::is_same_v<element_type, int32_t>,
        "the selector SIMD vector must have an int32_t element type.");
    static_assert(width == N, "the selector SIMD vector width must be equal of "
                              "the operand SIMD width.");
    if constexpr (isEmulated) {
      SIMDVector<T, N> result;
      for (int i = 0, e = size(); i < e; ++i)
        result[i] = data()[i] ? lhs[i] : rhs[i];
      return result;
    } else if constexpr (sizeof(T) == sizeof(element_type)) {
      // This is a special case optimization for 32-bit types. For example, when
      // the element_type is a float, then compilers such as GCC produces better
      // assembly instructions.
      auto lhsInt = *reinterpret_cast<const vector_type *>(&lhs.value());
      auto rhsInt = *reinterpret_cast<const vector_type *>(&rhs.value());
      auto values = (value() & lhsInt) | (~value() & rhsInt);
      SIMDVector<T, N> result;
      std::memcpy(result.data(), &values, result.byte_count);
      return result;
    } else {
      // Otherwise, we use the mask as the condition and use the ternary
      // operator.
      auto mask = value();
      SIMDVector<T, N> result = mask ? lhs.value() : rhs.value();
      return result;
    }
  }

  /// Reduces the values within the vector using the addition operator.
  element_type reduceAdd() const {
    if constexpr (isEmulated || !__has_builtin(__builtin_reduce_add))
      return std::reduce(data(), data() + size(), element_type(0),
                         std::plus<>());
    else
      return __builtin_reduce_add(value());
  }

  /// Reduces the values within the vector using the multiplication operator.
  element_type reduceMul() const {
    if constexpr (isEmulated || !__has_builtin(__builtin_reduce_mul))
      return std::reduce(data(), data() + size(), element_type(1),
                         std::multiplies<>());
    else
      return __builtin_reduce_mul(value());
  }

  /// Reduces the values within the vector using the max operator.
  element_type reduceMax() const {
    if constexpr (isEmulated || !__has_builtin(__builtin_reduce_max))
      return std::reduce(data(), data() + size(),
                         std::numeric_limits<element_type>::lowest(),
                         [](auto a, auto b) { return std::max(a, b); });
    else
      return __builtin_reduce_max(value());
  }

  /// Reduces the values within the vector using the min operator.
  element_type reduceMin() const {
    if constexpr (isEmulated || !__has_builtin(__builtin_reduce_min))
      return std::reduce(data(), data() + size(),
                         std::numeric_limits<element_type>::max(),
                         [](auto a, auto b) { return std::min(a, b); });
    else
      return __builtin_reduce_min(value());
  }

  /// Gets the underlying vector value.
  const vector_type &value() const { return vectorData; }
  vector_type &value() { return vectorData; }

  /// Get the underlying data of the simd vector as a pointer.
  element_type *data() {
    if constexpr (std::is_array_v<vector_type>)
      return vectorData.data();
    return (element_type *)&vectorData;
  }

  /// Get the underlying data of the simd vector as a const pointer.
  const element_type *data() const {
    if constexpr (std::is_array_v<vector_type>)
      return vectorData.data();
    return (const element_type *)&vectorData;
  }

  void print(raw_ostream &os) const {
    os << "SIMDVector([";
    llvm::interleaveComma(llvm::makeArrayRef(data(), size()), os);
    os << "], dtype=" << llvm::getTypeName<element_type>()
       << ", width=" << width << ")";
  }
  void dump() const { print(llvm::errs()); }

private:
  /// If the isEmulated flag is true, then we are not actually explicitly using
  /// SIMD instructions. Instead, we are looping over the elements of the vector
  /// and performing the operation.
  static constexpr bool isEmulated = LLCL_SIMD_EMULATED == 1;

  /// The underlying simd vector data.
  vector_type vectorData;
};

template <typename ElementType, size_t Width>
inline raw_ostream &operator<<(raw_ostream &os,
                               const SIMDVector<ElementType, Width> &value) {
  value.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// simd_transform
//===----------------------------------------------------------------------===//
//
// This is a helper function that is used to transform a range of values in SIMD
// fashion. It takes 2 functions, one that is used to transform the values on
// SIMD vectors and another that is used to transform values on scalar values.
// The function then applies those two functions on the ranges provided. It is
// supposed to look like std::transform.
//
//===----------------------------------------------------------------------===//
template <typename InputElemTy, typename OutputElemTy,
          typename UnarySIMDFunctionTy, typename UnaryScalarFunctionTy,
          size_t Width = kPreferredSIMDBitWidth /
                         (CHAR_BIT * sizeof(InputElemTy))>
OutputElemTy *simd_transform(const InputElemTy *first, const InputElemTy *last,
                             OutputElemTy *result, UnarySIMDFunctionTy simdFunc,
                             UnaryScalarFunctionTy scalarFunc) {
  // If we are emulating the SIMD vector, then there is no point of using
  // simd_transform.. So, fallback to std::transform.
#if LLCL_SIMD_EMULATED == 0
  using InputSIMDType = SIMDVector<InputElemTy, Width>;
  using OutputSIMDType = SIMDVector<OutputElemTy, Width>;
  // Apply the simd function on the input buffer in batches of simd size.
  for (; first <= last - InputSIMDType::size();
       first += InputSIMDType::size(), result += OutputSIMDType::size()) {
    InputSIMDType input(InputSIMDType::loadFrom(first));
    OutputSIMDType output(simdFunc(input));
    output.storeTo(result);
  }
  // Handle the remaining elements when the buffer is not a multiple of simd
  // size.
#endif // LLCL_SIMD_EMULATED
  return std::transform(first, last, result, scalarFunc);
}

template <typename InputElemTy, typename OutputElemTy,
          typename BinarySIMDFunctionTy, typename BinaryScalarFunctionTy,
          size_t Width = kPreferredSIMDBitWidth /
                         (CHAR_BIT * sizeof(InputElemTy))>
OutputElemTy *simd_transform(const InputElemTy *first1,
                             const InputElemTy *last1,
                             const InputElemTy *first2, OutputElemTy *result,
                             BinarySIMDFunctionTy simdFunc,
                             BinaryScalarFunctionTy scalarFunc) {
  // If we are emulating the SIMD vector, then there is no point of using
  // simd_transform.. So, fallback to std::transform.
#if LLCL_SIMD_EMULATED == 0
  using InputSIMDType = SIMDVector<InputElemTy, Width>;
  using OutputSIMDType = SIMDVector<OutputElemTy, Width>;
  // Apply the simd function on the input buffers in batches of simd size.
  for (; first1 <= last1 - InputSIMDType::size();
       first1 += InputSIMDType::size(), first2 += InputSIMDType::size(),
       result += OutputSIMDType::size()) {
    InputSIMDType input1(InputSIMDType::loadFrom(first1));
    InputSIMDType input2(InputSIMDType::loadFrom(first2));
    OutputSIMDType output(simdFunc(input1, input2));
    output.storeTo(result);
  }
  // Handle the remaining elements when the buffer is not a multiple of simd
  // size.
#endif // LLCL_SIMD_EMULATED
  return std::transform(first1, last1, first2, result, scalarFunc);
}

} // namespace M

#endif // GENERICML_SUPPORT_SIMD_H
