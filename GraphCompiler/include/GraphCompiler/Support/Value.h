//===----------------------------------------------------------------------===//
// Copyright (c) 2026, Modular Inc. All rights reserved.
//
// Licensed under the MAX License:
// https://modular.com/legal/max-license
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//===----------------------------------------------------------------------===//

#ifndef GENERICML_SUPPORT_VALUE_H
#define GENERICML_SUPPORT_VALUE_H

#include "AsyncRT/Runtime/AnyAsyncValueRef.h"
#include "MLRT/Core/Tensor/Tensor.h"

namespace M::GML {

/// Base class for `Dict`, `List`, `Tuple`: lightweight stand-ins for the Python
/// types that glue the C and Python APIs to non-tensor model values. Model
/// internals can't escape execution, and `nanobind::dict` and friends need RTTI
/// so can't go below the bindings. These avoid MLIR, Python, and framework
/// dependencies (symbol/RTTI conflicts) and need no dynamic casts.
class Container {};

/// Pythonic `None`, tunneled between the C/Python API and compiled models.
class None {
public:
  None() = default;

  None(const None &other) = default;

  None &operator=(const None &other) = default;
  None &operator=(None &&other);

  /// Returns the string representation "<none>".
  std::string to_string() const;

  /// Returns true for all instances of `None`.
  bool operator==(const None &other) const;

  /// Returns false for all instances of `None`.
  bool operator<(const None &other) const;
};

raw_ostream &operator<<(raw_ostream &os, const None &none);
std::ostream &operator<<(std::ostream &os, const None &none);

// Enum class to flag special keys used for `DenseMapInfo`.
enum class DenseMapInfoKey {
  Empty = 0,
  Tombstone = 1,
  Value = 2,
};

raw_ostream &operator<<(raw_ostream &os, const DenseMapInfoKey &info);
std::ostream &operator<<(std::ostream &os, const DenseMapInfoKey &info);

/// Wraps `AnyAsyncValueRef` for use as a map key (`hash`, `to_string`, `==`)
/// and for string and YAML representations.
class Variant {
public:
  Variant();

  Variant(const AsyncRT::AnyAsyncValueRef &other);
  Variant(const Variant &other);
  Variant(const DenseMapInfoKey &info);

  /// Returns the string representation of the wrapped `AnyAsyncValueRef`.
  std::string to_string() const;

  DenseMapInfoKey getInfo() const;

  /// Returns a reference to the wrapped `AnyAsyncValueRef`.
  const AsyncRT::AnyAsyncValueRef &getValue() const;

  /// Copies the value ref, incrementing the ref count.
  Variant &operator=(const Variant &other);
  Variant &operator=(Variant &&other);

private:
  DenseMapInfoKey info;
  AsyncRT::AnyAsyncValueRef value;
};

/// Defines equality on the underlying `AnyAsyncValueRef`s. See `Value.h` for
/// details.
bool operator==(const Variant &lhs, const Variant &rhs);

/// Defines inequality on the underlying `AnyAsyncValueRef`s. See `Value.h` for
/// details.
bool operator!=(const Variant &lhs, const Variant &rhs);

/// Defines the less-than operator on the underlying `AnyAsyncValueRef`s. See
/// `Value.h` for details.
bool operator<(const Variant &lhs, const Variant &rhs);

/// Writes the string representation of `Variant` to a stream.
raw_ostream &operator<<(raw_ostream &os, const Variant &variant);
std::ostream &operator<<(std::ostream &os, const Variant &variant);

/// Prints (nested) values; leaf tensors use `Kernels::printTensor`.
ErrorOrSuccess printGenericValue(raw_ostream &os,
                                 const AsyncRT::AnyAsyncValueRef &value,
                                 ResultOutputStyle style, unsigned precision);

/// Stringifies a supported composed type or tensor. Not `to_string` because
/// it isn't generic.
std::string valueToString(const AsyncRT::AnyAsyncValueRef &value);

/// PyTorch/Python-like equality: containers by content, tensors by buffer
/// pointer. Not `operator==` because it isn't generic.
bool valueEqual(const AsyncRT::AnyAsyncValueRef &lhs,
                const AsyncRT::AnyAsyncValueRef &rhs);

/// PyTorch/Python-like less-than: containers by content, tensors by buffer
/// pointer. Not `operator<` because it isn't generic.
bool valueLessThan(const AsyncRT::AnyAsyncValueRef &lhs,
                   const AsyncRT::AnyAsyncValueRef &rhs);

/// Hashes containers by content and tensors by buffer pointer; unsupported
/// types hash to `std::numeric_limits<size_t>::max()`. Not a hash-struct
/// specialization because it isn't generic.
size_t valueHash(const AsyncRT::AnyAsyncValueRef &value);

/// Golden-ratio hash combine.
size_t hashCombine(size_t seed, size_t value);

} // namespace M::GML

namespace std {

std::string to_string(const M::GML::DenseMapInfoKey &value);

/// Computes the hash for `None`, which is constant 0.
template <>
struct hash<M::GML::None> {
  size_t operator()(const M::GML::None &none) const { return 0; }
};

/// Returns the string representation "<none>".
std::string to_string(const M::GML::None &none);

/// Computes the hash function of the underlying `AnyAsyncValueRef`.
template <>
struct hash<M::GML::Variant> {
  size_t operator()(const M::GML::Variant &value) const;
};

/// Delegates to the individual key types.
std::string to_string(const M::GML::Variant &variant);

} // namespace std

#endif // GENERICML_SUPPORT_VALUE_H
