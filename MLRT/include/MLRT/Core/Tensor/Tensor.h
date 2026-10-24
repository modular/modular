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

#ifndef MLRT_CORE_TENSOR_TENSOR_H
#define MLRT_CORE_TENSOR_TENSOR_H

#include "AsyncRT/Runtime/CompactCPUDevicePtr.h"
#include "MLRT/Core/Tensor/TensorBufferRef.h"
#include "Support/ML/DebugPrint.h"
#include "Support/ML/TensorBase.h"
#include "Support/ML/TensorSpec.h"

namespace M::MLRT {

//===----------------------------------------------------------------------===//
// Tensor Printing
//===----------------------------------------------------------------------===//

/// Tensors in CPU memory using a dense, row-major element order.
///
/// Tensors are just a pair of a TensorBufferRef and a TensorSpec. Tensors are
/// movable and explicitly deep copyable.
///
/// CAUTION: Though movable, this is 48 bytes on 64-bit systems so best not
///          to consider it a simple handle.
class Tensor : public TensorBase<TensorBufferRef> {
public:
  using TensorBase::TensorBase;

  //===--------------------------------------------------------------------===//
  // Creation methods.
  //===--------------------------------------------------------------------===//

  /// Creates a tensor whose buffer is allocated from the allocator in
  /// cpuDevice with size appropriate for spec.
  ///
  /// TODO(#5799): We're requiring all callers to be explicit about alignment
  /// (or lack thereof) to help track down 'forgotten alignment' problems.
  static Tensor createOwned(TensorSpec spec,
                            AsyncRT::CompactCPUDevicePtr cpuDevice,
                            std::optional<size_t> alignment);

  /// Creates an owned tensor whose buffer is from a taken void* pointer -- This
  /// taken pointer is allocated by caller and taken to construct the underlying
  /// bufferRef which has control over the lifetime. There is an assumption that
  /// the takeFrom pointer is allocated by caller via cpuDevice->getAllocator().
  /// Lifetime of the allocated memory follows any Owned Tensor (via bufferRef)
  static Tensor takeOwned(TensorSpec spec,
                          AsyncRT::CompactCPUDevicePtr cpuDevice,
                          void *takeFrom);

  /// Creates a tensor whose buffer is at the given pointer. It is assumed the
  /// pointer has size appropriate for spec and lifetime exceeding that of
  /// the result tensor. The alignment will be used should the tensor need
  /// to be deep copied.
  static Tensor createBorrowed(void *buffer, TensorSpec spec,
                               std::optional<size_t> alignment);
  static Tensor createBorrowed(const void *buffer, TensorSpec spec,
                               std::optional<size_t> alignment);

  //===--------------------------------------------------------------------===//
  // Core APIs for working with a Tensor.
  //===--------------------------------------------------------------------===//

  /// Returns a pointer to the underlying buffer.
  void *getMutableBuffer() { return bufferRef.getBuffer(); }
  const void *getBuffer() const { return bufferRef.getBuffer(); }
  template <typename T>
  T *getMutableBuffer() {
    return static_cast<T *>(getMutableBuffer());
  }
  template <typename T>
  const T *getBuffer() const {
    return static_cast<const T *>(getBuffer());
  }

  /// Release the ownership of the underlying buffer to the returned reference.
  /// This should be only happen when the tensor will be immediately destroyed.
  TensorBufferRef releaseBufferRef() { return std::move(bufferRef); }

  /// Simple accessor to the buffer ref. To maintain the value semantics of this
  /// tensor it should not be modified.
  TensorBufferRef &getBufferRef() { return bufferRef; }

  /// Returns a shallow copy of this tensor, with the underlying buffer borrowed
  /// from this tensor. This tensor may owning or itself be borrowing the
  /// buffer, and the underlying buffer must outlive the returned tensor. The
  /// result will have the given borrowLifetime.
  Tensor borrow() const;

  /// This prints the specified tensor data to stdout.
  ErrorOrSuccess print(ResultOutputStyle style, unsigned precision = 6) const;

  /// This prints the specified tensor data to stream.
  ErrorOrSuccess print(raw_ostream &os, ResultOutputStyle style,
                       unsigned precision = 6) const;

  /// This prints a debug representation of the specified tensor data to stream.
  void printDebug(raw_ostream &os) const;
};

inline raw_ostream &operator<<(raw_ostream &os, const Tensor &value) {
  (void)value.print(os, ResultOutputStyle::kFull);
  return os;
}

} // namespace M::MLRT

#endif // MLRT_CORE_TENSOR_TENSOR_H
