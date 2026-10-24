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

#ifndef GENERICML_SUPPORT_COMPILEDKERNELSCONTEXT_H
#define GENERICML_SUPPORT_COMPILEDKERNELSCONTEXT_H

#include "Mojo/Compiler/ObjectCompiler.h"
#include "Mojo/ExecutionEngine/ExecutionEngine.h"
#include "Support/AssertStream.h"
#include "llvm/ADT/StringExtras.h"

namespace M::GML {

//===----------------------------------------------------------------------===//
// CompiledKernelCall
//===----------------------------------------------------------------------===//

// Data structures supporting the MOGG/MGP ABI
namespace CompiledKernelABI {

// ABI version of GML::List. Layout must match the Mojo `ABI_List`.
struct ABIList {
  size_t numElems;
  void *elements; // Currently always points to ABITensors

  ABIList(size_t numElems, void *elements)
      : numElems(numElems), elements(elements) {}
};

// ABI version of Tensor. Layout must match the Mojo `ABI_Tensor`.
struct ABITensor {
  ssize_t *dims; // Shape, with implied rank
  void *dataPtr; // The underlying buffer

  ABITensor(ssize_t *dims, void *dataPtr) : dims(dims), dataPtr(dataPtr) {}
};

/// Holds marshalled data until the ABI call returns; the called kernel must
/// unmarshal it before returning, even if asynchronous.
///
/// TODO: Remove once values have a unified representation across the ABI.
struct CompiledKernelCall {
  /// The accumulated pointer arguments.
  SmallVector<void *, 8> inputPointers;

  // `inputPointers` and the other memory vectors point into these, so the
  // pointees must stay put when a vector grows: hence `std::vector<T>` and
  // `std::unique_ptr<T>` rather than `SmallVector<T>` and `T`.

  /// ABI Shapes, in the form of arrays of ssize_ts.
  SmallVector<std::vector<ssize_t>> dimsMemory;
  /// ABI Tensors, which combine raw buffers with shapes.
  SmallVector<std::vector<ABITensor>> tensorsMemory;
  /// ABI Lists, which hold pointers to arbitrary marshalled objects.
  SmallVector<std::unique_ptr<ABIList>> listMemory;

  CompiledKernelCall(size_t numPointers) { inputPointers.reserve(numPointers); }

  /// Pushes the marshalled form of `input` onto the pointers vector, adding
  /// intermediate data to the fields above. Errors on unsupported input.
  ErrorOrSuccess pushObject(AnyAsyncValueRef input, char inOutType,
                            StringRef summary, const EncodedLocation &loc);
};

} // namespace CompiledKernelABI

// This is the ABI for calling a compiled kernel:
//   I = number of 'in' arguments
//   O = number of 'out' tensor buffers
//   N = number of 'in' dimension values (for dynamic dims only)
//   M = number of 'out' dimension values (for compiled shape functions only)
//   void* args[I + O], ssize_t dims[N + M], Runtime*
//
// Both args and dims take read-only 'in' args followed by write-only 'out'
// args. A tensor is passed as its buffer (in args) plus one entry per dynamic
// dimension (in dims); static dims are baked into the kernel. Only static
// ranks are supported, so no ranks are passed.
//
// Each void* arg is either:
//  - A TensorBufferRef's buffer (for `!mgp.buffer`s), kept alive by the C++
//    runtime, possibly by moving references into a sub-task closure.
//  - A `CompiledKernelABI::ABIList*` (for `!mgp.list`s), owned by the
//    `CompiledKernelCall`; the kernel shim must unmarshal it before returning.
//
// If the called kernel is asynchronous then the C++ runtime will keep all
// operands alive until the kernel's coroutine completes.
using CompiledKernelFn = void(void **, ssize_t *);

// Alternate ABI taking a list of async value pointers that the kernel unpacks
// itself.
using CompiledKernelAsyncABIFn = void(void **);
//===----------------------------------------------------------------------===//
// CompiledKernelsContext
//===----------------------------------------------------------------------===//

// Holds all loaded compiled kernels, in a state slot of the mgp.context.
class CompiledKernelsContext : public ReferenceCounted<CompiledKernelsContext> {
public:
  CompiledKernelsContext(KGEN::ExecutionEngine *engine, bool loadDebuginfo)
      : engine(engine), loadDebuginfo(loadDebuginfo) {}

  ~CompiledKernelsContext() = default;

  /// Loads `binary`'s kernels, retrievable by their index in `kernelNames`.
  ErrorOrSuccess load(ArrayRef<uint8_t> binary, ArrayRef<StringRef> kernelNames,
                      size_t kernelIndex = 0);

  /// Returns the compiled kernel at the given index.
  CompiledKernelFn *getKernel(size_t kernelIndex) const {
    ASSERT_STREAM(kernelIndex < kernels.size(),
                  << "invalid kernel index " << kernelIndex);
    return reinterpret_cast<CompiledKernelFn *>(kernels[kernelIndex]);
  }

  CompiledKernelAsyncABIFn *getKernelAsyncABI(size_t kernelIndex) const {
    ASSERT_STREAM(kernelIndex < kernels.size(),
                  << "invalid kernel index " << kernelIndex);
    return reinterpret_cast<CompiledKernelAsyncABIFn *>(kernels[kernelIndex]);
  }

  /// Returns the symbol name for the compiled kernel at the given index.
  StringRef getKernelName(size_t kernelIndex) const {
    ASSERT_STREAM(kernelIndex < kernels.size(),
                  << "invalid kernel index " << kernelIndex);
    return symbolTable[kernelIndex];
  }

private:
  void addKernel(StringRef name, size_t index, void *kernel);

  /// Borrowed from the runtime. Must outlive the MGP Context holding this
  /// object: AsyncRT tasks running loaded code may not finish until the
  /// Runtime's WorkQueue shuts down.
  KGEN::ExecutionEngine *engine = nullptr;

  bool loadDebuginfo;

  SmallVector<void *, 16> kernels;

  // For debugging and profiling only.
  SmallVector<StringRef, 16> symbolTable;
};

} // namespace M::GML

#endif // GENERICML_SUPPORT_COMPILEDKERNELSCONTEXT_H
